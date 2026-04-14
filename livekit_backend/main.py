"""
main.py – LiveKit worker entrypoint + FastAPI HTTP server
=========================================================

Two server roles run in the same process:

1. **LiveKit Agent Worker** (`livekit.agents.WorkerOptions`)
   Connects to the LiveKit server, receives room-dispatch events, and
   spins up a ForegroundAgent + BackgroundAgent pair per room.

2. **FastAPI HTTP server** (port 8000 by default)
   Exposes:
   - POST /token   → generate a LiveKit access token for a web client
   - GET  /health  → simple liveness probe

Run locally:
    python main.py dev          # LiveKit dev mode (auto-reconnect)
    python main.py start        # production worker

Or via uvicorn for the HTTP server only (no worker):
    uvicorn main:http_app --reload
"""

from __future__ import annotations

import logging
import secrets
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LiveKit Agents SDK
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.api import AccessToken, VideoGrants

# Project modules
from agents.background_agent import BackgroundAgent
from agents.foreground_agent import ForegroundAgent
from config import cfg
from message_bus import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── LiveKit entrypoint ────────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    """
    Called by the LiveKit worker once per dispatched room.

    Boots both agents and wires them through the shared message bus.
    """
    room_name = ctx.room.name
    logger.info("[worker] room connected: %s", room_name)

    try:
        # 1. Start background agent (registers bus subscriptions)
        bg = BackgroundAgent(room_name)
        bg.start()

        # 2. Start foreground voice agent (connects to room, says hello)
        fg = ForegroundAgent(ctx)
        await fg.start()

    finally:
        # Clean up bus when the room ends
        registry.remove(room_name)
        logger.info("[worker] room disconnected: %s", room_name)


# ── FastAPI HTTP app ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HTTP server started on %s:%s", cfg.HOST, cfg.PORT)
    yield
    logger.info("HTTP server shutting down")


http_app = FastAPI(title="LiveKit Dual-Agent Backend", lifespan=lifespan)

http_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


class TokenRequest(BaseModel):
    room_name: str
    participant_identity: str | None = None


class TokenResponse(BaseModel):
    token: str
    livekit_url: str
    room_name: str
    identity: str


@http_app.post("/token", response_model=TokenResponse)
async def create_token(req: TokenRequest) -> TokenResponse:
    """
    Generate a short-lived LiveKit access token for a browser/mobile client.

    The token grants:
    - canPublish  (microphone)
    - canSubscribe (agent audio)
    - canPublishData
    """
    identity = req.participant_identity or f"user-{secrets.token_hex(4)}"

    token = (
        AccessToken(cfg.LIVEKIT_API_KEY, cfg.LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=req.room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )

    return TokenResponse(
        token=token,
        livekit_url=cfg.LIVEKIT_URL,
        room_name=req.room_name,
        identity=identity,
    )


@http_app.get("/health")
async def health() -> dict:
    return {"status": "ok", "timestamp": time.time()}


# ── CLI entry (LiveKit worker) ────────────────────────────────────────────────

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Worker connects to LiveKit server and waits for room dispatches
            api_key=cfg.LIVEKIT_API_KEY,
            api_secret=cfg.LIVEKIT_API_SECRET,
            ws_url=cfg.LIVEKIT_URL,
        )
    )
