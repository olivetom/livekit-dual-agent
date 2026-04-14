"""
Centralised configuration loaded from environment variables.
Copy .env.example to .env and fill in your credentials.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── LiveKit ────────────────────────────────────────────────────────────
    LIVEKIT_URL: str = os.environ["LIVEKIT_URL"]
    LIVEKIT_API_KEY: str = os.environ["LIVEKIT_API_KEY"]
    LIVEKIT_API_SECRET: str = os.environ["LIVEKIT_API_SECRET"]

    # ── OpenAI  (foreground agent – fast STT/LLM/TTS) ─────────────────────
    OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
    # Small, fast model for the real-time foreground voice turn
    FOREGROUND_LLM_MODEL: str = os.getenv("FOREGROUND_LLM_MODEL", "gpt-4o-mini")

    # ── Anthropic  (background agent – deep reasoning + tool calls) ────────
    ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
    BACKGROUND_LLM_MODEL: str = os.getenv(
        "BACKGROUND_LLM_MODEL", "claude-sonnet-4-6"
    )

    # ── Deepgram STT ───────────────────────────────────────────────────────
    DEEPGRAM_API_KEY: str = os.environ["DEEPGRAM_API_KEY"]

    # ── ElevenLabs TTS ────────────────────────────────────────────────────
    ELEVENLABS_API_KEY: str = os.environ["ELEVENLABS_API_KEY"]
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

    # ── Background agent timeout (seconds) ────────────────────────────────
    BACKGROUND_TIMEOUT: float = float(os.getenv("BACKGROUND_TIMEOUT", "8.0"))

    # ── FastAPI ────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


cfg = Config()
