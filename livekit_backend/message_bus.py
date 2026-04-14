"""
Lightweight in-process message bus that bridges the foreground voice agent
and the background reasoning agent running in the same worker process.

Design
------
Each LiveKit room gets one AgentSession.  The ForegroundAgent publishes
"user utterances" onto the bus; the BackgroundAgent subscribes, processes
them (guardrails, tool calls, deep reasoning) and writes back an enriched
"BackgroundResult" that the ForegroundAgent can optionally inject into its
next system-prompt update.

All communication is async-safe and uses asyncio primitives so it works
naturally inside LiveKit's event loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class UserUtterance:
    """Published by the ForegroundAgent when the user finishes a turn."""
    room_name: str
    participant_identity: str
    text: str
    timestamp: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass
class BackgroundResult:
    """Published by the BackgroundAgent after it has processed an utterance."""
    room_name: str
    original_utterance: str
    # Structured data the background agent wants to surface
    reasoning_summary: str          # Short paragraph of its chain-of-thought
    guardrail_passed: bool          # False if the request violated a policy
    guardrail_reason: str           # Human-readable reason when blocked
    tool_outputs: list[dict]        # Results from any tool calls
    suggested_context_update: str   # Injected into FG system-prompt on next turn
    timestamp: float = dataclasses.field(default_factory=time.time)


BackgroundHandler = Callable[[UserUtterance], Awaitable[None]]


class AgentMessageBus:
    """
    One bus instance per room.  Fore- and background agents share a reference.

      fg.publish_utterance(utt)     → background handler invoked
      bg.publish_result(result)     → foreground handler invoked
    """

    def __init__(self, room_name: str) -> None:
        self.room_name = room_name
        self._utt_handlers: list[BackgroundHandler] = []
        self._result_handlers: list[Callable[[BackgroundResult], Awaitable[None]]] = []
        # Latest enriched context from the background agent
        self._latest_result: BackgroundResult | None = None
        self._result_event: asyncio.Event = asyncio.Event()

    # ── Subscription ────────────────────────────────────────────────────────

    def subscribe_utterances(self, handler: BackgroundHandler) -> None:
        self._utt_handlers.append(handler)

    def subscribe_results(
        self, handler: Callable[[BackgroundResult], Awaitable[None]]
    ) -> None:
        self._result_handlers.append(handler)

    # ── Publishing ──────────────────────────────────────────────────────────

    async def publish_utterance(self, utt: UserUtterance) -> None:
        logger.debug("[bus] utterance → background | room=%s", self.room_name)
        for h in self._utt_handlers:
            asyncio.create_task(h(utt))

    async def publish_result(self, result: BackgroundResult) -> None:
        logger.debug(
            "[bus] background result → foreground | room=%s guardrail_passed=%s",
            self.room_name,
            result.guardrail_passed,
        )
        self._latest_result = result
        self._result_event.set()
        for h in self._result_handlers:
            asyncio.create_task(h(result))

    # ── Polling helper used by ForegroundAgent ───────────────────────────────

    async def wait_for_result(self, timeout: float) -> BackgroundResult | None:
        """Wait up to *timeout* seconds for a fresh BackgroundResult."""
        self._result_event.clear()
        try:
            await asyncio.wait_for(self._result_event.wait(), timeout=timeout)
            return self._latest_result
        except asyncio.TimeoutError:
            return None


class MessageBusRegistry:
    """Process-wide registry: one bus per room name."""

    def __init__(self) -> None:
        self._buses: dict[str, AgentMessageBus] = {}

    def get_or_create(self, room_name: str) -> AgentMessageBus:
        if room_name not in self._buses:
            self._buses[room_name] = AgentMessageBus(room_name)
            logger.info("[registry] created bus for room=%s", room_name)
        return self._buses[room_name]

    def remove(self, room_name: str) -> None:
        self._buses.pop(room_name, None)


# Module-level singleton
registry = MessageBusRegistry()
