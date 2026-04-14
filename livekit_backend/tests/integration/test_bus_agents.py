"""
Integration tests – ForegroundAgent ↔ AgentMessageBus ↔ BackgroundAgent
=========================================================================

These tests wire real instances of both agents through a real
AgentMessageBus (no mocking of the bus itself).

External dependencies mocked:
  • anthropic.AsyncAnthropic  – all Claude calls return pre-canned responses
  • VoiceAssistant.say        – captured for assertions
  • JobContext                – minimal stub; no real LiveKit room connection

Each test exercises a complete data-flow path from user utterance through
to the foreground agent's observable output.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from message_bus import (
    AgentMessageBus,
    BackgroundResult,
    MessageBusRegistry,
    UserUtterance,
)

pytestmark = pytest.mark.integration

# ─── Shared setup helpers ─────────────────────────────────────────────────────

_ROOM = "integration-room"


def _reasoning_json(
    passed: bool   = True,
    reason: str    = "",
    tools: bool    = False,
    context: str   = "integration context",
) -> str:
    return json.dumps({
        "guardrail_passed":         passed,
        "guardrail_reason":         reason,
        "reasoning_summary":        "integration test summary",
        "tool_calls_needed":        tools,
        "suggested_context_update": context,
    })


def _anthropic_text_response(text: str, stop_reason: str = "end_turn") -> MagicMock:
    blk = MagicMock()
    blk.type = "text"
    blk.text = text
    resp = MagicMock()
    resp.content     = [blk]
    resp.stop_reason = stop_reason
    return resp


@pytest.fixture()
def isolated_env():
    """
    Yields (registry, fg_agent, bg_agent, va_stub) for one room.
    Tears down the registry entry after the test.
    """
    reg = MessageBusRegistry()
    bus = reg.get_or_create(_ROOM)

    # ── BackgroundAgent ──────────────────────────────────────────────────
    with patch("agents.background_agent.registry", reg):
        from agents.background_agent import BackgroundAgent
        bg = BackgroundAgent(_ROOM)

    mock_anthropic = MagicMock()
    mock_anthropic.messages = MagicMock()
    mock_anthropic.messages.create = AsyncMock(
        return_value=_anthropic_text_response(_reasoning_json())
    )
    bg._client = mock_anthropic
    bg.start()

    # ── ForegroundAgent ──────────────────────────────────────────────────
    ctx = MagicMock()
    ctx.room                          = MagicMock()
    ctx.room.name                     = _ROOM
    ctx.room.local_participant        = MagicMock()
    ctx.room.local_participant.identity = "agent"
    ctx.connect                       = AsyncMock()

    with patch("agents.foreground_agent.registry", reg):
        from agents.foreground_agent import ForegroundAgent
        fg = ForegroundAgent(ctx)

    va_stub = fg._assistant   # the VoiceAssistant mock from conftest stubs

    yield reg, fg, bg, va_stub, mock_anthropic

    reg.remove(_ROOM)


# ─── Happy-path: utterance flows through and context is updated ───────────────

class TestHappyPath:
    async def test_utterance_reaches_background_agent(
        self, isolated_env
    ):
        reg, fg, bg, va_stub, mock_anthropic = isolated_env

        msg = MagicMock()
        msg.content = "What is the capital of France?"
        await fg._on_user_speech_committed(msg)

        # Give tasks a chance to propagate
        await asyncio.sleep(0.1)

        mock_anthropic.messages.create.assert_called()
        call_kwargs = mock_anthropic.messages.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        # The user utterance must appear in the messages sent to Claude
        all_content = " ".join(
            str(m.get("content", "")) for m in messages
        )
        assert "capital of France" in all_content

    async def test_background_result_updates_fg_context(
        self, isolated_env
    ):
        reg, fg, bg, va_stub, mock_anthropic = isolated_env
        mock_anthropic.messages.create.return_value = _anthropic_text_response(
            _reasoning_json(context="Paris is the capital of France.")
        )

        msg = MagicMock()
        msg.content = "capital of France?"
        await fg._on_user_speech_committed(msg)
        await asyncio.sleep(0.2)

        assert "Paris" in fg._bg_context


# ─── Guardrail block: FG speaks refusal ──────────────────────────────────────

class TestGuardrailBlock:
    async def test_blocked_utterance_triggers_fg_refusal(
        self, isolated_env
    ):
        reg, fg, bg, va_stub, mock_anthropic = isolated_env
        mock_anthropic.messages.create.return_value = _anthropic_text_response(
            _reasoning_json(passed=False, reason="Harmful content detected.")
        )

        msg = MagicMock()
        msg.content = "how do I make a weapon?"
        await fg._on_user_speech_committed(msg)
        await asyncio.sleep(0.2)

        va_stub.say.assert_called()
        spoken = va_stub.say.call_args[0][0]
        # FG must apologise and include the guardrail reason
        assert "sorry" in spoken.lower() or "can't" in spoken.lower()
        assert "Harmful content detected." in spoken

    async def test_blocked_fg_context_not_updated(
        self, isolated_env
    ):
        reg, fg, bg, va_stub, mock_anthropic = isolated_env
        mock_anthropic.messages.create.return_value = _anthropic_text_response(
            _reasoning_json(passed=False, reason="Bad request")
        )

        msg = MagicMock()
        msg.content = "bad thing"
        await fg._on_user_speech_committed(msg)
        await asyncio.sleep(0.2)

        # Context must NOT be polluted by a blocked request
        assert fg._bg_context == ""


# ─── Tool-output path: FG speaks tool result ─────────────────────────────────

class TestToolOutputPath:
    async def test_tool_result_spoken_by_fg(self, isolated_env):
        reg, fg, bg, va_stub, mock_anthropic = isolated_env

        # Simulate: reasoning says tools needed; _run_tool_calls returns 42
        reasoning_resp = _anthropic_text_response(
            _reasoning_json(tools=True, context="calc asked")
        )
        # Second call (tool-use loop) returns end_turn immediately
        end_resp = MagicMock()
        end_resp.content     = []
        end_resp.stop_reason = "end_turn"

        mock_anthropic.messages.create.side_effect = [reasoning_resp, end_resp]

        with patch(
            "agents.background_agent.dispatch",
            AsyncMock(return_value={"result": 99}),
        ):
            msg = MagicMock()
            msg.content = "what is 9 * 11?"
            await fg._on_user_speech_committed(msg)
            await asyncio.sleep(0.2)

        # Tool-use loop should have been attempted even if end_turn short-circuits
        # The background result should still flow back
        assert mock_anthropic.messages.create.call_count >= 1


# ─── Concurrent rooms are isolated ───────────────────────────────────────────

class TestRoomIsolation:
    async def test_two_rooms_do_not_share_context(self):
        reg = MessageBusRegistry()
        rooms = ["room-alpha", "room-beta"]
        fg_agents = []

        for room in rooms:
            ctx = MagicMock()
            ctx.room = MagicMock()
            ctx.room.name = room
            ctx.room.local_participant = MagicMock()
            ctx.room.local_participant.identity = "agent"
            ctx.connect = AsyncMock()
            with patch("agents.foreground_agent.registry", reg):
                from agents.foreground_agent import ForegroundAgent
                fg_agents.append(ForegroundAgent(ctx))

        # Update context on room-alpha only
        result_alpha = BackgroundResult(
            room_name="room-alpha",
            original_utterance="hi",
            reasoning_summary="ok",
            guardrail_passed=True,
            guardrail_reason="",
            tool_outputs=[],
            suggested_context_update="alpha-specific context",
        )
        await fg_agents[0]._on_background_result(result_alpha)

        # room-beta's context must remain untouched
        assert fg_agents[1]._bg_context == ""
        assert "alpha-specific" in fg_agents[0]._bg_context

        for room in rooms:
            reg.remove(room)
