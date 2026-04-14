"""
Unit tests – agents/foreground_agent.py
==========================================

Tests focus on the pure-logic parts that don't need a live LiveKit room:
  • _build_system_prompt()   – context injection into the system prompt
  • _on_background_result()  – guardrail refusal, context update, tool-output TTS
  • _on_user_speech_committed() – utterance published onto the bus

The VoiceAssistant, STT, TTS and VAD are all provided by the stubs
injected in tests/conftest.py; no real audio I/O happens.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from message_bus import (
    AgentMessageBus,
    BackgroundResult,
    MessageBusRegistry,
    UserUtterance,
)

pytestmark = pytest.mark.unit


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_fg(room_name: str = "fg-unit-room"):
    """
    Build a ForegroundAgent with an isolated registry so tests are
    independent.  The VoiceAssistant stub (from conftest) records .say() calls.
    """
    reg = MessageBusRegistry()

    ctx = MagicMock()
    ctx.room       = MagicMock()
    ctx.room.name  = room_name
    ctx.room.local_participant       = MagicMock()
    ctx.room.local_participant.identity = "agent"
    ctx.connect    = AsyncMock()

    with patch("agents.foreground_agent.registry", reg):
        from agents.foreground_agent import ForegroundAgent
        fg = ForegroundAgent(ctx)

    return fg, reg


def _make_bg_result(
    room: str = "fg-unit-room",
    passed: bool = True,
    reason: str  = "",
    context: str = "",
    tools: list  | None = None,
) -> BackgroundResult:
    return BackgroundResult(
        room_name=room,
        original_utterance="test utterance",
        reasoning_summary="ok",
        guardrail_passed=passed,
        guardrail_reason=reason,
        tool_outputs=tools or [],
        suggested_context_update=context,
    )


# ─── _build_system_prompt ────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    def test_empty_context_has_no_background_block(self):
        from agents.foreground_agent import _build_system_prompt
        prompt = _build_system_prompt()
        assert "[Background context]" not in prompt

    def test_with_context_includes_background_block(self):
        from agents.foreground_agent import _build_system_prompt
        prompt = _build_system_prompt("The user is a premium subscriber.")
        assert "[Background context]" in prompt
        assert "premium subscriber" in prompt

    def test_empty_string_treated_as_no_context(self):
        from agents.foreground_agent import _build_system_prompt
        prompt = _build_system_prompt("")
        assert "[Background context]" not in prompt

    def test_base_instructions_always_present(self):
        from agents.foreground_agent import _build_system_prompt
        prompt = _build_system_prompt()
        assert "concise" in prompt.lower() or "short" in prompt.lower()


# ─── _on_background_result – guardrail blocking ──────────────────────────────

class TestOnBackgroundResultGuardrail:
    async def test_guardrail_block_triggers_say(self):
        fg, _ = _make_fg()
        result = _make_bg_result(passed=False, reason="violence detected")

        await fg._on_background_result(result)

        fg._assistant.say.assert_called_once()
        spoken: str = fg._assistant.say.call_args[0][0]
        assert "can't help" in spoken.lower() or "sorry" in spoken.lower()

    async def test_guardrail_block_includes_reason(self):
        fg, _ = _make_fg()
        result = _make_bg_result(passed=False, reason="phishing attempt")

        await fg._on_background_result(result)

        spoken = fg._assistant.say.call_args[0][0]
        assert "phishing attempt" in spoken

    async def test_guardrail_pass_does_not_speak_refusal(self):
        fg, _ = _make_fg()
        result = _make_bg_result(passed=True)

        await fg._on_background_result(result)

        # No call to .say() when guardrail passes and no tool outputs
        fg._assistant.say.assert_not_called()


# ─── _on_background_result – context update ──────────────────────────────────

class TestOnBackgroundResultContextUpdate:
    async def test_context_stored_on_fg(self):
        fg, _ = _make_fg()
        result = _make_bg_result(context="user prefers metric units")

        await fg._on_background_result(result)

        assert fg._bg_context == "user prefers metric units"

    async def test_system_prompt_message_updated(self):
        fg, _ = _make_fg()
        # Ensure there is at least one message to update
        assert len(fg._chat_ctx.messages) >= 1

        result = _make_bg_result(context="some new context")
        await fg._on_background_result(result)

        # The first message should now contain the new context
        first_msg = fg._chat_ctx.messages[0]
        assert "some new context" in first_msg.content

    async def test_empty_context_update_does_not_speak(self):
        fg, _ = _make_fg()
        result = _make_bg_result(context="")

        await fg._on_background_result(result)

        fg._assistant.say.assert_not_called()


# ─── _on_background_result – tool outputs ────────────────────────────────────

class TestOnBackgroundResultToolOutputs:
    async def test_tool_output_triggers_say(self):
        fg, _ = _make_fg()
        result = _make_bg_result(
            tools=[{"tool": "calculate", "result": {"result": 42}}]
        )

        await fg._on_background_result(result)

        fg._assistant.say.assert_called_once()
        spoken = fg._assistant.say.call_args[0][0]
        assert "42" in spoken or "result" in spoken.lower()

    async def test_multiple_tool_outputs_all_included(self):
        fg, _ = _make_fg()
        result = _make_bg_result(
            tools=[
                {"tool": "calculate",     "result": {"result": 10}},
                {"tool": "get_current_time", "result": {"utc_time": "2025-01-01T00:00:00Z"}},
            ]
        )

        await fg._on_background_result(result)

        spoken = fg._assistant.say.call_args[0][0]
        assert "10" in spoken
        assert "2025" in spoken


# ─── _on_user_speech_committed – utterance published ─────────────────────────

class TestOnUserSpeechCommitted:
    async def test_utterance_published_to_bus(self):
        fg, reg = _make_fg()
        bus: AgentMessageBus = reg.get_or_create("fg-unit-room")

        received: list[UserUtterance] = []

        async def capture(utt): received.append(utt)
        bus.subscribe_utterances(capture)

        msg = MagicMock()
        msg.content = "what time is it?"
        await fg._on_user_speech_committed(msg)
        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0].text == "what time is it?"

    async def test_utterance_room_name_matches(self):
        fg, reg = _make_fg(room_name="special-room")
        bus: AgentMessageBus = reg.get_or_create("special-room")
        received: list[UserUtterance] = []

        async def capture(utt): received.append(utt)
        bus.subscribe_utterances(capture)

        msg = MagicMock()
        msg.content = "hello"
        await fg._on_user_speech_committed(msg)
        await asyncio.sleep(0)

        assert received[0].room_name == "special-room"

    async def test_empty_content_still_published(self):
        fg, reg = _make_fg()
        bus = reg.get_or_create("fg-unit-room")
        received: list[UserUtterance] = []
        async def capture(utt): received.append(utt)
        bus.subscribe_utterances(capture)

        msg = MagicMock()
        msg.content = ""
        await fg._on_user_speech_committed(msg)
        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0].text == ""
