"""
Unit tests – agents/background_agent.py
=========================================

The Anthropic client is replaced by an AsyncMock so no real API calls are
made.  Tests cover:
  • Fallback result construction (_timeout_result, _error_result)
  • JSON parsing / markdown-fence stripping in _reason()
  • Guardrail block path in _process()
  • Tool-call accumulation path in _process()
  • Conversation history trimming
  • Timeout and exception handling in _handle_utterance()
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

pytestmark = pytest.mark.unit

# ── Helpers shared across test classes ───────────────────────────────────────

def _make_registry_and_agent(room: str = "unit-room"):
    """
    Build an isolated registry so global state is never touched,
    create and start a BackgroundAgent wired to it.
    """
    reg = MessageBusRegistry()

    # Patch the module-level singleton used by BackgroundAgent.__init__
    with patch("agents.background_agent.registry", reg):
        from agents.background_agent import BackgroundAgent
        agent = BackgroundAgent(room)

    # Replace the real Anthropic client with an AsyncMock
    agent._client = MagicMock()
    agent._client.messages = MagicMock()
    agent._client.messages.create = AsyncMock()
    return reg, agent


def _make_utt(room: str = "unit-room", text: str = "hello") -> UserUtterance:
    return UserUtterance(
        room_name=room,
        participant_identity="tester",
        text=text,
    )


def _text_block(text: str) -> MagicMock:
    blk = MagicMock()
    blk.type = "text"
    blk.text = text
    return blk


def _tool_use_block(name: str, tid: str, inputs: dict) -> MagicMock:
    blk = MagicMock()
    blk.type  = "tool_use"
    blk.name  = name
    blk.id    = tid
    blk.input = inputs
    return blk


def _anthropic_resp(blocks: list, stop_reason: str = "end_turn") -> MagicMock:
    resp = MagicMock()
    resp.content     = blocks
    resp.stop_reason = stop_reason
    return resp


def _reasoning_payload(
    *,
    passed: bool    = True,
    reason: str     = "",
    summary: str    = "all good",
    tools: bool     = False,
    context: str    = "ctx",
) -> dict:
    return {
        "guardrail_passed":         passed,
        "guardrail_reason":         reason,
        "reasoning_summary":        summary,
        "tool_calls_needed":        tools,
        "suggested_context_update": context,
    }


# ─── Fallback result factories ────────────────────────────────────────────────

class TestFallbackResults:
    def test_timeout_result_fields(self):
        _, agent = _make_registry_and_agent()
        utt = _make_utt(text="slow query")
        r = agent._timeout_result(utt)
        assert isinstance(r, BackgroundResult)
        assert r.guardrail_passed is True
        assert r.room_name == "unit-room"
        assert r.original_utterance == "slow query"
        assert "timed out" in r.reasoning_summary.lower()
        assert r.tool_outputs == []

    def test_error_result_fields(self):
        _, agent = _make_registry_and_agent()
        utt = _make_utt(text="boom")
        r = agent._error_result(utt)
        assert isinstance(r, BackgroundResult)
        assert r.guardrail_passed is True
        assert "error" in r.reasoning_summary.lower()


# ─── _reason() – JSON parsing ────────────────────────────────────────────────

class TestReason:
    async def _call_reason(self, agent, payload, raw: str | None = None):
        """Set the mock response and call _reason()."""
        text = raw if raw is not None else json.dumps(payload)
        agent._client.messages.create.return_value = _anthropic_resp(
            [_text_block(text)]
        )
        return await agent._reason("what is 2+2?")

    async def test_valid_json_returned_as_dict(self):
        _, agent = _make_registry_and_agent()
        payload = _reasoning_payload()
        result = await self._call_reason(agent, payload)
        assert result["guardrail_passed"] is True
        assert result["reasoning_summary"] == "all good"

    async def test_markdown_fenced_json_is_stripped(self):
        _, agent = _make_registry_and_agent()
        payload = _reasoning_payload()
        fenced = f"```json\n{json.dumps(payload)}\n```"
        result = await self._call_reason(agent, payload, raw=fenced)
        assert result["guardrail_passed"] is True

    async def test_malformed_json_falls_back_gracefully(self):
        _, agent = _make_registry_and_agent()
        result = await self._call_reason(agent, {}, raw="not valid json at all")
        # Falls back to safe defaults
        assert result["guardrail_passed"] is True
        assert result["tool_calls_needed"] is False

    async def test_empty_content_returns_safe_dict(self):
        _, agent = _make_registry_and_agent()
        # Response with no text blocks → next() falls back to "{}" → json.loads → {}
        # All downstream .get() calls have safe defaults so {} is acceptable.
        agent._client.messages.create.return_value = _anthropic_resp([])
        result = await agent._reason("hi")
        assert isinstance(result, dict)
        # Safe defaults still hold: guardrail passes, no tools triggered
        assert result.get("guardrail_passed", True) is True
        assert result.get("tool_calls_needed", False) is False

    async def test_conversation_history_grows(self):
        _, agent = _make_registry_and_agent()
        agent._client.messages.create.return_value = _anthropic_resp(
            [_text_block(json.dumps(_reasoning_payload()))]
        )
        assert len(agent._conversation_history) == 0
        await agent._reason("first turn")
        assert len(agent._conversation_history) == 2   # user + assistant

    async def test_conversation_history_trimmed_at_20(self):
        _, agent = _make_registry_and_agent()
        agent._client.messages.create.return_value = _anthropic_resp(
            [_text_block(json.dumps(_reasoning_payload()))]
        )
        for _ in range(12):
            await agent._reason("repeat")
        assert len(agent._conversation_history) <= 20

    async def test_anthropic_called_with_system_prompt(self):
        _, agent = _make_registry_and_agent()
        agent._client.messages.create.return_value = _anthropic_resp(
            [_text_block(json.dumps(_reasoning_payload()))]
        )
        await agent._reason("test")
        call_kwargs = agent._client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0


# ─── _process() – guardrail block path ───────────────────────────────────────

class TestProcessGuardrailBlock:
    async def test_guardrail_blocked_result_returned(self):
        _, agent = _make_registry_and_agent()
        payload = _reasoning_payload(passed=False, reason="Detected violence.")
        agent._client.messages.create.return_value = _anthropic_resp(
            [_text_block(json.dumps(payload))]
        )
        utt = _make_utt(text="hurt someone")
        result = await agent._process(utt)
        assert result.guardrail_passed is False
        assert result.guardrail_reason == "Detected violence."

    async def test_guardrail_blocked_has_no_tool_outputs(self):
        _, agent = _make_registry_and_agent()
        payload = _reasoning_payload(passed=False, reason="spam")
        agent._client.messages.create.return_value = _anthropic_resp(
            [_text_block(json.dumps(payload))]
        )
        result = await agent._process(_make_utt())
        assert result.tool_outputs == []


# ─── _process() – tool-call path ─────────────────────────────────────────────

class TestProcessToolCalls:
    async def test_tool_outputs_appended_to_context(self):
        _, agent = _make_registry_and_agent()
        # First call: reasoning says tools needed
        reasoning = _reasoning_payload(tools=True, context="user asked for calc")
        agent._client.messages.create.return_value = _anthropic_resp(
            [_text_block(json.dumps(reasoning))]
        )

        # Patch _run_tool_calls to return a fake result
        fake_tools = [{"tool": "calculate", "result": {"result": 42}}]
        with patch.object(agent, "_run_tool_calls", AsyncMock(return_value=fake_tools)):
            result = await agent._process(_make_utt(text="what is 6*7"))

        assert result.guardrail_passed is True
        assert result.tool_outputs == fake_tools
        assert "calculate" in result.suggested_context_update

    async def test_no_tool_calls_when_not_needed(self):
        _, agent = _make_registry_and_agent()
        reasoning = _reasoning_payload(tools=False)
        agent._client.messages.create.return_value = _anthropic_resp(
            [_text_block(json.dumps(reasoning))]
        )
        with patch.object(agent, "_run_tool_calls", AsyncMock()) as mock_tools:
            await agent._process(_make_utt())
        mock_tools.assert_not_called()


# ─── _handle_utterance() – timeout / error paths ─────────────────────────────

class TestHandleUtterance:
    async def test_publishes_result_on_success(self):
        reg, agent = _make_registry_and_agent()
        bus: AgentMessageBus = reg.get_or_create("unit-room")
        received: list[BackgroundResult] = []
        bus.subscribe_results(lambda r: received.append(r) or asyncio.sleep(0))

        reasoning = _reasoning_payload()
        agent._client.messages.create.return_value = _anthropic_resp(
            [_text_block(json.dumps(reasoning))]
        )
        utt = _make_utt()
        await agent._handle_utterance(utt)
        assert len(received) == 1
        assert received[0].guardrail_passed is True

    async def test_publishes_timeout_result_on_timeout(self):
        reg, agent = _make_registry_and_agent()
        bus: AgentMessageBus = reg.get_or_create("unit-room")
        received: list[BackgroundResult] = []

        async def capture(r): received.append(r)
        bus.subscribe_results(capture)

        # Make _process hang forever
        async def hang(_utt):
            await asyncio.sleep(999)
        with patch.object(agent, "_process", hang):
            # Patch timeout to be tiny
            with patch("agents.background_agent.cfg") as mock_cfg:
                mock_cfg.BACKGROUND_TIMEOUT = 0.05
                await agent._handle_utterance(_make_utt())

        await asyncio.sleep(0)
        assert len(received) == 1
        assert "timed out" in received[0].reasoning_summary.lower()

    async def test_publishes_error_result_on_exception(self):
        reg, agent = _make_registry_and_agent()
        bus: AgentMessageBus = reg.get_or_create("unit-room")
        received: list[BackgroundResult] = []

        async def capture(r): received.append(r)
        bus.subscribe_results(capture)

        async def boom(_utt):
            raise RuntimeError("unexpected!")
        with patch.object(agent, "_process", boom):
            await agent._handle_utterance(_make_utt())
        # publish_result creates tasks; yield to the event loop to let them run
        await asyncio.sleep(0)

        assert len(received) == 1
        assert "error" in received[0].reasoning_summary.lower()


# ─── start() wires subscription ──────────────────────────────────────────────

class TestStart:
    def test_start_registers_utterance_handler(self):
        reg, agent = _make_registry_and_agent()
        bus: AgentMessageBus = reg.get_or_create("unit-room")
        assert len(bus._utt_handlers) == 0
        with patch("agents.background_agent.registry", reg):
            agent.start()
        assert len(bus._utt_handlers) == 1
