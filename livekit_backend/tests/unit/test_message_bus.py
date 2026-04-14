"""
Unit tests – message_bus.py
===========================

Pure asyncio tests; zero external dependencies.
All imports are from stdlib or the message_bus module itself.
"""

import asyncio
import time

import pytest

from message_bus import (
    AgentMessageBus,
    BackgroundResult,
    MessageBusRegistry,
    UserUtterance,
    registry as global_registry,
)

pytestmark = pytest.mark.unit


# ─── Dataclass smoke tests ────────────────────────────────────────────────────

class TestUserUtterance:
    def test_fields_are_set(self):
        utt = UserUtterance(
            room_name="room-1",
            participant_identity="alice",
            text="Hello world",
        )
        assert utt.room_name == "room-1"
        assert utt.participant_identity == "alice"
        assert utt.text == "Hello world"

    def test_timestamp_defaults_to_now(self):
        before = time.time()
        utt = UserUtterance(room_name="r", participant_identity="u", text="t")
        after = time.time()
        assert before <= utt.timestamp <= after

    def test_explicit_timestamp(self):
        utt = UserUtterance(room_name="r", participant_identity="u", text="t", timestamp=1.0)
        assert utt.timestamp == 1.0


class TestBackgroundResult:
    def test_fields_are_set(self):
        res = BackgroundResult(
            room_name="room-1",
            original_utterance="hi",
            reasoning_summary="all good",
            guardrail_passed=True,
            guardrail_reason="",
            tool_outputs=[{"tool": "calculate", "result": 42}],
            suggested_context_update="ctx",
        )
        assert res.guardrail_passed is True
        assert res.tool_outputs[0]["tool"] == "calculate"

    def test_guardrail_failure(self):
        res = BackgroundResult(
            room_name="r",
            original_utterance="bad request",
            reasoning_summary="blocked",
            guardrail_passed=False,
            guardrail_reason="violence",
            tool_outputs=[],
            suggested_context_update="",
        )
        assert res.guardrail_passed is False
        assert res.guardrail_reason == "violence"


# ─── AgentMessageBus ─────────────────────────────────────────────────────────

class TestAgentMessageBus:
    def _make_bus(self, name: str = "test-room") -> AgentMessageBus:
        return AgentMessageBus(name)

    def _make_utt(self, text: str = "hello") -> UserUtterance:
        return UserUtterance(room_name="test-room", participant_identity="u", text=text)

    def _make_result(self, passed: bool = True) -> BackgroundResult:
        return BackgroundResult(
            room_name="test-room",
            original_utterance="hello",
            reasoning_summary="ok",
            guardrail_passed=passed,
            guardrail_reason="" if passed else "bad",
            tool_outputs=[],
            suggested_context_update="",
        )

    # ── subscribe / publish utterances ────────────────────────────────────

    async def test_utterance_handler_is_called(self):
        bus = self._make_bus()
        received: list[UserUtterance] = []

        async def handler(utt: UserUtterance) -> None:
            received.append(utt)

        bus.subscribe_utterances(handler)
        await bus.publish_utterance(self._make_utt("hi"))
        # give the task a chance to run
        await asyncio.sleep(0)
        assert len(received) == 1
        assert received[0].text == "hi"

    async def test_multiple_utterance_handlers_all_called(self):
        bus = self._make_bus()
        log: list[str] = []

        async def h1(utt): log.append("h1")
        async def h2(utt): log.append("h2")

        bus.subscribe_utterances(h1)
        bus.subscribe_utterances(h2)
        await bus.publish_utterance(self._make_utt())
        await asyncio.sleep(0)
        assert "h1" in log
        assert "h2" in log

    # ── subscribe / publish results ───────────────────────────────────────

    async def test_result_handler_is_called(self):
        bus = self._make_bus()
        received: list[BackgroundResult] = []

        async def handler(r): received.append(r)

        bus.subscribe_results(handler)
        res = self._make_result()
        await bus.publish_result(res)
        await asyncio.sleep(0)
        assert len(received) == 1
        assert received[0].guardrail_passed is True

    async def test_latest_result_is_stored(self):
        bus = self._make_bus()
        await bus.publish_result(self._make_result(passed=True))
        await asyncio.sleep(0)
        assert bus._latest_result is not None
        assert bus._latest_result.guardrail_passed is True

    async def test_result_event_is_set_after_publish(self):
        bus = self._make_bus()
        assert not bus._result_event.is_set()
        await bus.publish_result(self._make_result())
        await asyncio.sleep(0)
        assert bus._result_event.is_set()

    # ── wait_for_result ───────────────────────────────────────────────────

    async def test_wait_for_result_returns_result_when_published(self):
        bus = self._make_bus()

        async def publish_later():
            await asyncio.sleep(0.05)
            await bus.publish_result(self._make_result())

        asyncio.create_task(publish_later())
        result = await bus.wait_for_result(timeout=1.0)
        assert result is not None
        assert result.guardrail_passed is True

    async def test_wait_for_result_returns_none_on_timeout(self):
        bus = self._make_bus()
        result = await bus.wait_for_result(timeout=0.05)
        assert result is None

    async def test_wait_for_result_clears_event_before_waiting(self):
        bus = self._make_bus()
        # Set the event manually first; wait_for_result should clear it
        bus._result_event.set()
        # Now there is NO new publish → should time out
        result = await bus.wait_for_result(timeout=0.05)
        assert result is None

    # ── isolated per-room ─────────────────────────────────────────────────

    async def test_handlers_scoped_to_bus_instance(self):
        bus_a = AgentMessageBus("room-a")
        bus_b = AgentMessageBus("room-b")
        log: list[str] = []

        async def ha(utt): log.append("a")
        async def hb(utt): log.append("b")

        bus_a.subscribe_utterances(ha)
        bus_b.subscribe_utterances(hb)

        await bus_a.publish_utterance(self._make_utt())
        await asyncio.sleep(0)
        assert log == ["a"]   # "b" must NOT appear


# ─── MessageBusRegistry ───────────────────────────────────────────────────────

class TestMessageBusRegistry:
    def _make_registry(self) -> MessageBusRegistry:
        return MessageBusRegistry()

    def test_get_or_create_returns_bus(self):
        reg = self._make_registry()
        bus = reg.get_or_create("room-1")
        assert isinstance(bus, AgentMessageBus)
        assert bus.room_name == "room-1"

    def test_get_or_create_is_idempotent(self):
        reg = self._make_registry()
        b1 = reg.get_or_create("room-x")
        b2 = reg.get_or_create("room-x")
        assert b1 is b2

    def test_different_rooms_get_different_buses(self):
        reg = self._make_registry()
        b1 = reg.get_or_create("room-alpha")
        b2 = reg.get_or_create("room-beta")
        assert b1 is not b2

    def test_remove_deletes_bus(self):
        reg = self._make_registry()
        reg.get_or_create("room-del")
        reg.remove("room-del")
        # After remove, a fresh bus should be created
        b2 = reg.get_or_create("room-del")
        assert isinstance(b2, AgentMessageBus)

    def test_remove_nonexistent_is_safe(self):
        reg = self._make_registry()
        reg.remove("does-not-exist")   # should not raise

    def test_global_registry_is_singleton(self):
        b1 = global_registry.get_or_create("global-room")
        b2 = global_registry.get_or_create("global-room")
        assert b1 is b2
        global_registry.remove("global-room")
