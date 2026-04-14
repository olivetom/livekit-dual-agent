"""
tests/conftest.py
=================

Bootstraps the test environment BEFORE any application module is imported:

  1. Injects dummy environment variables – config.py hard-requires them at
     class-body evaluation time, so they must exist before the first import.
  2. Stubs out the LiveKit SDK families (livekit.*) with MagicMock objects
     injected into sys.modules.  This lets application code that does
       ``from livekit.agents import JobContext``
     succeed in a Python environment that doesn't have the SDK installed
     and without any real WebRTC connections.

Everything below runs at *module load time* (not inside fixtures) so the
stubs are present for every subsequent import.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# ─── 1. Required environment variables ───────────────────────────────────────
_DEFAULTS = {
    "LIVEKIT_URL":        "wss://test.livekit.cloud",
    "LIVEKIT_API_KEY":    "APItest123",
    "LIVEKIT_API_SECRET": "test_secret_xxxxxxxxxxxxxxxx",
    "OPENAI_API_KEY":     "sk-test-openai",
    "ANTHROPIC_API_KEY":  "sk-ant-test",
    "DEEPGRAM_API_KEY":   "test_deepgram_key",
    "ELEVENLABS_API_KEY": "test_elevenlabs_key",
    "BACKGROUND_TIMEOUT": "5.0",
}
for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)


# ─── 2. LiveKit SDK stubs ─────────────────────────────────────────────────────

def _fresh_chat_context() -> MagicMock:
    """Factory: each call returns an independent chat-context stub."""
    ctx = MagicMock(name="ChatContext-instance")
    ctx.messages = [MagicMock(role="system", content="system prompt")]
    ctx.append = MagicMock(return_value=ctx)
    return ctx


def _build_livekit_stubs() -> dict[str, MagicMock]:
    """
    Return a mapping of dotted module path → stub suitable for sys.modules.
    All stubs are created once and reused for the entire test session.
    """
    # ── livekit.agents ────────────────────────────────────────────────────
    lk_agents = MagicMock(name="livekit.agents")
    lk_agents.AutoSubscribe    = MagicMock()
    lk_agents.JobContext       = MagicMock()
    lk_agents.WorkerOptions    = MagicMock()
    lk_agents.cli              = MagicMock()
    lk_agents.Agent            = MagicMock()
    lk_agents.AgentSession     = MagicMock()

    # livekit.agents.llm
    lk_llm = MagicMock(name="livekit.agents.llm")
    lk_llm.ChatContext  = MagicMock(side_effect=lambda: _fresh_chat_context())
    lk_llm.ChatMessage  = MagicMock(
        side_effect=lambda role, content: MagicMock(role=role, content=content)
    )
    lk_agents.llm = lk_llm

    # livekit.agents.voice_assistant
    # Use side_effect so every ForegroundAgent.__init__ call gets a FRESH
    # VoiceAssistant stub – prevents .say() call history leaking between tests.
    def _fresh_va(*_a, **_kw) -> MagicMock:
        inst = MagicMock(name="VoiceAssistant-instance")
        inst.on    = MagicMock()
        inst.start = MagicMock()
        inst.say   = AsyncMock()
        return inst

    lk_va = MagicMock(name="livekit.agents.voice_assistant")
    lk_va.VoiceAssistant = MagicMock(side_effect=_fresh_va)

    # livekit.plugins.*
    lk_plugins   = MagicMock(name="livekit.plugins")
    lk_deepgram  = MagicMock(name="livekit.plugins.deepgram")
    lk_elevenlabs= MagicMock(name="livekit.plugins.elevenlabs")
    lk_openai    = MagicMock(name="livekit.plugins.openai")
    lk_silero    = MagicMock(name="livekit.plugins.silero")
    lk_silero.VAD      = MagicMock()
    lk_silero.VAD.load = MagicMock(return_value=MagicMock(name="VAD-instance"))

    # livekit.api (AccessToken / VideoGrants)
    lk_api   = MagicMock(name="livekit.api")
    _lk_tok  = MagicMock(name="AccessToken-instance")
    _lk_tok.with_identity = MagicMock(return_value=_lk_tok)
    _lk_tok.with_name     = MagicMock(return_value=_lk_tok)
    _lk_tok.with_grants   = MagicMock(return_value=_lk_tok)
    _lk_tok.to_jwt        = MagicMock(return_value="header.payload.signature")
    lk_api.AccessToken    = MagicMock(return_value=_lk_tok)
    lk_api.VideoGrants    = MagicMock()

    return {
        "livekit":                           MagicMock(name="livekit"),
        "livekit.agents":                    lk_agents,
        "livekit.agents.llm":                lk_llm,
        "livekit.agents.voice_assistant":    lk_va,
        "livekit.plugins":                   lk_plugins,
        "livekit.plugins.deepgram":          lk_deepgram,
        "livekit.plugins.elevenlabs":        lk_elevenlabs,
        "livekit.plugins.openai":            lk_openai,
        "livekit.plugins.silero":            lk_silero,
        "livekit.api":                       lk_api,
    }


_LK_STUBS: dict[str, MagicMock] = _build_livekit_stubs()

for _path, _stub in _LK_STUBS.items():
    sys.modules.setdefault(_path, _stub)


# ─── 3. Anthropic response helpers ───────────────────────────────────────────

def make_text_block(text: str) -> MagicMock:
    blk = MagicMock()
    blk.type = "text"
    blk.text = text
    return blk


def make_tool_use_block(
    name: str, tool_id: str, inputs: dict
) -> MagicMock:
    blk = MagicMock()
    blk.type  = "tool_use"
    blk.name  = name
    blk.id    = tool_id
    blk.input = inputs
    return blk


def make_anthropic_response(
    content_blocks: list,
    stop_reason: str = "end_turn",
) -> MagicMock:
    resp = MagicMock()
    resp.content     = content_blocks
    resp.stop_reason = stop_reason
    return resp


def make_reasoning_response(payload: dict) -> MagicMock:
    """Convenience: text block that contains a JSON reasoning payload."""
    return make_anthropic_response([make_text_block(json.dumps(payload))])


# ─── 4. Shared pytest fixtures ────────────────────────────────────────────────

@pytest.fixture()
def lk_va_instance() -> MagicMock:
    """The VoiceAssistant singleton stub; resets .say() between tests."""
    inst = _LK_STUBS["livekit.agents.voice_assistant"].VoiceAssistant.return_value
    inst.say.reset_mock()
    return inst


@pytest.fixture()
def mock_job_ctx() -> MagicMock:
    """A minimal JobContext with a pre-named room."""
    ctx = MagicMock()
    ctx.room                          = MagicMock()
    ctx.room.name                     = "test-room"
    ctx.room.local_participant        = MagicMock()
    ctx.room.local_participant.identity = "agent-identity"
    ctx.connect                       = AsyncMock()
    return ctx
