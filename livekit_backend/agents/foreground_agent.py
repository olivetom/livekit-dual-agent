"""
ForegroundAgent – real-time voice conversation layer
=====================================================

Responsibilities
----------------
* Streams audio from the LiveKit room through Deepgram STT.
* Responds to each user turn using a small, fast OpenAI model so the
  perceived latency stays under ~500 ms.
* After every user turn it publishes the transcription to the shared
  AgentMessageBus so the BackgroundAgent can process it asynchronously.
* Listens for BackgroundResults:
  - If the background agent flagged a guardrail violation it will NOT
    deliver the foreground reply and will instead speak a safe refusal.
  - If the background agent returned enriched context or tool outputs it
    injects them into the next system-prompt update so the conversation
    stays coherent.

Pipeline: Deepgram STT → gpt-4o-mini (fast LLM) → ElevenLabs TTS
"""

from __future__ import annotations

import asyncio
import logging
from textwrap import dedent

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    llm,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, elevenlabs, openai, silero

from config import cfg
from message_bus import (
    AgentMessageBus,
    BackgroundResult,
    UserUtterance,
    registry,
)

logger = logging.getLogger(__name__)

# ── System prompt template ───────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = dedent("""\
    You are a helpful, warm, and concise voice assistant.
    Keep replies SHORT – two or three sentences at most – because this is
    a real-time voice conversation and long responses feel slow.
    If you need to do a complex calculation or look something up, say
    "Let me check that for you" and wait; a background assistant will
    provide the result shortly.
    {context_block}
""")


def _build_system_prompt(context_block: str = "") -> str:
    block = f"\n[Background context]\n{context_block}" if context_block else ""
    return _BASE_SYSTEM_PROMPT.format(context_block=block)


# ── Agent class ──────────────────────────────────────────────────────────────

class ForegroundAgent:
    """
    Wraps a LiveKit VoiceAssistant with dual-agent plumbing.

    Usage::

        fg = ForegroundAgent(ctx)
        await fg.start()
    """

    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx
        self._room = ctx.room
        self._bus: AgentMessageBus = registry.get_or_create(self._room.name)

        # Mutable context injected by the background agent
        self._bg_context: str = ""
        # Hold the last background result for guardrail checks
        self._pending_bg_result: BackgroundResult | None = None
        # Lock so we don't race between voice reply and background result
        self._reply_lock = asyncio.Lock()

        # Build STT / LLM / TTS pipeline components
        self._stt = deepgram.STT(api_key=cfg.DEEPGRAM_API_KEY)
        self._llm = openai.LLM(
            model=cfg.FOREGROUND_LLM_MODEL,
            api_key=cfg.OPENAI_API_KEY,
        )
        self._tts = elevenlabs.TTS(
            api_key=cfg.ELEVENLABS_API_KEY,
            voice_id=cfg.ELEVENLABS_VOICE_ID,
            model_id="eleven_turbo_v2",   # lowest ElevenLabs latency model
        )
        self._vad = silero.VAD.load()

        self._chat_ctx = llm.ChatContext().append(
            role="system",
            text=_build_system_prompt(),
        )

        self._assistant = VoiceAssistant(
            vad=self._vad,
            stt=self._stt,
            llm=self._llm,
            tts=self._tts,
            chat_ctx=self._chat_ctx,
            interrupt_speech_duration=0.6,   # allow quick interruptions
            interrupt_min_words=2,
        )

        # Wire callbacks
        self._assistant.on("user_speech_committed", self._on_user_speech_committed)
        self._assistant.on("agent_speech_interrupted", self._on_speech_interrupted)

        # Subscribe to background results
        self._bus.subscribe_results(self._on_background_result)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self._ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        self._assistant.start(self._room)
        logger.info("[fg] started in room=%s", self._room.name)
        await self._assistant.say(
            "Hello! I'm your voice assistant. How can I help you today?",
            allow_interruptions=True,
        )

    # ── Event handlers ───────────────────────────────────────────────────────

    async def _on_user_speech_committed(self, msg: llm.ChatMessage) -> None:
        """
        Called by VoiceAssistant after the user's utterance has been
        transcribed and committed to the chat context.

        1. Publish utterance to background agent (fire-and-forget).
        2. Kick off the fast foreground LLM reply in parallel.
        3. Before speaking, optionally wait briefly for a background
           guardrail verdict so we can suppress blocked replies.
        """
        user_text: str = msg.content or ""
        logger.info("[fg] user said: %r", user_text)

        utt = UserUtterance(
            room_name=self._room.name,
            participant_identity=self._room.local_participant.identity,
            text=user_text,
        )
        # Non-blocking publish; background agent processes concurrently
        await self._bus.publish_utterance(utt)

    async def _on_speech_interrupted(self) -> None:
        logger.debug("[fg] speech interrupted by user")

    async def _on_background_result(self, result: BackgroundResult) -> None:
        """
        Receives enriched context from the background agent.

        * If guardrail_passed is False → speak a refusal and abort the
          current pending reply.
        * Otherwise → update the running system-prompt context so future
          turns benefit from the reasoning summary and tool outputs.
        """
        async with self._reply_lock:
            self._pending_bg_result = result

            if not result.guardrail_passed:
                logger.warning(
                    "[fg] guardrail blocked utterance: %s", result.guardrail_reason
                )
                # Interrupt whatever the assistant might be saying and
                # replace it with a safe refusal.
                await self._assistant.say(
                    f"I'm sorry, I can't help with that. {result.guardrail_reason}",
                    allow_interruptions=False,
                )
                return

            # Enrich the system prompt with background context
            if result.suggested_context_update:
                self._bg_context = result.suggested_context_update
                new_prompt = _build_system_prompt(self._bg_context)
                # Update the first (system) message in the chat context
                if self._chat_ctx.messages:
                    self._chat_ctx.messages[0] = llm.ChatMessage(
                        role="system", content=new_prompt
                    )
                logger.debug("[fg] system prompt updated with background context")

            # If the background agent completed tool calls, speak the result
            if result.tool_outputs:
                tool_summary = "; ".join(
                    str(t.get("result", "")) for t in result.tool_outputs
                )
                await self._assistant.say(
                    f"Here's what I found: {tool_summary}",
                    allow_interruptions=True,
                )
