"""
BackgroundAgent – deep reasoning, guardrails, and tool calls
============================================================

Responsibilities
----------------
* Subscribes to UserUtterance events from the AgentMessageBus.
* Runs every utterance through a Claude model with:
    1. A **guardrails pass** – checks policy violations before anything else.
    2. **Tool use** – lets Claude call registered tools (calculator,
       knowledge lookup, …) via Anthropic's native tool-use API.
    3. **Reasoning summary** – produces a short paragraph the foreground
       agent can inject into its system prompt to stay coherent.
* Publishes a BackgroundResult back to the bus.

The background agent never speaks directly; it only enriches state that
the ForegroundAgent consumes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from textwrap import dedent
from typing import Any

import anthropic

from config import cfg
from message_bus import (
    AgentMessageBus,
    BackgroundResult,
    UserUtterance,
    registry,
)
from tools.definitions import dispatch, get_all_schemas

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = dedent("""\
    You are a background reasoning assistant operating in a dual-agent voice
    system.  A fast foreground voice agent handles real-time conversation;
    you run silently in parallel to provide:

    1. GUARDRAILS – detect harmful, unsafe, or policy-violating requests.
    2. TOOL CALLS – execute calculations, knowledge lookups, or API calls.
    3. CONTEXT ENRICHMENT – produce a short, structured summary the
       foreground agent can use to give a better next reply.

    Rules:
    - Respond ONLY with valid JSON matching the schema below – no prose.
    - Keep reasoning_summary under 120 words.
    - suggested_context_update must be one short paragraph the foreground
      agent can paste verbatim into its system prompt.

    Response schema:
    {
      "guardrail_passed": true | false,
      "guardrail_reason": "<empty string if passed>",
      "reasoning_summary": "<brief chain-of-thought>",
      "tool_calls_needed": true | false,
      "suggested_context_update": "<paragraph for foreground system prompt>"
    }
""")

_GUARDRAIL_CATEGORIES = [
    "violence or self-harm",
    "sexual content involving minors",
    "instructions for illegal weapons or drugs",
    "personal data exfiltration",
    "social engineering or phishing",
]


# ── Agent class ──────────────────────────────────────────────────────────────

class BackgroundAgent:
    """
    One instance per room.  Subscribes to the bus and processes utterances.

    Usage::

        bg = BackgroundAgent(room_name)
        bg.start()        # registers subscriptions, no await needed
    """

    def __init__(self, room_name: str) -> None:
        self._room_name = room_name
        self._bus: AgentMessageBus = registry.get_or_create(room_name)
        self._client = anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY)
        self._conversation_history: list[dict] = []
        self._semaphore = asyncio.Semaphore(3)   # max 3 concurrent calls

    def start(self) -> None:
        self._bus.subscribe_utterances(self._handle_utterance)
        logger.info("[bg] background agent started for room=%s", self._room_name)

    # ── Core processing ──────────────────────────────────────────────────────

    async def _handle_utterance(self, utt: UserUtterance) -> None:
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    self._process(utt),
                    timeout=cfg.BACKGROUND_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("[bg] processing timed out for utterance: %r", utt.text)
                result = self._timeout_result(utt)
            except Exception:
                logger.exception("[bg] unexpected error processing utterance")
                result = self._error_result(utt)

            await self._bus.publish_result(result)

    async def _process(self, utt: UserUtterance) -> BackgroundResult:
        logger.info("[bg] processing utterance: %r", utt.text)

        # ── Step 1: initial reasoning + guardrail check ──────────────────
        initial_response = await self._reason(utt.text)

        if not initial_response.get("guardrail_passed", True):
            return BackgroundResult(
                room_name=self._room_name,
                original_utterance=utt.text,
                reasoning_summary=initial_response.get("reasoning_summary", ""),
                guardrail_passed=False,
                guardrail_reason=initial_response.get("guardrail_reason", "Policy violation."),
                tool_outputs=[],
                suggested_context_update="",
            )

        # ── Step 2: tool calls (if needed) ──────────────────────────────
        tool_outputs: list[dict] = []
        if initial_response.get("tool_calls_needed", False):
            tool_outputs = await self._run_tool_calls(utt.text)

        # ── Step 3: synthesise final context update ──────────────────────
        context_update = initial_response.get("suggested_context_update", "")
        if tool_outputs:
            tool_text = "; ".join(
                f"{t['tool']}={json.dumps(t['result'])}" for t in tool_outputs
            )
            context_update = (
                f"{context_update}  Tool results: {tool_text}"
            ).strip()

        return BackgroundResult(
            room_name=self._room_name,
            original_utterance=utt.text,
            reasoning_summary=initial_response.get("reasoning_summary", ""),
            guardrail_passed=True,
            guardrail_reason="",
            tool_outputs=tool_outputs,
            suggested_context_update=context_update,
        )

    # ── Anthropic helpers ────────────────────────────────────────────────────

    async def _reason(self, user_text: str) -> dict:
        """
        Call Claude to get a structured JSON reasoning verdict.
        We use extended thinking for higher-quality guardrail + context results.
        """
        messages = self._conversation_history + [
            {"role": "user", "content": user_text}
        ]

        response = await self._client.messages.create(
            model=cfg.BACKGROUND_LLM_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=messages,
            # Enable extended thinking for richer reasoning
            thinking={
                "type": "enabled",
                "budget_tokens": 512,
            },
        )

        # Extract the text block (thinking blocks are filtered automatically)
        text_block = next(
            (b.text for b in response.content if hasattr(b, "text")), "{}"
        )
        # Strip markdown fences if present
        text_block = text_block.strip().removeprefix("```json").removesuffix("```").strip()

        try:
            result = json.loads(text_block)
        except json.JSONDecodeError:
            logger.warning("[bg] could not parse JSON from Claude: %r", text_block)
            result = {
                "guardrail_passed": True,
                "guardrail_reason": "",
                "reasoning_summary": text_block[:120],
                "tool_calls_needed": False,
                "suggested_context_update": "",
            }

        # Persist to rolling history (keep last 10 turns to avoid token bloat)
        self._conversation_history.append({"role": "user", "content": user_text})
        self._conversation_history.append(
            {"role": "assistant", "content": text_block}
        )
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        return result

    async def _run_tool_calls(self, user_text: str) -> list[dict]:
        """
        Second Claude call with tools exposed.  Claude decides which tools
        to call; we execute them and return the aggregated results.
        """
        messages: list[dict] = [{"role": "user", "content": user_text}]
        tool_schemas = get_all_schemas()
        tool_outputs: list[dict] = []

        # Agentic loop: keep going while Claude wants more tool calls
        for _ in range(5):   # safety cap
            response = await self._client.messages.create(
                model=cfg.BACKGROUND_LLM_MODEL,
                max_tokens=1024,
                tools=tool_schemas,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason != "tool_use":
                break

            # Collect tool-use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            # Append assistant message
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool and build tool_result content
            tool_results_content: list[dict] = []
            for block in tool_use_blocks:
                try:
                    result = await dispatch(block.name, block.input)
                    logger.info("[bg] tool %s returned: %s", block.name, result)
                    tool_outputs.append({"tool": block.name, "result": result})
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
                except Exception as exc:
                    logger.error("[bg] tool %s failed: %s", block.name, exc)
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "is_error": True,
                        "content": str(exc),
                    })

            messages.append({"role": "user", "content": tool_results_content})

        return tool_outputs

    # ── Fallback results ─────────────────────────────────────────────────────

    def _timeout_result(self, utt: UserUtterance) -> BackgroundResult:
        return BackgroundResult(
            room_name=self._room_name,
            original_utterance=utt.text,
            reasoning_summary="Processing timed out.",
            guardrail_passed=True,
            guardrail_reason="",
            tool_outputs=[],
            suggested_context_update="",
        )

    def _error_result(self, utt: UserUtterance) -> BackgroundResult:
        return BackgroundResult(
            room_name=self._room_name,
            original_utterance=utt.text,
            reasoning_summary="An internal error occurred.",
            guardrail_passed=True,
            guardrail_reason="",
            tool_outputs=[],
            suggested_context_update="",
        )
