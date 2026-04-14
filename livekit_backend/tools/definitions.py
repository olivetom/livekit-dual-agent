"""
Tool definitions exposed to the BackgroundAgent.

Each tool is a plain async function.  The BackgroundAgent builds an
Anthropic-compatible tool schema from the docstrings and type hints and
passes the list to the Claude API's `tools` parameter.

Add your own domain tools here (calendar lookups, database queries, etc.)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import operator as op
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Tool registry ────────────────────────────────────────────────────────────

_TOOLS: dict[str, "ToolDef"] = {}


class ToolDef:
    def __init__(self, fn, schema: dict) -> None:
        self.fn = fn
        self.schema = schema  # Anthropic tool schema dict

    async def call(self, **kwargs: Any) -> Any:
        return await self.fn(**kwargs)


def tool(description: str, input_schema: dict):
    """Decorator that registers an async function as a background tool."""

    def decorator(fn):
        _TOOLS[fn.__name__] = ToolDef(
            fn,
            {
                "name": fn.__name__,
                "description": description,
                "input_schema": input_schema,
            },
        )
        return fn

    return decorator


def get_all_schemas() -> list[dict]:
    return [t.schema for t in _TOOLS.values()]


async def dispatch(name: str, arguments: dict) -> Any:
    if name not in _TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    return await _TOOLS[name].call(**arguments)


# ── Concrete tools ───────────────────────────────────────────────────────────

@tool(
    description=(
        "Evaluate a safe arithmetic expression and return the numeric result. "
        "Supports +, -, *, /, **, sqrt(), and parentheses."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression to evaluate, e.g. '2 ** 10 + sqrt(144)'",
            }
        },
        "required": ["expression"],
    },
)
async def calculate(expression: str) -> dict:
    """Safe arithmetic evaluator – no exec/eval on arbitrary code."""
    # Whitelist: digits, operators, spaces, sqrt, pi, e, parens, dot
    allowed = re.compile(r"^[\d\s\+\-\*\/\.\(\)\^sqrtepi]+$", re.IGNORECASE)
    if not allowed.match(expression):
        return {"error": "Expression contains disallowed characters."}

    safe_expr = expression.replace("^", "**")
    safe_globals = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
    }
    try:
        result = eval(safe_expr, safe_globals)  # noqa: S307 – restricted env
        return {"result": result, "expression": expression}
    except Exception as exc:
        return {"error": str(exc)}


@tool(
    description="Look up a definition or short factual answer from a knowledge base.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or term to look up.",
            }
        },
        "required": ["query"],
    },
)
async def knowledge_lookup(query: str) -> dict:
    """
    Stub – replace with a real vector-store / RAG retrieval call.
    For demo purposes returns a placeholder answer.
    """
    await asyncio.sleep(0.1)   # simulate I/O
    return {
        "query": query,
        "answer": (
            f"(Demo) I found a relevant passage about '{query}'. "
            "Replace this stub with your real retrieval logic."
        ),
    }


@tool(
    description="Get the current UTC time and date.",
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
async def get_current_time() -> dict:
    import datetime
    now = datetime.datetime.utcnow()
    return {"utc_time": now.isoformat() + "Z", "weekday": now.strftime("%A")}
