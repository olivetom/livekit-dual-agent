"""
Unit tests – tools/definitions.py
===================================

Tests the tool registry, dispatch mechanism, and every concrete tool.
No external I/O; all network/DB calls in the tools are stubs already.
"""

import math
import re

import pytest

from tools.definitions import (
    _TOOLS,
    calculate,
    dispatch,
    get_all_schemas,
    get_current_time,
    knowledge_lookup,
)

pytestmark = pytest.mark.unit


# ─── Tool registry ────────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_expected_tools_registered(self):
        names = set(_TOOLS.keys())
        assert {"calculate", "knowledge_lookup", "get_current_time"}.issubset(names)

    def test_get_all_schemas_returns_list_of_dicts(self):
        schemas = get_all_schemas()
        assert isinstance(schemas, list)
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "input_schema" in s

    def test_schema_has_required_anthropic_keys(self):
        for schema in get_all_schemas():
            assert schema["input_schema"]["type"] == "object"
            assert "properties" in schema["input_schema"]

    async def test_dispatch_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            await dispatch("nonexistent_tool", {})

    async def test_dispatch_known_tool_returns_result(self):
        result = await dispatch("get_current_time", {})
        assert "utc_time" in result


# ─── calculate ────────────────────────────────────────────────────────────────

class TestCalculate:
    async def test_simple_addition(self):
        r = await calculate("2 + 2")
        assert r["result"] == 4

    async def test_subtraction(self):
        r = await calculate("10 - 3")
        assert r["result"] == 7

    async def test_multiplication(self):
        r = await calculate("6 * 7")
        assert r["result"] == 42

    async def test_division(self):
        r = await calculate("10 / 4")
        assert abs(r["result"] - 2.5) < 1e-9

    async def test_exponentiation_double_star(self):
        r = await calculate("2 ** 10")
        assert r["result"] == 1024

    async def test_exponentiation_caret_alias(self):
        r = await calculate("2^10")
        assert r["result"] == 1024

    async def test_sqrt(self):
        r = await calculate("sqrt(144)")
        assert abs(r["result"] - 12.0) < 1e-9

    async def test_combined_expression(self):
        r = await calculate("2 ** 10 + sqrt(36)")
        assert abs(r["result"] - 1030.0) < 1e-9

    async def test_uses_pi(self):
        r = await calculate("pi")
        assert abs(r["result"] - math.pi) < 1e-9

    async def test_parentheses(self):
        r = await calculate("(3 + 4) * 2")
        assert r["result"] == 14

    async def test_expression_preserved_in_result(self):
        r = await calculate("1 + 1")
        assert r["expression"] == "1 + 1"

    async def test_injection_attempt_is_blocked(self):
        # Attempt to smuggle arbitrary code via the expression
        r = await calculate("__import__('os').system('id')")
        assert "error" in r

    async def test_disallowed_characters_return_error(self):
        r = await calculate("open('/etc/passwd').read()")
        assert "error" in r

    async def test_division_by_zero_returns_error(self):
        r = await calculate("1 / 0")
        assert "error" in r

    async def test_malformed_expression_returns_error(self):
        r = await calculate("(((")
        assert "error" in r


# ─── knowledge_lookup ─────────────────────────────────────────────────────────

class TestKnowledgeLookup:
    async def test_returns_dict_with_query(self):
        r = await knowledge_lookup("Python")
        assert r["query"] == "Python"

    async def test_returns_dict_with_answer(self):
        r = await knowledge_lookup("machine learning")
        assert "answer" in r
        assert isinstance(r["answer"], str)
        assert len(r["answer"]) > 0

    async def test_query_reflected_in_answer(self):
        r = await knowledge_lookup("photosynthesis")
        assert "photosynthesis" in r["answer"]


# ─── get_current_time ─────────────────────────────────────────────────────────

class TestGetCurrentTime:
    async def test_returns_utc_time_key(self):
        r = await get_current_time()
        assert "utc_time" in r

    async def test_utc_time_ends_with_z(self):
        r = await get_current_time()
        assert r["utc_time"].endswith("Z")

    async def test_utc_time_is_iso_format(self):
        r = await get_current_time()
        # Basic ISO-8601 pattern: YYYY-MM-DDTHH:MM:SS...Z
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", r["utc_time"])

    async def test_weekday_is_valid(self):
        r = await get_current_time()
        valid_days = {
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        }
        assert r["weekday"] in valid_days
