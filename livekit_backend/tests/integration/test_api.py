"""
Integration tests – FastAPI HTTP endpoints
==========================================

Tests the /token and /health endpoints using httpx's async test client.

The LiveKit AccessToken builder is already stubbed by the livekit.api mock
injected in tests/conftest.py, so no real JWT signing takes place.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration


# ─── App fixture ──────────────────────────────────────────────────────────────

@pytest.fixture()
async def api_client():
    """AsyncClient pointed at the FastAPI app under test."""
    from main import http_app

    async with AsyncClient(
        transport=ASGITransport(app=http_app),
        base_url="http://test",
    ) as client:
        yield client


# ─── GET /health ──────────────────────────────────────────────────────────────

class TestHealth:
    async def test_health_returns_200(self, api_client):
        r = await api_client.get("/health")
        assert r.status_code == 200

    async def test_health_body_has_status_ok(self, api_client):
        r = await api_client.get("/health")
        assert r.json()["status"] == "ok"

    async def test_health_body_has_timestamp(self, api_client):
        r = await api_client.get("/health")
        body = r.json()
        assert "timestamp" in body
        assert isinstance(body["timestamp"], float)


# ─── POST /token ──────────────────────────────────────────────────────────────

class TestToken:
    async def test_returns_200(self, api_client):
        r = await api_client.post(
            "/token", json={"room_name": "my-room"}
        )
        assert r.status_code == 200

    async def test_response_has_required_fields(self, api_client):
        r = await api_client.post(
            "/token", json={"room_name": "my-room"}
        )
        body = r.json()
        assert "token" in body
        assert "livekit_url" in body
        assert "room_name" in body
        assert "identity" in body

    async def test_room_name_echoed_in_response(self, api_client):
        r = await api_client.post(
            "/token", json={"room_name": "conversation-room"}
        )
        assert r.json()["room_name"] == "conversation-room"

    async def test_explicit_identity_preserved(self, api_client):
        r = await api_client.post(
            "/token",
            json={"room_name": "r", "participant_identity": "alice"},
        )
        assert r.json()["identity"] == "alice"

    async def test_auto_generated_identity_when_not_provided(self, api_client):
        r = await api_client.post(
            "/token", json={"room_name": "r"}
        )
        identity = r.json()["identity"]
        assert identity.startswith("user-")
        assert len(identity) > len("user-")

    async def test_two_calls_without_identity_get_different_identities(
        self, api_client
    ):
        r1 = await api_client.post("/token", json={"room_name": "r"})
        r2 = await api_client.post("/token", json={"room_name": "r"})
        assert r1.json()["identity"] != r2.json()["identity"]

    async def test_livekit_url_matches_config(self, api_client):
        import os
        r = await api_client.post("/token", json={"room_name": "r"})
        assert r.json()["livekit_url"] == os.environ["LIVEKIT_URL"]

    async def test_token_is_non_empty_string(self, api_client):
        r = await api_client.post("/token", json={"room_name": "r"})
        token = r.json()["token"]
        assert isinstance(token, str)
        assert len(token) > 0

    async def test_missing_room_name_returns_422(self, api_client):
        r = await api_client.post("/token", json={})
        assert r.status_code == 422

    async def test_invalid_json_returns_422(self, api_client):
        r = await api_client.post(
            "/token",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422
