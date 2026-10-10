from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from backend import app as backend_app


def test_missing_bearer_token_is_rejected_before_analysis() -> None:
    client = TestClient(backend_app.app)

    response = client.post(
        "/api/analyze",
        files={"file": ("image.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 401


def test_auth_verification_forwards_the_signed_in_users_token(monkeypatch) -> None:
    token = "user-access-token"
    monkeypatch.setattr(backend_app, "SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setattr(backend_app, "SUPABASE_ANON_KEY", "public-key")

    class StubClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url, *, headers):
            assert headers["Authorization"] == f"Bearer {token}"
            return httpx.Response(200, json={"id": "user-123"})

    monkeypatch.setattr(backend_app.httpx, "AsyncClient", lambda **_kwargs: StubClient())

    user = asyncio.run(backend_app.require_user(f"Bearer {token}"))

    assert user.user_id == "user-123"
    assert user.access_token == token


def test_supabase_insert_uses_the_owner_token(monkeypatch) -> None:
    token = "user-access-token"
    user = backend_app.AuthenticatedUser(user_id="user-123", access_token=token)
    monkeypatch.setattr(backend_app, "SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setattr(backend_app, "SUPABASE_ANON_KEY", "public-key")

    class StubClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, _url, *, params, headers, json):
            assert params == {"select": "*"}
            assert headers["Authorization"] == f"Bearer {token}"
            assert json["user_id"] == user.user_id
            return httpx.Response(
                201,
                json=[{"id": "result-1", "created_at": "2026-10-08T00:00:00Z"}],
            )

    monkeypatch.setattr(backend_app.httpx, "AsyncClient", lambda **_kwargs: StubClient())

    saved = asyncio.run(
        backend_app.save_analysis(
            user,
            {"user_id": user.user_id, "media_type": "image"},
        )
    )

    assert saved["id"] == "result-1"


def test_analysis_is_saved_for_authenticated_owner(monkeypatch) -> None:
    from backend.app import AuthenticatedUser

    user = AuthenticatedUser(user_id="user-123", access_token="test-access-token")
    backend_app.app.dependency_overrides[backend_app.require_user] = lambda: user
    monkeypatch.setattr(
        backend_app,
        "get_detector",
        AsyncMock(return_value={"model_id": "test-model"}),
    )
    monkeypatch.setattr(backend_app, "analyze_image_bytes", lambda *_args: 0.73)
    save_mock = AsyncMock(
        return_value={"id": "result-1", "created_at": "2026-10-08T00:00:00Z"}
    )
    monkeypatch.setattr(backend_app, "save_analysis", save_mock)

    try:
        client = TestClient(backend_app.app)
        response = client.post(
            "/api/analyze",
            files={"file": ("sample.png", b"image-content", "image/png")},
        )
    finally:
        backend_app.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["positive_percent"] == 73.0
    assert response.json()["real_percent"] == 27.0
    assert response.json()["score_label"] == "ai_generated"
    saved_user, saved_record = save_mock.await_args.args
    assert saved_user.user_id == "user-123"
    assert saved_record["user_id"] == "user-123"
    assert "image-content" not in response.text
