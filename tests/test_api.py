"""
Integration tests for FastAPI REST routes.
"""

from fastapi.testclient import TestClient
from cortex.api.server import create_app
from cortex.api.routes import router


def test_health_check_endpoint():
    app = create_app()
    app.include_router(router)
    client = TestClient(app)

    res = client.get("/v1/healthz")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_chat_completions_endpoint():
    app = create_app()
    app.include_router(router)
    client = TestClient(app)

    res = client.post("/v1/chat/completions", json={"message": "Ping", "session_id": "test_sess"})
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "test_sess"
    assert "reply" in data
