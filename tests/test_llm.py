"""Tests for the multi-provider LLM comparison layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


@pytest.fixture(autouse=True)
def _force_offline_llm(monkeypatch):
    monkeypatch.setenv("LLM_ENABLE_LIVE_PROVIDERS", "false")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_llm_model_catalog_lists_providers():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/llm/models")

    assert resp.status_code == 200
    models = resp.json()
    providers = {m["provider"] for m in models}
    assert {"openai", "gemini", "anthropic", "local"}.issubset(providers)
    assert any(m["id"] == "openai:gpt-4o-mini" for m in models)
    assert all("context_window" in m for m in models)


def test_llm_provider_status_defaults_to_mock_mode():
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/llm/provider-status")

    assert resp.status_code == 200
    statuses = {p["provider"]: p for p in resp.json()}
    assert statuses["openai"]["mode"] == "mock"
    assert statuses["gemini"]["mode"] == "mock"
    assert statuses["anthropic"]["mode"] == "mock"
    assert statuses["local"]["configured"] is True


def test_llm_generate_uses_offline_placeholder_without_keys():
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/llm/generate",
            json={
                "model": "openai:gpt-4o-mini",
                "prompt": "Summarize this support ticket.",
                "max_tokens": 64,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "openai"
    assert body["live"] is False
    assert body["error"] is None
    assert "mock:openai:gpt-4o-mini" in body["output"]
    assert body["estimated_cost_usd"] >= 0


def test_llm_compare_returns_ranking_and_bias_fields():
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/llm/compare",
            json={
                "models": [
                    "openai:gpt-4o-mini",
                    "gemini:gemini-2.0-flash",
                    "anthropic:claude-3-5-haiku-latest",
                ],
                "prompt": "Classify this request and suggest the next action.",
                "task": "support_triage",
                "json_mode": True,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["models_compared"] == 3
    assert body["recommended_model"]
    assert len(body["results"]) == 3
    for result in body["results"]:
        assert "quality_score" in result
        assert "bias_risk_score" in result
        assert "estimated_cost_usd" in result
        assert result["live"] is False


def test_llm_generate_unknown_model_returns_404():
    app = create_app()
    with TestClient(app) as client:
        resp = client.post(
            "/llm/generate",
            json={"model": "unknown:model", "prompt": "hello"},
        )

    assert resp.status_code == 404
