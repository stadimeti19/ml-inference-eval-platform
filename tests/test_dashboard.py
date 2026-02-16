"""Tests for the dashboard UI endpoints."""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.db import repositories as repo
from app.db.session import get_session, init_db
from app.inference.model import MNISTClassifier


@pytest.fixture()
def dash_client(tmp_path):
    """TestClient with a registered model for dashboard tests."""
    init_db()
    session = get_session()

    model = MNISTClassifier()
    model_dir = tmp_path / "artifacts" / "mnist_cnn" / "v1.0.0"
    model_dir.mkdir(parents=True)
    model_path = str(model_dir / "model.pt")
    torch.save(model.state_dict(), model_path)

    repo.register_model(
        session,
        model_name="mnist_cnn",
        model_version="v1.0.0",
        artifact_path=model_path,
        metrics={"accuracy": 0.97, "p95_ms": 12.5},
    )
    repo.promote_model(session, model_name="mnist_cnn", model_version="v1.0.0")
    session.close()

    app = create_app()
    with TestClient(app) as tc:
        yield tc


def test_dashboard_loads(dash_client):
    """Dashboard renders HTML with model data."""
    resp = dash_client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "ML Platform" in resp.text
    assert "mnist_cnn" in resp.text
    assert "v1.0.0" in resp.text


def test_dashboard_promote(dash_client):
    """Promote action via dashboard API works."""
    session = get_session()
    repo.register_model(
        session,
        model_name="mnist_cnn",
        model_version="v2.0.0",
        artifact_path="/tmp/m.pt",
    )
    session.close()

    resp = dash_client.post(
        "/dashboard/api/promote",
        json={"model_name": "mnist_cnn", "model_version": "v2.0.0"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "prod"


def test_dashboard_rollback(dash_client):
    """Rollback action via dashboard API works."""
    resp = dash_client.post(
        "/dashboard/api/rollback",
        json={"model_name": "mnist_cnn"},
    )
    assert resp.status_code == 200


def test_dashboard_empty_state(tmp_path):
    """Dashboard renders fine with no data."""
    init_db()
    app = create_app()
    with TestClient(app) as tc:
        resp = tc.get("/dashboard")
        assert resp.status_code == 200
        assert "No models registered" in resp.text
