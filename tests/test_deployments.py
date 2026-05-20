"""Tests for deployment audit events and rollback flow."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.db import repositories as repo
from app.db.session import get_session, init_db
from app.registry.manager import promote, register, rollback_with_summary


def test_register_promote_and_rollback_write_deployment_events(sample_model):
    init_db()
    register("mnist_cnn", "v1.0.0", sample_model)
    register("mnist_cnn", "v2.0.0", sample_model)
    promote("mnist_cnn", "v1.0.0")
    promote("mnist_cnn", "v2.0.0")

    summary = rollback_with_summary("mnist_cnn")
    assert summary.rolled_back is True
    assert summary.previous_prod_version == "v2.0.0"
    assert summary.new_prod_version == "v1.0.0"

    session = get_session()
    try:
        events = repo.list_deployment_events(session, model_name="mnist_cnn")
        event_types = [e.event_type for e in events]
        assert "register" in event_types
        assert "promote" in event_types
        assert "rollback" in event_types
    finally:
        session.close()


def test_deployment_events_and_rollback_endpoints(sample_model):
    init_db()
    register("mnist_cnn", "v1.0.0", sample_model)
    register("mnist_cnn", "v2.0.0", sample_model)
    promote("mnist_cnn", "v1.0.0")
    promote("mnist_cnn", "v2.0.0")

    app = create_app()
    with TestClient(app) as client:
        events = client.get("/deployments/events")
        assert events.status_code == 200
        assert len(events.json()) >= 4

        resp = client.post("/models/mnist_cnn/rollback")
        assert resp.status_code == 200
        body = resp.json()
        assert body["rolled_back"] is True
        assert body["new_prod_version"] == "v1.0.0"
