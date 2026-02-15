"""Tests for FastAPI endpoints using TestClient."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import create_app
from app.db import repositories as repo
from app.db.session import get_session, init_db
from app.inference.model import MNISTClassifier


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """FastAPI TestClient with temporary DB."""
    init_db()

    # Register and promote a dummy model
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
    )
    repo.promote_model(session, model_name="mnist_cnn", model_version="v1.0.0")
    session.close()

    app = create_app()
    with TestClient(app) as tc:
        yield tc


def _make_image_b64() -> str:
    arr = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "request_latency_seconds" in resp.text


def test_list_models(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["model_name"] == "mnist_cnn"


def test_predict(client):
    image_b64 = _make_image_b64()
    resp = client.post(
        "/predict",
        json={"model_name": "mnist_cnn", "image_b64": image_b64},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["prediction"] <= 9
    assert body["latency_ms"] >= 0
    assert body["model_version"] == "v1.0.0"


def test_predict_missing_model(client):
    resp = client.post(
        "/predict",
        json={"model_name": "nonexistent", "image_b64": "aGVsbG8="},
    )
    assert resp.status_code == 404
