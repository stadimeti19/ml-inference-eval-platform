"""Tests for shadow/canary deployment endpoints."""

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
def shadow_client(tmp_path):
    """TestClient with a prod model (v1.0.0) and a shadow candidate (v2.0.0)."""
    init_db()
    session = get_session()

    for version in ("v1.0.0", "v2.0.0"):
        model = MNISTClassifier()
        model_dir = tmp_path / "artifacts" / "mnist_cnn" / version
        model_dir.mkdir(parents=True)
        model_path = str(model_dir / "model.pt")
        torch.save(model.state_dict(), model_path)
        repo.register_model(
            session,
            model_name="mnist_cnn",
            model_version=version,
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


def test_predict_without_shadow(shadow_client):
    """Normal predict should work exactly as before (no shadow field)."""
    resp = shadow_client.post(
        "/predict",
        json={"model_name": "mnist_cnn", "image_b64": _make_image_b64()},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["prediction"] <= 9
    assert body["model_version"] == "v1.0.0"
    assert body["shadow"] is None


def test_predict_with_shadow(shadow_client):
    """When shadow_version is provided, response includes shadow detail."""
    resp = shadow_client.post(
        "/predict",
        json={
            "model_name": "mnist_cnn",
            "image_b64": _make_image_b64(),
            "shadow_version": "v2.0.0",
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["model_version"] == "v1.0.0"
    assert 0 <= body["prediction"] <= 9

    shadow = body["shadow"]
    assert shadow is not None
    assert shadow["shadow_version"] == "v2.0.0"
    assert 0 <= shadow["shadow_prediction"] <= 9
    assert shadow["shadow_latency_ms"] >= 0
    assert isinstance(shadow["agreed"], bool)


def test_predict_with_missing_shadow_version(shadow_client):
    """Shadow version that doesn't exist should not break prod prediction."""
    resp = shadow_client.post(
        "/predict",
        json={
            "model_name": "mnist_cnn",
            "image_b64": _make_image_b64(),
            "shadow_version": "v99.0.0",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "v1.0.0"
    assert 0 <= body["prediction"] <= 9
    assert body["shadow"] is None


def test_shadow_results_empty(shadow_client):
    """Shadow results endpoint returns zero when no shadow traffic yet."""
    resp = shadow_client.get(
        "/shadow/results",
        params={"model_name": "mnist_cnn", "shadow_version": "v2.0.0"},
    )
    assert resp.status_code == 200
    assert resp.json()["total_comparisons"] == 0


def test_shadow_results_after_traffic(shadow_client):
    """After sending shadow traffic, summary should reflect it."""
    image = _make_image_b64()
    for _ in range(3):
        shadow_client.post(
            "/predict",
            json={
                "model_name": "mnist_cnn",
                "image_b64": image,
                "shadow_version": "v2.0.0",
            },
        )

    resp = shadow_client.get(
        "/shadow/results",
        params={"model_name": "mnist_cnn", "shadow_version": "v2.0.0"},
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_comparisons"] == 3
    assert body["agreements"] + body["disagreements"] == 3
    assert 0.0 <= body["agreement_rate"] <= 1.0
    assert 0.0 <= body["disagreement_rate"] <= 1.0
    assert body["avg_prod_latency_ms"] >= 0
    assert body["avg_shadow_latency_ms"] >= 0


def test_shadow_db_records(shadow_client):
    """Verify shadow results are actually persisted in the database."""
    shadow_client.post(
        "/predict",
        json={
            "model_name": "mnist_cnn",
            "image_b64": _make_image_b64(),
            "shadow_version": "v2.0.0",
        },
    )

    session = get_session()
    try:
        from app.db.models import ShadowResult
        results = session.query(ShadowResult).all()
        assert len(results) == 1
        r = results[0]
        assert r.model_name == "mnist_cnn"
        assert r.prod_version == "v1.0.0"
        assert r.shadow_version == "v2.0.0"
        assert isinstance(r.agreed, bool)
    finally:
        session.close()
