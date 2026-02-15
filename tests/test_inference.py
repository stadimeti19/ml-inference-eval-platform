"""Tests for inference predict logic."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from app.inference.model import MNISTClassifier, load_model
from app.inference.predict import predict_single


def _make_test_image_b64() -> str:
    arr = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_predict_single(sample_model: str):
    model = load_model(sample_model)
    image_b64 = _make_test_image_b64()
    prediction, latency_ms = predict_single(model, image_b64)
    assert 0 <= prediction <= 9
    assert latency_ms >= 0


def test_model_loads(sample_model: str):
    model = load_model(sample_model)
    assert isinstance(model, MNISTClassifier)
