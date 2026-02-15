"""Prediction utilities for single and batch inference."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from app.datasets.mnist import MNIST_TRANSFORM
from app.inference.model import MNISTClassifier


@dataclass
class BatchResult:
    predictions: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    total_time_s: float = 0.0


def _preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Decode raw image bytes, convert to grayscale 28x28 tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    tensor = MNIST_TRANSFORM(img)  # (1, 28, 28)
    return tensor.unsqueeze(0)  # (1, 1, 28, 28)


def predict_single(
    model: MNISTClassifier, image_b64: str
) -> tuple[int, float]:
    """Run single-image inference.

    Args:
        model: Loaded MNISTClassifier.
        image_b64: Base64-encoded image.

    Returns:
        (predicted_class, latency_ms)
    """
    image_bytes = base64.b64decode(image_b64)
    tensor = _preprocess_image(image_bytes)

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    predicted_class = int(output.argmax(dim=1).item())
    return predicted_class, round(elapsed_ms, 3)


def predict_batch(
    model: MNISTClassifier,
    dataloader: DataLoader,
) -> BatchResult:
    """Run inference over an entire DataLoader.

    Returns:
        BatchResult with per-sample predictions, labels, and latencies.
    """
    device = next(model.parameters()).device
    result = BatchResult()
    overall_start = time.perf_counter()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            batch_start = time.perf_counter()
            outputs = model(images)
            batch_ms = (time.perf_counter() - batch_start) * 1000.0

            preds = outputs.argmax(dim=1).cpu().tolist()
            per_sample_ms = batch_ms / len(preds)

            result.predictions.extend(preds)
            result.labels.extend(labels.tolist())
            result.latencies_ms.extend([per_sample_ms] * len(preds))

    result.total_time_s = time.perf_counter() - overall_start
    return result
