#!/usr/bin/env python3
"""Train a small MNIST classifier and register it as v1.0.0."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    from app.inference.model import train_mnist_model
    from app.registry.manager import register, promote

    print("=== Training MNIST classifier ===")
    saved_path = train_mnist_model(epochs=3, lr=1e-3)

    git_sha: str | None = None
    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        pass

    print("=== Registering model ===")
    mv = register(
        model_name="mnist_cnn",
        model_version="v1.0.0",
        artifact_path=saved_path,
        git_sha=git_sha,
        tags={"framework": "pytorch", "dataset": "mnist"},
    )
    print(f"Registered: {mv.model_name}@{mv.model_version}")

    print("=== Promoting to prod ===")
    promote(model_name="mnist_cnn", model_version="v1.0.0")
    print("Done. Model is now in production.")


if __name__ == "__main__":
    main()
