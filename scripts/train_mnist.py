#!/usr/bin/env python3
"""Train an MNIST classifier and register it in the model registry.

Usage:
    python scripts/train_mnist.py                                   # default v1.0.0
    python scripts/train_mnist.py --model_version v2.0.0 --architecture large
    python scripts/train_mnist.py --model_version v2.0.0 --architecture large --epochs 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register an MNIST model")
    parser.add_argument("--model_name", default="mnist_cnn", help="Model name")
    parser.add_argument("--model_version", default="v1.0.0", help="Version string")
    parser.add_argument(
        "--architecture",
        default="default",
        choices=["default", "large"],
        help="Model architecture: 'default' (2 conv) or 'large' (3 conv, wider)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--promote", action="store_true", default=False,
        help="Promote this version to prod after registration",
    )
    args = parser.parse_args()

    from app.inference.model import train_mnist_model
    from app.registry.manager import register, promote

    print(f"=== Training {args.model_name}@{args.model_version} "
          f"(arch={args.architecture}, epochs={args.epochs}) ===")

    saved_path = train_mnist_model(
        epochs=args.epochs,
        lr=args.lr,
        architecture=args.architecture,
        model_name=args.model_name,
        model_version=args.model_version,
    )

    git_sha: str | None = None
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        pass

    print("=== Registering model ===")
    mv = register(
        model_name=args.model_name,
        model_version=args.model_version,
        artifact_path=saved_path,
        git_sha=git_sha,
        tags={
            "framework": "pytorch",
            "dataset": "mnist",
            "architecture": args.architecture,
        },
    )
    print(f"Registered: {mv.model_name}@{mv.model_version} (arch={args.architecture})")

    if args.promote:
        print("=== Promoting to prod ===")
        promote(model_name=args.model_name, model_version=args.model_version)
        print(f"Done. {args.model_name}@{args.model_version} is now in production.")
    else:
        print(f"Done. Model registered as staging. Use 'promote' to push to prod.")


if __name__ == "__main__":
    main()
