"""MNIST classifier model definition, training, and loading."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Architecture registry: name -> class
ARCHITECTURES: dict[str, type[nn.Module]] = {}


def _register_arch(name: str):
    """Decorator to register a model class under *name*."""
    def wrapper(cls: type[nn.Module]) -> type[nn.Module]:
        ARCHITECTURES[name] = cls
        return cls
    return wrapper


@_register_arch("default")
class MNISTClassifier(nn.Module):
    """Small convolutional network for MNIST digit classification (2 conv layers)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # (B,16,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # (B,32,7,7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@_register_arch("large")
class MNISTClassifierLarge(nn.Module):
    """Larger convolutional network (3 conv layers, wider).

    More parameters → potentially higher accuracy but slower inference.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After pool-pool: 7x7, third conv keeps 7x7 (no pool after it)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B,32,14,14)
        x = self.pool(F.relu(self.conv2(x)))   # (B,64,7,7)
        x = F.relu(self.conv3(x))              # (B,64,7,7)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def _get_arch_class(architecture: str) -> type[nn.Module]:
    """Look up a model class by architecture name."""
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available: {list(ARCHITECTURES.keys())}"
        )
    return ARCHITECTURES[architecture]


def train_mnist_model(
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 64,
    save_dir: str | None = None,
    architecture: str = "default",
    model_name: str = "mnist_cnn",
    model_version: str = "v1.0.0",
) -> str:
    """Train an MNIST classifier and save the checkpoint.

    Args:
        architecture: One of 'default' or 'large'.
        model_name: Name used in the artifact path.
        model_version: Version used in the artifact path.

    Returns:
        Path to the saved model file.
    """
    from app.datasets.mnist import get_mnist_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = _get_arch_class(architecture)
    model = model_cls().to(device)

    logger.info(
        "training_started",
        architecture=architecture,
        params=sum(p.numel() for p in model.parameters()),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = get_mnist_loader(batch_size=batch_size, train=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        logger.info(
            "training_epoch",
            epoch=epoch + 1,
            loss=round(avg_loss, 4),
            accuracy=round(accuracy, 4),
        )

    if save_dir is None:
        settings = get_settings()
        save_dir = os.path.join(settings.model_artifacts_dir, model_name, model_version)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info("model_saved", path=save_path, architecture=architecture)
    return save_path


def load_model(artifact_path: str, architecture: str = "default") -> nn.Module:
    """Load a trained model from *artifact_path*.

    Args:
        artifact_path: Path to the .pt state dict file.
        architecture: Architecture name ('default' or 'large') to instantiate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = _get_arch_class(architecture)
    model = model_cls()
    model.load_state_dict(torch.load(artifact_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model
