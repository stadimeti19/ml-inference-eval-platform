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


class MNISTClassifier(nn.Module):
    """Small convolutional network for MNIST digit classification."""

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


def train_mnist_model(
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 64,
    save_dir: str | None = None,
) -> str:
    """Train the MNIST classifier and save the checkpoint.

    Returns:
        Path to the saved model file.
    """
    from app.datasets.mnist import get_mnist_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTClassifier().to(device)
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
        save_dir = os.path.join(settings.model_artifacts_dir, "mnist_cnn", "v1.0.0")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info("model_saved", path=save_path)
    return save_path


def load_model(artifact_path: str) -> MNISTClassifier:
    """Load a trained MNISTClassifier from *artifact_path*."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTClassifier()
    model.load_state_dict(torch.load(artifact_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model
