"""MNIST dataset download and loading utilities."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def get_mnist_train(download_dir: str = "./data") -> datasets.MNIST:
    """Return the full MNIST training set."""
    return datasets.MNIST(
        root=download_dir, train=True, download=True, transform=MNIST_TRANSFORM
    )


def get_mnist_test(download_dir: str = "./data") -> datasets.MNIST:
    """Return the full MNIST test set."""
    return datasets.MNIST(
        root=download_dir, train=False, download=True, transform=MNIST_TRANSFORM
    )


def get_mnist_loader(
    batch_size: int = 64,
    train: bool = True,
    download_dir: str = "./data",
) -> DataLoader:
    """Return a DataLoader over the MNIST train or test split."""
    ds = get_mnist_train(download_dir) if train else get_mnist_test(download_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=train)


def get_mnist_subset(
    n: int = 1000,
    train: bool = False,
    download_dir: str = "./data",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first *n* images and labels from the MNIST test set.

    Returns:
        (images, labels) each of shape (n, ...).
    """
    ds = get_mnist_test(download_dir) if not train else get_mnist_train(download_dir)
    subset = Subset(ds, list(range(min(n, len(ds)))))
    loader = DataLoader(subset, batch_size=n, shuffle=False)
    images, labels = next(iter(loader))
    return images, labels
