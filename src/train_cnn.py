"""Pelatihan CNN kecil PyTorch yang dirancang untuk CPU dan penghentian dini."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int, cpu_threads: int = 4) -> None:
    """Menetapkan seed Python, NumPy, dan PyTorch untuk pengulangan."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(cpu_threads)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageDataset(Dataset):
    """Mengubah RGB ke CHW dan memakai flip/rotasi 90° pada data latih."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255.0
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.augment = augment

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, dims=(2,))
            if random.random() < 0.5:
                image = torch.flip(image, dims=(1,))
            image = torch.rot90(image, k=random.randrange(4), dims=(1, 2))
        return image, self.labels[index]


class SmallCNN(nn.Module):
    """CNN tiga blok konvolusi dengan pooling adaptif agar mudah dipahami."""

    def __init__(self, class_count: int, base_channels: int = 16):
        super().__init__()
        first, second, third = base_channels, base_channels * 2, base_channels * 4
        self.features = nn.Sequential(
            nn.Conv2d(3, first, kernel_size=3, padding=1),
            nn.BatchNorm2d(first),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(first, second, kernel_size=3, padding=1),
            nn.BatchNorm2d(second),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(second, third, kernel_size=3, padding=1),
            nn.BatchNorm2d(third),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(third, class_count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


@dataclass
class CNNTrainingResult:
    """Model terlatih dan informasi waktu/parameter/early stopping."""

    model: SmallCNN
    training_seconds: float
    parameter_count: int
    best_validation_loss: float
    epochs_trained: int
    best_epoch: int


def _evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Menghitung loss validasi tanpa memperbarui parameter."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_function(model(images), labels)
            total_loss += float(loss.item()) * len(labels)
            total_items += len(labels)
    return total_loss / max(total_items, 1)


def train_cnn_model(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    settings: dict[str, Any],
    class_count: int,
    seed: int,
) -> CNNTrainingResult:
    """Melatih CNN dengan augmentasi dan early stopping pada validasi."""
    set_seed(seed, int(settings["cpu_threads"]))
    device = torch.device("cpu")
    batch_size = int(settings["batch_size"])
    train_loader = DataLoader(
        ImageDataset(train_images, train_labels, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        ImageDataset(validation_images, validation_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = SmallCNN(class_count, int(settings["base_channels"])).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    start = perf_counter()
    maximum_epochs = int(settings["epochs"])

    for epoch in range(maximum_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
        validation_loss = _evaluate_loss(model, validation_loader, loss_function, device)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(settings["patience"]):
                break

    training_seconds = perf_counter() - start
    model.load_state_dict(best_weights)
    return CNNTrainingResult(
        model=model,
        training_seconds=training_seconds,
        parameter_count=parameter_count,
        best_validation_loss=best_loss,
        epochs_trained=epoch + 1,
        best_epoch=best_epoch,
    )


def predict_cnn(
    model: SmallCNN,
    images: np.ndarray,
    batch_size: int = 128,
) -> tuple[np.ndarray, float]:
    """Menghasilkan probabilitas kelas dan mengukur waktu inferensi."""
    loader = DataLoader(
        ImageDataset(images, np.zeros(len(images), dtype=np.int64)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    probabilities: list[np.ndarray] = []
    start = perf_counter()
    with torch.inference_mode():
        for batch, _ in loader:
            logits = model(batch)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    elapsed = perf_counter() - start
    return np.concatenate(probabilities), elapsed
