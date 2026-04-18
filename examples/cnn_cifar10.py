#!/usr/bin/env python3
"""VGG-style CNN on CIFAR-10."""

import numpy as np
from whitematter import Tensor, no_grad
from whitematter.nn import (
    Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear,
    Dropout, CrossEntropyLoss,
)
from whitematter.optim import Adam, CosineAnnealingLR
from whitematter.data import CIFAR10, DataLoader

np.random.seed(42)

train_data = CIFAR10(root="./data/cifar10", train=True)
test_data = CIFAR10(root="./data/cifar10", train=False)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# VGG-style architecture
model = Sequential(
    # Block 1
    Conv2d(3, 64, 3, padding=1), BatchNorm2d(64), ReLU(),
    Conv2d(64, 64, 3, padding=1), BatchNorm2d(64), ReLU(),
    MaxPool2d(2),  # -> 16x16
    # Block 2
    Conv2d(64, 128, 3, padding=1), BatchNorm2d(128), ReLU(),
    Conv2d(128, 128, 3, padding=1), BatchNorm2d(128), ReLU(),
    MaxPool2d(2),  # -> 8x8
    # Block 3
    Conv2d(128, 256, 3, padding=1), BatchNorm2d(256), ReLU(),
    Conv2d(256, 256, 3, padding=1), BatchNorm2d(256), ReLU(),
    MaxPool2d(2),  # -> 4x4
    # Classifier
    Flatten(),
    Linear(256 * 4 * 4, 512), ReLU(), Dropout(0.5),
    Linear(512, 10),
)
print(f"Parameters: {model.num_parameters():,}")

optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=50)
loss_fn = CrossEntropyLoss()
EPOCHS = 50
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    n_batches = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with no_grad():
        for images, labels in test_loader:
            logits = model(images)
            preds = logits.data.argmax(axis=1)
            correct += (preds == labels.data).sum()
            total += len(labels)

    acc = correct / total * 100
    best_acc = max(best_acc, acc)
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/n_batches:.4f} | "
          f"Acc: {acc:.2f}% | Best: {best_acc:.2f}% | LR: {optimizer.lr:.6f}")
