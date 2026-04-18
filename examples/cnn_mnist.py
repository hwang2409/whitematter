#!/usr/bin/env python3
"""CNN on MNIST — demonstrates convolutions, pooling, batch norm."""

import numpy as np
from whitematter import Tensor, no_grad
from whitematter.nn import (
    Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear,
    CrossEntropyLoss,
)
from whitematter.optim import Adam
from whitematter.data import MNIST, DataLoader

np.random.seed(42)

# Load data
train_data = MNIST(root="./data/mnist", train=True, flatten=False)
test_data = MNIST(root="./data/mnist", train=False, flatten=False)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# Model: Conv -> BN -> ReLU -> Pool, repeat, then FC
model = Sequential(
    Conv2d(1, 16, 3, padding=1),   # -> (N, 16, 28, 28)
    BatchNorm2d(16),
    ReLU(),
    MaxPool2d(2),                   # -> (N, 16, 14, 14)
    Conv2d(16, 32, 3, padding=1),  # -> (N, 32, 14, 14)
    BatchNorm2d(32),
    ReLU(),
    MaxPool2d(2),                   # -> (N, 32, 7, 7)
    Flatten(),                      # -> (N, 1568)
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Linear(128, 10),
)
print(f"Parameters: {model.num_parameters():,}")

optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = CrossEntropyLoss()
EPOCHS = 5

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

    # Validation
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
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/n_batches:.4f} | Test Acc: {acc:.2f}%")
