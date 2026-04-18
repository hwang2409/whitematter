#!/usr/bin/env python3
"""Basic MLP tutorial — train on XOR to verify everything works."""

import numpy as np
from whitematter import Tensor
from whitematter.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from whitematter.optim import SGD

np.random.seed(42)

# XOR dataset
X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
Y = Tensor(np.array([0, 1, 1, 0]))  # class labels

# Model
model = Sequential(
    Linear(2, 32),
    ReLU(),
    Linear(32, 2),
)
print(f"Parameters: {model.num_parameters()}")

# Training
optimizer = SGD(model.parameters(), lr=0.5, momentum=0.9)
loss_fn = CrossEntropyLoss()

for epoch in range(200):
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, Y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        preds = logits.data.argmax(axis=1)
        acc = (preds == Y.data).mean() * 100
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Accuracy: {acc:.0f}%")

# Final evaluation
logits = model(X)
preds = logits.data.argmax(axis=1)
print(f"\nPredictions: {preds}")
print(f"Targets:     {Y.data.astype(int)}")
print(f"Correct:     {(preds == Y.data).all()}")
