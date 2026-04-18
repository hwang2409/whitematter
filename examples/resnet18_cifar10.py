#!/usr/bin/env python3
"""ResNet-18 on CIFAR-10 — demonstrates residual connections."""

import numpy as np
from whitematter import Tensor, no_grad
from whitematter.nn import (
    Module, Conv2d, BatchNorm2d, ReLU, Sequential,
    AdaptiveAvgPool2d, Flatten, Linear, CrossEntropyLoss,
)
from whitematter.optim import AdamW, LinearWarmupCosineDecay
from whitematter.data import CIFAR10, DataLoader


class BasicBlock(Module):
    """ResNet basic block with skip connection."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(out_ch)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(out_ch)

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = Sequential(
                Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(x)
        out = out + identity
        return self.relu(out)


class ResNet18(Module):
    """ResNet-18 for CIFAR-10 (smaller initial conv, no maxpool)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()

        self.layer1 = Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer4 = Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))

        self.pool = AdaptiveAvgPool2d(1)
        self.flatten = Flatten()
        self.fc = Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)


if __name__ == "__main__":
    np.random.seed(42)

    train_data = CIFAR10(root="./data/cifar10", train=True)
    test_data = CIFAR10(root="./data/cifar10", train=False)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    model = ResNet18()
    print(f"Parameters: {model.num_parameters():,}")

    EPOCHS = 200
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = LinearWarmupCosineDecay(optimizer, warmup_steps=5, total_steps=EPOCHS)
    loss_fn = CrossEntropyLoss()
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
        correct = total = 0
        with no_grad():
            for images, labels in test_loader:
                preds = model(images).data.argmax(axis=1)
                correct += (preds == labels.data).sum()
                total += len(labels)

        acc = correct / total * 100
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {total_loss/n_batches:.4f} | "
                  f"Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
