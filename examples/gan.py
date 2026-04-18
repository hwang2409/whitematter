#!/usr/bin/env python3
"""Simple GAN on MNIST — generates handwritten digits."""

import numpy as np
from whitematter import Tensor
from whitematter.nn import (
    Module, Sequential, Linear, ReLU, Sigmoid, Tanh,
    BCEWithLogitsLoss,
)
from whitematter.optim import Adam
from whitematter.data import MNIST, DataLoader


class Generator(Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = Sequential(
            Linear(latent_dim, 256), ReLU(),
            Linear(256, 512), ReLU(),
            Linear(512, 784), Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.net = Sequential(
            Linear(784, 512), ReLU(),
            Linear(512, 256), ReLU(),
            Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    np.random.seed(42)
    LATENT_DIM = 64
    BATCH_SIZE = 64
    EPOCHS = 50

    train_data = MNIST(root="./data/mnist", train=True, flatten=True)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator(LATENT_DIM)
    D = Discriminator()
    print(f"G params: {G.num_parameters():,} | D params: {D.num_parameters():,}")

    opt_G = Adam(G.parameters(), lr=2e-4)
    opt_D = Adam(D.parameters(), lr=2e-4)
    loss_fn = BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        d_loss_total = g_loss_total = 0
        n = 0

        for real_imgs, _ in train_loader:
            bs = real_imgs.shape[0]
            # Scale to [-1, 1]
            real_imgs = real_imgs * 2 - 1

            # Train Discriminator
            opt_D.zero_grad()
            z = Tensor(np.random.randn(bs, LATENT_DIM).astype(np.float32))
            fake_imgs = G(z)

            real_preds = D(real_imgs)
            fake_preds = D(fake_imgs.detach())

            real_labels = Tensor(np.ones((bs, 1), dtype=np.float32))
            fake_labels = Tensor(np.zeros((bs, 1), dtype=np.float32))

            d_loss = loss_fn(real_preds, real_labels) + loss_fn(fake_preds, fake_labels)
            d_loss.backward()
            opt_D.step()

            # Train Generator
            opt_G.zero_grad()
            z = Tensor(np.random.randn(bs, LATENT_DIM).astype(np.float32))
            fake_imgs = G(z)
            fake_preds = D(fake_imgs)
            g_loss = loss_fn(fake_preds, real_labels)
            g_loss.backward()
            opt_G.step()

            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
            n += 1

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | D Loss: {d_loss_total/n:.4f} | G Loss: {g_loss_total/n:.4f}")
