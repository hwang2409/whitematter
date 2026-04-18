#!/usr/bin/env python3
"""Convolutional autoencoder on MNIST."""

import numpy as np
from whitematter import Tensor, no_grad
from whitematter.nn import (
    Module, Sequential, Conv2d, ConvTranspose2d, ReLU, Sigmoid, Flatten, Linear,
    MSELoss,
)
from whitematter.optim import Adam
from whitematter.data import MNIST, DataLoader


class Encoder(Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = Sequential(
            Conv2d(1, 16, 3, stride=2, padding=1),  # -> 14x14
            ReLU(),
            Conv2d(16, 32, 3, stride=2, padding=1),  # -> 7x7
            ReLU(),
            Flatten(),
            Linear(32 * 7 * 7, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = Linear(latent_dim, 32 * 7 * 7)
        self.relu = ReLU()
        self.deconv1 = ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.relu2 = ReLU()
        self.deconv2 = ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        self.sigmoid = Sigmoid()

    def forward(self, z):
        x = self.relu(self.fc(z))
        x = x.reshape(-1, 32, 7, 7)
        x = self.relu2(self.deconv1(x))
        return self.sigmoid(self.deconv2(x))


class Autoencoder(Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


if __name__ == "__main__":
    np.random.seed(42)

    train_data = MNIST(root="./data/mnist", train=True, flatten=False)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = Autoencoder(latent_dim=32)
    print(f"Parameters: {model.num_parameters():,}")

    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = MSELoss()
    EPOCHS = 10

    for epoch in range(EPOCHS):
        total_loss = 0
        n_batches = 0
        for images, _ in train_loader:
            optimizer.zero_grad()
            recon = model(images)
            loss = loss_fn(recon, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch+1}/{EPOCHS} | Recon Loss: {total_loss/n_batches:.6f}")
