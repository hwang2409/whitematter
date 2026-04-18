#!/usr/bin/env python3
"""Transformer example — simple sequence-to-sequence learning."""

import numpy as np
from whitematter import Tensor, no_grad
from whitematter.nn import (
    Module, Embedding, Linear, RMSNorm, ReLU,
    MultiHeadAttention, CrossEntropyLoss,
)
from whitematter.nn.attention import causal_mask
from whitematter.optim import Adam


class TransformerLM(Module):
    """Small transformer language model."""

    def __init__(self, vocab_size=50, embed_dim=64, num_heads=4, num_layers=2, max_len=32):
        super().__init__()
        self.tok_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(max_len, embed_dim)
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = RMSNorm(embed_dim)
        self.ff1 = Linear(embed_dim, embed_dim * 4)
        self.relu = ReLU()
        self.ff2 = Linear(embed_dim * 4, embed_dim)
        self.out_norm = RMSNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, idx):
        N, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(Tensor(np.arange(T)))
        mask = causal_mask(T)
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ff2(self.relu(self.ff1(self.norm2(x))))
        return self.head(self.out_norm(x))


if __name__ == "__main__":
    np.random.seed(42)

    # Learn to repeat: input "123" -> predict "1234"
    VOCAB = 50
    SEQ_LEN = 16
    BATCH = 16

    model = TransformerLM(vocab_size=VOCAB, embed_dim=64, num_heads=4, max_len=SEQ_LEN)
    print(f"Parameters: {model.num_parameters():,}")

    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss()

    for step in range(1, 2001):
        # Random sequences
        data = np.random.randint(0, VOCAB, (BATCH, SEQ_LEN + 1)).astype(np.int64)
        x = Tensor(data[:, :-1])
        y = Tensor(data[:, 1:])

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")

    print("\nTraining complete. Loss should approach ~log(50) ≈ 3.91 (random baseline)")
    print("For a real task, you'd train on structured data where patterns exist.")
