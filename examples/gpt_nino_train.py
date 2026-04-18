#!/usr/bin/env python3
"""GPT-Nino training — BPE language model on dialogue data.

Requires: data/nino/train.bin, data/nino/val.bin, data/nino/tokenizer.txt
Generate with: python examples/preprocess_nino.py
"""

import os
import sys
import numpy as np
from whitematter import Tensor, no_grad
from whitematter.nn import (
    Module, Embedding, Linear, RMSNorm, SiLU, Dropout,
    MultiHeadAttention, CrossEntropyLoss,
)
from whitematter.nn.attention import causal_mask
from whitematter.optim import AdamW, LinearWarmupCosineDecay, clip_grad_norm_, GradientAccumulator
import whitematter.serialization as serial

sys.path.insert(0, os.path.dirname(__file__))
from bpe_tokenizer import load_tokenizer

# --- Model ---

class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = RMSNorm(embed_dim)
        self.mlp_up = Linear(embed_dim, embed_dim * 4)
        self.mlp_act = SiLU()
        self.mlp_down = Linear(embed_dim * 4, embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        h = self.mlp_down(self.mlp_act(self.mlp_up(self.norm2(x))))
        x = x + self.dropout(h)
        return x


class NinoGPT(Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.tok_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(max_seq_len, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        for i, b in enumerate(self.blocks):
            self.register_module(f"block_{i}", b)
        self.norm = RMSNorm(embed_dim)
        # Weight-tied head: shares tok_emb.weight
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def forward(self, idx):
        N, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(Tensor(np.arange(T)))
        x = tok + pos
        mask = causal_mask(T)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        # Tied head: logits = x @ tok_emb.weight^T
        return x.matmul(self.tok_emb.weight.transpose())


# --- Data ---

def load_bin(path):
    data = np.fromfile(path, dtype=np.uint16).astype(np.int64)
    return data


def sample_batch(data, batch_size, seq_len):
    starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts])
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return Tensor(x), Tensor(y)


if __name__ == "__main__":
    np.random.seed(42)

    VOCAB_SIZE = 1024
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    SEQ_LEN = 256
    BATCH_SIZE = 32
    ACCUM_STEPS = 4
    TOTAL_STEPS = 30000
    WARMUP = 500
    LR = 1e-4
    SAVE_EVERY = 1000
    SAMPLE_EVERY = 500
    MODEL_PATH = "models/nino_gpt.npz"

    train_data = load_bin("data/nino/train.bin")
    val_data = load_bin("data/nino/val.bin")
    merges, vocab = load_tokenizer("data/nino/tokenizer.txt")
    print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    model = NinoGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN)
    print(f"Parameters: {model.num_parameters():,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = LinearWarmupCosineDecay(optimizer, WARMUP, TOTAL_STEPS)
    loss_fn = CrossEntropyLoss()
    accumulator = GradientAccumulator(ACCUM_STEPS)

    os.makedirs("models", exist_ok=True)

    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        optimizer.zero_grad()

        total_loss = 0
        for _ in range(ACCUM_STEPS):
            x, y = sample_batch(train_data, BATCH_SIZE, SEQ_LEN)
            logits = model(x).reshape(-1, VOCAB_SIZE)
            loss = loss_fn(logits, y.reshape(-1))
            accumulator.backward(loss)
            total_loss += loss.item()
        accumulator.reset()

        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        avg_loss = total_loss / ACCUM_STEPS

        if step % 100 == 0:
            model.eval()
            with no_grad():
                vx, vy = sample_batch(val_data, BATCH_SIZE, SEQ_LEN)
                vloss = loss_fn(model(vx).reshape(-1, VOCAB_SIZE), vy.reshape(-1))
            print(f"Step {step:5d} | Train: {avg_loss:.4f} | Val: {vloss.item():.4f} | LR: {optimizer.lr:.6f}")

        if step % SAVE_EVERY == 0:
            serial.save(MODEL_PATH, model)
            print(f"  Saved {MODEL_PATH}")
