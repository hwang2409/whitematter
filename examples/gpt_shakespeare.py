#!/usr/bin/env python3
"""GPT on Shakespeare — character-level language model."""

import os
import numpy as np
from whitematter import Tensor, no_grad
from whitematter.nn import (
    Module, Embedding, Linear, RMSNorm, SiLU, Dropout,
    MultiHeadAttention, CrossEntropyLoss,
)
from whitematter.nn.attention import causal_mask
from whitematter.optim import AdamW, LinearWarmupCosineDecay, clip_grad_norm_


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


class GPT(Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.tok_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(max_seq_len, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        for i, b in enumerate(self.blocks):
            self.register_module(f"block_{i}", b)
        self.norm = RMSNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        N, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(Tensor(np.arange(T)))
        x = tok + pos
        mask = causal_mask(T)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.norm(x)
        return self.head(x)


def load_data(path="data/shakespeare/input.txt"):
    """Load TinyShakespeare as uint8 byte tokens."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)
    with open(path, "r") as f:
        text = f.read()
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)
    n = int(len(data) * 0.9)
    return data[:n], data[n:]


def sample_batch(data, batch_size, seq_len):
    starts = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts])
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return Tensor(x), Tensor(y)


def generate(model, prompt_ids, max_tokens=200, temperature=0.8):
    model.eval()
    ids = list(prompt_ids)
    with no_grad():
        for _ in range(max_tokens):
            context = ids[-model.max_seq_len :]
            x = Tensor(np.array([context], dtype=np.int64))
            logits = model(x).data[0, -1, :]  # last token
            logits /= temperature
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            next_id = np.random.choice(len(probs), p=probs)
            ids.append(int(next_id))
    return bytes(ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    np.random.seed(42)

    VOCAB_SIZE = 256
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    SEQ_LEN = 128
    BATCH_SIZE = 32
    TOTAL_STEPS = 5000
    WARMUP = 100
    LR = 3e-4

    train_data, val_data = load_data()
    print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    model = GPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN)
    print(f"Parameters: {model.num_parameters():,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = LinearWarmupCosineDecay(optimizer, WARMUP, TOTAL_STEPS)
    loss_fn = CrossEntropyLoss()

    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        optimizer.zero_grad()

        x, y = sample_batch(train_data, BATCH_SIZE, SEQ_LEN)
        logits = model(x)
        # Reshape for cross-entropy: (N*T, vocab) vs (N*T,)
        logits_flat = logits.reshape(-1, VOCAB_SIZE)
        targets_flat = y.reshape(-1)
        loss = loss_fn(logits_flat, targets_flat)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            # Validation
            model.eval()
            with no_grad():
                vx, vy = sample_batch(val_data, BATCH_SIZE, SEQ_LEN)
                vlogits = model(vx).reshape(-1, VOCAB_SIZE)
                vloss = loss_fn(vlogits, vy.reshape(-1))
            print(f"Step {step:5d} | Train: {loss.item():.4f} | Val: {vloss.item():.4f} | LR: {optimizer.lr:.6f}")

        if step % 500 == 0:
            prompt = "ROMEO:"
            text = generate(model, list(prompt.encode("utf-8")))
            print(f"--- Sample ---\n{text[:300]}\n--------------")
