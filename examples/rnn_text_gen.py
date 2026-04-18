#!/usr/bin/env python3
"""RNN/LSTM text generation — character-level on Shakespeare."""

import os
import numpy as np
from whitematter import Tensor, no_grad
from whitematter.nn import Module, Embedding, LSTM, Linear, CrossEntropyLoss
from whitematter.optim import Adam, clip_grad_norm_


class CharLSTM(Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_size, num_layers=num_layers)
        self.head = Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)  # (N, T, embed_dim)
        out, hidden = self.lstm(emb, hidden)  # (N, T, hidden_size)
        logits = self.head(out)  # (N, T, vocab_size)
        return logits, hidden


if __name__ == "__main__":
    np.random.seed(42)

    # Load data
    data_path = "data/shakespeare/input.txt"
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    with open(data_path) as f:
        text = f.read()

    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(chars)
    data = np.array([char2idx[c] for c in text], dtype=np.int64)
    print(f"Vocab: {vocab_size} chars | Data: {len(data):,} chars")

    SEQ_LEN = 100
    BATCH_SIZE = 32
    HIDDEN = 256
    EMBED = 64
    STEPS = 5000
    LR = 2e-3

    model = CharLSTM(vocab_size, EMBED, HIDDEN, num_layers=2)
    print(f"Parameters: {model.num_parameters():,}")

    optimizer = Adam(model.parameters(), lr=LR)
    loss_fn = CrossEntropyLoss()

    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        starts = np.random.randint(0, len(data) - SEQ_LEN - 1, BATCH_SIZE)
        x = Tensor(np.stack([data[s : s + SEQ_LEN] for s in starts]))
        y = Tensor(np.stack([data[s + 1 : s + SEQ_LEN + 1] for s in starts]))

        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step:5d} | Loss: {loss.item():.4f}")

        if step % 1000 == 0:
            # Generate sample
            model.eval()
            seed = "ROMEO: "
            ids = [char2idx[c] for c in seed]
            hidden = None
            with no_grad():
                for _ in range(200):
                    x = Tensor(np.array([[ids[-1]]], dtype=np.int64))
                    logits, hidden = model(x, hidden)
                    probs = np.exp(logits.data[0, -1] / 0.8)
                    probs /= probs.sum()
                    next_id = np.random.choice(vocab_size, p=probs)
                    ids.append(int(next_id))
            sample = "".join(idx2char[i] for i in ids)
            print(f"--- Sample ---\n{sample}\n--------------")
            model.train()
