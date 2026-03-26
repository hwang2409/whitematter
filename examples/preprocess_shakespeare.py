#!/usr/bin/env python3
"""Download TinyShakespeare and save as raw byte tokens for GPT training."""

import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    txt_path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(txt_path):
        print(f"Downloading TinyShakespeare to {txt_path} ...")
        urllib.request.urlretrieve(URL, txt_path)
    else:
        print(f"Already downloaded: {txt_path}")

    with open(txt_path, "r") as f:
        text = f.read()
    print(f"Total characters: {len(text)}")

    # Character-level tokenization: each byte is a token (0-255)
    tokens = text.encode("utf-8")  # bytes object
    print(f"Total tokens (bytes): {len(tokens)}")

    # Print vocabulary stats
    unique = sorted(set(tokens))
    print(f"Unique byte values: {len(unique)}  (min={min(unique)}, max={max(unique)})")

    # 90/10 train/val split
    split = int(len(tokens) * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Val tokens:   {len(val_tokens)}")

    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")

    with open(train_path, "wb") as f:
        f.write(train_tokens)
    with open(val_path, "wb") as f:
        f.write(val_tokens)

    print(f"Saved {train_path} ({len(train_tokens)} bytes)")
    print(f"Saved {val_path} ({len(val_tokens)} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
