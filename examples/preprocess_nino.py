#!/usr/bin/env python3
"""
Preprocess Nino Nakano dialogue data for GPT training with BPE tokenization.

Input:  data/nino/raw_dialogue.jsonl  (one JSON object per line)
Output: data/nino/train.bin, data/nino/val.bin  (uint16 BPE token IDs)
        data/nino/tokenizer.txt  (BPE merge rules for C++ loader)

Each JSONL line is either:
  Single turn:  {"user": "...", "nino": "..."}
  Multi-turn:   {"turns": [{"user": "...", "nino": "..."}, ...]}
"""

import json
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from bpe_tokenizer import train_bpe, encode, save_tokenizer, save_tokens_bin, load_tokenizer

DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "nino")
INPUT_FILE = os.path.join(DATA_DIR, "raw_dialogue.jsonl")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.txt")

SYSTEM_TAG = "<|system|>"
USER_TAG = "<|user|>"
NINO_TAG = "<|nino|>"

BPE_VOCAB_SIZE = 1024

# System prompts (varied for data augmentation)
SYSTEM_PROMPTS = [
    "You are Nino Nakano from The Quintessential Quintuplets. You are tsundere, proud, initially cold but secretly caring. You love cooking and are fiercely protective of your sisters.",
    "You are Nino Nakano, the second of the Nakano quintuplets. You have a sharp tongue and a proud personality, but deep down you care deeply about the people close to you. You are passionate about cooking.",
    "You are Nino Nakano. You speak bluntly and don't hide your opinions. You're tsundere — cold on the outside, warm on the inside. You love your sisters more than anything and you're an excellent cook.",
    "You are Nino, one of the Nakano quintuplets. You're strong-willed, fashionable, and fiercely loyal. You express affection through cooking and acts of care, even when your words say otherwise.",
    "You are Nino Nakano from Quintessential Quintuplets. You are proud and competitive. You don't trust people easily but once you do, you're devoted. You love cooking for the people you care about.",
]


def build_conversations(entries):
    """Convert raw dialogue entries into full conversation strings."""
    conversations = []

    for entry in entries:
        system_prompt = random.choice(SYSTEM_PROMPTS)

        if "turns" in entry:
            conv = f"{SYSTEM_TAG}{system_prompt}"
            for turn in entry["turns"]:
                conv += f"{USER_TAG}{turn['user']}{NINO_TAG}{turn['nino']}"
            conversations.append(conv)
        else:
            conv = f"{SYSTEM_TAG}{system_prompt}"
            conv += f"{USER_TAG}{entry['user']}{NINO_TAG}{entry['nino']}"
            conversations.append(conv)

    return conversations


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
        print("Create this file with Nino dialogue in JSONL format.")
        print('Each line: {"user": "...", "nino": "..."}')
        print('Or multi-turn: {"turns": [{"user": "...", "nino": "..."}, ...]}')
        return

    # Load dialogue entries
    entries = []
    with open(INPUT_FILE, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping line {line_num}: {e}")

    print(f"Loaded {len(entries)} dialogue entries")

    # Build conversations with system prompts
    # Shuffle and repeat with different system prompts for augmentation
    all_conversations = []
    for _epoch in range(5):  # 5x augmentation with varied system prompts
        random.shuffle(entries)
        all_conversations.extend(build_conversations(entries))

    random.shuffle(all_conversations)

    # Join all conversations into one long text
    full_text = "".join(all_conversations)
    print(f"Total conversation text: {len(full_text)} characters")

    # Train BPE tokenizer
    print(f"\nTraining BPE tokenizer (vocab_size={BPE_VOCAB_SIZE})...")
    merges, vocab = train_bpe(full_text, vocab_size=BPE_VOCAB_SIZE)
    save_tokenizer(TOKENIZER_PATH, merges, vocab)

    # Encode all text to BPE tokens
    print("\nEncoding text to BPE tokens...")
    token_ids = encode(full_text, merges)
    print(f"Total BPE tokens: {len(token_ids)}")
    print(f"Compression ratio: {len(full_text) / len(token_ids):.2f}x (chars/token)")
    print(f"Byte compression: {len(full_text.encode('utf-8')) / len(token_ids):.2f}x (bytes/token)")

    # Verify round-trip
    from bpe_tokenizer import decode
    decoded = decode(token_ids, vocab)
    if decoded == full_text:
        print("Round-trip verification: PASSED")
    else:
        # Find first mismatch
        for i, (a, b) in enumerate(zip(decoded, full_text)):
            if a != b:
                print(f"Round-trip verification: FAILED at position {i}")
                print(f"  Expected: {repr(full_text[max(0,i-10):i+10])}")
                print(f"  Got:      {repr(decoded[max(0,i-10):i+10])}")
                break

    # Stats
    unique_tokens = sorted(set(token_ids))
    print(f"Unique token IDs used: {len(unique_tokens)} (min={min(unique_tokens)}, max={max(unique_tokens)})")

    # 90/10 train/val split
    split = int(len(token_ids) * 0.9)
    train_tokens = token_ids[:split]
    val_tokens = token_ids[split:]
    print(f"\nTrain tokens: {len(train_tokens)}")
    print(f"Val tokens:   {len(val_tokens)}")

    # Write binary files (uint16 little-endian)
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")

    save_tokens_bin(train_path, train_tokens)
    save_tokens_bin(val_path, val_tokens)

    print("\nDone.")


if __name__ == "__main__":
    random.seed(42)
    main()
