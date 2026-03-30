#!/usr/bin/env python3
"""
Preprocess Nino Nakano dialogue data for GPT training.

Input:  data/nino/raw_dialogue.jsonl  (one JSON object per line)
Output: data/nino/train.bin, data/nino/val.bin  (raw byte tokens)

Each JSONL line is either:
  Single turn:  {"user": "...", "nino": "..."}
  Multi-turn:   {"turns": [{"user": "...", "nino": "..."}, ...]}

The script wraps conversations with system prompts and delimiters:
  <|system|>...<|nino|>...<|user|>...<|nino|>...<|user|>
"""

import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "nino")
INPUT_FILE = os.path.join(DATA_DIR, "raw_dialogue.jsonl")

SYSTEM_TAG = "<|system|>"
USER_TAG = "<|user|>"
NINO_TAG = "<|nino|>"

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
        # Pick a random system prompt for variety
        system_prompt = random.choice(SYSTEM_PROMPTS)

        if "turns" in entry:
            # Multi-turn conversation
            conv = f"{SYSTEM_TAG}{system_prompt}"
            for turn in entry["turns"]:
                conv += f"{USER_TAG}{turn['user']}{NINO_TAG}{turn['nino']}"
            conversations.append(conv)
        else:
            # Single turn
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

    # Encode to bytes
    tokens = full_text.encode("utf-8")
    print(f"Total tokens (bytes): {len(tokens)}")

    # Print some stats
    unique = sorted(set(tokens))
    print(f"Unique byte values: {len(unique)} (min={min(unique)}, max={max(unique)})")

    # 90/10 train/val split
    split = int(len(tokens) * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Val tokens:   {len(val_tokens)}")

    # Write binary files
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
    random.seed(42)
    main()
