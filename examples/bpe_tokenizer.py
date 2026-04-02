#!/usr/bin/env python3
"""
Minimal BPE tokenizer for NinoGPT.

Trains byte-pair encoding merges from raw text, with reserved special tokens
for conversation delimiters. Saves/loads a simple text format parseable in C++.

Token layout:
  0-255:   raw byte tokens
  256:     <|system|>
  257:     <|user|>
  258:     <|nino|>
  259-1023: BPE merge tokens
"""

import struct
from collections import Counter

SPECIAL_TOKENS = {
    "<|system|>": 256,
    "<|user|>": 257,
    "<|nino|>": 258,
}
NUM_SPECIAL = len(SPECIAL_TOKENS)
FIRST_MERGE_ID = 256 + NUM_SPECIAL  # 259


def get_pairs(ids):
    """Count consecutive pairs in a list of token IDs."""
    counts = Counter()
    for i in range(len(ids) - 1):
        counts[(ids[i], ids[i + 1])] += 1
    return counts


def merge_pair(ids, pair, new_id):
    """Replace all occurrences of pair with new_id."""
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


def replace_special_tokens(text):
    """Replace delimiter strings with placeholder bytes, return (processed_bytes, special_positions)."""
    # We replace special token strings with single-byte placeholders that won't
    # appear in normal UTF-8 text, then post-process to map to special IDs
    result = []
    i = 0
    while i < len(text):
        matched = False
        for token_str, token_id in SPECIAL_TOKENS.items():
            if text[i:i + len(token_str)] == token_str:
                result.append(token_id)  # Store special ID directly
                i += len(token_str)
                matched = True
                break
        if not matched:
            result.append(text[i])  # Store character (will be converted to byte later)
            i += 1
    return result


def train_bpe(text, vocab_size=1024):
    """Train BPE merges on text, returning (merges, vocab)."""
    num_merges = vocab_size - FIRST_MERGE_ID

    # Replace special tokens and convert to token IDs
    mixed = replace_special_tokens(text)
    ids = []
    for item in mixed:
        if isinstance(item, int):
            ids.append(item)  # Special token ID
        else:
            ids.append(item.encode("utf-8")[0] if ord(item) < 128 else None)
            if ids[-1] is None:
                # Multi-byte UTF-8 character
                ids.pop()
                ids.extend(item.encode("utf-8"))

    # Collect merges
    merges = []
    for i in range(num_merges):
        pairs = get_pairs(ids)
        if not pairs:
            break
        # Only merge non-special pairs (both IDs must not be special tokens)
        valid_pairs = {p: c for p, c in pairs.items()
                       if p[0] not in SPECIAL_TOKENS.values() and p[1] not in SPECIAL_TOKENS.values()}
        if not valid_pairs:
            break
        best_pair = max(valid_pairs, key=valid_pairs.get)
        new_id = FIRST_MERGE_ID + i
        ids = merge_pair(ids, best_pair, new_id)
        merges.append(best_pair)
        if (i + 1) % 100 == 0:
            print(f"  merge {i+1}/{num_merges}: {best_pair} -> {new_id} (tokens remaining: {len(ids)})")

    # Build vocab: token_id -> byte sequence
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for token_str, token_id in SPECIAL_TOKENS.items():
        vocab[token_id] = token_str.encode("utf-8")
    for i, (a, b) in enumerate(merges):
        vocab[FIRST_MERGE_ID + i] = vocab[a] + vocab[b]

    print(f"Trained {len(merges)} merges, final vocab size: {FIRST_MERGE_ID + len(merges)}")
    return merges, vocab


def encode(text, merges):
    """Encode text to BPE token IDs."""
    # Replace special tokens first
    mixed = replace_special_tokens(text)
    ids = []
    for item in mixed:
        if isinstance(item, int):
            ids.append(item)
        else:
            for b in item.encode("utf-8"):
                ids.append(b)

    # Apply merges in order
    for i, (a, b) in enumerate(merges):
        new_id = FIRST_MERGE_ID + i
        ids = merge_pair(ids, (a, b), new_id)

    return ids


def decode(ids, vocab):
    """Decode BPE token IDs back to text."""
    raw_bytes = b""
    for token_id in ids:
        if token_id in vocab:
            raw_bytes += vocab[token_id]
        else:
            raw_bytes += b"?"
    return raw_bytes.decode("utf-8", errors="replace")


def save_tokenizer(path, merges, vocab):
    """Save tokenizer to a simple text file.

    Format:
      Line 1: vocab_size
      Line 2: num_merges
      Lines 3..2+num_merges: "token_a token_b" (merge pairs)
    """
    vocab_size = FIRST_MERGE_ID + len(merges)
    with open(path, "w") as f:
        f.write(f"{vocab_size}\n")
        f.write(f"{len(merges)}\n")
        for a, b in merges:
            f.write(f"{a} {b}\n")
    print(f"Saved tokenizer to {path} ({vocab_size} vocab, {len(merges)} merges)")


def load_tokenizer(path):
    """Load tokenizer from text file, return (merges, vocab)."""
    with open(path, "r") as f:
        vocab_size = int(f.readline().strip())
        num_merges = int(f.readline().strip())
        merges = []
        for _ in range(num_merges):
            a, b = f.readline().strip().split()
            merges.append((int(a), int(b)))

    # Rebuild vocab
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for token_str, token_id in SPECIAL_TOKENS.items():
        vocab[token_id] = token_str.encode("utf-8")
    for i, (a, b) in enumerate(merges):
        vocab[FIRST_MERGE_ID + i] = vocab[a] + vocab[b]

    return merges, vocab


def save_tokens_bin(path, token_ids):
    """Save token IDs as uint16 little-endian binary."""
    with open(path, "wb") as f:
        for tid in token_ids:
            f.write(struct.pack("<H", tid))
    print(f"Saved {path} ({len(token_ids)} tokens, {len(token_ids)*2} bytes)")
