#!/usr/bin/env python3
"""Interactive chat with trained NinoGPT model."""

import os
import sys
import numpy as np
from whitematter import Tensor, no_grad
from whitematter.nn import (
    Module, Embedding, Linear, RMSNorm, SiLU, Dropout,
    MultiHeadAttention,
)
from whitematter.nn.attention import causal_mask
import whitematter.serialization as serial

sys.path.insert(0, os.path.dirname(__file__))
from bpe_tokenizer import encode, decode, load_tokenizer

# Import model definition
from gpt_nino_train import NinoGPT


def generate(model, tokenizer_merges, tokenizer_vocab, prompt_ids, max_tokens=200,
             temperature=0.7, top_p=0.9, rep_penalty=1.2, stop_tokens=None):
    """Generate text with top-p sampling and repetition penalty."""
    model.eval()
    ids = list(prompt_ids)
    if stop_tokens is None:
        stop_tokens = set()

    with no_grad():
        for _ in range(max_tokens):
            context = ids[-model.max_seq_len :]
            x = Tensor(np.array([context], dtype=np.int64))
            logits = model(x).data[0, -1, :].copy()

            # Repetition penalty
            recent = set(ids[-50:])
            for tok_id in recent:
                if tok_id < len(logits):
                    if logits[tok_id] > 0:
                        logits[tok_id] /= rep_penalty
                    else:
                        logits[tok_id] *= rep_penalty

            # Temperature
            logits /= temperature

            # Top-p (nucleus) sampling
            logits -= logits.max()
            probs = np.exp(logits) / np.exp(logits).sum()
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = sorted_idx[cumsum <= top_p]
            if len(cutoff) == 0:
                cutoff = sorted_idx[:1]

            masked_probs = np.zeros_like(probs)
            masked_probs[cutoff] = probs[cutoff]
            masked_probs /= masked_probs.sum()

            next_id = int(np.random.choice(len(masked_probs), p=masked_probs))
            if next_id in stop_tokens:
                break
            ids.append(next_id)

    return decode(ids, tokenizer_vocab)


if __name__ == "__main__":
    MODEL_PATH = "models/nino_gpt.npz"
    TOKENIZER_PATH = "data/nino/tokenizer.txt"

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Train first with gpt_nino_train.py")
        sys.exit(1)

    merges, vocab = load_tokenizer(TOKENIZER_PATH)

    VOCAB_SIZE = 1024
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    SEQ_LEN = 256

    model = NinoGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN)
    serial.load(MODEL_PATH, model)
    model.eval()
    print("NinoGPT loaded. Type 'quit' to exit, 'temp <value>' to change temperature.")

    temperature = 0.7
    # Stop tokens: <|system|>=256, <|user|>=257, <|nino|>=258
    stop_tokens = {256, 257}

    context = ""

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "reset":
            context = ""
            print("Context reset.")
            continue
        if user_input.lower().startswith("temp "):
            temperature = float(user_input.split()[1])
            print(f"Temperature set to {temperature}")
            continue
        if not user_input:
            continue

        context += f"<|user|>{user_input}<|nino|>"
        prompt_ids = encode(context, merges)

        # Truncate context if too long
        if len(prompt_ids) > SEQ_LEN - 50:
            # Keep most recent context
            prompt_ids = prompt_ids[-(SEQ_LEN - 50) :]
            context = decode(prompt_ids, vocab)

        full_text = generate(
            model, merges, vocab, prompt_ids,
            max_tokens=200, temperature=temperature, top_p=0.9,
            rep_penalty=1.2, stop_tokens=stop_tokens,
        )

        # Extract Nino's response
        response = full_text[len(decode(prompt_ids, vocab)) :]
        for stop in ["<|user|>", "<|system|>", "<|nino|>"]:
            if stop in response:
                response = response[: response.index(stop)]

        print(f"Nino: {response.strip()}")
        context += response.strip()
