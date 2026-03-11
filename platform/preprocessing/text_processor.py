"""
Text preprocessing for language model training.
Supports character-level and word-level tokenization.
"""
import json
import struct
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class TokenizerType(str, Enum):
    CHARACTER = "character"
    WORD = "word"


@dataclass
class TextVocabulary:
    """Vocabulary for text tokenization."""
    token_to_idx: Dict[str, int]
    idx_to_token: Dict[int, str]
    vocab_size: int
    tokenizer_type: str
    special_tokens: Dict[str, int]  # <pad>, <unk>, <eos>, etc.

    def encode(self, text: str) -> List[int]:
        """Encode text to token indices."""
        if self.tokenizer_type == TokenizerType.CHARACTER.value:
            tokens = list(text)
        else:
            tokens = text.split()

        unk_idx = self.special_tokens.get('<unk>', 0)
        return [self.token_to_idx.get(t, unk_idx) for t in tokens]

    def decode(self, indices: List[int]) -> str:
        """Decode token indices to text."""
        tokens = [self.idx_to_token.get(i, '<unk>') for i in indices]
        if self.tokenizer_type == TokenizerType.CHARACTER.value:
            return ''.join(tokens)
        else:
            return ' '.join(tokens)

    def to_dict(self) -> dict:
        return {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': {str(k): v for k, v in self.idx_to_token.items()},
            'vocab_size': self.vocab_size,
            'tokenizer_type': self.tokenizer_type,
            'special_tokens': self.special_tokens
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TextVocabulary':
        return cls(
            token_to_idx=data['token_to_idx'],
            idx_to_token={int(k): v for k, v in data['idx_to_token'].items()},
            vocab_size=data['vocab_size'],
            tokenizer_type=data['tokenizer_type'],
            special_tokens=data['special_tokens']
        )


class TextProcessor:
    """Process text data for language model training."""

    def __init__(
        self,
        tokenizer_type: TokenizerType = TokenizerType.CHARACTER,
        seq_length: int = 128,
        min_freq: int = 1
    ):
        self.tokenizer_type = tokenizer_type
        self.seq_length = seq_length
        self.min_freq = min_freq

    def build_vocabulary(self, text: str) -> TextVocabulary:
        """Build vocabulary from text."""
        # Tokenize
        if self.tokenizer_type == TokenizerType.CHARACTER:
            tokens = list(text)
        else:
            tokens = text.split()

        # Count frequencies
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        # Filter by min frequency and sort
        filtered = [(t, f) for t, f in freq.items() if f >= self.min_freq]
        filtered.sort(key=lambda x: (-x[1], x[0]))  # Sort by freq desc, then alphabetically

        # Build mappings with special tokens first
        special_tokens = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        token_to_idx = dict(special_tokens)
        idx_to_token = {v: k for k, v in special_tokens.items()}

        for token, _ in filtered:
            if token not in token_to_idx:
                idx = len(token_to_idx)
                token_to_idx[token] = idx
                idx_to_token[idx] = token

        return TextVocabulary(
            token_to_idx=token_to_idx,
            idx_to_token=idx_to_token,
            vocab_size=len(token_to_idx),
            tokenizer_type=self.tokenizer_type.value,
            special_tokens=special_tokens
        )

    def process_text(
        self,
        text: str,
        vocab: TextVocabulary,
        train_ratio: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process text into training sequences.

        Returns:
            train_inputs, train_targets, test_inputs, test_targets
            Each is [num_sequences, seq_length]
        """
        # Encode full text
        encoded = vocab.encode(text)

        # Create sequences: input[i] predicts target[i] (next token)
        # For autoregressive: input = tokens[:-1], target = tokens[1:]
        num_tokens = len(encoded)

        # Create overlapping sequences
        sequences_input = []
        sequences_target = []

        stride = self.seq_length // 2  # 50% overlap
        for i in range(0, num_tokens - self.seq_length - 1, stride):
            seq_input = encoded[i:i + self.seq_length]
            seq_target = encoded[i + 1:i + self.seq_length + 1]
            sequences_input.append(seq_input)
            sequences_target.append(seq_target)

        if not sequences_input:
            # Text too short, use what we have
            pad_len = self.seq_length - (num_tokens - 1)
            if pad_len > 0:
                seq_input = encoded[:-1] + [vocab.special_tokens['<pad>']] * pad_len
                seq_target = encoded[1:] + [vocab.special_tokens['<pad>']] * pad_len
            else:
                seq_input = encoded[:self.seq_length]
                seq_target = encoded[1:self.seq_length + 1]
            sequences_input.append(seq_input)
            sequences_target.append(seq_target)

        # Convert to numpy
        inputs = np.array(sequences_input, dtype=np.float32)
        targets = np.array(sequences_target, dtype=np.float32)

        # Split train/test
        n_train = int(len(inputs) * train_ratio)
        n_train = max(1, n_train)

        train_inputs = inputs[:n_train]
        train_targets = targets[:n_train]
        test_inputs = inputs[n_train:] if n_train < len(inputs) else inputs[-1:]
        test_targets = targets[n_train:] if n_train < len(targets) else targets[-1:]

        return train_inputs, train_targets, test_inputs, test_targets

    def save_processed_data(
        self,
        output_dir: Path,
        train_inputs: np.ndarray,
        train_targets: np.ndarray,
        test_inputs: np.ndarray,
        test_targets: np.ndarray,
        vocab: TextVocabulary,
        text_stats: dict
    ) -> dict:
        """Save processed data to binary files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        def save_tensor(path: Path, data: np.ndarray):
            """Save tensor in whitematter format."""
            TENSOR_MAGIC = 0x54454E53  # 'TENS'
            with open(path, 'wb') as f:
                f.write(struct.pack('I', TENSOR_MAGIC))
                f.write(struct.pack('I', len(data.shape)))
                for dim in data.shape:
                    f.write(struct.pack('Q', dim))
                data = np.ascontiguousarray(data, dtype=np.float32)
                f.write(data.tobytes())

        # Save tensors
        save_tensor(output_dir / "train_inputs.bin", train_inputs)
        save_tensor(output_dir / "train_targets.bin", train_targets)
        save_tensor(output_dir / "test_inputs.bin", test_inputs)
        save_tensor(output_dir / "test_targets.bin", test_targets)

        # Save vocabulary
        with open(output_dir / "vocabulary.json", 'w') as f:
            json.dump(vocab.to_dict(), f, indent=2)

        # Save config
        config = {
            "data_type": "text",
            "tokenizer_type": vocab.tokenizer_type,
            "vocab_size": vocab.vocab_size,
            "seq_length": self.seq_length,
            "train_sequences": len(train_inputs),
            "test_sequences": len(test_inputs),
            "total_tokens": text_stats.get('total_tokens', 0),
            "unique_tokens": text_stats.get('unique_tokens', 0),
            "special_tokens": vocab.special_tokens,
            # For compatibility with existing code
            "input_shape": [self.seq_length],
            "num_classes": vocab.vocab_size,
            "class_names": list(vocab.token_to_idx.keys())[:100],  # First 100 for display
        }

        with open(output_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        return config

    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        train_ratio: float = 0.9
    ) -> dict:
        """Process a text file end-to-end."""
        # Read text
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        # Clean text (basic normalization)
        text = text.strip()

        # Build vocabulary
        vocab = self.build_vocabulary(text)

        # Get stats
        if self.tokenizer_type == TokenizerType.CHARACTER:
            total_tokens = len(text)
        else:
            total_tokens = len(text.split())

        text_stats = {
            'total_tokens': total_tokens,
            'unique_tokens': vocab.vocab_size - len(vocab.special_tokens)
        }

        # Process into sequences
        train_in, train_tgt, test_in, test_tgt = self.process_text(
            text, vocab, train_ratio
        )

        # Save
        config = self.save_processed_data(
            output_dir,
            train_in, train_tgt, test_in, test_tgt,
            vocab, text_stats
        )

        return config


def process_text_dataset(
    raw_text: str,
    output_dir: Path,
    tokenizer_type: str = "character",
    seq_length: int = 128
) -> dict:
    """
    Convenience function to process text data.

    Args:
        raw_text: The text content to process
        output_dir: Directory to save processed files
        tokenizer_type: "character" or "word"
        seq_length: Sequence length for training

    Returns:
        Config dictionary with dataset info
    """
    processor = TextProcessor(
        tokenizer_type=TokenizerType(tokenizer_type),
        seq_length=seq_length
    )

    # Build vocabulary
    vocab = processor.build_vocabulary(raw_text)

    # Get stats
    if tokenizer_type == "character":
        total_tokens = len(raw_text)
    else:
        total_tokens = len(raw_text.split())

    text_stats = {
        'total_tokens': total_tokens,
        'unique_tokens': vocab.vocab_size - len(vocab.special_tokens)
    }

    # Process
    train_in, train_tgt, test_in, test_tgt = processor.process_text(raw_text, vocab)

    # Save
    config = processor.save_processed_data(
        output_dir,
        train_in, train_tgt, test_in, test_tgt,
        vocab, text_stats
    )

    return config
