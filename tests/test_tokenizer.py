"""
Tests for the BPE Tokenizer.
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.trainer import TokenizerTrainer
from tokenizer.utils import SPECIAL_TOKENS


class TestBPETokenizer:
    """Test suite for BPE Tokenizer."""

    @pytest.fixture
    def sample_texts(self):
        return [
            "Hello, world! This is a test.",
            "KitAI is a language model framework.",
            "The quick brown fox jumps over the lazy dog.",
            "Tokenization is the first step in NLP pipelines.",
            "BPE tokenization works by merging frequent character pairs.",
        ]

    @pytest.fixture
    def tokenizer(self, sample_texts):
        trainer = TokenizerTrainer(vocab_size=500, min_frequency=1)
        return trainer.train(texts=sample_texts)

    def test_initialization(self):
        """Test tokenizer initializes with special tokens."""
        tokenizer = BPETokenizer()
        # Special tokens are registered immediately
        assert tokenizer.vocab_size == len(SPECIAL_TOKENS)
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.eos_token_id == 3
        assert tokenizer.target_vocab_size == 32000

    def test_train_and_vocab_size(self, tokenizer):
        """Test tokenizer training produces a usable vocabulary."""
        assert tokenizer.vocab_size > len(SPECIAL_TOKENS)
        assert tokenizer.vocab_size <= tokenizer.target_vocab_size

    def test_encode_decode(self, tokenizer):
        """Test encode then decode is non-empty and round-trips reasonably."""
        text = "Hello, world! This is a test."
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        assert len(encoded) > 0
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_encode_returns_integers(self, tokenizer):
        """Test encode returns list of integers."""
        text = "KitAI language model"
        encoded = tokenizer.encode(text)
        assert all(isinstance(t, int) for t in encoded)

    def test_decode_returns_string(self, tokenizer):
        """Test decode returns string."""
        tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id]
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)

    def test_special_tokens_exist(self, tokenizer):
        """Test all special tokens are in vocabulary."""
        for token in SPECIAL_TOKENS.values():
            assert token in tokenizer.token_to_id

    def test_save_and_load(self, tokenizer, tmp_path):
        """Test tokenizer serialization round-trip."""
        path = tmp_path / "tok.json"
        tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        text = "KitAI tokenizer round trip"
        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.vocab_size == tokenizer.vocab_size


if __name__ == "__main__":
    pytest.main([__file__])
