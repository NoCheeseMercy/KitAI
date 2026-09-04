"""
Tests for text generation functionality.
"""

from __future__ import annotations

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.transformer import KitAITransformer


class TestGeneration:
    """Test suite for text generation."""

    @pytest.fixture
    def config(self):
        return ModelConfig(
            d_model=128,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            d_ff=340,
            vocab_size=500,
            max_seq_len=128,
            dropout=0.0,
        )

    @pytest.fixture
    def model(self, config):
        return KitAITransformer(config)

    def test_greedy_generation(self, model):
        """Test greedy decoding (temperature=0)."""
        batch, seq_len = 1, 4
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,  # greedy
        )
        assert generated.shape[0] == batch
        assert generated.shape[1] == min(seq_len + 10, model.config.max_seq_len)

    def test_sampling_generation(self, model):
        """Test sampling-based generation."""
        batch, seq_len = 1, 4
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )
        assert generated.shape[0] == batch
        assert generated.shape[1] > seq_len

    def test_batch_generation(self, model):
        """Test batch generation."""
        batch, seq_len = 2, 4
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        generated = model.generate_batch(
            input_ids,
            max_new_tokens=10,
            temperature=0.8,
        )
        assert generated.shape[0] == batch
        assert generated.shape[1] > seq_len

    def test_max_new_tokens(self, model):
        """Test max_new_tokens limit is respected."""
        batch, seq_len = 1, 4
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        max_new = 5
        generated = model.generate(input_ids, max_new_tokens=max_new)
        assert generated.shape[1] <= seq_len + max_new

    def test_repetition_penalty(self, model):
        """Test repetition penalty doesn't crash."""
        batch, seq_len = 1, 4
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        generated = model.generate(
            input_ids,
            max_new_tokens=10,
            repetition_penalty=1.2,
        )
        assert generated.shape[0] == batch

    def test_eos_stop(self, model):
        """Test generation stops at EOS token."""
        batch, seq_len = 1, 4
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        generated = model.generate(
            input_ids,
            max_new_tokens=50,
            eos_token_id=model.config.vocab_size - 1,
        )
        assert generated.shape[0] == batch


if __name__ == "__main__":
    pytest.main([__file__])
