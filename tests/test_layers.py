"""
Tests for Transformer Block and full model.
"""

from __future__ import annotations

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.layers import TransformerBlock
from models.transformer import KitAITransformer
from models.kv_cache import KVCache


class TestTransformerBlock:
    """Test suite for TransformerBlock."""

    @pytest.fixture
    def config(self):
        return ModelConfig(
            d_model=256,
            n_heads=8,
            n_kv_heads=4,
            n_layers=2,
            d_ff=682,
            max_seq_len=512,
            dropout=0.0,
        )

    @pytest.fixture
    def block(self, config):
        return TransformerBlock(0, config)

    @pytest.fixture
    def model(self, config):
        return KitAITransformer(config)

    def test_block_forward_shape(self, block):
        """Test block forward pass produces correct shape."""
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, block.config.d_model)
        out, kv_cache = block(x)
        assert out.shape == (batch, seq_len, block.config.d_model)
        assert kv_cache is None

    def test_block_with_cache(self, block):
        """Test block with KV cache."""
        batch = 2
        config = block.config
        device = next(block.parameters()).device
        x = torch.randn(batch, 1, config.d_model, device=device)
        cache = KVCache(
            max_batch_size=batch,
            max_seq_len=config.max_seq_len,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device=device,
        )
        out, cache = block(x, kv_cache=cache)
        assert out.shape == (batch, 1, config.d_model)

    def test_block_residual(self, block):
        """Test residual connections work."""
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, block.config.d_model)

        # Output should be in a reasonable range due to residual
        out, _ = block(x)
        assert torch.isfinite(out).all()
        assert out.shape == x.shape

    def test_model_forward_shape(self, model):
        """Test full model forward pass."""
        batch, seq_len = 2, 16
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        logits, _ = model(input_ids)
        assert logits.shape == (batch, seq_len, model.config.vocab_size)

    def test_model_generate(self, model):
        """Test model generation."""
        batch, seq_len = 2, 8
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

    def test_model_loss(self, model):
        """Test model forward_with_loss."""
        batch, seq_len = 2, 16
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        labels = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        loss, perplexity = model.forward_with_loss(input_ids, labels)
        assert loss.ndim == 0
        assert loss > 0
        assert perplexity > 0

    def test_model_parameter_count(self, model):
        """Test model parameter calculation."""
        num_params = model.get_trainable_parameters()
        assert num_params > 0

    def test_gradient_checkpointing(self, config):
        """Test gradient checkpointing works."""
        config.gradient_checkpointing = True
        model = KitAITransformer(config)
        batch, seq_len = 2, 16
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        logits, _ = model(input_ids)
        assert logits.shape == (batch, seq_len, model.config.vocab_size)

        loss = logits.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break


if __name__ == "__main__":
    pytest.main([__file__])
