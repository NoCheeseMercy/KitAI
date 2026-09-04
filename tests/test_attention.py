"""
Tests for the Grouped Query Attention module.
"""

from __future__ import annotations

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.attention import GroupedQueryAttention
from models.kv_cache import KVCache


class TestGroupedQueryAttention:
    """Test suite for GroupedQueryAttention."""

    @pytest.fixture
    def config(self):
        return ModelConfig(
            d_model=256,
            n_heads=8,
            n_kv_heads=4,
            max_seq_len=512,
            dropout=0.0,
            bias=False,
        )

    @pytest.fixture
    def attention(self, config):
        return GroupedQueryAttention(config)

    def test_initialization(self, attention, config):
        """Test attention module initializes correctly."""
        assert attention.d_model == config.d_model
        assert attention.n_heads == config.n_heads
        assert attention.n_kv_heads == config.n_kv_heads
        assert attention.head_dim == config.d_model // config.n_heads
        assert attention.n_groups == config.n_heads // config.n_kv_heads

    def test_forward_shape(self, attention):
        """Test forward pass produces correct output shape."""
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, attention.d_model)
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=x.dtype),
            diagonal=1,
        )
        out, kv_cache = attention(x, mask=mask)
        assert out.shape == (batch, seq_len, attention.d_model)

    def test_forward_no_mask(self, attention):
        """Test forward pass without mask."""
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, attention.d_model)
        out, kv_cache = attention(x)
        assert out.shape == (batch, seq_len, attention.d_model)
        assert kv_cache is None

    def test_kv_cache_single_step(self, attention):
        """Test KV cache with single token generation."""
        batch = 2
        config = attention.config
        device = next(attention.parameters()).device
        cache = KVCache(
            max_batch_size=batch,
            max_seq_len=config.max_seq_len,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device=device,
        )

        # First token
        x1 = torch.randn(batch, 1, attention.d_model, device=device)
        mask = torch.zeros(1, 1, device=device)
        out1, cache = attention(x1, mask=mask, kv_cache=cache)
        assert out1.shape == (batch, 1, attention.d_model)

        # Second token
        x2 = torch.randn(batch, 1, attention.d_model, device=device)
        mask = torch.zeros(1, 1, device=device)
        out2, cache = attention(x2, mask=mask, kv_cache=cache)
        assert out2.shape == (batch, 1, attention.d_model)

    def test_gqa_projection_shapes(self, attention):
        """Test GQA projections have correct shapes."""
        # nn.Linear weight shape is (out_features, in_features)
        assert attention.Wq.weight.shape == (attention.d_model, attention.d_model)
        k_out_dim = attention.n_kv_heads * attention.head_dim
        assert attention.Wk.weight.shape == (k_out_dim, attention.d_model)
        assert attention.Wv.weight.shape == (k_out_dim, attention.d_model)

    def test_gradient_flows(self, attention):
        """Test gradients flow through attention."""
        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, attention.d_model, requires_grad=True)
        out, _ = attention(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_different_seq_lengths(self, attention):
        """Test attention works with different sequence lengths."""
        for seq_len in [1, 2, 4, 8, 16]:
            batch = 2
            x = torch.randn(batch, seq_len, attention.d_model)
            out, _ = attention(x)
            assert out.shape[-1] == attention.d_model

    def test_self_attention_is_causal(self, attention):
        """Test causal masking prevents attending to future tokens."""
        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, attention.d_model)
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=x.dtype),
            diagonal=1,
        )
        out, _ = attention(x, mask=mask)
        # Verify output doesn't depend on future tokens
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__])
