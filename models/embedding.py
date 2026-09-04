"""
Token Embeddings and Rotary Positional Embeddings (RoPE).

Implements:
1. TokenEmbedding - Standard token embeddings with optional weight tying support
2. RotaryEmbedding - Rotary Positional Embeddings as used in Llama, GPT-NeoX, etc.

RoPE applies rotation to query and key vectors based on position, encoding
relative position information directly in the attention computation.

Reference: https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE).

    Encodes position information by rotating query and key vectors in
    attention heads. This allows the model to capture relative position
    information naturally.

    The rotation frequencies follow a geometric schedule:
    theta_i = base^(-2i/d) for i in [0, d/2)

    Args:
        dim: Dimension of the rotary embeddings (should be head_dim).
        max_seq_len: Maximum sequence length supported.
        base: Base frequency (theta) for the rotations. 10000.0 is standard.
        dtype: Torch dtype for the precomputed frequencies.

    Examples:
        >>> rope = RotaryEmbedding(dim=64)
        >>> q = torch.randn(2, 8, 16, 64)  # (batch, heads, seq, head_dim)
        >>> q_rotated = rope.rotate_queries_or_keys(q, offset=0)
        >>> q_rotated.shape
        torch.Size([2, 8, 16, 64])
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.dtype = dtype

        # Precompute frequency for each dimension pair
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin for all positions
        self._update_cos_sin_cache(max_seq_len, offset=0)

    def _update_cos_sin_cache(
        self,
        seq_len: int,
        offset: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        """Precompute cos and sin values for all positions up to seq_len."""
        total_len = offset + seq_len
        device = device or self.inv_freq.device

        # Reuse cache when length and device already match.
        if (
            hasattr(self, "_cos_cache")
            and self._cos_cache is not None
            and self._cos_cache.device == device
            and self._cos_cache.shape[0] >= total_len
        ):
            return

        # Generate position indices on the correct device
        t = torch.arange(total_len, device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)

        # Compute frequencies: (total_len, dim/2)
        freqs = torch.outer(t, inv_freq)

        # Interleave style: [freqs, freqs] matches rotate_half pairing
        emb = torch.cat((freqs, freqs), dim=-1)

        # Store cos and sin as non-persistent buffers
        cos = emb.cos().to(dtype=self.dtype)
        sin = emb.sin().to(dtype=self.dtype)
        self.register_buffer("_cos_cache", cos, persistent=False)
        self.register_buffer("_sin_cache", sin, persistent=False)

    def rotate_queries_or_keys(
        self, x: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to queries or keys.

        Args:
            x: Input tensor of shape (batch, num_heads, seq_len, head_dim).
            offset: Position offset for the first token in the sequence.

        Returns:
            Rotated tensor of same shape as input.
        """
        seq_len = x.shape[2]

        # Ensure cache is up to date on the same device as x
        self._update_cos_sin_cache(seq_len, offset, device=x.device)

        # Slice the relevant positions
        cos = self._cos_cache[offset : offset + seq_len].to(dtype=x.dtype, device=x.device)
        sin = self._sin_cache[offset : offset + seq_len].to(dtype=x.dtype, device=x.device)

        # Reshape for broadcasting: (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation: RoPE(x) = x * cos + rotate_half(x) * sin
        return x * cos + self._rotate_half(x) * sin

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the second half of the last dimension.

        This implements the rotation operation:
        rotate_half([x1, x2]) = [-x2, x1]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def extra_repr(self) -> str:
        """String representation."""
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"


class TokenEmbedding(nn.Module):
    """
    Token Embedding layer.

    Maps token IDs to dense vectors. Supports weight tying with the
    output projection layer for parameter efficiency.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Embedding dimension.
        padding_idx: Index for the padding token (if any).
        dtype: Torch dtype for the embedding weights.

    Examples:
        >>> emb = TokenEmbedding(32000, 768)
        >>> x = torch.randint(0, 32000, (2, 16))
        >>> y = emb(x)
        >>> y.shape
        torch.Size([2, 16, 768])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(
            torch.empty(vocab_size, d_model, dtype=dtype)
        )

        # Initialize with normal distribution
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding weights."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            x: Token IDs of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, d_model).
        """
        return F.embedding(x, self.weight, padding_idx=self.padding_idx)

    def extra_repr(self) -> str:
        """String representation."""
        return f"vocab_size={self.vocab_size}, d_model={self.d_model}"

