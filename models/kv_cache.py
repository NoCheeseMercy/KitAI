"""
Key-Value (KV) Cache for Efficient Autoregressive Generation.

Stores keys and values from previous decoding steps to avoid
recomputation. Uses Grouped Query Attention (GQA) compatible shapes
for memory efficiency.

The cache pre-allocates a fixed-size buffer and uses slicing for
O(1) updates, minimizing memory allocations during generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class KVCache:
    """
    Efficient key-value cache for autoregressive generation.

    Pre-allocates a contiguous memory buffer for keys and values,
    avoiding repeated memory allocations during token-by-token generation.

    The cache supports both Multi-Head Attention (MHA) and
    Grouped Query Attention (GQA) shapes.

    Args:
        max_batch_size: Maximum batch size for generation.
        max_seq_len: Maximum sequence length to cache.
        n_kv_heads: Number of key/value heads.
        head_dim: Dimension of each attention head.
        dtype: Torch data type for cache tensors.
        device: Device for cache tensors (default: auto-detect).

    Examples:
        >>> cache = KVCache(max_batch_size=2, max_seq_len=2048, n_kv_heads=4, head_dim=64)
        >>> k = torch.randn(2, 4, 10, 64)  # (batch, kv_heads, seq_len, head_dim)
        >>> v = torch.randn(2, 4, 10, 64)
        >>> k_out, v_out = cache.update(k, v)
        >>> k_out.shape
        torch.Size([2, 4, 10, 64])
        >>> # Second call concatenates
        >>> k2 = torch.randn(2, 4, 1, 64)
        >>> v2 = torch.randn(2, 4, 1, 64)
        >>> k_out2, v_out2 = cache.update(k2, v2)
        >>> k_out2.shape
        torch.Size([2, 4, 11, 64])
    """

    def __init__(
        self,
        max_batch_size: int = 1,
        max_seq_len: int = 2048,
        n_kv_heads: int = 4,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Pre-allocate cache buffers
        self.cache_k = torch.zeros(
            max_batch_size,
            n_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.cache_v = torch.zeros(
            max_batch_size,
            n_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )

        # Track the current sequence length in the cache
        self._seq_len: int = 0

    @property
    def seq_len(self) -> int:
        """Current sequence length stored in the cache."""
        return self._seq_len

    @property
    def is_empty(self) -> bool:
        """Check if cache is empty (no tokens stored)."""
        return self._seq_len == 0

    @property
    def is_full(self) -> bool:
        """Check if cache has reached max sequence length."""
        return self._seq_len >= self.max_seq_len

    def reset(self) -> None:
        """
        Reset the cache to empty state.

        This zeroes the memory and resets the sequence counter.
        Used between different generation sequences.
        """
        self._seq_len = 0
        self.cache_k.zero_()
        self.cache_v.zero_()

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new keys and values.

        New KV pairs are appended to the existing cache. The method
        returns the concatenated cache (existing + new) for attention
        computation.

        Args:
            k: Key tensor of shape (batch, n_kv_heads, new_seq_len, head_dim).
            v: Value tensor of shape (batch, n_kv_heads, new_seq_len, head_dim).

        Returns:
            Tuple of (full_k, full_v) tensors with all cached KV pairs.
        """
        batch, n_kv_heads, new_len, head_dim = k.shape

        # Validate dimensions
        assert batch <= self.max_batch_size, (
            f"Batch size {batch} exceeds cache max {self.max_batch_size}"
        )
        assert n_kv_heads == self.n_kv_heads, (
            f"KV heads {n_kv_heads} doesn't match cache {self.n_kv_heads}"
        )
        assert head_dim == self.head_dim, (
            f"Head dim {head_dim} doesn't match cache {self.head_dim}"
        )

        # Check if we need to handle overflow
        new_seq_len = self._seq_len + new_len
        if new_seq_len > self.max_seq_len:
            # Truncate from the beginning (oldest tokens)
            # This is a simple windowed cache strategy
            overflow = new_seq_len - self.max_seq_len
            # Shift existing cache left by overflow amount
            self.cache_k[:batch, :, :-overflow] = self.cache_k[
                :batch, :, overflow:
            ].clone()
            self.cache_v[:batch, :, :-overflow] = self.cache_v[
                :batch, :, overflow:
            ].clone()
            self._seq_len = self.max_seq_len - new_len

        # Store new KV pairs at the current position
        start = self._seq_len
        end = start + new_len
        self.cache_k[:batch, :, start:end] = k
        self.cache_v[:batch, :, start:end] = v

        # Update sequence length
        self._seq_len += new_len

        # Return the full cached sequence up to current length
        return (
            self.cache_k[:batch, :, : self._seq_len],
            self.cache_v[:batch, :, : self._seq_len],
        )

    def slice(
        self, batch_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached KV values for specific batch indices.

        Useful for methods like beam search where batch ordering changes.

        Args:
            batch_indices: Indices of batch elements to retrieve.

        Returns:
            Tuple of (k, v) tensors for selected batch indices.
        """
        return (
            self.cache_k[batch_indices, :, : self._seq_len],
            self.cache_v[batch_indices, :, : self._seq_len],
        )

    def get_max_cache_len(self) -> int:
        """
        Get the maximum sequence length this cache can hold.

        Returns:
            Maximum sequence length.
        """
        return self.max_seq_len

    def to(self, device: torch.device) -> "KVCache":
        """
        Move cache to a different device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.cache_k = self.cache_k.to(device)
        self.cache_v = self.cache_v.to(device)
        self.device = device
        return self

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"KVCache(batch={self.max_batch_size}, "
            f"seq_len={self._seq_len}/{self.max_seq_len}, "
            f"kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
            f"device={self.device})"
        )

