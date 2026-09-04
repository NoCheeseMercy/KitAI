"""
Grouped Query Attention (GQA).

Implements multi-query grouped attention where multiple query heads share
a single key/value head. This significantly reduces KV cache memory usage
while maintaining model quality.

Reference: https://arxiv.org/abs/2305.13245

Inspired by: Llama 2, Mistral, Gemma, and GPT-NeoX attention mechanisms.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelConfig
from .embedding import RotaryEmbedding
from .kv_cache import KVCache


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with RoPE and optional Flash Attention.

    In GQA, n_heads query heads are divided into groups, with each group
    sharing one key head and one value head. This reduces the KV cache
    size by a factor of n_heads // n_kv_heads.

    The attention computation supports both standard PyTorch attention
    and Flash Attention (when available and configured).

    Args:
        config: Model configuration.

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_groups = config.n_query_groups  # queries per kv head
        self.n_query_groups = self.n_groups
        self.use_flash = config.use_flash_attention
        self.dropout = config.dropout

        # Q, K, V projections
        self.q_proj = nn.Linear(
            self.d_model,
            self.n_heads * self.head_dim,
            bias=config.bias,
        )
        self.k_proj = nn.Linear(
            self.d_model,
            self.n_kv_heads * self.head_dim,
            bias=config.bias,
        )
        self.v_proj = nn.Linear(
            self.d_model,
            self.n_kv_heads * self.head_dim,
            bias=config.bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            self.d_model,
            bias=config.bias,
        )

        # Rotary Positional Embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_theta,
        )

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    @property
    def Wq(self) -> nn.Linear:
        """Alias for q_proj (test compatibility)."""
        return self.q_proj

    @property
    def Wk(self) -> nn.Linear:
        """Alias for k_proj (test compatibility)."""
        return self.k_proj

    @property
    def Wv(self) -> nn.Linear:
        """Alias for v_proj (test compatibility)."""
        return self.v_proj

    @property
    def Wo(self) -> nn.Linear:
        """Alias for o_proj (test compatibility)."""
        return self.o_proj

    def _init_weights(self) -> None:
        """Initialize projections using normal distribution."""
        std = self.config.init_std
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Positional Embeddings to queries and keys.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim).
            k: Key tensor (batch, n_kv_heads, seq_len, head_dim).
            offset: Position offset for generation.

        Returns:
            Tuple of rotated (q, k) tensors.
        """
        q = self.rotary_emb.rotate_queries_or_keys(q, offset=offset)
        k = self.rotary_emb.rotate_queries_or_keys(k, offset=offset)
        return q, k

    def _expand_kv_for_gqa(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand KV heads to match the number of query heads for GQA.

        For GQA, each KV head is repeated n_query_groups times to match
        the number of query heads.

        Args:
            k: Key tensor (batch, n_kv_heads, seq_len, head_dim).
            v: Value tensor (batch, n_kv_heads, seq_len, head_dim).

        Returns:
            Tuple of expanded (k, v) tensors with shape
            (batch, n_heads, seq_len, head_dim).
        """
        if self.n_query_groups == 1:
            return k, v

        # Expand KV heads by repeating
        k = k.unsqueeze(2).expand(-1, -1, self.n_query_groups, -1, -1)
        v = v.unsqueeze(2).expand(-1, -1, self.n_query_groups, -1, -1)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        k = k.reshape(k.shape[0], -1, k.shape[3], k.shape[4])
        v = v.reshape(v.shape[0], -1, v.shape[3], v.shape[4])

        return k, v

    def _prepare_mask(
        self,
        mask: Optional[torch.Tensor],
        batch: int,
        n_heads: int,
        q_seq_len: int,
        k_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Prepare attention mask in the correct shape.

        Handles 2D (seq_len, seq_len), 3D (batch, seq_len, seq_len),
        and 4D (batch, 1, seq_len, seq_len) masks.

        Args:
            mask: Input mask tensor.
            batch: Batch size.
            n_heads: Number of attention heads.
            q_seq_len: Query sequence length.
            k_seq_len: Key sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Mask tensor broadcast to (batch, 1, q_seq_len, k_seq_len) or None.
        """
        if mask is None:
            return None

        if mask.dim() == 2:
            # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
            mask = mask.unsqueeze(1)

        # Ensure on correct device and dtype
        mask = mask.to(device=device, dtype=dtype)

        # Slice to the correct dimensions
        mask = mask[:, :, :q_seq_len, :k_seq_len]

        return mask

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard PyTorch attention computation.

        Computes attention scores via dot product, applies causal mask,
        softmax, and dropout.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim).
            k: Key tensor (batch, n_heads, seq_len, head_dim).
            v: Value tensor (batch, n_heads, seq_len, head_dim).
            mask: Optional attention mask (broadcastable to batch, 1, q_len, k_len).

        Returns:
            Attention output (batch, n_heads, seq_len, head_dim).
        """
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply mask
        if mask is not None:
            attn_weights = attn_weights + mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        output = torch.matmul(attn_weights, v)
        return output

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flash Attention computation (memory-efficient).

        Requires PyTorch 2.0+'s scaled_dot_product_attention.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim).
            k: Key tensor (batch, n_heads, seq_len, head_dim).
            v: Value tensor (batch, n_heads, seq_len, head_dim).
            mask: Optional attention mask.

        Returns:
            Attention output (batch, n_heads, seq_len, head_dim).
        """
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if mask is not None:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        return output

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass for Grouped Query Attention.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            mask: Optional causal attention mask.
            kv_cache: Optional KV cache for generation.
            position_offset: Position offset when using KV cache.

        Returns:
            Tuple of (attention output (batch, seq_len, d_model), updated KV cache).
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head format
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self._apply_rotary(q, k, offset=position_offset)

        # Update KV cache if provided
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Get current sequence lengths after cache update
        cur_k_seq_len = k.shape[2]

        # Expand KV heads for GQA
        k, v = self._expand_kv_for_gqa(k, v)

        # Prepare mask
        prepared_mask = self._prepare_mask(
            mask, batch, self.n_heads, seq_len, cur_k_seq_len, q.device, q.dtype
        )

        # Compute attention
        if self.use_flash and seq_len > 1:
            attn_output = self._flash_attention(q, k, v, prepared_mask)
        else:
            attn_output = self._standard_attention(q, k, v, prepared_mask)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.d_model)

        # Output projection
        output = self.o_proj(attn_output)

        return output, kv_cache
