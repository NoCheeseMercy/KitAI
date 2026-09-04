"""
Transformer Block (Decoder Layer).

Implements a single transformer decoder block with:
1. Pre-Norm architecture (normalize before sublayers)
2. Grouped Query Attention + RoPE
3. SwiGLU Feed-Forward
4. Residual connections
5. Optional dropout

The Pre-Norm architecture is the modern standard (used in Llama, GPT-NeoX, etc.)
as it provides more stable training gradients compared to Post-Norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import ModelConfig
from .normalization import RMSNorm
from .attention import GroupedQueryAttention
from .feed_forward import get_feed_forward
from .kv_cache import KVCache


class TransformerBlock(nn.Module):
    """
    A single transformer decoder block.

    Architecture:
        x -> RMSNorm -> Attention -> Residual(+x) -> RMSNorm -> FFN -> Residual(+x)

    Both attention and feed-forward sublayers use pre-normalization and
    residual connections. This structure is proven stable for deep
    transformer training.

    Args:
        layer_idx: Index of this layer (0-indexed).
        config: Model configuration.

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model) + optional updated KV cache

    Examples:
        >>> config = ModelConfig.tiny()
        >>> block = TransformerBlock(0, config)
        >>> x = torch.randn(2, 16, 256)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 16, 256])
    """

    def __init__(self, layer_idx: int, config: ModelConfig) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Pre-attention normalization
        self.attention_norm = RMSNorm(
            config.d_model, eps=config.norm_eps
        )

        # Grouped Query Attention
        self.attention = GroupedQueryAttention(config)

        # Pre-FFN normalization
        self.ffn_norm = RMSNorm(
            config.d_model, eps=config.norm_eps
        )

        # Feed-Forward Network (SwiGLU by default)
        self.feed_forward = get_feed_forward(config)

        # Residual dropout
        self.residual_dropout = (
            nn.Dropout(config.dropout)
            if config.dropout > 0.0
            else nn.Identity()
        )

        # Gradient checkpointing support
        self.gradient_checkpointing = config.gradient_checkpointing

    def _forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass without gradient checkpointing wrapper.

        This method exists so we can wrap just the forward logic
        with torch's checkpoint function.
        """
        # Attention sublayer with Pre-Norm
        residual = x
        x = self.attention_norm(x)
        x, kv_cache = self.attention(
            x,
            mask=mask,
            kv_cache=kv_cache,
            position_offset=position_offset,
        )
        x = self.residual_dropout(x)
        x = residual + x

        # Feed-forward sublayer with Pre-Norm
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = self.residual_dropout(x)
        x = residual + x

        return x, kv_cache

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            mask: Optional causal attention mask.
            kv_cache: Optional KV cache for generation.
            position_offset: Position offset for KV cache generation.

        Returns:
            Tuple of (output tensor (batch, seq_len, d_model), updated KV cache).
        """
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing to save memory.
            # Only safe when not using a KV cache (training / full-context
            # forward). torch.utils.checkpoint re-runs the function during
            # backward, which would corrupt a stateful KV cache.
            if kv_cache is None:
                x, _ = torch.utils.checkpoint.checkpoint(
                    self._forward,
                    x,
                    mask,
                    None,  # kv_cache is None during training
                    position_offset,
                    use_reentrant=False,
                )
                return x, None
            # Fall through to normal forward when using cache

        return self._forward(x, mask, kv_cache, position_offset)

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"layer={self.layer_idx}, "
            f"d_model={self.config.d_model}, "
            f"checkpointing={self.gradient_checkpointing}"
        )

