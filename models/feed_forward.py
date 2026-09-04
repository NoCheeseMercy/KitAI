"""
Feed-Forward Networks for Transformer Blocks.

Implements:
1. SwiGLU Feed-Forward - The modern standard used in Llama, Mistral, etc.
2. GELU Feed-Forward - Standard GELU activation (alternative).
3. ReLU Feed-Forward - Simple ReLU activation (fallback).

SwiGLU is the default and recommended activation as it consistently
outperforms standard GELU and ReLU in language modeling tasks.

Reference: https://arxiv.org/abs/2002.05202 (GLU Variants)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

from .config import ModelConfig


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Implements: FFN(x) = (silu(x @ W_gate) * (x @ W_up)) @ W_down

    Where silu(x) = x * sigmoid(x)

    This is more expressive than standard FFN since the gating mechanism
    learns to control information flow. Used by Llama, Mistral, Qwen, etc.

    Args:
        config: Model configuration containing d_model, d_ff, etc.

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, d_model)

    Examples:
        >>> config = ModelConfig(d_model=768, d_ff=2048)
        >>> ffn = SwiGLUFeedForward(config)
        >>> x = torch.randn(2, 16, 768)
        >>> y = ffn(x)
        >>> y.shape
        torch.Size([2, 16, 768])
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.dropout = config.dropout

        # SwiGLU uses 3 weight matrices (gate, up, down)
        # gate and up project to d_ff, down projects back to d_model
        self.gate_proj = nn.Linear(
            self.d_model, self.d_ff, bias=config.bias
        )
        self.up_proj = nn.Linear(
            self.d_model, self.d_ff, bias=config.bias
        )
        self.down_proj = nn.Linear(
            self.d_ff, self.d_model, bias=config.bias
        )

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights with normal distribution."""
        std = 0.02
        for proj in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU FFN.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        # Compute gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # SwiGLU activation: silu(gate) * up
        hidden = F.silu(gate) * up

        # Apply dropout
        hidden = self.dropout_layer(hidden)

        # Project back to d_model
        output = self.down_proj(hidden)

        return output

    def extra_repr(self) -> str:
        """String representation."""
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


class GELUFeedForward(nn.Module):
    """
    GELU Feed-Forward Network (GPT-style).

    Implements: FFN(x) = gelu(x @ W1) @ W2

    Uses GELU activation. Simpler than SwiGLU but less expressive.
    Included for compatibility with models trained with GELU.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        self.fc1 = nn.Linear(self.d_model, self.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(self.d_ff, self.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        std = 0.02
        for layer in [self.fc1, self.fc2]:
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        hidden = F.gelu(self.fc1(x))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


class ReLUFeedForward(nn.Module):
    """
    ReLU Feed-Forward Network (original Transformer).

    Implements: FFN(x) = relu(x @ W1) @ W2

    Simplest FFN variant. Included for completeness and testing.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        self.fc1 = nn.Linear(self.d_model, self.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(self.d_ff, self.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        std = 0.02
        for layer in [self.fc1, self.fc2]:
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Output tensor (batch, seq_len, d_model).
        """
        hidden = F.relu(self.fc1(x))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


def get_feed_forward(config: ModelConfig) -> nn.Module:
    """
    Factory function to get the appropriate FFN based on config.

    Args:
        config: Model configuration with .activation field.

    Returns:
        FFN module instance.

    Raises:
        ValueError: If activation type is unknown.
    """
    activation_map: dict[str, Type[nn.Module]] = {
        "swiglu": SwiGLUFeedForward,
        "gelu": GELUFeedForward,
        "relu": ReLUFeedForward,
    }

    if config.activation not in activation_map:
        raise ValueError(
            f"Unknown activation '{config.activation}'. "
            f"Choose from {list(activation_map.keys())}"
        )

    return activation_map[config.activation](config)

