"""
Model Configuration.

Defines a dataclass-based configuration system for the KitAI transformer model.
All hyperparameters are configurable with sensible defaults optimized for
consumer GPUs (6GB VRAM).

Inspired by: Llama, Mistral, GPT-NeoX, and modern transformer configs.
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """
    Configuration for the KitAI decoder-only transformer.

    All model hyperparameters are defined here with defaults optimized for
    training on a 6GB VRAM GPU (e.g., NVIDIA RTX 3050 Laptop).

    Attributes:
        vocab_size: Size of the vocabulary.
        d_model: Hidden dimension of the model.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads (query).
        n_kv_heads: Number of key/value heads (for GQA). If None, equals n_heads.
        d_ff: Feed-forward hidden dimension. If None, computed as 8/3 * d_model.
        max_seq_len: Maximum sequence length for positional encoding and KV cache.
        dropout: Dropout probability (0.0 disables dropout).
        activation: Activation function ('swiglu', 'gelu', 'relu').
        norm_eps: Epsilon for RMSNorm numerical stability.
        rope_theta: Base frequency for Rotary Positional Embeddings.
        weight_tying: Whether to tie input and output embeddings.
        bias: Whether to use bias in linear layers.
        init_std: Standard deviation for weight initialization.
        init_mean: Mean for weight initialization.
        gradient_checkpointing: Whether to use gradient checkpointing.
        use_flash_attention: Whether to use Flash Attention (if available).
    """

    # Vocabulary & dimensions
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: Optional[int] = None  # GQA: if None, equals n_heads (MHA)
    d_ff: Optional[int] = None  # If None, computed as 8/3 * d_model
    max_seq_len: int = 2048

    # Regularization
    dropout: float = 0.0
    activation: str = "swiglu"  # swiglu, gelu, relu
    norm_eps: float = 1e-6

    # Positional encoding
    rope_theta: float = 10000.0

    # Weight tying
    weight_tying: bool = True

    # Linear options
    bias: bool = False

    # Initialization
    init_std: float = 0.02
    init_mean: float = 0.0

    # Training optimizations
    gradient_checkpointing: bool = True
    use_flash_attention: bool = False  # Set True if hardware supports it

    def __post_init__(self) -> None:
        """Validate and compute derived parameters."""
        # Ensure dimensions are compatible
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        # Set default n_kv_heads to n_heads if not specified (standard MHA)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # Verify GQA compatibility
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

        # Compute default d_ff using the SwiGLU convention (8/3 * d_model)
        if self.d_ff is None:
            self.d_ff = int(8 / 3 * self.d_model)

        # Ensure d_ff is rounded to a multiple of 256 for tensor core efficiency
        self.d_ff = ((self.d_ff + 255) // 256) * 256

        # Validate activation
        valid_activations = {"swiglu", "gelu", "relu"}
        if self.activation not in valid_activations:
            raise ValueError(
                f"activation must be one of {valid_activations}, got '{self.activation}'"
            )

        # Validate dimensions > 0
        for name, val in [
            ("vocab_size", self.vocab_size),
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("n_kv_heads", self.n_kv_heads),
            ("d_ff", self.d_ff),
            ("max_seq_len", self.max_seq_len),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    @property
    def n_query_groups(self) -> int:
        """
        Number of query groups for GQA.
        Each group of query heads shares one set of KV heads.
        """
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize to JSON string or file."""
        data = self.to_dict()
        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return None
        return json.dumps(data, indent=2)

    def to_yaml(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize to YAML string or file."""
        data = self.to_dict()
        if path:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
            return None
        return yaml.dump(data, default_flow_style=False)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load configuration from a YAML file.

        Accepts either a flat model config or a nested ``{model: {...}}`` file
        (as used by ``configs/*.yaml``).
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and "model" in data and isinstance(data["model"], dict):
            data = data["model"]
        # Drop unknown keys so training-only fields don't crash construction
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def small(cls) -> "ModelConfig":
        """Small model config optimized for 6GB VRAM."""
        return cls(
            vocab_size=32000,
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=4,
            d_ff=2048,
            max_seq_len=2048,
            dropout=0.0,
            activation="swiglu",
        )

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Tiny model config for testing."""
        return cls(
            vocab_size=32000,
            d_model=256,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            d_ff=1024,
            max_seq_len=512,
            dropout=0.0,
            activation="swiglu",
        )

    @classmethod
    def medium(cls) -> "ModelConfig":
        """Medium model config (requires >6GB VRAM)."""
        return cls(
            vocab_size=32000,
            d_model=1024,
            n_layers=24,
            n_heads=16,
            n_kv_heads=4,
            d_ff=2816,
            max_seq_len=2048,
            dropout=0.0,
            activation="swiglu",
        )

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"ModelConfig(\n"
            f"  vocab_size={self.vocab_size}, d_model={self.d_model},\n"
            f"  n_layers={self.n_layers}, n_heads={self.n_heads},\n"
            f"  n_kv_heads={self.n_kv_heads}, d_ff={self.d_ff},\n"
            f"  max_seq_len={self.max_seq_len}, activation={self.activation},\n"
            f"  weight_tying={self.weight_tying}, dropout={self.dropout},\n"
            f"  params={self.total_params / 1e6:.2f}M\n"
            f")"
        )

    @property
    def total_params(self) -> int:
        """
        Estimate total number of parameters.
        Useful for quick memory estimation.
        """
        # Embedding: vocab_size * d_model
        emb = self.vocab_size * self.d_model

        # Per transformer block:
        # Self-attention: 4 * d_model^2 (Q,K,V,O projections)
        attn = 4 * self.d_model * self.d_model

        # Feed-forward (SwiGLU): 3 * d_model * d_ff (gate, up, down)
        ffn = 3 * self.d_model * self.d_ff

        # RMSNorm: 2 * d_model (two norms per block)
        norms = 2 * self.d_model

        block_params = attn + ffn + norms
        total = emb + self.n_layers * block_params

        # Output head (if no weight tying)
        if not self.weight_tying:
            total += self.vocab_size * self.d_model

        return total

