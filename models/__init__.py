"""
KitAI Model Package.

Provides the complete decoder-only transformer architecture including:
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- SwiGLU Feed-Forward Network
- Grouped Query Attention (GQA)
- KV Cache
- Config-driven model assembly
"""

from .config import ModelConfig
from .normalization import RMSNorm
from .embedding import RotaryEmbedding, TokenEmbedding
from .attention import GroupedQueryAttention
from .feed_forward import SwiGLUFeedForward
from .kv_cache import KVCache
from .layers import TransformerBlock
from .output import OutputHead
from .transformer import KitAITransformer

__all__ = [
    "ModelConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "TokenEmbedding",
    "GroupedQueryAttention",
    "SwiGLUFeedForward",
    "KVCache",
    "TransformerBlock",
    "OutputHead",
    "KitAITransformer",
]

