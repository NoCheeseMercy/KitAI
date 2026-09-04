"""
KitAI - Production-quality local language model framework.

A modular, configurable decoder-only transformer library designed for
consumer GPUs (6GB VRAM). Supports training, inference, and experimentation.

Key features:
- Modern transformer architecture (RoPE, RMSNorm, SwiGLU, GQA)
- Complete training pipeline with mixed precision and gradient checkpointing
- BPE tokenizer with training and inference support
- Streaming generation with top-k, top-p, temperature, and penalties
- Speculative decoding interface for faster inference
- Checkpointing and resume training
- Live console dashboard for training metrics

Usage:
    from kitai import KitAITransformer, ModelConfig, BPETokenizer
    config = ModelConfig.tiny()
    model = KitAITransformer(config)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "KitAI Contributors"

from kitai.generation import speculative_decode, stream_generate

__all__ = [
    "speculative_decode",
    "stream_generate",
    "__version__",
    "__author__",
]