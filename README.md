# KitAI

**Production-Quality Local Language Model Framework**

KitAI is a from-scratch implementation of a modern decoder-only transformer language model framework. It combines the best architectural ideas from GPT, Llama, Qwen, DeepSeek, Mistral, Phi, and Gemma into a cohesive, modular, and efficient system optimized for consumer GPUs with **6GB VRAM**.

## Features

### Model Architecture
- **Rotary Positional Embeddings (RoPE)** - Efficient position encoding
- **RMSNorm** - Pre-Norm architecture for stable training
- **SwiGLU Feed-Forward** - Gated activation for better performance
- **Grouped Query Attention (GQA)** - Memory-efficient attention with KV head sharing
- **Weight Tying** - Shared embedding and output weights
- **KV Cache** - Efficient autoregressive generation
- **Config-driven** - YAML configuration for all parameters

### Memory Optimization (6GB VRAM)
- Gradient checkpointing (activation checkpointing)
- Automatic Mixed Precision (AMP)
- Gradient accumulation
- Fused AdamW optimizer
- Memory-efficient KV cache with GQA

### BPE Tokenizer
- Train from scratch on any text corpus
- Special tokens support (PAD, UNK, BOS, EOS, MASK)
- Vocabulary export/import (JSON format)
- Fast encoding and decoding

### Training Pipeline
- Multi-format dataset support (txt, jsonl, parquet, csv)
- Alpaca, ShareGPT, and OpenAI message formats
- Automatic sequence packing
- Cosine LR scheduler with linear warmup
- Gradient clipping
- Label smoothing
- Live console dashboard with rich metrics
- Automatic checkpointing (best + latest)
- Resume training from checkpoints

### Inference
- Interactive chat interface
- Streaming generation
- Top-k, Top-p, Temperature sampling
- Repetition, presence, and frequency penalties
- Stop sequences
- Batch generation
- Conversation history management

## Architecture

```
KitAI/
├── models/          # Transformer architecture
│   ├── config.py         # Model configuration
│   ├── transformer.py    # Main model class
│   ├── attention.py      # GQA attention
│   ├── feed_forward.py   # SwiGLU FFN
│   ├── embedding.py      # Token embeddings + RoPE
│   ├── normalization.py  # RMSNorm
│   ├── kv_cache.py       # KV cache
│   ├── layers.py         # Transformer blocks
│   └── output.py         # LM head + loss
├── tokenizer/       # BPE tokenizer
├── training/        # Training pipeline
├── datasets/        # Dataset formats
├── utils/           # Utilities
├── configs/         # YAML configuration files
├── scripts/         # Entry points
├── tests/           # Test suite
└── examples/        # Usage examples
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/NoCheeseMercy/kitai.git
cd kitai

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Train a Tokenizer

```bash
python -m scripts.tokenize \
    --data /path/to/text/data.txt \
    --output tokenizer.json \
    --vocab-size 32000
```

### Train a Model

```bash
python -m scripts.train \
    --config configs/small.yaml \
    --data /path/to/training/data \
    --val-data /path/to/validation/data
```

### Resume Training

```bash
python -m scripts.resume \
    --checkpoint checkpoints/latest.pt \
    --data /path/to/training/data \
    --max-steps 200000
```

### Chat with a Model

```bash
python -m scripts.chat \
    --model checkpoints/best.pt \
    --tokenizer tokenizer.json \
    --interactive
```

### Generate Text

```bash
python -m scripts.generate \
    --model checkpoints/best.pt \
    --tokenizer tokenizer.json \
    --prompt "Once upon a time" \
    --max-tokens 256
```

### Evaluate

```bash
python -m scripts.evaluate \
    --model checkpoints/best.pt \
    --tokenizer tokenizer.json \
    --data /path/to/test/data \
    --perplexity
```

## Configuration

All model and training parameters are configurable via YAML files in `configs/`:

| Config | Parameters | VRAM Usage |
|--------|-----------|------------|
| `small.yaml` | ~85M (d_model=512, n_layers=8) | ~2-3 GB |
| `default.yaml` | ~150M (d_model=768, n_layers=12) | ~4-5 GB |
| `medium.yaml` | ~350M (d_model=1024, n_layers=16) | ~6 GB |

### Recommended Hyperparameters

| Parameter | Small | Default | Medium |
|-----------|-------|---------|--------|
| d_model | 512 | 768 | 1024 |
| n_layers | 8 | 12 | 16 |
| n_heads | 8 | 12 | 16 |
| n_kv_heads | 4 | 4 | 8 |
| d_ff | 1365 | 2048 | 2730 |
| max_seq_len | 1024 | 2048 | 2048 |
| batch_size | 8 | 4 | 4 |
| learning_rate | 5e-4 | 3e-4 | 3e-4 |

## Model Architecture Details

### Grouped Query Attention (GQA)
Uses fewer key-value heads than query heads, significantly reducing KV cache memory while maintaining model quality. For example, with 12 query heads and 4 KV heads, each KV head serves 3 query heads.

### Rotary Positional Embeddings (RoPE)
Applies rotation to query and key vectors based on position, allowing the model to learn relative position dependencies naturally without learned position embeddings.

### SwiGLU
A gated activation function: `SwiGLU(x) = (x * sigmoid(xW1)) * V * W2`. Provides better performance than standard ReLU or GELU in transformer models.

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=models --cov=training --cov=tokenizer --cov=datasets --cov=utils

# Run specific test file
pytest tests/test_attention.py -v
```

## Future Roadmap

- [ ] Mixture of Experts (MoE)
- [ ] Vision encoder integration
- [ ] Speech encoder integration
- [ ] Multimodal support
- [ ] Tool calling / Function calling
- [ ] RAG (Retrieval-Augmented Generation)
- [ ] RLHF pipeline
- [ ] LoRA / QLoRA fine-tuning
- [ ] 4-bit / 8-bit quantization (GPTQ, AWQ)
- [ ] Distributed training (FSDP, DDP)
- [ ] Speculative decoding

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA support for GPU training)
- 6GB+ VRAM GPU recommended (RTX 3050, RTX 3060, RTX 4060, etc.)
- 16GB+ System RAM

**Hardware tested:** NVIDIA RTX 3050 Laptop GPU (6GB VRAM), Intel Core i7-13650HX, 32GB DDR5 RAM, Windows 11.

## License

MIT

## Citation

```bibtex
@software{kitai2026,
  title = {KitAI: Production-Quality Local Language Model Framework},
  author = {NoCheeseMercy On Github},
  year = {2026},
  url = {https://github.com/NoCheeseMercy/kitai}
}
