# KitAI Architecture Plan

## Overview
KitAI is a production-quality local language model framework built from scratch, combining the best ideas from GPT, Llama, Qwen, DeepSeek, Mistral, Phi, and Gemma architectures. Optimized for consumer GPUs with 6GB VRAM.

---

## 1. Project Structure

```
KitAI/
├── configs/                  # Configuration files
│   ├── default.yaml          # Default model config
│   ├── small.yaml            # Small model (for 6GB VRAM)
│   ├── medium.yaml           # Medium model
│   └── training.yaml         # Training-specific config
├── models/                   # Model architecture
│   ├── __init__.py
│   ├── config.py             # ModelConfig dataclass
│   ├── transformer.py        # Main transformer model
│   ├── attention.py          # Multi-head + GQA attention
│   ├── layers.py             # Transformer blocks
│   ├── embedding.py          # Token embeddings + RoPE
│   ├── normalization.py      # RMSNorm
│   ├── feed_forward.py       # SwiGLU FFN
│   ├── kv_cache.py           # KV cache implementation
│   └── output.py             # LM head + loss
├── tokenizer/                # BPE tokenizer
│   ├── __init__.py
│   ├── bpe_tokenizer.py      # BPE implementation
│   ├── trainer.py            # Tokenizer training
│   └── utils.py              # Tokenizer utilities
├── training/                 # Training pipeline
│   ├── __init__.py
│   ├── dataset.py            # Dataset loading & preprocessing
│   ├── trainer.py            # Main training loop
│   ├── optimizer.py          # Fused AdamW
│   ├── scheduler.py          # Cosine LR scheduler with warmup
│   ├── checkpoint.py         # Checkpoint saving/loading
│   ├── metrics.py            # Training metrics tracking
│   ├── evaluation.py         # Evaluation pipeline
│   ├── logger.py             # Logging (console + file)
│   ├── validation.py         # Validation logic
│   └── resume.py             # Resume training from checkpoint
├── datasets/                 # Dataset support
│   ├── __init__.py
│   ├── base_dataset.py       # Base dataset class
│   ├── text_dataset.py       # txt files
│   ├── jsonl_dataset.py      # jsonl files
│   ├── parquet_dataset.py    # parquet files
│   ├── csv_dataset.py        # csv files
│   ├── alpaca_dataset.py     # Alpaca format
│   ├── sharegpt_dataset.py   # ShareGPT format
│   ├── openai_dataset.py     # OpenAI messages format
│   └── packer.py             # Sequence packing
├── utils/                    # Utilities
│   ├── __init__.py
│   ├── device.py             # Device management
│   ├── memory.py             # Memory optimization
│   ├── distributed.py        # (Future) DDP support
│   ├── seeding.py            # Reproducibility
│   └── io_utils.py           # File I/O helpers
├── scripts/                  # Entry points
│   ├── train.py              # Training entry point
│   ├── chat.py               # Inference/chat CLI
│   ├── evaluate.py           # Evaluation script
│   ├── resume.py             # Resume training script
│   ├── tokenize.py           # Tokenizer training script
│   └── generate.py           # Generation script
├── tests/                    # Tests
│   ├── test_attention.py
│   ├── test_layers.py
│   ├── test_tokenizer.py
│   ├── test_training.py
│   └── test_generation.py
├── examples/                 # Usage examples
│   └── basic_usage.py
├── requirements.txt
├── setup.py
├── README.md
└── pyproject.toml
```

---

## 2. Model Architecture (Decoder-Only Transformer)

### 2.1 Core Components

#### Rotary Positional Embeddings (RoPE)
- Implements frequency-based rotary embeddings
- Applied to query and key vectors in attention
- Supports `theta=10000.0` base frequency
- Configurable `rope_theta` parameter

#### RMSNorm (Root Mean Square Normalization)
- Pre-Norm architecture (normalize before sublayers)
- `RMSNorm(x) = x * rms(x) * gamma` where `rms(x) = sqrt(mean(x^2) + eps)`
- `eps = 1e-6` for numerical stability

#### SwiGLU Feed-Forward Network
- `FFN(x) = (swish(xW1) * xV) W2` where `swish(x) = x * sigmoid(x)`
- Hidden dimension: `8/3 * d_model` (as in Llama/Mistral)
- Supports 3 weight matrices (gate, up, down)

#### Grouped Query Attention (GQA)
- Number of key-value heads < number of query heads
- Typical ratio: `num_kv_heads = num_attention_heads / 4`
- All query heads share KV heads in groups
- Significantly reduces KV cache memory

#### KV Cache
- Stores keys and values during generation
- Memory-efficient with GQA
- Supports `max_batch_size` and `max_seq_len`
- `dtype = torch.float16` for memory efficiency

### 2.2 Default Configuration (Small - 6GB VRAM)

```yaml
model:
  vocab_size: 32000
  d_model: 768
  n_layers: 12
  n_heads: 12
  n_kv_heads: 4
  d_ff: 2048
  max_seq_len: 2048
  dropout: 0.0
  activation: swiglu
  norm_eps: 1e-6
  rope_theta: 10000.0
  weight_tying: true
  bias: false
  init_std: 0.02
```

### 2.3 Memory Budget (6GB VRAM)

| Component            | Memory (~) |
|---------------------|------------|
| Model Parameters    | ~1.5 GB    |
| Optimizer States    | ~3.0 GB    |
| Activations + Grads | ~0.8 GB    |
| KV Cache            | ~0.3 GB    |
| Misc (buffers, etc) | ~0.4 GB    |
| **Total**           | **~6.0 GB**|

Techniques to stay within budget:
1. **Gradient Checkpointing** - Trade compute for memory
2. **Mixed Precision (AMP)** - Use `torch.float16`
3. **Gradient Accumulation** - Accumulate over micro-batches
4. **Activation Checkpointing** - Selective layer checkpointing
5. **Fused AdamW** - Memory-efficient optimizer
6. **Gradient Clipping** - For training stability

---

## 3. Training Pipeline

### 3.1 Dataset Module
- Support multiple formats (txt, jsonl, parquet, csv, Alpaca, ShareGPT, OpenAI)
- Automatic sequence packing to `max_seq_len`
- Efficient shuffling (smart shuffle + buffer)
- Streaming support for large datasets
- `IterableDataset` for memory efficiency

### 3.2 Optimizer
- Fused AdamW implementation
- Weight decay support (no decay on bias/layer norms)
- `betas = (0.9, 0.95)`, `eps = 1e-8`
- Implements parameter groups for weight decay

### 3.3 Scheduler
- Cosine learning rate decay with linear warmup
- Configurable warmup_steps (recommended: 5-10% of total)
- Configurable min_lr (recommended: 10% of max_lr)
- Supports multiple schedule types (cosine, linear, constant)

### 3.4 Logging/Monitoring
- Rich live console dashboard
- Metrics: loss, lr, tokens/sec, GPU%, VRAM, ETA, epoch, step
- Gradient norm monitoring
- TensorBoard-compatible logging
- JSON log file

### 3.5 Checkpointing
- Automatic saving every N steps
- Best checkpoint (lowest validation loss)
- Latest checkpoint (resume support)
- Checkpoint averaging (optional)
- Contains: model state, optimizer state, scheduler state, step, epoch

---

## 4. Tokenizer

### 4.1 BPE Tokenizer Implementation
- Train from text corpus
- Support special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<mask>`
- Configurable vocab size (default: 32000)
- Efficient encoding/decoding
- Vocabulary export/import

### 4.2 Features
- Pre-tokenization (by whitespace + punctuation)
- Merge ranking by frequency
- Byte-level BPE (for universal encoding)
- Regex-based splitting patterns
- Fast inference path (cached merges)

---

## 5. Inference

### 5.1 Chat Interface
- Interactive CLI with streaming output
- Conversation history management
- System prompt support
- Chat templates (Llama, Mistral, Alpaca)
- Markdown rendering (optional)

### 5.2 Generation Parameters
- Top-k sampling (`k=40-50`)
- Top-p (nucleus) sampling (`p=0.9-0.95`)
- Temperature scaling (`t=0.1-1.5`)
- Repetition penalty (`1.0-1.2`)
- Presence penalty (`0.0-0.5`)
- Frequency penalty (`0.0-0.5`)
- Stop sequences (customizable)
- Max tokens limit

### 5.3 Generation Modes
- Greedy decoding (temperature=0)
- Sampling (stochastic)
- Beam search (future)
- Speculative decoding interface (future-ready)

---

## 6. Implementation Order

### Phase 1: Foundation (Files 1-10)
1. `models/config.py` - Model configuration system
2. `models/normalization.py` - RMSNorm
3. `models/embedding.py` - Embeddings + RoPE
4. `models/attention.py` - GQA attention
5. `models/feed_forward.py` - SwiGLU FFN
6. `models/kv_cache.py` - KV cache
7. `models/layers.py` - Transformer block
8. `models/output.py` - LM head + loss
9. `models/transformer.py` - Full model assembly
10. `utils/*.py` - Utility modules

### Phase 2: Tokenizer (Files 11-13)
11. `tokenizer/bpe_tokenizer.py` - BPE implementation
12. `tokenizer/trainer.py` - Tokenizer training
13. `tokenizer/utils.py` - Tokenizer utilities

### Phase 3: Training Pipeline (Files 14-25)
14. `training/dataset.py` - Dataset system
15. `training/optimizer.py` - Fused AdamW
16. `training/scheduler.py` - LR scheduler
17. `training/checkpoint.py` - Checkpoint system
18. `training/metrics.py` - Metrics tracking
19. `training/logger.py` - Logging system
20. `training/evaluation.py` - Evaluation
21. `training/validation.py` - Validation
22. `training/resume.py` - Resume training
23. `training/trainer.py` - Main trainer
24. `datasets/*.py` - Dataset format support
25. `configs/*.yaml` - Configuration files

### Phase 4: Scripts & Entry Points (Files 26-31)
26. `scripts/train.py` - Training script
27. `scripts/chat.py` - Chat interface
28. `scripts/generate.py` - Generation script
29. `scripts/evaluate.py` - Evaluation script
30. `scripts/resume.py` - Resume script
31. `scripts/tokenize.py` - Tokenizer training

### Phase 5: Tests & Examples (Files 32-37)
32. `tests/test_attention.py`
33. `tests/test_layers.py`
34. `tests/test_tokenizer.py`
35. `tests/test_training.py`
36. `tests/test_generation.py`
37. `examples/basic_usage.py`
38. `requirements.txt`, `setup.py`, `pyproject.toml`

---

## 7. Dependencies

```txt
torch>=2.0.0
rich>=13.0.0
pyyaml>=6.0
datasets>=2.0.0 (optional, for HuggingFace datasets)
tqdm>=4.65.0
regex>=2023.0.0
tokenizers==0.15.0 (for comparison, not used internally)
numpy>=1.24.0
psutil>=5.9.0 (for system monitoring)
pandas>=2.0.0 (for CSV/Parquet support)
pyarrow>=12.0.0 (for Parquet support)
```

---

## 8. Future Roadmap

- [ ] Mixture of Experts (MoE)
- [ ] Vision encoder
- [ ] Speech encoder
- [ ] Multimodal inputs
- [ ] Tool calling
- [ ] RAG support
- [ ] Function calling
- [ ] RLHF pipeline
- [ ] LoRA/QLoRA fine-tuning
- [ ] Quantization (GPTQ, AWQ)
- [ ] Distributed training (FSDP)

---

*This architecture is designed to be modular, extensible, and production-ready from day one.*

