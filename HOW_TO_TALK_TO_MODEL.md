# How to Talk to KitAI Model

KitAI is a decoder-only transformer language model designed for consumer GPUs (like RTX 3050 6GB). Here's how to interact with it:

## Quick Start (After Training)

### 1. Interactive Chat Mode
```powershell
python -m scripts.chat --model checkpoints/best.pt --tokenizer demo_data/kitai_tokenizer.json --interactive
```

### 2. Single Prompt Generation
```powershell
python -m scripts.chat --model checkpoints/best.pt --tokenizer demo_data/kitai_tokenizer.json --prompt "Explain quantum computing in simple terms"
```

### 3. Batch Generation from File
```powershell
python -m scripts.generate --model checkpoints/best.pt --tokenizer demo_data/kitai_tokenizer.json --prompt-file prompts.txt --output results.jsonl
```

### 4. Custom Generation Parameters
```powershell
python -m scripts.generate --model checkpoints/best.pt --tokenizer demo_data/kitai_tokenizer.json --prompt "The future of AI is" --temperature 0.7 --top-k 50 --top-p 0.95 --max-tokens 256
```

## Training Your Own Model

### Prepare Your Data
KitAI supports multiple formats:
- Plain text (.txt)
- JSONL (.jsonl) 
- Parquet (.parquet)
- CSV (.csv)
- Alpaca format
- ShareGPT format
- OpenAI messages format

Place your data in `demo_data/` or specify a path with `--data`.

### Start Training
```powershell
# Small model (~28M params) - fits easily on 6GB VRAM
python -m scripts.train --config configs/small.yaml --data demo_data/your_data.txt

# Medium model (~85M params) 
python -m scripts.train --config configs/medium.yaml --data demo_data/your_data.txt

# Large model (~180M params) - may need gradient accumulation
python -m scripts.train --config configs/large.yaml --data demo_data/your_data.txt
```

### Resume Training
```powershell
python -m scripts.resume --checkpoint checkpoints/checkpoint_last.pt --data demo_data/your_data.txt
```

## Model Capabilities

### Supported Features
- **Architecture**: Decoder-only transformer with RoPE, RMSNorm, SwiGLU, GQA
- **Efficiency**: Flash Attention, gradient checkpointing, mixed precision, KV caching
- **Generation Controls**: 
  - Temperature (0.0 - 2.0)
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Repetition penalty
  - Presence penalty
  - Frequency penalty
  - Stop sequences
- **Interfaces**: 
  - Interactive chat
  - Batch generation
  - Streaming output (via Python API)
  - Speculative decoding (future-ready)

### Python API
```python
from kitai import KitAITransformer, ModelConfig, BPETokenizer, stream_generate

# Load model and tokenizer
config = ModelConfig.from_pretrained("checkpoints/best.pt")
model = KitAITransformer.from_pretrained("checkpoints/best.pt")
tokenizer = BPETokenizer.load("demo_data/kitai_tokenizer.json")

# Generate with full control
input_ids = tokenizer.encode("Hello, how are you?")
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
    presence_penalty=0.1,
    frequency_penalty=0.1
)
print(tokenizer.decode(output[0]))

# Streaming generation (yields tokens one at a time)
for token_id, token in stream_generate(model, input_ids, max_new_tokens=50):
    print(tokenizer.decode([token_id]), end="", flush=True)
```

## Expected Results

**Important**: The model's output quality directly correlates with:
1. **Training data quality and quantity** - More diverse, high-quality text = better results
2. **Training steps** - For the small model (~28M params), aim for at least 5k-10k steps on decent data
3. **Model size** - Larger models (85M+) capture more patterns but need more VRAM/time

With the small demo dataset (64 lines), the model will produce mostly nonsensical output as seen in your tests. This is expected - it's like asking a person who's only read 64 lines of text to write an essay.

## Next Steps for Better Results

1. **Get more training data**: 
   - Download datasets from Hugging Face (e.g., TinyStories, TinyLlama sources)
   - Combine multiple text sources (books, articles, code, conversations)
   - Aim for at least 100MB-1GB of clean text for meaningful results

2. **Train longer**:
   - Start with `max_steps: 10000` in your config
   - Monitor loss - stop when it plateaus
   - Use learning rate warmup and cosine decay (already configured)

3. **Experiment with configurations**:
   - Try `configs/medium.yaml` for better quality (still fits 6GB)
   - Adjust `d_model`, `n_layers`, `n_heads` based on your VRAM

4. **Use the provided scripts**:
   - `train.py` - main training loop with logging
   - `resume.py` - continue from checkpoints
   - `chat.py` - interactive inference
   - `generate.py` - batch generation
   - `evaluate.py` - compute perplexity
   - `demo_full_pipeline.py` - end-to-end test

## Troubleshooting

**CUDA out of memory**: 
- Reduce `batch_size` or increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing` (already on in configs)
- Use mixed precision (`use_amp: true`)

**Slow training**:
- Ensure you're using a CUDA-enabled PyTorch build
- Check GPU utilization with `nvidia-smi`
- Consider reducing `max_seq_len` if sequences are very long

**Poor generation quality**:
- Train longer with more data
- Check data quality (remove garbage, ensure proper encoding)
- Try different temperature/top-p values
- Ensure tokenizer was trained on similar data

## Memory Usage Estimates (RTX 3050 6GB)

| Model Size | Parameters | VRAM Usage (Training) | VRAM Usage (Inference) |
|------------|------------|----------------------|------------------------|
| Small      | ~28M       | ~3.5GB               | ~1.2GB                 |
| Medium     | ~85M       | ~5.0GB               | ~2.0GB                 |
| Large      | ~180M      | ~5.8GB (with grad checkpointing) | ~3.5GB |

All configurations include:
- AdamW optimizer states
- Gradient buffers  
- Activation storage (with checkpointing)
- KV cache for generation
- Mixed precision training

## Community & Support

For questions, issues, or collaboration:
- Check the `docs/` directory for detailed documentation
- Look at `examples/` for usage patterns
- Review `tests/` for implementation verification
- The project follows standard Python packaging - install with `pip install -e .` for development

Remember: This is a foundation model. With sufficient quality data and training, it can become a capable local AI assistant for your specific use cases.