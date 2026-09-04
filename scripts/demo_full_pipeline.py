"""
KitAI full pipeline smoke demo.

Trains a tiny tokenizer, builds a tiny model, runs forward / generate /
one training step. Use this to verify the install is healthy.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("\n[1/6] Creating sample training data...")
data_dir = ROOT / "demo_data"
data_dir.mkdir(exist_ok=True)

sample_data = """The KitAI language model is a production-quality transformer built from scratch.
It uses Rotary Positional Embeddings (RoPE) for position encoding.
The architecture features Grouped Query Attention for efficient memory usage.
SwiGLU activation functions are used in the feed-forward network.
RMSNorm provides stable training with pre-normalization.
Weight tying reduces the total parameter count by sharing embedding weights.
The model is designed for consumer GPUs with 6GB of VRAM.
Gradient checkpointing trades computation for memory during training.
Mixed precision training with float16 reduces memory usage by half.
The BPE tokenizer efficiently encodes text into subword tokens.
Cosine learning rate scheduling with warmup provides stable training.
The Fused AdamW optimizer combines weight decay with adaptive learning rates.
KV caching accelerates text generation by storing previous key-value pairs.
The model supports top-k sampling, top-p sampling, and temperature scaling.
Repetition penalty helps avoid repetitive text during generation.
KitAI can be trained on text data, JSONL, CSV, Parquet, and more formats.
The checkpoint system saves both the best and latest model states.
Training can be resumed from any checkpoint with the resume module.
The rich dashboard shows live metrics during training.
Model evaluation computes perplexity and loss on validation data.
Future versions will support Mixture of Experts and multimodal inputs."""

data_file = data_dir / "kitai_training_data.txt"
data_file.write_text(sample_data, encoding="utf-8")
print(f"  Created {data_file} ({len(sample_data)} chars)")

print("\n[2/6] Checking environment...")
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

print("\n[3/6] Training BPE tokenizer...")
from tokenizer.trainer import TokenizerTrainer
from tokenizer.utils import SPECIAL_TOKENS

trainer = TokenizerTrainer(
    vocab_size=200,
    min_frequency=1,
    special_tokens=SPECIAL_TOKENS,
)

tokenizer = trainer.train(files=[data_file], verbose=True)

tokenizer_path = data_dir / "kitai_tokenizer.json"
tokenizer.save(str(tokenizer_path))
print(f"  Tokenizer saved to {tokenizer_path}")
print(f"  Vocabulary size: {tokenizer.vocab_size}")

test_text = "KitAI language model with RoPE and GQA"
encoded = tokenizer.encode(test_text, add_special_tokens=False)
decoded = tokenizer.decode(encoded)
print(f"  Tokenizer test: '{test_text}' -> {encoded[:10]}... ({len(encoded)} tokens)")
print(f"  Decoded: '{decoded}'")

print("\n[4/6] Initializing KitAI model (tiny config)...")
from models.config import ModelConfig
from models.transformer import KitAITransformer

config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    n_layers=2,
    n_heads=2,
    n_kv_heads=1,
    d_ff=128,
    max_seq_len=512,
    dropout=0.0,
    weight_tying=True,
    gradient_checkpointing=False,
)

model = KitAITransformer(config).to(device)
param_count = model.get_trainable_parameters()
print(f"  Model parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

print("\n[5/6] Testing forward pass and generation...")
prompt = "The KitAI language model"
input_ids = torch.tensor(
    tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.long
).unsqueeze(0).to(device)
print(f"  Input: '{prompt}' -> {input_ids.shape}")

with torch.no_grad():
    logits, _ = model(input_ids)
    print(f"  Forward pass output shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

print("\n  Generating text (untrained model - will be random)...")
generated = model.generate(
    input_ids,
    max_new_tokens=20,
    temperature=0.8,
    top_k=20,
    top_p=0.9,
    presence_penalty=0.1,
    frequency_penalty=0.1,
)
generated_text = tokenizer.decode(generated[0].tolist())
print(f"  Generated: '{generated_text}'")

print("\n[6/6] Testing training step...")
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.logger import setup_logger

logger = setup_logger(log_dir=str(data_dir / "logs"), experiment_name="demo")

optimizer = build_optimizer(model, learning_rate=3e-4, weight_decay=0.01)
print(f"  Optimizer: {type(optimizer).__name__}")

scheduler = build_scheduler(
    optimizer,
    total_steps=100,
    warmup_steps=10,
    schedule_type="cosine",
    min_lr=3e-5,
)
print("  Scheduler: cosine, 100 steps, 10 warmup")

model.train()
tokens = torch.tensor(
    tokenizer.encode(sample_data, add_special_tokens=False), dtype=torch.long
)
# Keep sequences within max_seq_len
seq = tokens[: min(len(tokens), config.max_seq_len + 1)]
inputs = seq[:-1].unsqueeze(0).to(device)
targets = seq[1:].unsqueeze(0).to(device)

optimizer.zero_grad(set_to_none=True)
loss, ppl = model.forward_with_loss(inputs, targets)
loss.backward()
optimizer.step()
scheduler.step()

print(f"  Training step completed! Loss: {loss.item():.4f}  PPL: {ppl.item():.2f}")
logger.log_metrics({"loss": loss.item(), "perplexity": ppl.item()}, title="Demo Step")
logger.close()

print("\n" + "=" * 70)
print("DEMO COMPLETE - KITAI IS WORKING!")
print("=" * 70)
print("\n  [OK] Tokenizer: {} vocab".format(tokenizer.vocab_size))
print("  [OK] Model: {} parameters on {}".format(param_count, device))
print("  [OK] Forward pass: OK")
print("  [OK] Text generation: OK")
print("  [OK] Training step: OK")
print(f"\nArtifacts in: {data_dir}")
print("\nFull training command:")
print(
    "  python -m scripts.train --config configs/small.yaml "
    "--data demo_data/kitai_training_data.txt"
)
