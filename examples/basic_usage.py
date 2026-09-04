"""
Basic Usage Examples for KitAI.

This script demonstrates the fundamental capabilities of KitAI:
1. Creating a model configuration
2. Instantiating the model
3. Forward pass with random data
4. Text generation
5. Tokenizer training and usage

Run: python examples/basic_usage.py
"""

import sys
from pathlib import Path

import torch

# Add KitAI root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.transformer import KitAITransformer
from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.trainer import TokenizerTrainer
from training.optimizer import create_optimizer
from training.scheduler import create_scheduler


def example_1_model_creation():
    """Create a model with default configuration."""
    print("=" * 60)
    print("Example 1: Model Creation")
    print("=" * 60)

    # Use tiny config for demonstration
    config = ModelConfig.tiny()
    model = KitAITransformer(config)
    print(f"Model created with {model.get_trainable_parameters():,} parameters")
    print(f"Configuration: d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"               n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}")
    return model, config


def example_2_forward_pass(model, config):
    """Run a forward pass through the model."""
    print("\n" + "=" * 60)
    print("Example 2: Forward Pass")
    print("=" * 60)

    batch, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    logits, hidden_states = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    return logits


def example_3_loss_computation(model, config):
    """Compute loss and perplexity."""
    print("\n" + "=" * 60)
    print("Example 3: Loss Computation")
    print("=" * 60)

    batch, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch, seq_len))

    loss, perplexity = model.forward_with_loss(input_ids, labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.4f}")


def example_4_text_generation(model, config):
    """Generate text from a prompt."""
    print("\n" + "=" * 60)
    print("Example 4: Text Generation")
    print("=" * 60)

    batch, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    generated = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    print(f"Input tokens: {input_ids.tolist()}")
    print(f"Generated tokens: {generated.tolist()}")
    print(f"Generated {generated.shape[1] - seq_len} new tokens")


def example_5_tokenizer_training():
    """Train and use a simple BPE tokenizer."""
    print("\n" + "=" * 60)
    print("Example 5: Tokenizer Training")
    print("=" * 60)

    # Sample training texts
    texts = [
        "Hello, world! Welcome to KitAI.",
        "KitAI is a production-quality local language model framework.",
        "It implements modern transformer architecture with RoPE, GQA, and SwiGLU.",
        "Train your own language model on consumer GPUs.",
        "BPE tokenization enables efficient text processing.",
    ]

    # Train tokenizer
    trainer = TokenizerTrainer(vocab_size=500, min_frequency=1)
    tokenizer = trainer.train(texts=texts)

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Encode and decode
    test_text = "Hello, KitAI!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")


def example_6_optimizer_and_scheduler(model, config):
    """Create optimizer and learning rate scheduler."""
    print("\n" + "=" * 60)
    print("Example 6: Optimizer & Scheduler")
    print("=" * 60)

    # Create optimizer with weight decay
    optimizer = create_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    # Create cosine scheduler
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=100,
        max_steps=1000,
        min_lr=3e-5,
    )

    print(f"Optimizer: {type(optimizer).__name__}")
    print(f"Scheduler: Cosine with warmup")
    print(f"Initial LR: {scheduler.get_last_lr()[0]:.6f}")

    # Simulate training steps
    for step in range(5):
        optimizer.zero_grad()

        # Forward
        batch, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch, seq_len))
        loss, _ = model.forward_with_loss(input_ids, labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Step
        optimizer.step()
        scheduler.step()

        print(f"Step {step+1}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.6f}")


if __name__ == "__main__":
    print("KitAI Basic Usage Examples")
    print("=" * 60)

    # Run examples
    model, config = example_1_model_creation()
    example_2_forward_pass(model, config)
    example_3_loss_computation(model, config)
    example_4_text_generation(model, config)
    example_5_tokenizer_training()
    example_6_optimizer_and_scheduler(model, config)

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
