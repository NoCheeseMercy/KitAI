"""Estimate decoder-only KitAI parameter counts and from-scratch token budgets."""
from __future__ import annotations


def parameter_count(vocab: int, d_model: int, layers: int, kv_heads: int, heads: int, d_ff: int) -> int:
    head_dim = d_model // heads
    kv_dim = kv_heads * head_dim
    embeddings = vocab * d_model
    attention = 2 * d_model * d_model + 2 * d_model * kv_dim
    swiglu = 3 * d_model * d_ff
    norms = 2 * d_model
    return embeddings + layers * (attention + swiglu + norms)


def main() -> None:
    candidates = [
        ("current_67M", 32000, 640, 10, 5, 10, 1792),
        ("proposed_198M", 32000, 768, 20, 4, 12, 3072),
        ("one_billion_class", 32000, 1536, 24, 8, 16, 6144),
    ]
    print("name,parameters,fp32_model_plus_grads_adam_gb,20x_token_budget,tokens_per_second,days_at_budget")
    for name, vocab, d_model, layers, kv_heads, heads, d_ff in candidates:
        params = parameter_count(vocab, d_model, layers, kv_heads, heads, d_ff)
        # fp32 parameters + fp32 gradients + Adam m/v, excluding activations and framework overhead.
        optimizer_floor_gb = params * 16 / 1_000_000_000
        token_budget = params * 20
        # Conservative proportional extrapolation from the measured 67M run at ~8,100 tok/s.
        estimated_tps = 8100 * parameter_count(32000, 640, 10, 5, 10, 1792) / params
        days = token_budget / estimated_tps / 86400
        print(f"{name},{params},{optimizer_floor_gb:.2f},{token_budget},{estimated_tps:.0f},{days:.1f}")


if __name__ == "__main__":
    main()
