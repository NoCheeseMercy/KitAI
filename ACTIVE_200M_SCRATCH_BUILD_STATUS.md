# KitAI 200M From-Scratch Build Status

**Updated:** 2026-08-20, local workstation time

## User constraints

This build must use **no pretrained weights and no other model weights**. It is a new KitAI model trained from scratch, followed by a training phase that teaches the same new KitAI weights to respond in the required chat format.

## Architecture

The isolated architecture configuration is `configs/kitai_200m_scratch_pretrain.yaml`.

| Setting | Value |
|---|---:|
| Target parameters | 197,621,760 |
| Layers | 20 |
| Model width | 768 |
| Attention heads / KV heads | 12 / 4 |
| Feed-forward width | 3,072 |
| Context length | 256 |
| Micro-batch / accumulation | 1 / 64 |
| Optimizer compatibility | AMP enabled; fused AdamW disabled |

## Active data

| Corpus | Source | Status |
|---|---|---|
| General pretraining text | `HuggingFaceFW/fineweb-edu`, `sample-10BT` configuration | 3.1B source tokens retained across 15 text shards; acquisition stopped at a memory guardrail before the original 4B target. |
| Chat training | `HuggingFaceTB/smol-smoltalk` | 363,974 conversations retained after exclusion of tool/function/agent-oriented sources and forbidden trace markers. |

The active dataset directory is `datasets/kitai_200m_scratch/`. The data manifest is `DATA_MANIFEST.json`.

## Current gate

`scripts/validate_kitai_200m_data.py` is actively scanning the complete pretraining and chat corpus for prohibited trace patterns. Training, tokenizer creation, and GPU smoke tests must **not** start until it writes `datasets/kitai_200m_scratch/validation_report.json` with `"passed": true`.

## Next steps

1. Confirm the safety report passes.
2. Train `tokenizer/kitai_200m_chat_32k.json` using `scripts/train_kitai_hf_tokenizer.py`, preserving `<|system|>`, `<|user|>`, `<|assistant|>`, and `<|end|>` as real tokens.
3. Run a 200M one-step GPU memory smoke test at sequence length 256, batch size 1.
4. Start isolated from-scratch pretraining with no resume checkpoint.
5. Run response-supervised chat training with assistant-only loss labels and test `Hello`, `Who are you?`, and short factual prompts before any export.
