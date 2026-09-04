from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32)
    model.eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"))
    encoded = tokenizer.encode("KitAI export validation.", add_special_tokens=False)
    if not encoded:
        encoded = [config.bos_token_id]
    input_ids = torch.tensor([encoded[: min(len(encoded), 16)]], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
    if not torch.isfinite(logits).all():
        raise RuntimeError("Exported model produced non-finite logits")
    print("config_model_type:", config.model_type)
    print("input_shape:", tuple(input_ids.shape))
    print("logits_shape:", tuple(logits.shape))
    print("logits_finite:", bool(torch.isfinite(logits).all()))


if __name__ == "__main__":
    main()
