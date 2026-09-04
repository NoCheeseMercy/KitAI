"""Export a KitAI checkpoint into a Llama-compatible Hugging Face directory.

The resulting directory contains safetensors weights, Llama configuration, and
KitAI's tokenizer JSON.  It is intended as the deterministic input to the
official llama.cpp GGUF converter used by Ollama.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


DEFAULT_SPECIAL_TOKENS = {
    "pad": "<pad>",
    "unk": "<unk>",
    "bos": "<bos>",
    "eos": "<eos>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="KitAI .pt checkpoint")
    parser.add_argument("--tokenizer", required=True, help="KitAI tokenizer JSON")
    parser.add_argument("--output-dir", required=True, help="Destination Hugging Face model directory")
    parser.add_argument("--overwrite", action="store_true", help="Allow an existing destination directory")
    return parser.parse_args()


def get_special_tokens(tokenizer_path: Path) -> dict[str, str]:
    try:
        from tokenizer.bpe_tokenizer import BPETokenizer

        return {**DEFAULT_SPECIAL_TOKENS, **BPETokenizer.load(tokenizer_path).special_tokens}
    except Exception:
        return dict(DEFAULT_SPECIAL_TOKENS)


def get_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config") or checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a serialised model configuration")
    required = {"vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff", "max_seq_len"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Checkpoint model configuration is missing: {', '.join(missing)}")
    return config


def map_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target: dict[str, torch.Tensor] = {}
    key_map = {
        "embedding.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output_head.weight": "lm_head.weight",
    }
    for source_name, tensor in state.items():
        target_name = key_map.get(source_name)
        if target_name is None:
            if source_name.endswith(".attention_norm.weight"):
                layer_index = source_name.split(".")[1]
                target_name = f"model.layers.{layer_index}.input_layernorm.weight"
            elif source_name.endswith(".ffn_norm.weight"):
                layer_index = source_name.split(".")[1]
                target_name = f"model.layers.{layer_index}.post_attention_layernorm.weight"
            elif ".attention." in source_name:
                layer_index = source_name.split(".")[1]
                projection = source_name.split(".")[-2]
                target_name = f"model.layers.{layer_index}.self_attn.{projection}.weight"
            elif ".feed_forward." in source_name:
                layer_index = source_name.split(".")[1]
                projection = source_name.split(".")[-2]
                target_name = f"model.layers.{layer_index}.mlp.{projection}.weight"
            else:
                raise ValueError(f"No Hugging Face mapping for tensor: {source_name}")
        if target_name in target:
            raise ValueError(f"Duplicate target tensor name: {target_name}")
        # Clone to eliminate shared storage from tied embedding/output weights.
        target[target_name] = tensor.detach().to("cpu").contiguous().clone()
    return target


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Destination is not empty: {output_dir}; pass --overwrite to reuse it")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = get_model_config(checkpoint)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain model_state_dict")

    hf_state = map_state_dict(state)
    expected_layers = int(model_config["n_layers"])
    expected_tensor_count = 3 + expected_layers * 9
    if len(hf_state) != expected_tensor_count:
        raise ValueError(f"Expected {expected_tensor_count} exported tensors, found {len(hf_state)}")

    special_tokens = get_special_tokens(tokenizer_path)
    token_ids = {
        key: next((index for index, name in enumerate([])), None)
        for key in ()
    }
    # KitAI reserves IDs in special-token insertion order: pad, unk, bos, eos, mask.
    token_ids.update({"pad": 0, "unk": 1, "bos": 2, "eos": 3})

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": int(model_config["vocab_size"]),
        "hidden_size": int(model_config["d_model"]),
        "intermediate_size": int(model_config["d_ff"]),
        "num_hidden_layers": expected_layers,
        "num_attention_heads": int(model_config["n_heads"]),
        "num_key_value_heads": int(model_config["n_kv_heads"]),
        "max_position_embeddings": int(model_config["max_seq_len"]),
        "rms_norm_eps": float(model_config.get("norm_eps", 1e-6)),
        "rope_theta": float(model_config.get("rope_theta", 10000.0)),
        "hidden_act": "silu",
        "tie_word_embeddings": bool(model_config.get("weight_tying", True)),
        "attention_bias": bool(model_config.get("bias", False)),
        "mlp_bias": bool(model_config.get("bias", False)),
        "bos_token_id": token_ids["bos"],
        "eos_token_id": token_ids["eos"],
        "pad_token_id": token_ids["pad"],
        "torch_dtype": "float32",
        "transformers_version": "5.2.0",
    }
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "bos_token": special_tokens["bos"],
        "eos_token": special_tokens["eos"],
        "pad_token": special_tokens["pad"],
        "unk_token": special_tokens["unk"],
        "model_max_length": int(model_config["max_seq_len"]),
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
    }
    special_tokens_map = {
        "bos_token": special_tokens["bos"],
        "eos_token": special_tokens["eos"],
        "pad_token": special_tokens["pad"],
        "unk_token": special_tokens["unk"],
    }

    save_file(hf_state, str(output_dir / "model.safetensors"), metadata={"format": "pt"})
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    (output_dir / "config.json").write_text(json.dumps(hf_config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8")
    (output_dir / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2) + "\n", encoding="utf-8")
    (output_dir / "generation_config.json").write_text(
        json.dumps({"bos_token_id": token_ids["bos"], "eos_token_id": token_ids["eos"], "pad_token_id": token_ids["pad"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# KitAI Llama-Compatible Export\n\n"
        "This directory was generated from a KitAI checkpoint. Its tensor names and configuration are mapped to the Llama decoder architecture for conversion to GGUF with llama.cpp.\n",
        encoding="utf-8",
    )
    print(f"Exported {len(hf_state)} tensors to {output_dir}")


if __name__ == "__main__":
    main()
