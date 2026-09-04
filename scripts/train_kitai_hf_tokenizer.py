"""Train KitAI's from-scratch Hugging Face-compatible ByteLevel BPE tokenizer."""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<mask>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a KitAI ByteLevel BPE tokenizer")
    parser.add_argument("--files", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = [path.resolve() for path in args.files]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing tokenizer training input(s): " + ", ".join(missing))

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(path) for path in files], trainer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))

    loaded = Tokenizer.from_file(str(args.output))
    missing_special = [token for token in SPECIAL_TOKENS if loaded.token_to_id(token) is None]
    if missing_special:
        raise RuntimeError("Tokenizer did not preserve special tokens: " + ", ".join(missing_special))
    print(f"Saved {args.output} with vocab_size={loaded.get_vocab_size()}")
    for token in SPECIAL_TOKENS:
        print(f"{token}: {loaded.token_to_id(token)}")


if __name__ == "__main__":
    main()
