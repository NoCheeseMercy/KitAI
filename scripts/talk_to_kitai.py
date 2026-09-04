"""Talk to or smoke-test the latest completed KitAI checkpoint.

Examples:
    python scripts/talk_to_kitai.py
    python scripts/talk_to_kitai.py --prompt "Write one sentence about the moon."
    python scripts/talk_to_kitai.py --quick-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch
from tokenizers import Tokenizer as HFTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from models.config import ModelConfig
from models.transformer import KitAITransformer
from scripts.chat import ChatInterface


class HFTokenizerAdapter:
    """Expose the small tokenizer interface used by KitAI's ChatInterface."""

    def __init__(self, tokenizer_path: Path) -> None:
        self._tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        self.pad_token_id = self._id_for("<pad>", 0)
        self.unk_token_id = self._id_for("<unk>", 1)
        self.bos_token_id = self._id_for("<bos>", 2)
        self.eos_token_id = self._id_for("<eos>", 3)

    def _id_for(self, token: str, fallback: int) -> int:
        token_id = self._tokenizer.token_to_id(token)
        return fallback if token_id is None else int(token_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = list(self._tokenizer.encode(text, add_special_tokens=False).ids)
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=skip_special_tokens)


def default_checkpoint(checkpoint_dirs: Iterable[Path]) -> Path:
    """Return the newest checkpoint, preferring the isolated mixed-data run."""
    ordered_dirs = list(checkpoint_dirs)
    for checkpoint_dir in ordered_dirs:
        candidates = sorted(
            checkpoint_dir.glob("*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    searched = ", ".join(str(path) for path in ordered_dirs)
    raise FileNotFoundError(f"No checkpoint files were found in: {searched}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talk to the latest completed KitAI model")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional KitAI checkpoint override")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=PROJECT_DIR / "tokenizer" / "kitai_bpe_32k.json",
        help="Hugging Face tokenizer JSON",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Answer one prompt and exit")
    parser.add_argument("--quick-test", action="store_true", help="Run a short deterministic prompt suite and exit")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum generated tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature; use 0 for greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus-sampling threshold")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference instead of CUDA when available")
    return parser.parse_args()


def load_chat(args: argparse.Namespace) -> tuple[ChatInterface, Path, torch.device]:
    checkpoint_dirs = (
        PROJECT_DIR / "checkpoints" / "fable5_wikitext_mixed_smoke",
        PROJECT_DIR / "checkpoints" / "overnight_cache_20260817",
    )
    checkpoint_path = (args.checkpoint or default_checkpoint(checkpoint_dirs)).resolve()
    tokenizer_path = args.tokenizer.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config_dict = checkpoint.get("config")
    if not isinstance(model_config_dict, dict):
        raise ValueError("The checkpoint does not contain a usable model configuration")
    model_config = ModelConfig(**model_config_dict)
    model = KitAITransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    model.eval()

    tokenizer = HFTokenizerAdapter(tokenizer_path)
    chat = ChatInterface(
        model=model,
        tokenizer=tokenizer,
        config={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_tokens,
            "stream": False,
        },
    )
    return chat, checkpoint_path, device


def run_prompt(chat: ChatInterface, prompt: str) -> None:
    print(f"\nYou: {prompt}")
    print("KitAI:", chat.generate_response(prompt).strip())


def main() -> None:
    args = parse_args()
    chat, checkpoint_path, device = load_chat(args)
    print("KitAI local chat")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("Note: this is an early checkpoint, so responses are experimental.")

    if args.quick_test:
        for prompt in (
            "Complete this sentence: The capital of France is",
            "Write one short sentence about the Moon.",
            "What is 2 plus 2?",
        ):
            run_prompt(chat, prompt)
        return
    if args.prompt is not None:
        run_prompt(chat, args.prompt)
        return

    print("\nType a message. Commands: /clear, /quit, /help")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye.")
            break
        if prompt.lower() == "/clear":
            chat.clear_history()
            print("Conversation history cleared.")
            continue
        if prompt.lower() == "/help":
            print("Enter a prompt, /clear to reset context, or /quit to exit.")
            continue
        run_prompt(chat, prompt)


if __name__ == "__main__":
    main()
