"""
Chat Interface for KitAI Language Models.

Interactive CLI for chatting with trained models.

Usage:
    python -m scripts.chat --model checkpoints/best.pt --tokenizer tokenizer.json
    python -m scripts.chat --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.config import ModelConfig
from models.transformer import KitAITransformer
from tokenizer.bpe_tokenizer import BPETokenizer
from tokenizer.utils import get_chat_template
from utils.io_utils import read_yaml, resolve_path

logger = logging.getLogger(__name__)


class ChatInterface:
    """
    Interactive chat interface for KitAI models.

    Supports conversation history, system prompts, and
    various generation parameters.

    Args:
        model: The KitAI transformer model.
        tokenizer: The BPE tokenizer.
        config: Generation configuration.
    """

    def __init__(
        self,
        model: KitAITransformer,
        tokenizer: BPETokenizer,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}

        self.device = next(model.parameters()).device
        self.conversation_history: List[Dict[str, str]] = []

        # Generation parameters
        self.temperature: float = self.config.get("temperature", 0.7)
        self.top_k: int = self.config.get("top_k", 50)
        self.top_p: float = self.config.get("top_p", 0.9)
        self.max_new_tokens: int = self.config.get("max_new_tokens", 512)
        self.repetition_penalty: float = self.config.get("repetition_penalty", 1.1)
        self.system_prompt: Optional[str] = self.config.get("system_prompt", None)
        self.stream: bool = self.config.get("stream", False)

        # Chat template
        template_name = self.config.get("chat_template", "llama")
        self.chat_template = get_chat_template(template_name)

        # Set model to eval mode
        self.model.eval()

    def _format_prompt(self, user_input: str) -> str:
        """Format the prompt with conversation history."""
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self.conversation_history:
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})

        return self.chat_template(user_input, system_message=self.system_prompt, history=messages)

    def generate_response(self, user_input: str, stream: bool = False):
        """
        Generate a response to user input.

        Args:
            user_input: User's message.
            stream: If True, yield tokens one at a time.

        Returns:
            Generated response text (or generator if stream=True).
        """
        prompt = self._format_prompt(user_input)
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        if stream:
            return self._stream_generate(input_tensor, input_ids)

        with torch.no_grad():
            generated = self.model.generate(
                input_tensor,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )

        response_ids = generated[0].tolist()
        response_text = self.tokenizer.decode(response_ids)

        # Extract only the new response part
        if len(response_ids) > len(input_ids):
            new_ids = response_ids[len(input_ids):]
            response_text = self.tokenizer.decode(new_ids)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response_text})

        return response_text

    def _stream_generate(
        self, input_tensor: torch.Tensor, input_ids: list
    ):
        """Stream generation token by token."""
        model_input = input_tensor
        pos_offset = 0
        finished = False
        token_counts: dict = {}
        for tid in input_ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

        with torch.no_grad():
            for step in range(self.max_new_tokens):
                if finished:
                    break

                logits, _ = self.model.forward(
                    model_input,
                    kv_cache=None,
                    position_offset=pos_offset,
                )
                next_token_logits = logits[:, -1, :].float().clone()

                # Apply penalties
                if self.repetition_penalty != 1.0:
                    for tid, count in token_counts.items():
                        logit = next_token_logits[0, tid]
                        if logit > 0:
                            logit = logit / self.repetition_penalty
                        else:
                            logit = logit * self.repetition_penalty
                        next_token_logits[0, tid] = logit

                if self.temperature is not None and self.temperature > 0 and self.temperature != 1.0:
                    next_token_logits = next_token_logits / self.temperature

                if self.top_k is not None and self.top_k > 0:
                    k = min(self.top_k, next_token_logits.size(-1))
                    topk_vals, _ = torch.topk(next_token_logits, k, dim=-1)
                    cutoff = topk_vals[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < cutoff, float("-inf")
                    )

                if self.top_p is not None and 0.0 < self.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_mask = cumulative_probs > self.top_p
                    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                    sorted_mask[:, 0] = False
                    indices_to_remove = sorted_mask.scatter(
                        1, sorted_indices, sorted_mask
                    )
                    next_token_logits = next_token_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                tid = int(next_token[0].item())
                token_counts[tid] = token_counts.get(tid, 0) + 1

                yield self.tokenizer.decode([tid])

                model_input = next_token
                pos_offset += 1

                if tid == self.tokenizer.eos_token_id:
                    finished = True

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    def run_interactive(self) -> None:
        """Run interactive chat session."""
        print("\n" + "=" * 60)
        print("  KitAI Chat Interface")
        print("  Type 'exit' to quit, 'clear' to reset, 'help' for options")
        print("=" * 60 + "\n")

        if self.system_prompt:
            print(f"System: {self.system_prompt}\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break
                elif user_input.lower() == "clear":
                    self.clear_history()
                    print("Conversation history cleared.\n")
                    continue
                elif user_input.lower() == "help":
                    print("Commands: exit, clear, help")
                    print("Generation params: temperature, top_k, top_p")
                    continue

                response = self.generate_response(user_input)
                print(f"\nKitAI: {response}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chat with a KitAI model")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True,
        help="Path to tokenizer JSON file",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Model configuration file",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--system", type=str, default=None,
        help="System prompt",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Top-p (nucleus) sampling threshold",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream output token by token",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt to respond to (non-interactive)",
    )
    return parser.parse_args()


def main() -> None:
    """Main chat entry point."""
    args = parse_args()

    # Ensure stdout/stderr can handle Unicode (UTF-8) on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    # Load checkpoint first to get model config
    checkpoint = torch.load(args.model, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    if isinstance(checkpoint_config, dict) and "d_model" not in checkpoint_config:
        checkpoint_config = vars(checkpoint_config)

    # Load configuration from YAML (for training params), but use checkpoint config for model arch
    config_path = resolve_path(args.config)
    if config_path.exists():
        full_config = read_yaml(config_path)
    else:
        full_config = {}

    # Model config comes from checkpoint if available, otherwise from YAML
    if checkpoint_config:
        model_config_dict = checkpoint_config
        # Convert string-typed numeric values from checkpoint config
        for key in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff",
                     "max_seq_len", "vocab_size"):
            if key in model_config_dict and isinstance(model_config_dict[key], str):
                model_config_dict[key] = int(model_config_dict[key])
        for key in ("dropout", "norm_eps", "rope_theta", "init_std", "init_mean",
                     "learning_rate", "min_learning_rate", "weight_decay",
                     "label_smoothing", "gradient_clip_val"):
            if key in model_config_dict and isinstance(model_config_dict[key], str):
                model_config_dict[key] = float(model_config_dict[key])
    elif config_path.exists():
        model_config_dict = full_config.get("model", {})
    else:
        model_config_dict = {}

    model_config = ModelConfig(**model_config_dict) if model_config_dict else ModelConfig()

    # Load tokenizer
    tokenizer = BPETokenizer.load(args.tokenizer)

    # Initialize model
    model = KitAITransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Model loaded: {model.get_trainable_parameters():,} parameters")
    print(f"Device: {device}")

    # Create chat interface
    chat = ChatInterface(
        model=model,
        tokenizer=tokenizer,
        config={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_tokens,
            "system_prompt": args.system,
            "stream": args.stream,
        },
    )

    if args.interactive or args.prompt is None:
        chat.run_interactive()
    elif args.prompt:
        response = chat.generate_response(args.prompt)
        print(f"\nPrompt: {args.prompt}")
        print(f"\nResponse: {response}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
