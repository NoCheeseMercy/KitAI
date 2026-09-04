"""Streaming assistant-only chat SFT dataset for the production KitAI corpus.

This implementation mirrors ``ChatSFTDataset`` serialization and label masking,
but reads JSONL records on demand instead of materializing all conversations and
tensors in memory. Every yielded sample comes from exactly one independent
conversation; conversations are never concatenated or packed together.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch.utils.data import IterableDataset, get_worker_info


SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<mask>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
)
ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
OVERLENGTH_STRATEGIES = ("drop", "final_pair")


@dataclass(frozen=True)
class FormattedConversation:
    """The token-level form of one conversation before shifting and padding."""

    tokens: List[int]
    active_target_positions: List[bool]
    text: str
    record_id: str


class StreamingChatSFTDataset(IterableDataset):
    """Stream assistant-only-supervised conversations from JSONL files.

    A fitting conversation is serialized exactly as the established KitAI chat
    format: ``<bos><|role|>content<|end|>`` with one encoded newline between
    successive messages and a final ``<eos>``. The target label is enabled only
    for assistant spans, including each assistant role delimiter, its end marker,
    and the final EOS after the required final assistant turn. System/user text,
    separators, and padding receive ``-100`` labels.

    The ``final_pair`` overlength strategy retains a bounded final user/assistant
    exchange when a complete multi-turn conversation exceeds the context window.
    It uses the same KitAI control tokens and assistant-only mask, never joins
    records, and prioritizes the final user context plus the beginning of the
    corresponding assistant response. ``drop`` provides strict parity with the
    older materialized loader.
    """

    def __init__(
        self,
        file_paths: Union[str, Path, Sequence[Union[str, Path]]],
        tokenizer: Any,
        block_size: int,
        drop_overlength: bool = True,
        shuffle_buffer_size: int = 1024,
        seed: int = 42,
        overlength_strategy: str = "drop",
        max_prompt_tokens: int = 96,
        record_modulus: Optional[int] = None,
        record_remainders: Optional[Iterable[int]] = None,
    ) -> None:
        super().__init__()
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        if overlength_strategy not in OVERLENGTH_STRATEGIES:
            raise ValueError(
                f"Unsupported overlength strategy {overlength_strategy!r}; "
                f"choose from {OVERLENGTH_STRATEGIES}"
            )
        if max_prompt_tokens < 0:
            raise ValueError("max_prompt_tokens must be non-negative")
        if record_modulus is not None and int(record_modulus) < 2:
            raise ValueError("record_modulus must be at least 2 when supplied")
        if record_remainders is not None and record_modulus is None:
            raise ValueError("record_remainders requires record_modulus")
        remainder_set: Optional[Set[int]] = None
        if record_remainders is not None:
            remainder_set = {int(value) for value in record_remainders}
            if not remainder_set:
                raise ValueError("record_remainders must not be empty when supplied")
            invalid = [value for value in remainder_set if value < 0 or value >= int(record_modulus)]
            if invalid:
                raise ValueError("record_remainders contains values outside record_modulus")
        self.file_paths = (
            [Path(file_paths)]
            if isinstance(file_paths, (str, Path))
            else [Path(path) for path in file_paths]
        )
        if not self.file_paths:
            raise ValueError("At least one chat JSONL file is required")
        for path in self.file_paths:
            if not path.is_file():
                raise FileNotFoundError(path)

        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.drop_overlength = bool(drop_overlength)
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.seed = int(seed)
        self.overlength_strategy = overlength_strategy
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.record_modulus = int(record_modulus) if record_modulus is not None else None
        self.record_remainders = remainder_set
        self._iteration_index = 0
        self.special_ids = self._require_special_ids(tokenizer)
        self.pad_id = self.special_ids["<pad>"]
        self.stats: Dict[str, int] = {
            "records_seen": 0,
            "records_kept": 0,
            "records_dropped_overlength": 0,
            "records_compressed_final_pair": 0,
            "assistant_target_tokens": 0,
            "padded_tokens": 0,
        }

    @staticmethod
    def _require_special_ids(tokenizer: Any) -> Dict[str, int]:
        ids: Dict[str, int] = {}
        for token in SPECIAL_TOKENS:
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"Tokenizer is missing required special token: {token}")
            ids[token] = int(token_id)
        return ids

    def _encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text).ids)

    def format_messages(
        self, messages: Iterable[Dict[str, Any]], record_id: str
    ) -> FormattedConversation:
        """Format one full conversation with existing ``ChatSFTDataset`` semantics."""
        messages = list(messages)
        if not messages:
            raise ValueError("Conversation has no messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError(
                "Conversation must end with an assistant message for assistant-only SFT"
            )

        tokens: List[int] = [self.special_ids["<bos>"]]
        active_target_positions: List[bool] = [False]
        text_parts: List[str] = ["<bos>"]

        for index, message in enumerate(messages):
            role = str(message.get("role", ""))
            if role not in ROLE_TOKENS:
                raise ValueError(f"Unsupported role {role!r}")
            content = str(message.get("content", ""))
            is_assistant = role == "assistant"
            role_token = ROLE_TOKENS[role]

            tokens.append(self.special_ids[role_token])
            active_target_positions.append(is_assistant)
            content_ids = self._encode(content)
            tokens.extend(content_ids)
            active_target_positions.extend([is_assistant] * len(content_ids))
            tokens.append(self.special_ids["<|end|>"])
            active_target_positions.append(is_assistant)
            text_parts.append(role_token + content + "<|end|>")

            if index != len(messages) - 1:
                newline_ids = self._encode("\n")
                tokens.extend(newline_ids)
                active_target_positions.extend([False] * len(newline_ids))
                text_parts.append("\n")

        tokens.append(self.special_ids["<eos>"])
        active_target_positions.append(True)
        text_parts.append("<eos>")
        return FormattedConversation(
            tokens=tokens,
            active_target_positions=active_target_positions,
            text="".join(text_parts),
            record_id=record_id,
        )

    def format_final_pair(
        self, messages: Iterable[Dict[str, Any]], record_id: str
    ) -> FormattedConversation:
        """Create a bounded final user/assistant training example from one record."""
        messages = list(messages)
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Fallback formatting requires a final assistant message")

        final_assistant_content = str(messages[-1].get("content", ""))
        final_user_content = ""
        for message in reversed(messages[:-1]):
            if message.get("role") == "user":
                final_user_content = str(message.get("content", ""))
                break

        bos_id = self.special_ids["<bos>"]
        eos_id = self.special_ids["<eos>"]
        end_id = self.special_ids["<|end|>"]
        assistant_id = self.special_ids["<|assistant|>"]
        assistant_content_ids = self._encode(final_assistant_content)
        max_source_tokens = self.block_size + 1

        tokens: List[int] = [bos_id]
        active_target_positions: List[bool] = [False]
        text_parts: List[str] = ["<bos>"]

        if final_user_content:
            user_id = self.special_ids["<|user|>"]
            newline_ids = self._encode("\n")
            # Reserve the assistant delimiter, end marker, EOS, and at least one
            # slot for assistant content whenever the sequence budget permits.
            fixed_after_user = len(newline_ids) + 3
            prompt_budget = max(0, max_source_tokens - len(tokens) - 2 - fixed_after_user - 1)
            prompt_budget = min(self.max_prompt_tokens, prompt_budget)
            user_content_ids = self._encode(final_user_content)[-prompt_budget:] if prompt_budget else []
            tokens.extend([user_id] + user_content_ids + [end_id])
            active_target_positions.extend([False] * (len(user_content_ids) + 2))
            tokens.extend(newline_ids)
            active_target_positions.extend([False] * len(newline_ids))
            text_parts.append("<|user|><truncated-final-user><|end|>\n")

        assistant_overhead = 3  # role delimiter, end marker, final EOS
        assistant_budget = max_source_tokens - len(tokens) - assistant_overhead
        if assistant_budget < 0:
            raise RuntimeError("Final-pair overhead exceeds the configured context window")
        assistant_content_ids = assistant_content_ids[:assistant_budget]
        tokens.extend([assistant_id] + assistant_content_ids + [end_id, eos_id])
        active_target_positions.extend([True] * (len(assistant_content_ids) + 3))
        text_parts.append("<|assistant|><truncated-final-assistant><|end|><eos>")

        if len(tokens) > max_source_tokens:
            raise RuntimeError("Final-pair fallback exceeded the configured context window")
        return FormattedConversation(
            tokens=tokens,
            active_target_positions=active_target_positions,
            text="".join(text_parts),
            record_id=record_id,
        )

    def _to_sample(
        self, formatted: FormattedConversation
    ) -> Dict[str, torch.Tensor]:
        if len(formatted.tokens) > self.block_size + 1:
            raise RuntimeError("Conversation exceeds the configured block size")

        input_ids = formatted.tokens[:-1]
        labels = [
            token if is_active else -100
            for token, is_active in zip(
                formatted.tokens[1:], formatted.active_target_positions[1:]
            )
        ]
        unpadded_length = len(input_ids)
        padding = self.block_size - unpadded_length
        if padding < 0:
            raise RuntimeError("Conversation exceeds the configured block size")

        labels += [-100] * padding
        self.stats["records_kept"] += 1
        self.stats["assistant_target_tokens"] += sum(label != -100 for label in labels)
        self.stats["padded_tokens"] += padding
        return {
            "input_ids": torch.tensor(
                input_ids + [self.pad_id] * padding, dtype=torch.long
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * unpadded_length + [0] * padding, dtype=torch.long
            ),
        }

    def _iter_records(
        self, paths: Sequence[Path], worker_id: int, worker_count: int
    ) -> Iterator[Tuple[Dict[str, Any], str]]:
        """Yield a disjoint, record-level shard for each data-loader worker."""
        record_index = 0
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    global_record_index = record_index
                    assigned_worker = global_record_index % worker_count
                    record_index += 1
                    if assigned_worker != worker_id:
                        continue
                    if (
                        self.record_modulus is not None
                        and self.record_remainders is not None
                        and global_record_index % self.record_modulus not in self.record_remainders
                    ):
                        continue
                    self.stats["records_seen"] += 1
                    record = json.loads(line)
                    record_id = str(record.get("id", f"{path.name}:{line_number}"))
                    yield record, record_id

    @staticmethod
    def _iter_shuffled(
        records: Iterable[Tuple[Dict[str, Any], str]],
        rng: random.Random,
        buffer_size: int,
    ) -> Iterator[Tuple[Dict[str, Any], str]]:
        buffer: List[Tuple[Dict[str, Any], str]] = []
        for record in records:
            buffer.append(record)
            if len(buffer) >= buffer_size:
                yield buffer.pop(rng.randrange(len(buffer)))
        while buffer:
            yield buffer.pop(rng.randrange(len(buffer)))

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        paths = list(self.file_paths)
        iteration_index = self._iteration_index
        self._iteration_index += 1
        rng = random.Random(self.seed + worker_id + (iteration_index * 100003))
        rng.shuffle(paths)

        records = self._iter_records(paths, worker_id, worker_count)
        for record, record_id in self._iter_shuffled(
            records, rng, self.shuffle_buffer_size
        ):
            messages = record.get("messages", [])
            formatted = self.format_messages(messages, record_id)
            if len(formatted.tokens) > self.block_size + 1:
                if self.overlength_strategy == "drop":
                    self.stats["records_dropped_overlength"] += 1
                    continue
                formatted = self.format_final_pair(messages, record_id)
                self.stats["records_compressed_final_pair"] += 1
            yield self._to_sample(formatted)
