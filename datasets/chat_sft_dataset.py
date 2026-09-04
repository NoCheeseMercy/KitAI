"""Conversation-aware assistant-only SFT dataset for KitAI.

Each JSONL record must contain a ``messages`` list with ``role`` and ``content``
fields. A record becomes one fixed-length sample and is never concatenated with a
second conversation. The next-token label is active only when its target token is
part of an assistant span, including the assistant delimiter, its end-of-turn
marker, and the final EOS token after a final assistant turn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>", "<mask>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>")
ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


@dataclass(frozen=True)
class FormattedConversation:
    tokens: List[int]
    active_target_positions: List[bool]
    text: str
    record_id: str


class ChatSFTDataset(Dataset):
    """One conversation per padded, assistant-only-supervised sample."""

    def __init__(
        self,
        file_paths: Union[str, Path, Sequence[Union[str, Path]]],
        tokenizer: Any,
        block_size: int,
        drop_overlength: bool = True,
    ) -> None:
        if block_size < 2:
            raise ValueError("block_size must be at least 2")
        self.tokenizer = tokenizer
        self.block_size = int(block_size)
        self.drop_overlength = bool(drop_overlength)
        self.special_ids = self._require_special_ids(tokenizer)
        self.pad_id = self.special_ids["<pad>"]
        self.samples: List[Dict[str, torch.Tensor]] = []
        self.sample_metadata: List[Dict[str, Any]] = []
        self.stats: Dict[str, int] = {
            "records_seen": 0,
            "records_kept": 0,
            "records_dropped_overlength": 0,
            "assistant_target_tokens": 0,
            "padded_tokens": 0,
        }
        paths = [Path(file_paths)] if isinstance(file_paths, (str, Path)) else [Path(p) for p in file_paths]
        for path in paths:
            self._load_jsonl(path)
        if not self.samples:
            raise ValueError("No usable chat conversations were loaded")

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

    def format_messages(self, messages: Iterable[Dict[str, Any]], record_id: str) -> FormattedConversation:
        messages = list(messages)
        if not messages:
            raise ValueError("Conversation has no messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError("Conversation must end with an assistant message for assistant-only SFT")

        tokens: List[int] = [self.special_ids["<bos>"]]
        active_target_positions: List[bool] = [False]
        text_parts: List[str] = ["<bos>"]

        for index, message in enumerate(messages):
            role = str(message.get("role", ""))
            if role not in ROLE_TOKENS:
                raise ValueError(f"Unsupported role {role!r}")
            content = str(message.get("content", ""))
            assistant = role == "assistant"
            role_token = ROLE_TOKENS[role]
            tokens.append(self.special_ids[role_token])
            active_target_positions.append(assistant)
            content_ids = self._encode(content)
            tokens.extend(content_ids)
            active_target_positions.extend([assistant] * len(content_ids))
            tokens.append(self.special_ids["<|end|>"])
            active_target_positions.append(assistant)
            text_parts.append(role_token + content + "<|end|>")
            if index != len(messages) - 1:
                newline_ids = self._encode("\n")
                tokens.extend(newline_ids)
                active_target_positions.extend([False] * len(newline_ids))
                text_parts.append("\n")

        tokens.append(self.special_ids["<eos>"])
        active_target_positions.append(True)
        text_parts.append("<eos>")
        return FormattedConversation(tokens, active_target_positions, "".join(text_parts), record_id)

    def _append_formatted(self, formatted: FormattedConversation) -> None:
        # Because labels are shifted, a block_size input needs block_size + 1 source tokens.
        if len(formatted.tokens) > self.block_size + 1:
            self.stats["records_dropped_overlength"] += 1
            if self.drop_overlength:
                return
            formatted = FormattedConversation(
                tokens=formatted.tokens[: self.block_size + 1],
                active_target_positions=formatted.active_target_positions[: self.block_size + 1],
                text=formatted.text,
                record_id=formatted.record_id,
            )

        input_ids = formatted.tokens[:-1]
        labels = [token if is_active else -100 for token, is_active in zip(formatted.tokens[1:], formatted.active_target_positions[1:])]
        unpadded_length = len(input_ids)
        pad_amount = self.block_size - unpadded_length
        if pad_amount < 0:
            raise RuntimeError("Unexpected negative padding after length handling")
        input_ids = input_ids + [self.pad_id] * pad_amount
        labels = labels + [-100] * pad_amount
        attention_mask = [1] * unpadded_length + [0] * pad_amount

        self.samples.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        })
        self.sample_metadata.append({
            "record_id": formatted.record_id,
            "formatted_text": formatted.text,
            "source_token_count": len(formatted.tokens),
            "input_length": unpadded_length,
            "padding_tokens": pad_amount,
            "active_label_count": sum(label != -100 for label in labels),
        })
        self.stats["records_kept"] += 1
        self.stats["assistant_target_tokens"] += sum(label != -100 for label in labels)
        self.stats["padded_tokens"] += pad_amount

    def _load_jsonl(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                self.stats["records_seen"] += 1
                record = json.loads(line)
                formatted = self.format_messages(record.get("messages", []), str(record.get("id", f"{path.name}:{line_number}")))
                self._append_formatted(formatted)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]
