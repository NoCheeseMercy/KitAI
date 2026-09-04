"""Tests for the production-safe streaming assistant-only chat loader."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.chat_sft_dataset import ChatSFTDataset
from datasets.streaming_chat_sft_dataset import StreamingChatSFTDataset


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


@dataclass
class _Encoding:
    ids: list[int]


class _CharacterTokenizer:
    """Tiny deterministic tokenizer sufficient to compare dataset semantics."""

    def __init__(self) -> None:
        self.vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .?!\n":
            self.vocab.setdefault(char, len(self.vocab))

    def token_to_id(self, token: str):
        return self.vocab.get(token)

    def encode(self, text: str) -> _Encoding:
        return _Encoding([self.vocab.get(char, self.vocab["<unk>"]) for char in text])


def _write_records(path: Path) -> None:
    records = [
        {
            "id": "with-system",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello!"},
            ],
        },
        {
            "id": "multi-turn",
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4."},
                {"role": "user", "content": "Thanks"},
                {"role": "assistant", "content": "You are welcome."},
            ],
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_streaming_dataset_matches_materialized_chat_semantics(tmp_path: Path) -> None:
    data_path = tmp_path / "fixture.jsonl"
    _write_records(data_path)
    tokenizer = _CharacterTokenizer()

    expected = ChatSFTDataset(data_path, tokenizer, block_size=128)
    streamed = StreamingChatSFTDataset(
        data_path, tokenizer, block_size=128, shuffle_buffer_size=1
    )
    actual_samples = list(streamed)

    assert len(actual_samples) == len(expected) == 2
    for expected_sample, actual_sample in zip(expected.samples, actual_samples):
        assert torch.equal(actual_sample["input_ids"], expected_sample["input_ids"])
        assert torch.equal(actual_sample["labels"], expected_sample["labels"])
        assert torch.equal(actual_sample["attention_mask"], expected_sample["attention_mask"])

    first = actual_samples[0]
    user_id = tokenizer.token_to_id("<|user|>")
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    user_position = (first["input_ids"] == user_id).nonzero(as_tuple=False).item()
    assistant_position = (first["input_ids"] == assistant_id).nonzero(as_tuple=False).item()
    # Labels are shifted: system/user targets are ignored; assistant delimiter is learned.
    assert first["labels"][user_position - 1].item() == -100
    assert first["labels"][assistant_position - 1].item() == assistant_id
    assert torch.all(first["labels"][first["attention_mask"] == 0] == -100)


def test_streaming_dataset_drops_overlength_conversations(tmp_path: Path) -> None:
    data_path = tmp_path / "overlong.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "x" * 100},
                    {"role": "assistant", "content": "y" * 100},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = StreamingChatSFTDataset(
        data_path, _CharacterTokenizer(), block_size=32, shuffle_buffer_size=1
    )

    assert list(dataset) == []
    assert dataset.stats["records_seen"] == 1
    assert dataset.stats["records_dropped_overlength"] == 1


def test_final_pair_strategy_retains_overlength_chat_with_assistant_masking(tmp_path: Path) -> None:
    data_path = tmp_path / "overlong-final-pair.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "z" * 80},
                    {"role": "user", "content": "u" * 100},
                    {"role": "assistant", "content": "a" * 100},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tokenizer = _CharacterTokenizer()
    dataset = StreamingChatSFTDataset(
        data_path,
        tokenizer,
        block_size=32,
        shuffle_buffer_size=1,
        overlength_strategy="final_pair",
        max_prompt_tokens=10,
    )

    samples = list(dataset)
    assert len(samples) == 1
    sample = samples[0]
    assert sample["input_ids"].shape == sample["labels"].shape == sample["attention_mask"].shape == (32,)
    assert dataset.stats["records_compressed_final_pair"] == 1
    assert dataset.stats["records_dropped_overlength"] == 0
    assert torch.any(sample["labels"] != -100)
    assert torch.all(sample["labels"][sample["attention_mask"] == 0] == -100)

    assistant_id = tokenizer.token_to_id("<|assistant|>")
    assistant_position = (sample["input_ids"] == assistant_id).nonzero(as_tuple=False).item()
    assert sample["labels"][assistant_position - 1].item() == assistant_id


def test_record_level_holdout_streams_are_disjoint_and_complete(tmp_path: Path) -> None:
    data_path = tmp_path / "holdout.jsonl"
    records = [
        {
            "id": f"conversation-{index}",
            "messages": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ],
        }
        for index in range(6)
    ]
    data_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    tokenizer = _CharacterTokenizer()

    train_dataset = StreamingChatSFTDataset(
        data_path,
        tokenizer,
        block_size=64,
        shuffle_buffer_size=1,
        record_modulus=3,
        record_remainders={1, 2},
    )
    validation_dataset = StreamingChatSFTDataset(
        data_path,
        tokenizer,
        block_size=64,
        shuffle_buffer_size=1,
        record_modulus=3,
        record_remainders={0},
    )

    train_samples = list(train_dataset)
    validation_samples = list(validation_dataset)
    assert len(train_samples) == 4
    assert len(validation_samples) == 2
    assert train_dataset.stats["records_seen"] + validation_dataset.stats["records_seen"] == len(records)
