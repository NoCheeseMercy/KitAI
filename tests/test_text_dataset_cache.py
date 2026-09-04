from __future__ import annotations

from pathlib import Path

import torch

from datasets.text_dataset import TextFileDataset


def test_text_dataset_builds_reusable_cache_and_samples_lazily(tmp_path: Path):
    source = tmp_path / "corpus.txt"
    source.write_text("alpha beta gamma delta epsilon zeta eta theta " * 20, encoding="utf-8")

    calls = {"count": 0}

    def encode(text: str):
        calls["count"] += 1
        return [ord(char) for char in text]

    dataset = TextFileDataset(
        source,
        block_size=16,
        tokenizer_encode_fn=encode,
        cache_key="test-tokenizer-v1",
        token_chunk_chars=32,
        token_batch_size=2,
    )

    assert dataset.token_count == len(source.read_text(encoding="utf-8"))
    assert dataset.cache_path.is_file()
    assert len(dataset) > 0
    inputs, labels = dataset[0]
    assert inputs.shape == (16,)
    assert labels.shape == (16,)
    assert torch.equal(inputs[1:], labels[:-1])

    call_count_after_initial_build = calls["count"]
    reused_dataset = TextFileDataset(
        source,
        block_size=16,
        tokenizer_encode_fn=encode,
        cache_key="test-tokenizer-v1",
        token_chunk_chars=32,
        token_batch_size=2,
    )

    assert calls["count"] == call_count_after_initial_build
    assert reused_dataset.token_count == dataset.token_count
    reused_inputs, reused_labels = reused_dataset[-1]
    assert reused_inputs.shape == (16,)
    assert reused_labels.shape == (16,)


def test_text_dataset_cache_invalidates_when_source_changes(tmp_path: Path):
    source = tmp_path / "corpus.txt"
    source.write_text("one two three four five " * 10, encoding="utf-8")

    def encode(text: str):
        return [ord(char) for char in text]

    first = TextFileDataset(source, block_size=8, tokenizer_encode_fn=encode, cache_key="test-tokenizer-v1")
    first_cache = first.cache_path
    source.write_text(source.read_text(encoding="utf-8") + "six seven eight ", encoding="utf-8")
    second = TextFileDataset(source, block_size=8, tokenizer_encode_fn=encode, cache_key="test-tokenizer-v1")

    assert second.cache_path != first_cache
    assert second.token_count > first.token_count
