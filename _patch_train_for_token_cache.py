from __future__ import annotations

import re
from pathlib import Path

path = Path(r"D:\EYAD\KitAI\scripts\train.py")
source = path.read_text(encoding="utf-8")

if "tokenizer_encode_batch_fn=encode_batch_fn" in source:
    raise SystemExit("train.py already contains the batched-tokenization patch")

needle = "    encode_fn = None\n"
replacement = "    encode_fn = None\n    encode_batch_fn = None\n    tokenizer_cache_key = None\n"
if source.count(needle) != 1:
    raise RuntimeError("Could not identify the tokenizer initialization block")
source = source.replace(needle, replacement, 1)

needle = "    if tokenizer_path:\n        tok_path_str = str(tokenizer_path)\n"
replacement = """    if tokenizer_path:
        tok_path_str = str(tokenizer_path)
        tok_resolved_path = resolve_path(tok_path_str)
        tok_stat = tok_resolved_path.stat()
        tokenizer_cache_key = (
            f\"{tok_resolved_path.resolve()}:{tok_stat.st_size}:{tok_stat.st_mtime_ns}\"
        )
"""
if source.count(needle) != 1:
    raise RuntimeError("Could not identify the tokenizer path block")
source = source.replace(needle, replacement, 1)

needle = "            hf_tok = _HFTokenizer.from_file(str(resolve_path(tok_path_str)))\n            encode_fn = lambda text, _t=hf_tok: _t.encode(text).ids\n"
replacement = """            hf_tok = _HFTokenizer.from_file(str(tok_resolved_path))
            encode_fn = lambda text, _t=hf_tok: _t.encode(text).ids
            encode_batch_fn = (
                lambda texts, _t=hf_tok: [encoding.ids for encoding in _t.encode_batch(texts)]
            )
"""
if source.count(needle) != 1:
    raise RuntimeError("Could not identify the Hugging Face tokenizer block")
source = source.replace(needle, replacement, 1)

needle = "            tok = BPETokenizer.load(resolve_path(tok_path_str))\n"
replacement = "            tok = BPETokenizer.load(tok_resolved_path)\n"
if source.count(needle) != 1:
    raise RuntimeError("Could not identify the native tokenizer fallback block")
source = source.replace(needle, replacement, 1)

pattern = re.compile(
    r"(?P<prefix>TextFileDataset\(\s*(?P<path>[^,\n]+), block_size=block_size, tokenizer_encode_fn=encode_fn)(?P<suffix>\s*\))"
)


def add_tokenization_settings(match: re.Match[str]) -> str:
    return (
        f"{match.group('prefix')},\n"
        "                tokenizer_encode_batch_fn=encode_batch_fn,\n"
        "                cache_key=tokenizer_cache_key"
        f"{match.group('suffix')}"
    )

source, substitutions = pattern.subn(add_tokenization_settings, source)
if substitutions < 2:
    raise RuntimeError(f"Expected at least 2 TextFileDataset calls, patched {substitutions}")

path.write_text(source, encoding="utf-8", newline="\n")
print("Patched scripts/train.py: batched HF tokenization and tokenizer-specific cache keys enabled.")
