from __future__ import annotations

from pathlib import Path

path = Path(r"D:\EYAD\KitAI\tools\llama.cpp\conversion\base.py")
source = path.read_text(encoding="utf-8")

if 'Using gpt-2 GGUF pre-tokenizer metadata for the KitAI ByteLevel tokenizer.' in source:
    raise SystemExit("llama.cpp conversion patch is already present")

needle = '            raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")\n'
replacement = '''            logger.warning(
                "Using gpt-2 GGUF pre-tokenizer metadata for the KitAI ByteLevel tokenizer."
            )
            # KitAI was trained with tokenizers.ByteLevel(add_prefix_space=False),
            # whose splitting behavior is GPT-2-compatible. Its vocabulary is
            # custom, so the converter's vocabulary-dependent fingerprint is new.
            res = "gpt-2"
'''
if source.count(needle) != 1:
    raise RuntimeError("Could not identify the unrecognized-BPE failure branch")

path.write_text(source.replace(needle, replacement, 1), encoding="utf-8", newline="\n")
print("Patched llama.cpp converter for the KitAI ByteLevel tokenizer override.")
