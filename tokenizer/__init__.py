"""
KitAI Tokenizer Package.

Implements a production-quality BPE (Byte Pair Encoding) tokenizer with:
- Training from text corpora
- Efficient encoding/decoding
- Special token support
- Vocabulary export/import
- Fast inference path
"""

from .bpe_tokenizer import BPETokenizer
from .trainer import TokenizerTrainer
from .utils import (
    get_special_token_map,
    get_chat_template,
    DEFAULT_PAD_TOKEN,
    DEFAULT_UNK_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_MASK_TOKEN,
)

__all__ = [
    "BPETokenizer",
    "TokenizerTrainer",
    "get_special_token_map",
    "get_chat_template",
    "DEFAULT_PAD_TOKEN",
    "DEFAULT_UNK_TOKEN",
    "DEFAULT_BOS_TOKEN",
    "DEFAULT_EOS_TOKEN",
    "DEFAULT_MASK_TOKEN",
]
