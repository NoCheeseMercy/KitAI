"""
Tokenizer Utility Functions.

Provides helper functions for:
- Special token configurations
- Chat template formatting
- Token counting for dataset preparation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Callable


# Default special tokens
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_BOS_TOKEN = "<bos>"
DEFAULT_EOS_TOKEN = "<eos>"
DEFAULT_MASK_TOKEN = "<mask>"

# Dictionary of default special tokens for easy import
SPECIAL_TOKENS = {
    "pad": DEFAULT_PAD_TOKEN,
    "unk": DEFAULT_UNK_TOKEN,
    "bos": DEFAULT_BOS_TOKEN,
    "eos": DEFAULT_EOS_TOKEN,
    "mask": DEFAULT_MASK_TOKEN,
}


def get_special_token_map(
    pad_token: str = DEFAULT_PAD_TOKEN,
    unk_token: str = DEFAULT_UNK_TOKEN,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    mask_token: str = DEFAULT_MASK_TOKEN,
    additional_tokens: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Get a dictionary mapping special token names to their string values.

    Args:
        pad_token: Padding token.
        unk_token: Unknown token.
        bos_token: Beginning of sequence token.
        eos_token: End of sequence token.
        mask_token: Mask token for masked language modeling.
        additional_tokens: Additional custom special tokens.

    Returns:
        Dictionary of special tokens.
    """
    tokens = {
        "pad": pad_token,
        "unk": unk_token,
        "bos": bos_token,
        "eos": eos_token,
        "mask": mask_token,
    }
    if additional_tokens:
        tokens.update(additional_tokens)
    return tokens


def get_chat_template(
    template_name: str = "llama",
) -> Callable[[str, Optional[str], Optional[List[Dict[str, str]]]], str]:
    """
    Get a chat template function for formatting conversations.

    Supports standard chat formats:
    - 'llama': Llama 2/3 chat format
    - 'mistral': Mistral chat format
    - 'alpaca': Alpaca instruction format
    - 'chatml': OpenAI ChatML format

    Args:
        template_name: Name of the template to use.

    Returns:
        A function that formats a conversation into a text string.

    Raises:
        ValueError: If template_name is not supported.

    Examples:
        >>> template = get_chat_template("llama")
        >>> text = template("Hello!", system="Be helpful.")
        >>> "Hello!" in text
        True
    """
    templates = {
        "llama": _format_llama,
        "mistral": _format_mistral,
        "alpaca": _format_alpaca,
        "chatml": _format_chatml,
    }

    if template_name not in templates:
        raise ValueError(
            f"Unknown template: {template_name}. "
            f"Supported: {list(templates.keys())}"
        )

    return templates[template_name]


def _format_llama(
    user_message: str,
    system_message: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Format conversation in Llama 2/3 chat format.

    Format:
        <s>[INST] <<SYS>>
        {{ system_message }}
        <</SYS>

        {{ user_message }} [/INST] {{ response }} </s>

    Args:
        user_message: The user's message.
        system_message: Optional system prompt.
        history: Optional conversation history.

    Returns:
        Formatted conversation text.
    """
    parts: List[str] = []

    if system_message:
        parts.append(f"<<SYS>>\n{system_message}\n<</SYS>\n\n")

    if history:
        for turn in history:
            if turn.get("role") == "user":
                parts.append(f"[INST] {turn['content']} [/INST]")
            elif turn.get("role") == "assistant":
                parts.append(f" {turn['content']} </s>")

    parts.append(f"[INST] {user_message} [/INST]")

    return "<s>" + "".join(parts)


def _format_mistral(
    user_message: str,
    system_message: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Format conversation in Mistral chat format.

    Format:
        <s>[INST] {user_message} [/INST]{response}</s>

    Args:
        user_message: The user's message.
        system_message: Optional (ignored in Mistral baseline).
        history: Optional conversation history.

    Returns:
        Formatted conversation text.
    """
    parts: List[str] = []

    if history:
        for turn in history:
            if turn.get("role") == "user":
                parts.append(f"[INST] {turn['content']} [/INST]")
            elif turn.get("role") == "assistant":
                parts.append(f" {turn['content']}</s>")

    parts.append(f"[INST] {user_message} [/INST]")

    return "<s>" + "".join(parts)


def _format_alpaca(
    user_message: str,
    system_message: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Format conversation in Alpaca instruction format.

    Format:
        Below is an instruction that describes a task...
        ### Instruction:
        {instruction}
        ### Response:
        {response}

    Args:
        user_message: The instruction.
        system_message: Optional system message (prepended).
        history: Optional conversation history.

    Returns:
        Formatted instruction text.
    """
    parts: List[str] = [
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    ]

    if system_message:
        parts.append(f"### System:\n{system_message}\n\n")

    if history:
        for turn in history:
            if turn.get("role") == "user":
                parts.append(f"### Instruction:\n{turn['content']}\n\n")
            elif turn.get("role") == "assistant":
                parts.append(f"### Response:\n{turn['content']}\n\n")

    parts.append(f"### Instruction:\n{user_message}\n\n")
    parts.append("### Response:\n")

    return "".join(parts)


def _format_chatml(
    user_message: str,
    system_message: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Format conversation in OpenAI ChatML format.

    Format:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {user}<|im_end|>
        <|im_start|>assistant
        {response}<|im_end|>

    Args:
        user_message: The user's message.
        system_message: Optional system message.
        history: Optional conversation history.

    Returns:
        Formatted chat text.
    """
    parts: List[str] = []

    if system_message:
        parts.append(f"<|im_start|>system\n{system_message}<|im_end|>\n")

    if history:
        for turn in history:
            role = turn.get("role", "user")
            parts.append(f"<|im_start|>{role}\n{turn['content']}<|im_end|>\n")

    parts.append(f"<|im_start|>user\n{user_message}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Roughly estimate the number of tokens in a text.

    Uses a simple heuristic: average English text has ~4 characters per token.
    This is useful for dataset planning before training.

    Args:
        text: The input text.
        chars_per_token: Average characters per token (default: 4.0).

    Returns:
        Estimated token count.
    """
    return max(1, int(len(text) / chars_per_token))


def count_tokens_in_dataset(
    texts: List[str],
    chars_per_token: float = 4.0,
) -> Dict[str, int]:
    """
    Count estimated tokens in a dataset.

    Args:
        texts: List of texts in the dataset.
        chars_per_token: Average characters per token.

    Returns:
        Dictionary with 'total_tokens', 'total_chars', 'num_texts', 'avg_tokens'.
    """
    total_chars = sum(len(text) for text in texts)
    total_tokens = sum(estimate_tokens(text, chars_per_token) for text in texts)

    return {
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "num_texts": len(texts),
        "avg_tokens": total_tokens // max(1, len(texts)),
        "avg_chars": total_chars // max(1, len(texts)),
    }
