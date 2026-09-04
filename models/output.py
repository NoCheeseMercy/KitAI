"""
Output Head and Loss Computation.

Implements the language model head that projects hidden states to
vocabulary logits, and the cross-entropy loss computation with
optional label smoothing support.

Supports weight tying between the input embedding and output projection
for parameter efficiency (as used in GPT, Llama, etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelConfig


class OutputHead(nn.Module):
    """
    Language Model Output Head.

    Projects the final hidden state to vocabulary logits.
    Supports weight tying with the input token embedding layer.

    Args:
        config: Model configuration.

    Shape:
        - Input: (batch, seq_len, d_model)
        - Output: (batch, seq_len, vocab_size)

    Examples:
        >>> config = ModelConfig.tiny()
        >>> head = OutputHead(config)
        >>> x = torch.randn(2, 16, 256)
        >>> logits = head(x)
        >>> logits.shape
        torch.Size([2, 16, 32000])
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        # Output projection (LM head)
        self.weight = nn.Parameter(
            torch.empty(self.vocab_size, self.d_model)
        )

        # Optional bias (not used in modern architectures)
        self.bias = nn.Parameter(torch.zeros(self.vocab_size)) if config.bias else None

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize output projection weights."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        """
        Tie the output head weights with input embedding weights.

        This shares the weight tensor, so changes to one affect the other.
        Standard practice for parameter-efficient transformer LMs.

        Args:
            embedding_weight: The weight tensor from TokenEmbedding.
        """
        self.weight = embedding_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.

        Args:
            x: Hidden states (batch, seq_len, d_model).

        Returns:
            Logits tensor (batch, seq_len, vocab_size).
        """
        logits = F.linear(x, self.weight, self.bias)
        return logits

    def extra_repr(self) -> str:
        """String representation."""
        return f"vocab_size={self.vocab_size}, d_model={self.d_model}, bias={self.bias is not None}"


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cross-entropy loss with optional label smoothing.

    Handles the shift between logits and targets for autoregressive
    language modeling (predict next token).

    Args:
        logits: Raw logits from the model (batch, seq_len, vocab_size).
        targets: Target token IDs (batch, seq_len).
        ignore_index: Index to ignore in loss computation (default: -100 for padding).
        label_smoothing: Label smoothing epsilon (0.0 disables).

    Returns:
        Tuple of (loss scalar, perplexity).

    Examples:
        >>> logits = torch.randn(2, 16, 32000)
        >>> targets = torch.randint(0, 32000, (2, 16))
        >>> loss, ppl = compute_loss(logits, targets)
        >>> loss.shape
        torch.Size([])
    """
    # Reshape logits and targets for cross-entropy
    # (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
    batch, seq_len, vocab_size = logits.shape
    logits_flat = logits.contiguous().view(-1, vocab_size)
    targets_flat = targets.contiguous().view(-1)

    # Compute cross-entropy loss
    if label_smoothing > 0.0:
        # Use label smoothing
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
    else:
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=ignore_index,
        )

    # Perplexity = exp(mean NLL over non-ignored tokens). For plain
    # cross-entropy the mean loss IS the mean NLL, so avoid a second
    # full softmax over the vocabulary (significant at large vocab sizes).
    with torch.no_grad():
        if label_smoothing > 0.0:
            # Label smoothing changes the objective; recompute exact NLL.
            nll = F.cross_entropy(
                logits_flat, targets_flat,
                ignore_index=ignore_index, reduction="none",
            )
            mask = targets_flat != ignore_index
            perplexity = (
                torch.exp(nll[mask].mean())
                if mask.any()
                else torch.tensor(float("inf"), device=logits.device)
            )
        else:
            perplexity = torch.exp(loss.detach())

    return loss, perplexity

