"""
Generation utilities for KitAI language models.

Provides speculative decoding interface and streaming generation support
for efficient inference on consumer GPUs.

Speculative decoding is a technique where a smaller draft model proposes
multiple tokens, and the main model verifies them in a single forward pass,
amortizing the cost of autoregressive decoding.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def speculative_decode(
    model: nn.Module,
    draft_model: Optional[nn.Module],
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    draft_steps: int = 5,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    stop_tokens: Optional[List[int]] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Perform speculative decoding with a draft model.

    The draft model proposes multiple tokens in a single forward pass,
    and the main model verifies them in parallel. Mismatched tokens are
    discarded and the main model continues autoregressively from the
    last verified token.

    Args:
        model: The main verification model.
        draft_model: The smaller draft model (must have same architecture).
            If None, falls back to standard autoregressive generation.
        input_ids: Input token IDs (batch, seq_len).
        max_new_tokens: Maximum number of tokens to generate.
        draft_steps: Number of tokens the draft model proposes per step.
        temperature: Sampling temperature.
        top_k: Top-k sampling threshold.
        top_p: Nucleus (top-p) sampling threshold.
        repetition_penalty: Multiplicative penalty for seen tokens.
        presence_penalty: Additive penalty for token presence.
        frequency_penalty: Additive penalty scaled by token frequency.
        stop_tokens: List of stop token IDs.
        eos_token_id: End-of-sequence token ID.

    Returns:
        Generated token IDs (batch, seq_len + generated_len).

    Examples:
        >>> # With a draft model for speculative decoding
        >>> generated = speculative_decode(
        ...     model, draft_model, input_ids,
        ...     max_new_tokens=50, draft_steps=4,
        ... )
    """
    if draft_model is None:
        return model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop_tokens=stop_tokens,
            eos_token_id=eos_token_id,
        )

    model.eval()
    draft_model.eval()
    batch_size = input_ids.shape[0]
    device = input_ids.device

    generated = input_ids.clone()
    token_counts: List[Dict[int, int]] = [{} for _ in range(batch_size)]
    for b in range(batch_size):
        for tid in generated[b].tolist():
            token_counts[b][tid] = token_counts[b].get(tid, 0) + 1

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        while not finished.all() and generated.shape[1] < input_ids.shape[1] + max_new_tokens:
            # Draft phase: draft model proposes multiple tokens
            draft_input = generated[:, -1:]
            draft_tokens = []

            for _ in range(draft_steps):
                if finished.all():
                    break
                draft_logits, _ = draft_model(draft_input)
                next_logits = draft_logits[:, -1, :].float()

                if temperature > 0 and temperature != 1.0:
                    next_logits = next_logits / temperature

                if top_k is not None and top_k > 0:
                    k = min(top_k, next_logits.size(-1))
                    topk_vals, _ = torch.topk(next_logits, k, dim=-1)
                    cutoff = topk_vals[:, -1].unsqueeze(-1)
                    next_logits = next_logits.masked_fill(
                        next_logits < cutoff, float("-inf")
                    )

                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                    sorted_mask[:, 0] = False
                    indices_to_remove = sorted_mask.scatter(
                        1, sorted_indices, sorted_mask
                    )
                    next_logits = next_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(next_token)
                draft_input = torch.cat([draft_input, next_token], dim=1)

            if not draft_tokens:
                break

            draft_output = torch.cat(draft_tokens, dim=1)

            # Verification phase: main model verifies draft tokens in parallel
            verify_input = torch.cat([generated, draft_output], dim=1)
            verify_logits, _ = model(verify_input)

            # Compare draft tokens with main model predictions
            verify_next = verify_logits[:, -draft_output.shape[1] - 1 : -1, :]
            draft_next = draft_output[:, :-1] if draft_output.shape[1] > 1 else draft_output

            # Find first mismatch
            match = (verify_next.argmax(dim=-1) == draft_next).all(dim=-1)

            # Accept all matching tokens from draft
            num_accepted = match.sum().item() if match.any() else 0

            if num_accepted > 0:
                accepted = draft_output[:, :num_accepted]
                generated = torch.cat([generated, accepted], dim=1)

                for b in range(batch_size):
                    if finished[b]:
                        continue
                    for t in accepted[b].tolist():
                        token_counts[b][t] = token_counts[b].get(t, 0) + 1

                # Check stop conditions
                for b in range(batch_size):
                    if finished[b]:
                        continue
                    last_tid = int(accepted[b, -1].item())
                    if eos_token_id is not None and last_tid == eos_token_id:
                        finished[b] = True
                    elif stop_tokens is not None and last_tid in stop_tokens:
                        finished[b] = True

            # If no tokens accepted or all finished, do standard autoregressive step
            if num_accepted == 0 or finished.all():
                last_logits = verify_logits[:, -1, :].float()

                if repetition_penalty != 1.0:
                    for b in range(batch_size):
                        if finished[b]:
                            continue
                        for tid, count in token_counts[b].items():
                            logit = last_logits[b, tid]
                            if logit > 0:
                                logit = logit / repetition_penalty
                            else:
                                logit = logit * repetition_penalty
                            last_logits[b, tid] = logit

                if presence_penalty != 0.0 or frequency_penalty != 0.0:
                    for b in range(batch_size):
                        if finished[b]:
                            continue
                        for tid, count in token_counts[b].items():
                            last_logits[b, tid] = (
                                last_logits[b, tid] - presence_penalty - frequency_penalty * count
                            )

                if temperature > 0 and temperature != 1.0:
                    last_logits = last_logits / temperature

                if top_k is not None and top_k > 0:
                    k = min(top_k, last_logits.size(-1))
                    topk_vals, _ = torch.topk(last_logits, k, dim=-1)
                    cutoff = topk_vals[:, -1].unsqueeze(-1)
                    last_logits = last_logits.masked_fill(
                        last_logits < cutoff, float("-inf")
                    )

                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        last_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_mask = cumulative_probs > top_p
                    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                    sorted_mask[:, 0] = False
                    indices_to_remove = sorted_mask.scatter(
                        1, sorted_indices, sorted_mask
                    )
                    last_logits = last_logits.masked_fill(
                        indices_to_remove, float("-inf")
                    )

                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                for b in range(batch_size):
                    if finished[b]:
                        continue
                    tid = int(next_token[b].item())
                    token_counts[b][tid] = token_counts[b].get(tid, 0) + 1

                    if eos_token_id is not None and tid == eos_token_id:
                        finished[b] = True
                    elif stop_tokens is not None and tid in stop_tokens:
                        finished[b] = True

    return generated


def stream_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    stop_tokens: Optional[List[int]] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Streaming generation that yields tokens as they are produced.

    This is useful for interactive applications where latency matters.
    The model generates one token at a time and yields it immediately.

    Args:
        model: The KitAI transformer model.
        input_ids: Input token IDs (batch, seq_len).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling threshold.
        top_p: Nucleus (top-p) sampling threshold.
        repetition_penalty: Multiplicative penalty for seen tokens.
        presence_penalty: Additive penalty for token presence.
        frequency_penalty: Additive penalty scaled by token frequency.
        stop_tokens: List of stop token IDs.
        eos_token_id: End-of-sequence token ID.

    Yields:
        Tuples of (token_id, accumulated_text) for each generated token.

    Examples:
        >>> for token_id, text in stream_generate(model, input_ids):
        ...     print(text, end="", flush=True)
    """
    model.eval()
    batch_size = input_ids.shape[0]
    device = input_ids.device

    generated = input_ids.clone()
    token_counts: List[Dict[int, int]] = [{} for _ in range(batch_size)]
    for b in range(batch_size):
        for tid in generated[b].tolist():
            token_counts[b][tid] = token_counts[b].get(tid, 0) + 1

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        for step in range(max_new_tokens):
            if finished.all():
                break

            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :].float().clone()

            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for tid, count in token_counts[b].items():
                        logit = next_token_logits[b, tid]
                        if logit > 0:
                            logit = logit / repetition_penalty
                        else:
                            logit = logit * repetition_penalty
                        next_token_logits[b, tid] = logit

            if presence_penalty != 0.0 or frequency_penalty != 0.0:
                for b in range(batch_size):
                    for tid, count in token_counts[b].items():
                        next_token_logits[b, tid] = (
                            next_token_logits[b, tid] - presence_penalty
                            - frequency_penalty * count
                        )

            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, next_token_logits.size(-1))
                topk_vals, _ = torch.topk(next_token_logits, k, dim=-1)
                cutoff = topk_vals[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < cutoff, float("-inf")
                )

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                indices_to_remove = sorted_mask.scatter(
                    1, sorted_indices, sorted_mask
                )
                next_token_logits = next_token_logits.masked_fill(
                    indices_to_remove, float("-inf")
                )

            if temperature == 0 or not temperature:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            for b in range(batch_size):
                if finished[b]:
                    continue
                tid = int(next_token[b].item())
                token_counts[b][tid] = token_counts[b].get(tid, 0) + 1

                if eos_token_id is not None and tid == eos_token_id:
                    finished[b] = True
                elif stop_tokens is not None and tid in stop_tokens:
                    finished[b] = True

            generated = torch.cat([generated, next_token], dim=1)

            yield next_token[0].item(), next_token


__all__ = [
    "speculative_decode",
    "stream_generate",
]