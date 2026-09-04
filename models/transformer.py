"""
KitAI Transformer Model.

Assembles all components into the complete decoder-only transformer:
1. Token Embedding + RoPE
2. N Transformer Blocks (Attention + FFN)
3. Output Head (LM head)
4. Weight tying (optional)
5. Causal mask generation
6. Gradient checkpointing

This is the main model class used for both training and inference.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .config import ModelConfig
from .normalization import RMSNorm
from .embedding import TokenEmbedding
from .layers import TransformerBlock
from .output import OutputHead, compute_loss
from .kv_cache import KVCache


class KitAITransformer(nn.Module):
    """
    KitAI Decoder-Only Transformer.

    The main model class that assembles all components into a complete
    autoregressive language model. Supports training with gradient
    checkpointing and efficient inference with KV cache.

    Args:
        config: Model configuration.

    Examples:
        >>> config = ModelConfig.tiny()
        >>> model = KitAITransformer(config)
        >>> x = torch.randint(0, 32000, (2, 16))
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([2, 16, 32000])
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embeddings
        self.embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(i, config)
            for i in range(config.n_layers)
        ])

        # Final normalization (Pre-Norm architecture)
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Output head
        self.output_head = OutputHead(config)

        # Weight tying: share embedding and output weights
        if config.weight_tying:
            self.output_head.tie_weights(self.embedding.weight)

        # Initialize all parameters
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all model parameters."""
        self.apply(self._init_module_weights)

    def _init_module_weights(self, module: nn.Module) -> None:
        """Initialize weights for a specific module."""
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create a causal attention mask.

        The mask prevents positions from attending to future positions.
        Uses -inf for masked positions and 0 for unmasked.

        Args:
            seq_len: Length of the sequence.
            device: Device for the mask tensor.
            dtype: Data type for the mask tensor.

        Returns:
            Causal mask of shape (1, 1, seq_len, seq_len).
        """
        mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        # Upper triangular mask (causal)
        mask = torch.triu(mask, diagonal=1)
        # Add batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[KVCache]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[List[KVCache]]]:
        """
        Forward pass through the full transformer model.

        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Optional attention mask (causal).
            kv_cache: Optional list of KV caches for each layer.
            position_offset: Position offset for generation.

        Returns:
            Tuple of (logits (batch, seq_len, vocab_size), updated KV cache list).
        """
        batch, seq_len = input_ids.shape

        # Create the causal mask when training without a KV cache. A caller may
        # supply a 2-D 1/0 padding mask; convert it into an additive mask that
        # preserves causality while preventing attention to right-padding.
        if attention_mask is None and kv_cache is None:
            attention_mask = self._create_causal_mask(
                seq_len, input_ids.device, dtype=self.embedding.weight.dtype
            )
        elif attention_mask is not None and attention_mask.dim() == 2 and kv_cache is None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError(
                    "2-D attention_mask must have shape (batch, seq_len); "
                    f"got {tuple(attention_mask.shape)} for {(batch, seq_len)}"
                )
            causal_mask = self._create_causal_mask(
                seq_len, input_ids.device, dtype=self.embedding.weight.dtype
            )
            key_padding_mask = torch.zeros(
                (batch, 1, 1, seq_len),
                device=input_ids.device,
                dtype=self.embedding.weight.dtype,
            )
            key_padding_mask = key_padding_mask.masked_fill(
                ~attention_mask.to(device=input_ids.device, dtype=torch.bool)[:, None, None, :],
                float("-inf"),
            )
            attention_mask = causal_mask + key_padding_mask

        # Token embeddings
        hidden_states = self.embedding(input_ids)

        # Pass through transformer blocks
        new_kv_cache: List[KVCache] = [] if kv_cache is not None else []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            hidden_states, updated_cache = layer(
                hidden_states,
                mask=attention_mask,
                kv_cache=layer_cache,
                position_offset=position_offset,
            )
            if kv_cache is not None:
                new_kv_cache.append(updated_cache)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Project to vocabulary
        logits = self.output_head(hidden_states)

        kv_cache_result = new_kv_cache if kv_cache is not None else None
        return logits, kv_cache_result

    def generate(
        self,
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
        do_sample: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively with modern sampling controls.

        Args:
            input_ids: Input token IDs (batch, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. 0 (or do_sample=False) => greedy.
            top_k: Top-k sampling threshold.
            top_p: Nucleus (top-p) sampling threshold.
            repetition_penalty: Multiplicative penalty for seen tokens (>1.0).
            presence_penalty: Additive penalty once a token has appeared.
            frequency_penalty: Additive penalty scaled by token frequency.
            stop_tokens: List of stop token IDs.
            eos_token_id: End-of-sequence token ID.
            do_sample: Override sampling. None => sample when temperature > 0.

        Returns:
            Generated token IDs (batch, prompt_len + generated_len).
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Clamp generation so prompt + new tokens never exceed the context
        # window. Beyond max_seq_len the KV cache starts sliding, which would
        # desynchronize RoPE positions and corrupt attention.
        max_ctx = self.config.max_seq_len
        prompt_len = input_ids.shape[1]
        if prompt_len > max_ctx:
            input_ids = input_ids[:, -max_ctx:]
            prompt_len = max_ctx
        max_new_tokens = max(0, min(max_new_tokens, max_ctx - prompt_len))
        if max_new_tokens == 0:
            return input_ids

        if do_sample is None:
            do_sample = temperature is not None and temperature > 0

        # Initialize KV cache (match model weight dtype)
        model_dtype = self.embedding.weight.dtype
        kv_cache = [
            KVCache(
                max_batch_size=batch_size,
                max_seq_len=self.config.max_seq_len,
                n_kv_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                dtype=model_dtype,
                device=device,
            )
            for _ in range(self.config.n_layers)
        ]

        generated = input_ids.clone()
        # Track full history (prompt + generated) for penalties
        token_counts: List[Dict[int, int]] = [
            {} for _ in range(batch_size)
        ]
        for b in range(batch_size):
            for tid in generated[b].tolist():
                token_counts[b][tid] = token_counts[b].get(tid, 0) + 1

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            if finished.all():
                break

            if step == 0:
                model_input = input_ids
                pos_offset = 0
            else:
                model_input = generated[:, -1:]
                pos_offset = generated.shape[1] - 1

            with torch.no_grad():
                logits, kv_cache = self.forward(
                    model_input,
                    kv_cache=kv_cache,
                    position_offset=pos_offset,
                )

            next_token_logits = logits[:, -1, :].float().clone()

            # Presence / frequency / repetition penalties
            if (
                repetition_penalty != 1.0
                or presence_penalty != 0.0
                or frequency_penalty != 0.0
            ):
                for b in range(batch_size):
                    for tid, count in token_counts[b].items():
                        logit = next_token_logits[b, tid]
                        # HF-style repetition penalty
                        if repetition_penalty != 1.0:
                            if logit > 0:
                                logit = logit / repetition_penalty
                            else:
                                logit = logit * repetition_penalty
                        # OpenAI-style additive penalties
                        logit = logit - presence_penalty - frequency_penalty * count
                        next_token_logits[b, tid] = logit

            # Temperature
            if do_sample and temperature is not None and temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-k
            if top_k is not None and top_k > 0:
                k = min(top_k, next_token_logits.size(-1))
                topk_vals, _ = torch.topk(next_token_logits, k, dim=-1)
                cutoff = topk_vals[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < cutoff, float("-inf")
                )

            # Top-p (nucleus)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_mask = cumulative_probs > top_p
                # Keep at least the first token
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                indices_to_remove = sorted_mask.scatter(
                    1, sorted_indices, sorted_mask
                )
                next_token_logits = next_token_logits.masked_fill(
                    indices_to_remove, float("-inf")
                )

            # Sample or greedy
            if not do_sample or temperature == 0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Force finished sequences to keep emitting pad-like last token
            if finished.any():
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    generated[:, -1:],
                    next_token,
                )

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

    def get_trainable_parameters(self) -> int:
        """
        Get the number of trainable parameters.

        Returns:
            Parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_with_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        label_smoothing: float = 0.0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass and compute loss with optional padding masking."""
        logits, _ = self.forward(input_ids, attention_mask=attention_mask)
        loss, perplexity = compute_loss(logits, labels, label_smoothing=label_smoothing)
        return loss, perplexity

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text for a batch of inputs.

        Args:
            input_ids: Batch of input token IDs.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional generation parameters.

        Returns:
            Generated token IDs for all batch items.
        """
        return self.generate(input_ids, max_new_tokens, **kwargs)

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"KitAI(d_model={self.config.d_model}, "
            f"n_layers={self.config.n_layers}, "
            f"n_heads={self.config.n_heads}, "
            f"n_kv_heads={self.config.n_kv_heads}, "
            f"params={self.get_trainable_parameters() / 1e6:.2f}M)"
        )
