"""
RMS Normalization (RMSNorm).

Implements Root Mean Square Layer Normalization as used in Llama, Mistral,
and other modern transformer architectures. RMSNorm is computationally
lighter than LayerNorm while providing comparable performance.

Reference: https://arxiv.org/abs/1910.07467
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes the input by its root mean square, then scales by a learned
    weight parameter. This is simpler and faster than LayerNorm since it
    doesn't compute the mean or the full variance.

    Args:
        d_model: Hidden dimension size.
        eps: Small constant for numerical stability.
        bias: Whether to include a bias term (not used in most implementations).
        dtype: Torch dtype for the weight parameter.

    Shape:
        - Input: (..., d_model)
        - Output: (..., d_model), same shape as input.

    Examples:
        >>> norm = RMSNorm(768)
        >>> x = torch.randn(2, 16, 768)
        >>> y = norm(x)
        >>> y.shape
        torch.Size([2, 16, 768])
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_model, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RMS normalization.

        rms(x) = sqrt(mean(x^2) + eps)
        output = x / rms(x)
        """
        # Compute x^2, then mean over last dimension
        # Keep the dimension for broadcasting
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Normalized tensor of same shape.
        """
        # Convert to float for numerical stability, then back to original dtype
        output = self._norm(x.float()).type_as(x)
        output = output * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        """String representation for print(model)."""
        return f"d_model={self.weight.shape[0]}, eps={self.eps}, bias={self.bias is not None}"

