"""
Gumbel-Softmax learnable masking for per-layer MLP width allocation.

Each layer gets a learnable logit vector that produces a soft mask in [0,1]
during training (via Gumbel-Sigmoid with temperature annealing) and a hard
mask in {0,1} at inference. A budget penalty regularizer keeps the total
active dimensions within a target budget.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class GumbelMaskLayer(nn.Module):
    """Per-layer learnable importance gate using Gumbel-Sigmoid."""

    def __init__(self, mlp_dim: int):
        super().__init__()
        self.mlp_dim = mlp_dim
        # Initialize logits to small positive values so masks start near 0.5
        self.logits = nn.Parameter(torch.zeros(mlp_dim))

    def forward(self, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Produce a mask in [0,1] (soft) or {0,1} (hard).

        Args:
            tau: Gumbel-Sigmoid temperature. High = soft, low = hard.
            hard: If True, use straight-through estimator for discrete masks.

        Returns:
            Tensor of shape (mlp_dim,) with values in [0,1] or {0,1}.
        """
        if self.training and not hard:
            # Gumbel-Sigmoid: sample Gumbel noise and apply sigmoid with temperature
            u = torch.rand_like(self.logits).clamp(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(u))
            y_soft = torch.sigmoid((self.logits + gumbel_noise) / tau)
        else:
            y_soft = torch.sigmoid(self.logits / tau)

        if hard:
            # Straight-through estimator: hard in forward, soft in backward
            y_hard = (y_soft > 0.5).float()
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def get_active_fraction(self) -> float:
        """Return fraction of dimensions that are 'on' (logits > 0)."""
        with torch.no_grad():
            return (self.logits > 0).float().mean().item()


class GumbelMaskManager(nn.Module):
    """Manages Gumbel masks for all layers in the model."""

    def __init__(self, n_layers: int, mlp_dim: int):
        super().__init__()
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim
        self.masks = nn.ModuleList([GumbelMaskLayer(mlp_dim) for _ in range(n_layers)])

    def get_mask(self, layer_idx: int, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """Get the mask for a specific layer."""
        return self.masks[layer_idx](tau=tau, hard=hard)

    def budget_loss(self, target: float) -> torch.Tensor:
        """
        Compute budget penalty: penalizes deviation from target active fraction.

        Args:
            target: Target fraction of total MLP dimensions to keep active (0, 1].

        Returns:
            Scalar penalty |mean_active_fraction - target|.
        """
        total_active = torch.tensor(0.0, device=self.masks[0].logits.device)
        for mask_layer in self.masks:
            total_active = total_active + torch.sigmoid(mask_layer.logits).sum()
        mean_fraction = total_active / (self.n_layers * self.mlp_dim)
        return torch.abs(mean_fraction - target)

    def get_layer_widths(self) -> Dict[int, int]:
        """
        Convert learned masks to discrete per-layer widths.

        Returns dict mapping layer_idx -> number of active dimensions.
        """
        widths = {}
        with torch.no_grad():
            for i, mask_layer in enumerate(self.masks):
                hard_mask = (torch.sigmoid(mask_layer.logits) > 0.5).float()
                widths[i] = int(hard_mask.sum().item())
        return widths

    def get_layer_factors(self, allowed_factors: Optional[List[int]] = None) -> Dict[int, int]:
        """
        Convert learned masks to discrete per-layer MatFormer factors.

        Maps each layer's active width to the nearest allowed factor.

        Args:
            allowed_factors: List of allowed factors (e.g., [1, 2, 4, 8]).
                            Defaults to powers of 2 up to mlp_dim.

        Returns:
            Dict mapping layer_idx -> chosen factor.
        """
        if allowed_factors is None:
            allowed_factors = [2**i for i in range(int(math.log2(self.mlp_dim)) + 1)]
        allowed_factors = sorted(allowed_factors)

        widths = self.get_layer_widths()
        factors = {}
        for layer_idx, width in widths.items():
            best_factor = allowed_factors[-1]
            best_diff = float("inf")
            for f in allowed_factors:
                target_width = self.mlp_dim // f
                diff = abs(width - target_width)
                if diff < best_diff:
                    best_diff = diff
                    best_factor = f
            factors[layer_idx] = best_factor
        return factors

    def log_summary(self) -> Dict[str, float]:
        """Return summary metrics for logging."""
        metrics = {}
        with torch.no_grad():
            total_active = 0.0
            for i, mask_layer in enumerate(self.masks):
                frac = torch.sigmoid(mask_layer.logits).mean().item()
                metrics[f"gumbel/layer_{i}_active_frac"] = frac
                total_active += frac
            metrics["gumbel/mean_active_frac"] = total_active / self.n_layers
        return metrics
