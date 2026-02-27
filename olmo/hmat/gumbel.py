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

    def __init__(
        self,
        mlp_dim: int,
        init_scale: float = 2.2,
        learnable: bool = True,
        init_mode: str = "linspace",
        init_value: float = 1.5,
    ):
        super().__init__()
        self.mlp_dim = mlp_dim
        # Build initial logit vector based on init_mode
        if init_mode == "zeros":
            init_logits = torch.zeros(mlp_dim)
        elif init_mode == "normal":
            init_logits = torch.randn(mlp_dim) * init_value
        elif init_mode == "constant":
            init_logits = torch.full((mlp_dim,), init_value)
        else:  # "linspace" (default)
            init_logits = torch.linspace(init_scale, -init_scale, mlp_dim)
        if learnable:
            self.logits = nn.Parameter(init_logits)
        else:
            # Register as buffer — not a parameter, not in optimizer
            self.register_buffer("logits", init_logits)

    def set_logits(self, new_logits: torch.Tensor):
        """Update logits from external source (e.g., Fisher EMA)."""
        with torch.no_grad():
            self.logits.copy_(new_logits)

    def forward(self, tau: float = 1.0, hard: bool = False, k: Optional[int] = None) -> torch.Tensor:
        """
        Produce a mask in [0,1] (soft) or {0,1} (hard).

        Supports two modes depending on whether k is provided:
        - k=None: Vanilla Gumbel-Sigmoid (per-element sigmoid, used with budget penalty).
        - k < mlp_dim: Gumbel-Top-K hybrid (top-k threshold on noisy logits,
          factor-dependent width, no budget penalty needed).
        - k >= mlp_dim: All-ones (no masking at full model width).

        Args:
            tau: Temperature. High = soft, low = hard.
            hard: If True, use straight-through estimator for discrete masks.
            k: Target sub-model width for Gumbel-Top-K. None for vanilla Gumbel-Sigmoid.

        Returns:
            Tensor of shape (mlp_dim,) with values in [0,1] or {0,1}.
        """
        if k is not None and k >= self.mlp_dim:
            # Full model — return all-ones, no mask overhead
            return torch.ones_like(self.logits)

        # Add Gumbel noise during training (shared by both modes)
        if self.training and not hard:
            u = torch.rand_like(self.logits).clamp(1e-6, 1.0 - 1e-6)
            gumbel_noise = -torch.log(-torch.log(u))
            noisy_logits = self.logits + gumbel_noise
        else:
            noisy_logits = self.logits

        if k is not None and k < self.mlp_dim:
            # Gumbel-Top-K: factor-dependent width via top-k threshold on noisy logits.
            # Selects ~k neurons by finding the k-th largest noisy logit as threshold.
            topk_vals, _ = noisy_logits.topk(k, sorted=False)
            threshold = topk_vals.min()
            y_soft = torch.sigmoid((noisy_logits - threshold) / tau)
        else:
            # Vanilla Gumbel-Sigmoid: per-element sigmoid
            y_soft = torch.sigmoid(noisy_logits / tau)

        if hard:
            # Straight-through estimator: hard in forward, soft in backward
            y_hard = (y_soft >= 0.5).float()
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def get_active_fraction(self) -> float:
        """Return fraction of dimensions that are 'on' (logits > 0)."""
        with torch.no_grad():
            return (self.logits > 0).float().mean().item()


class GumbelMaskManager(nn.Module):
    """Manages Gumbel masks for all layers in the model."""

    def __init__(
        self,
        n_layers: int,
        mlp_dim: int,
        init_scale: float = 2.2,
        learnable: bool = True,
        init_mode: str = "linspace",
        init_value: float = 1.5,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim
        self.masks = nn.ModuleList([
            GumbelMaskLayer(
                mlp_dim, init_scale=init_scale, learnable=learnable,
                init_mode=init_mode, init_value=init_value,
            )
            for _ in range(n_layers)
        ])

    def get_mask(self, layer_idx: int, tau: float = 1.0, hard: bool = False,
                 k: Optional[int] = None) -> torch.Tensor:
        """Get the mask for a specific layer at a given sub-model width."""
        return self.masks[layer_idx](tau=tau, hard=hard, k=k)

    def spread_loss(self, fisher_weights: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Spread penalty: maximize logit variance to sharpen mask boundaries.

        Args:
            fisher_weights: Optional per-layer Fisher saliency weights. When provided,
                computes Fisher-weighted variance (penalizes ambiguity more for important neurons).

        Returns negative mean per-layer variance (minimize this to maximize spread).
        """
        total = torch.tensor(0.0, device=self.masks[0].logits.device)
        for i, m in enumerate(self.masks):
            if fisher_weights is not None:
                w = fisher_weights[i] / (fisher_weights[i].sum() + 1e-8)
                weighted_var = (w * (m.logits - m.logits.mean()) ** 2).sum()
                total = total - weighted_var
            else:
                total = total - torch.var(m.logits)
        return total / self.n_layers

    def budget_loss(self, target: float) -> torch.Tensor:
        """
        Compute budget penalty: penalizes deviation from target active fraction.

        Args:
            target: Target fraction of total MLP dimensions to keep active (0, 1].

        Returns:
            Scalar penalty |mean_active_fraction - target|.
        """
        all_logits = torch.cat([m.logits for m in self.masks])
        mean_fraction = torch.sigmoid(all_logits).mean()
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
