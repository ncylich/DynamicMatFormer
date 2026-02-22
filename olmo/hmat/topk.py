"""
Soft Top-K structural sparsity for per-layer MLP width allocation.

Instead of per-element sigmoid (Gumbel-Sigmoid), this computes a differentiable
top-k threshold: exactly k dimensions are active by construction, eliminating
the need for a budget penalty. At factor=1 (full model), k = mlp_dim and the
mask is all-ones — zero overhead on the full model.

Gradient signal concentrates on boundary neurons near the k-th threshold,
rather than being spread across all neurons as in Gumbel-Sigmoid.
"""

from typing import Dict

import torch
import torch.nn as nn


class TopKMaskLayer(nn.Module):
    """
    Per-layer learnable importance gate using differentiable top-k.

    Computes a soft threshold at the k-th largest logit value and applies
    sigmoid relative to that threshold. Guarantees exactly k dimensions
    are active at convergence.
    """

    def __init__(self, mlp_dim: int, init_scale: float = 1.1):
        super().__init__()
        self.mlp_dim = mlp_dim
        # Same linear decay init as Phase 2 — first dims important, last dims not
        self.logits = nn.Parameter(torch.linspace(init_scale, -init_scale, mlp_dim))

    def forward(self, k: int, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Produce a mask with ~k active dimensions.

        Args:
            k: Target number of active dimensions for this sub-model width.
            tau: Temperature. Lower = sharper transition at threshold.
            hard: If True, use straight-through estimator for {0, 1} masks.

        Returns:
            Tensor of shape (mlp_dim,) with ~k values near 1 and the rest near 0.
        """
        if k >= self.mlp_dim:
            # Full model — return all-ones, no mask overhead
            return torch.ones_like(self.logits)

        # Find the soft threshold: the k-th largest logit
        # topk returns (values, indices), we want the smallest of the top-k
        topk_vals, _ = self.logits.topk(k, sorted=False)
        threshold = topk_vals.min()  # k-th largest logit

        # Soft mask: sigmoid of (logit - threshold) / tau
        # Dims above threshold → sigmoid > 0.5 → "on"
        # Dims below threshold → sigmoid < 0.5 → "off"
        y_soft = torch.sigmoid((self.logits - threshold) / tau)

        if hard:
            y_hard = (y_soft >= 0.5).float()
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def get_active_fraction(self, k: int) -> float:
        """Return fraction of dims that would be selected for top-k."""
        with torch.no_grad():
            mask = self.forward(k, tau=0.01, hard=True)
            return mask.mean().item()


class TopKMaskManager(nn.Module):
    """Manages top-k masks for all layers in the model."""

    def __init__(self, n_layers: int, mlp_dim: int, init_scale: float = 1.1):
        super().__init__()
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim
        self.masks = nn.ModuleList([
            TopKMaskLayer(mlp_dim, init_scale=init_scale)
            for _ in range(n_layers)
        ])

    def get_mask(self, layer_idx: int, k: int, tau: float = 1.0,
                 hard: bool = False) -> torch.Tensor:
        """Get the mask for a specific layer at a specific sub-model width."""
        return self.masks[layer_idx](k=k, tau=tau, hard=hard)

    def get_layer_widths(self, k: int) -> Dict[int, int]:
        """For a given target k, return actual active dim count per layer."""
        widths = {}
        with torch.no_grad():
            for i, mask_layer in enumerate(self.masks):
                hard_mask = mask_layer.forward(k, tau=0.01, hard=True)
                widths[i] = int(hard_mask.sum().item())
        return widths

    def log_summary(self, k: int) -> Dict[str, float]:
        """Return summary metrics for logging at a given sub-model width."""
        metrics = {}
        with torch.no_grad():
            for i, mask_layer in enumerate(self.masks):
                frac = mask_layer.get_active_fraction(k)
                metrics[f"topk/layer_{i}_active_frac_at_{k}"] = frac
        return metrics
