"""
Fisher saliency EMA accumulator for Phase 2.5 Fisher-Guided Gumbel masking.

Accumulates per-dimension squared-gradient saliency scores via exponential
moving average during training. Zero extra compute — uses gradients that
already exist after backward. Converts scores to Gumbel logits via rank-based
mapping.
"""

from typing import List

import torch
import torch.nn as nn


class FisherEMA:
    """
    Exponential moving average of per-dimension Fisher saliency scores.

    Updated each training step from existing gradients (zero extra compute).
    Converts to Gumbel logits via rank-based mapping for Phase 2.5.

    Note: Fisher scores accumulate from ALL factor iterations (1, 2, 4, 8).
    TODO: Investigate whether subset-specific Fisher accumulation (only from
    matching factor) improves allocation quality for specific sub-models.
    """

    def __init__(
        self,
        n_layers: int,
        mlp_dim: int,
        beta: float = 0.99,
        device: torch.device = None,
    ):
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim
        self.beta = beta
        self.scores = [torch.zeros(mlp_dim, device=device) for _ in range(n_layers)]
        self.steps = 0

    def update(self, model: nn.Module):
        """
        Accumulate Fisher scores from current gradients (call after backward).

        Computes per-dimension saliency from ff_proj and ff_out gradients:
            saliency[l][d] = ||grad(ff_proj.weight[d,:])||^2 + ||grad(ff_out.weight[:,d])||^2

        For activations with output_multiplier != 1 (e.g. SwiGLU), ff_proj has
        more rows than ff_out columns. We accumulate from ff_out (post-activation dim).
        """
        self.steps += 1
        for l, block in enumerate(model.transformer.blocks):
            grad_out = block.ff_out.weight.grad  # (d_model, mlp_out_dim)
            if grad_out is None:
                continue
            # ff_out columns correspond to post-activation dims (what masks operate on)
            fisher = (grad_out ** 2).sum(dim=0)
            self.scores[l] = self.beta * self.scores[l] + (1 - self.beta) * fisher.detach()

    def get_logits(self, scale: float = 1.1, mode: str = "rank") -> List[torch.Tensor]:
        """
        Convert Fisher EMA scores to Gumbel logits.

        Args:
            scale: Controls the logit range [-scale, +scale].
            mode: "rank" = rank-based linear mapping (default),
                  "log" = log-scaled Fisher scores normalized to [-scale, +scale].
        """
        logits = []
        for l in range(self.n_layers):
            if mode == "log":
                # Log-scale preserves magnitude differences between dims
                log_scores = torch.log1p(self.scores[l])
                if log_scores.max() > log_scores.min():
                    # Normalize to [-scale, +scale]
                    normalized = (log_scores - log_scores.min()) / (log_scores.max() - log_scores.min())
                    layer_logits = scale * (2 * normalized - 1)
                else:
                    layer_logits = torch.zeros_like(log_scores)
            else:
                # Rank-based: top-saliency → +scale, bottom → -scale
                ranks = self.scores[l].argsort(descending=True).argsort()
                layer_logits = scale - (2 * scale * ranks.float() / (self.mlp_dim - 1))
            logits.append(layer_logits)
        return logits

    def to(self, device: torch.device) -> "FisherEMA":
        """Move scores to device."""
        self.scores = [s.to(device) for s in self.scores]
        return self
