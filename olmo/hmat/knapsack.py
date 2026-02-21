"""
Multi-choice knapsack solver for per-layer MLP width allocation.

Given Fisher saliency scores and a target parameter budget, finds the optimal
per-layer slicing factors that maximize total retained saliency.
"""

import logging
from typing import Dict, List

import torch

log = logging.getLogger(__name__)

NEG_INF = float("-inf")


def solve_budget_allocation(
    saliency: Dict[int, torch.Tensor],
    budget_ratio: float,
    allowed_factors: List[int],
) -> Dict[int, int]:
    """
    Solve the multi-choice knapsack problem for per-layer width allocation.

    For each layer, chooses a slicing factor from `allowed_factors` such that:
    - Total MLP dimensions used across all layers <= budget
    - Total retained saliency is maximized

    Args:
        saliency: Per-layer saliency scores from compute_fisher_saliency.
            Maps layer_idx -> Tensor of shape (mlp_hidden_dim,).
        budget_ratio: Target fraction of total MLP parameters to use (0, 1].
        allowed_factors: List of allowed slicing factors (e.g., [1, 2, 4, 8]).
            Must be positive integers. Factor f means use mlp_dim/f dimensions.

    Returns:
        Dict mapping layer_idx -> chosen factor for that layer.
    """
    n_layers = len(saliency)
    mlp_dim = len(next(iter(saliency.values())))
    allowed_factors = sorted(allowed_factors)
    max_factor = allowed_factors[-1]

    # Total budget in MLP dimension units
    total_budget = int(budget_ratio * n_layers * mlp_dim)

    # Check if budget is feasible (can fit at least the minimum cost per layer)
    min_cost_per_layer = mlp_dim // max_factor
    min_total_cost = n_layers * min_cost_per_layer
    if total_budget <= 0 or total_budget < min_total_cost:
        return {l: max_factor for l in range(n_layers)}

    # For each layer and each allowed factor, compute cost and value
    choices = {}
    for l in range(n_layers):
        layer_choices = []
        for factor in allowed_factors:
            k = mlp_dim // factor
            # Value: sum of saliency for the first k dimensions (MatFormer nesting)
            value = saliency[l][:k].sum().item()
            cost = k
            layer_choices.append((factor, cost, value))
        choices[l] = layer_choices

    # Dynamic programming multi-choice knapsack
    # dp[l+1][b] = max saliency achievable using layers 0..l with budget exactly b
    # Use NEG_INF for infeasible states
    dp = [[NEG_INF] * (total_budget + 1) for _ in range(n_layers + 1)]
    dp[0][0] = 0.0  # base case: 0 layers, 0 budget used, 0 value

    choice_trace = [[max_factor] * (total_budget + 1) for _ in range(n_layers)]

    for l in range(n_layers):
        for b in range(total_budget + 1):
            best_val = NEG_INF
            best_factor = max_factor
            for factor, cost, value in choices[l]:
                prev_budget = b - cost
                if prev_budget >= 0 and dp[l][prev_budget] > NEG_INF:
                    candidate = dp[l][prev_budget] + value
                    if candidate > best_val:
                        best_val = candidate
                        best_factor = factor
            dp[l + 1][b] = best_val
            choice_trace[l][b] = best_factor

    # Find the best feasible budget <= total_budget
    best_b = 0
    best_val = NEG_INF
    for b in range(total_budget + 1):
        if dp[n_layers][b] > best_val:
            best_val = dp[n_layers][b]
            best_b = b

    # Traceback: reconstruct the chosen factors
    result = {}
    remaining = best_b
    for l in range(n_layers - 1, -1, -1):
        factor = choice_trace[l][remaining]
        result[l] = factor
        remaining -= mlp_dim // factor

    log.info(f"Budget allocation (ratio={budget_ratio}):")
    for l in sorted(result.keys()):
        log.info(f"  Layer {l}: factor={result[l]} (width={mlp_dim // result[l]})")
    total_used = sum(mlp_dim // result[l] for l in range(n_layers))
    total_full = n_layers * mlp_dim
    log.info(f"  Total: {total_used}/{total_full} dims ({total_used / total_full:.1%})")

    return result
