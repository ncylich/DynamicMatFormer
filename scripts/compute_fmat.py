"""
F-Mat: Post-training Fisher saliency analysis for heterogeneous MatFormer.

This script:
1. Loads a pre-trained uniform MatFormer checkpoint.
2. Computes Fisher saliency scores on calibration data.
3. Solves the knapsack problem for multiple budget ratios.
4. Outputs a JSON file mapping {budget_ratio: {layer_idx: factor}}.
5. Prints a summary table and per-layer sensitivity curve.

Usage:
    python scripts/compute_fmat.py \
        --checkpoint /path/to/step5-unsharded \
        --calibration-data /path/to/data.npy \
        --budget-ratios 0.125,0.25,0.5 \
        --output fmat_allocations.json
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from olmo.config import ModelConfig, TrainConfig
from olmo.data import DataCollator
from olmo.data.memmap_dataset import MemMapDataset
from olmo.hmat.fisher import compute_fisher_saliency
from olmo.hmat.knapsack import solve_budget_allocation
from olmo.model import Olmo
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_dir: str, device: str = "cpu") -> Olmo:
    """Load an OLMo model from an unsharded checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.yaml"

    if config_path.exists():
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)
    else:
        # Try loading from parent directory (some checkpoint layouts)
        parent_config = checkpoint_dir.parent / "config.yaml"
        if parent_config.exists():
            model_config = ModelConfig.load(parent_config, key="model", validate_paths=False)
        else:
            raise FileNotFoundError(f"No config.yaml found in {checkpoint_dir} or its parent")

    model_config.init_device = "cpu"
    model = Olmo(model_config)

    model_pt = checkpoint_dir / "model.pt"
    if model_pt.exists():
        state_dict = torch.load(model_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(model._make_state_dict_compatible(state_dict))
        log.info(f"Loaded model weights from {model_pt}")
    else:
        log.warning(f"No model.pt found in {checkpoint_dir}, using random weights")

    return model.to(device)


def build_calibration_dataloader(
    data_paths: list,
    batch_size: int = 8,
    max_sequence_length: int = 1024,
) -> DataLoader:
    """Build a simple DataLoader from .npy memmap files for calibration."""
    dataset = MemMapDataset(*data_paths, chunk_size=max_sequence_length)
    collator = DataCollator(pad_direction="right", pad_token_id=0)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )


def print_saliency_summary(saliency: dict, mlp_dim: int):
    """Print per-layer saliency summary and ASCII U-shape visualization."""
    n_layers = len(saliency)

    print("\n" + "=" * 60)
    print("Per-Layer Fisher Saliency Summary")
    print("=" * 60)

    # Compute per-layer total saliency (before normalization, use entropy as proxy)
    layer_scores = []
    for l in range(n_layers):
        scores = saliency[l]
        # Concentration: how much saliency is in the top 1/8 dimensions
        top_k = mlp_dim // 8
        top_fraction = scores[:top_k].sum().item()
        layer_scores.append(top_fraction)
        print(f"  Layer {l:2d}: top-1/8 concentration = {top_fraction:.4f}  "
              f"max = {scores.max().item():.6f}  "
              f"min = {scores.min().item():.6f}")

    # ASCII bar chart
    print("\nPer-Layer Sensitivity (top-1/8 concentration):")
    max_score = max(layer_scores) if layer_scores else 1.0
    for l in range(n_layers):
        bar_len = int(40 * layer_scores[l] / max_score) if max_score > 0 else 0
        print(f"  Layer {l:2d} |{'█' * bar_len}{' ' * (40 - bar_len)}| {layer_scores[l]:.4f}")

    print()


def print_allocation_table(allocations: dict, mlp_dim: int, n_layers: int):
    """Print a formatted table of budget allocations."""
    print("\n" + "=" * 60)
    print("Budget Allocation Results")
    print("=" * 60)

    for budget_ratio, layer_factors in sorted(allocations.items()):
        total_dims = sum(mlp_dim // layer_factors[l] for l in range(n_layers))
        total_full = n_layers * mlp_dim
        print(f"\n  Budget ratio = {budget_ratio} ({total_dims}/{total_full} = {total_dims/total_full:.1%})")
        print(f"  {'Layer':<8} {'Factor':<8} {'Width':<8} {'Fraction':<10}")
        print(f"  {'-'*34}")
        for l in range(n_layers):
            f = layer_factors[l]
            w = mlp_dim // f
            print(f"  {l:<8d} {f:<8d} {w:<8d} {1.0/f:<10.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="F-Mat: Fisher saliency analysis")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to unsharded checkpoint directory")
    parser.add_argument("--calibration-data", type=str, nargs="+", required=True,
                        help="Path(s) to .npy calibration data files")
    parser.add_argument("--num-batches", type=int, default=128,
                        help="Number of calibration batches")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for calibration")
    parser.add_argument("--budget-ratios", type=str, default="0.125,0.25,0.5",
                        help="Comma-separated budget ratios to solve for")
    parser.add_argument("--allowed-factors", type=str, default="1,2,4,8",
                        help="Comma-separated allowed slicing factors")
    parser.add_argument("--output", type=str, default="fmat_allocations.json",
                        help="Output JSON file path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budget_ratios = [float(x) for x in args.budget_ratios.split(",")]
    allowed_factors = [int(x) for x in args.allowed_factors.split(",")]

    # Load model
    log.info(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    n_layers = len(model.transformer.blocks)
    mlp_dim = model.transformer.blocks[0].ff_proj.weight.shape[0]
    log.info(f"Model: {n_layers} layers, MLP dim = {mlp_dim}")

    # Build calibration dataloader
    log.info(f"Building calibration dataloader from {args.calibration_data}...")
    dataloader = build_calibration_dataloader(
        args.calibration_data,
        batch_size=args.batch_size,
        max_sequence_length=model.config.max_sequence_length,
    )

    # Compute Fisher saliency
    log.info(f"Computing Fisher saliency over {args.num_batches} batches...")
    saliency = compute_fisher_saliency(
        model, dataloader, num_batches=args.num_batches,
        device=torch.device(args.device),
    )

    # Print saliency summary
    print_saliency_summary(saliency, mlp_dim)

    # Solve knapsack for each budget ratio
    allocations = {}
    for ratio in budget_ratios:
        log.info(f"Solving budget allocation for ratio={ratio}...")
        layer_factors = solve_budget_allocation(saliency, ratio, allowed_factors)
        # Convert keys to strings for JSON serialization
        allocations[str(ratio)] = {str(l): f for l, f in layer_factors.items()}

    # Print allocation table
    print_allocation_table(
        {float(k): {int(l): f for l, f in v.items()} for k, v in allocations.items()},
        mlp_dim, n_layers,
    )

    # Save saliency scores
    saliency_output = args.output.replace(".json", "_saliency.json")
    saliency_data = {
        str(l): scores.cpu().tolist() for l, scores in saliency.items()
    }

    # Save allocations
    output = {
        "model_info": {
            "n_layers": n_layers,
            "mlp_dim": mlp_dim,
            "checkpoint": args.checkpoint,
        },
        "allowed_factors": allowed_factors,
        "allocations": allocations,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Allocations saved to {output_path}")

    with open(saliency_output, "w") as f:
        json.dump(saliency_data, f)
    log.info(f"Saliency scores saved to {saliency_output}")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
