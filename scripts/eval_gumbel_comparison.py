"""
Evaluate the Gumbel-trained H-Mat model and compare with the uniform MatFormer baseline.

Evaluates:
1. Gumbel model at full width (no masking, uniform factor=1)
2. Gumbel model with learned masks applied (hard)
3. Gumbel model with uniform sub-model extraction (factors 1,2,4,8)
4. Baseline MatFormer model with uniform sub-model extraction (factors 1,2,4,8)

Usage:
    python scripts/eval_gumbel_comparison.py \
        --gumbel-checkpoint /path/to/hmat-gumbel/step540-unsharded \
        --baseline-checkpoint /path/to/matformer/step540-unsharded \
        --eval-data /path/to/val.npy \
        --num-batches 100
"""

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from olmo.config import ModelConfig
from olmo.data import DataCollator
from olmo.data.memmap_dataset import MemMapDataset
from olmo.hmat.gumbel import GumbelMaskManager
from olmo.model import Activation, MatformerManager, Olmo
from olmo.util import move_to_device, prepare_cli_environment

log = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_dir: str, device: str = "cpu") -> Olmo:
    """Load an OLMo model from an unsharded checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.yaml"
    if config_path.exists():
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)
    else:
        parent_config = checkpoint_dir.parent / "config.yaml"
        model_config = ModelConfig.load(parent_config, key="model", validate_paths=False)

    model_config.init_device = "cpu"
    model = Olmo(model_config)

    model_pt = checkpoint_dir / "model.pt"
    if model_pt.exists():
        state_dict = torch.load(model_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(model._make_state_dict_compatible(state_dict))
    return model.to(device).eval()


def load_gumbel_state(checkpoint_dir: str, model_config: ModelConfig, device: str = "cpu"):
    """Load GumbelMaskManager state from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)

    # Try dedicated gumbel.pt first (saved by save_unsharded_checkpoint)
    gumbel_pt = checkpoint_dir / "gumbel.pt"
    if gumbel_pt.exists():
        gumbel_state = torch.load(gumbel_pt, map_location="cpu", weights_only=False)
    else:
        # Fall back: search other checkpoint files for embedded gumbel_manager
        gumbel_state = None
        for fname in ["rank0.pt", "model.pt", "other.pt"]:
            fpath = checkpoint_dir / fname
            if fpath.exists():
                try:
                    data = torch.load(fpath, map_location="cpu", weights_only=False)
                    if isinstance(data, dict) and "gumbel_manager" in data:
                        gumbel_state = data["gumbel_manager"]
                        del data
                        break
                    del data
                except Exception:
                    continue

    if gumbel_state is None:
        log.warning("Could not find gumbel_manager state in checkpoint")
        return None

    act = Activation.build(model_config)
    mlp_dim = int(model_config.mlp_ratio * model_config.d_model * act.output_multiplier)
    mgr = GumbelMaskManager(n_layers=model_config.n_layers, mlp_dim=mlp_dim)
    mgr.load_state_dict(gumbel_state)
    return mgr.to(device).eval()


def evaluate_perplexity(
    model: Olmo,
    dataloader: DataLoader,
    num_batches: int,
    device: torch.device,
) -> float:
    """Compute perplexity on a dataset."""
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            batch = move_to_device(batch, device)
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")

            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            n_tokens = shift_labels.numel()
            total_loss += loss.item()
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare Gumbel H-Mat vs baseline MatFormer")
    parser.add_argument("--gumbel-checkpoint", type=str, required=True)
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--eval-data", type=str, nargs="+", required=True)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    results = {}

    # ================================================================
    # 1. Evaluate BASELINE MatFormer model (uniform sub-models)
    # ================================================================
    print("\n" + "=" * 60)
    print("Evaluating BASELINE MatFormer model")
    print("=" * 60)

    mgr = MatformerManager.get_instance()

    log.info(f"Loading baseline model from {args.baseline_checkpoint}...")
    baseline_model = load_model_from_checkpoint(args.baseline_checkpoint, device=args.device)
    n_layers = len(baseline_model.transformer.blocks)
    mlp_dim = baseline_model.transformer.blocks[0].ff_proj.weight.shape[0]

    dataset = MemMapDataset(*args.eval_data, chunk_size=baseline_model.config.max_sequence_length)
    collator = DataCollator(pad_direction="right", pad_token_id=baseline_model.config.pad_token_id)
    eval_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    for factor in [1, 2, 4, 8]:
        mgr.mode = "uniform"
        mgr.current_factor = factor
        mgr.layer_factors = None
        mgr.gumbel_masks = None

        ppl = evaluate_perplexity(baseline_model, eval_loader, args.num_batches, device)
        label = f"baseline-uniform-1/{factor}"
        width = mlp_dim // factor
        total_dims = n_layers * width
        results[label] = {"perplexity": ppl, "total_dims": total_dims}
        log.info(f"{label}: PPL = {ppl:.2f} (width={width})")

    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ================================================================
    # 2. Evaluate GUMBEL H-Mat model
    # ================================================================
    print("\n" + "=" * 60)
    print("Evaluating GUMBEL H-Mat model")
    print("=" * 60)

    # Reset singleton
    MatformerManager._instance = None
    mgr = MatformerManager.get_instance()

    log.info(f"Loading gumbel model from {args.gumbel_checkpoint}...")
    gumbel_model = load_model_from_checkpoint(args.gumbel_checkpoint, device=args.device)
    n_layers_g = len(gumbel_model.transformer.blocks)
    mlp_dim_g = gumbel_model.transformer.blocks[0].ff_proj.weight.shape[0]

    # 2a. Evaluate gumbel model with uniform sub-models (no masks)
    for factor in [1, 2, 4, 8]:
        mgr.mode = "uniform"
        mgr.current_factor = factor
        mgr.layer_factors = None
        mgr.gumbel_masks = None

        ppl = evaluate_perplexity(gumbel_model, eval_loader, args.num_batches, device)
        label = f"gumbel-uniform-1/{factor}"
        width = mlp_dim_g // factor
        total_dims = n_layers_g * width
        results[label] = {"perplexity": ppl, "total_dims": total_dims}
        log.info(f"{label}: PPL = {ppl:.2f} (width={width})")

    # 2b. Evaluate gumbel model with learned masks
    gumbel_mgr = load_gumbel_state(args.gumbel_checkpoint, gumbel_model.config, device=args.device)
    if gumbel_mgr is not None:
        gumbel_mgr.eval()
        mgr.mode = "gumbel"
        mgr.gumbel_masks = gumbel_mgr
        mgr.gumbel_tau = 0.01  # Very low tau for hard masks

        ppl = evaluate_perplexity(gumbel_model, eval_loader, args.num_batches, device)

        # Get learned widths
        widths = gumbel_mgr.get_layer_widths()
        factors = gumbel_mgr.get_layer_factors([1, 2, 4, 8])
        total_active = sum(widths.values())
        act = Activation.build(gumbel_model.config)
        total_possible = n_layers_g * int(mlp_dim_g * act.output_multiplier)

        label = "gumbel-learned-masks"
        results[label] = {
            "perplexity": ppl,
            "total_dims": total_active,
            "layer_widths": widths,
            "layer_factors": factors,
            "active_fraction": total_active / total_possible,
        }
        log.info(f"{label}: PPL = {ppl:.2f} (total_active={total_active}/{total_possible})")
        for i in sorted(widths.keys()):
            log.info(f"  Layer {i}: width={widths[i]}, factor≈{factors[i]}")

        # Also log mask summary
        summary = gumbel_mgr.log_summary()
        for k, v in summary.items():
            log.info(f"  {k}: {v:.4f}")

    else:
        log.warning("No gumbel masks found — skipping learned mask evaluation")

    # Reset
    mgr.mode = "uniform"
    mgr.current_factor = 1
    mgr.layer_factors = None
    mgr.gumbel_masks = None

    # ================================================================
    # Print comparison table
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Gumbel H-Mat vs Baseline MatFormer")
    print("=" * 70)
    print(f"  {'Configuration':<35} {'PPL':<12} {'MLP Dims':<12} {'vs Base'}")
    print(f"  {'-'*65}")

    # Get baseline full PPL for reference
    base_full_ppl = results.get("baseline-uniform-1/1", {}).get("perplexity", 0)

    for label, data in sorted(results.items()):
        ppl = data["perplexity"]
        dims = data.get("total_dims", "?")
        if base_full_ppl > 0:
            diff = ppl - base_full_ppl
            diff_str = f"{diff:+.2f}"
        else:
            diff_str = ""
        print(f"  {label:<35} {ppl:<12.2f} {str(dims):<12} {diff_str}")
    print()

    # Save results
    output_path = Path(args.gumbel_checkpoint) / "eval_comparison_results.json"
    serializable = {}
    for k, v in results.items():
        entry = {"perplexity": v["perplexity"]}
        if "total_dims" in v:
            entry["total_dims"] = v["total_dims"]
        if "layer_widths" in v:
            entry["layer_widths"] = {str(i): w for i, w in v["layer_widths"].items()}
        if "layer_factors" in v:
            entry["layer_factors"] = {str(i): f for i, f in v["layer_factors"].items()}
        if "active_fraction" in v:
            entry["active_fraction"] = v["active_fraction"]
        serializable[k] = entry

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
