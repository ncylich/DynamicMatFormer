"""
Evaluate a pre-trained MatFormer model with heterogeneous per-layer widths.

Compares uniform vs F-Mat heterogeneous allocations at matching parameter budgets.

Usage:
    python scripts/eval_hmat.py \
        --checkpoint /path/to/step5-unsharded \
        --eval-data /path/to/val.npy \
        --allocations fmat_allocations.json \
        --num-batches 50
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
from olmo.model import MatformerManager, Olmo
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

    parser = argparse.ArgumentParser(description="Evaluate H-Mat allocations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-data", type=str, nargs="+", required=True)
    parser.add_argument("--allocations", type=str, default=None,
                        help="F-Mat allocations JSON file (from compute_fmat.py)")
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    log.info(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    n_layers = len(model.transformer.blocks)
    mlp_dim = model.transformer.blocks[0].ff_proj.weight.shape[0]
    log.info(f"Model: {n_layers} layers, MLP dim = {mlp_dim}")

    # Build eval dataloader
    dataset = MemMapDataset(*args.eval_data, chunk_size=model.config.max_sequence_length)
    collator = DataCollator(pad_direction="right", pad_token_id=model.config.pad_token_id)
    eval_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    mgr = MatformerManager.get_instance()
    results = {}

    # 1. Evaluate uniform allocations
    for factor in [1, 2, 4, 8]:
        mgr.mode = "uniform"
        mgr.current_factor = factor
        mgr.layer_factors = None

        ppl = evaluate_perplexity(model, eval_loader, args.num_batches, device)
        label = f"uniform-1/{factor}"
        results[label] = {"perplexity": ppl, "factor": factor}
        width = mlp_dim // factor
        total_dims = n_layers * width
        log.info(f"{label}: PPL = {ppl:.2f} (width={width}, total_dims={total_dims})")

    # 2. Evaluate F-Mat heterogeneous allocations
    if args.allocations:
        alloc_path = Path(args.allocations)
        if alloc_path.exists():
            with open(alloc_path) as f:
                alloc_data = json.load(f)

            for budget_str, layer_factors_str in alloc_data.get("allocations", {}).items():
                layer_factors = {int(l): int(f) for l, f in layer_factors_str.items()}

                mgr.mode = "heterogeneous"
                mgr.layer_factors = layer_factors
                mgr.current_factor = 1  # fallback

                ppl = evaluate_perplexity(model, eval_loader, args.num_batches, device)
                label = f"fmat-budget-{budget_str}"
                total_dims = sum(mlp_dim // layer_factors[l] for l in range(n_layers))
                results[label] = {
                    "perplexity": ppl,
                    "layer_factors": layer_factors,
                    "total_dims": total_dims,
                }
                log.info(f"{label}: PPL = {ppl:.2f} (total_dims={total_dims})")
        else:
            log.warning(f"Allocations file not found: {alloc_path}")

    # Reset manager
    mgr.mode = "uniform"
    mgr.current_factor = 1
    mgr.layer_factors = None

    # Print comparison table
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"  {'Configuration':<30} {'Perplexity':<15} {'MLP Dims':<10}")
    print(f"  {'-'*55}")
    for label, data in results.items():
        ppl = data["perplexity"]
        if "total_dims" in data:
            dims = data["total_dims"]
        elif "factor" in data:
            dims = n_layers * (mlp_dim // data["factor"])
        else:
            dims = "?"
        print(f"  {label:<30} {ppl:<15.2f} {dims}")
    print()

    # Save results
    output_path = Path(args.checkpoint) / "eval_hmat_results.json"
    serializable = {}
    for k, v in results.items():
        entry = {"perplexity": v["perplexity"]}
        if "factor" in v:
            entry["factor"] = v["factor"]
        if "total_dims" in v:
            entry["total_dims"] = v["total_dims"]
        if "layer_factors" in v:
            entry["layer_factors"] = {str(l): f for l, f in v["layer_factors"].items()}
        serializable[k] = entry

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
