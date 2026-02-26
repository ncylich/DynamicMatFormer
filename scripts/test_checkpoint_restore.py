"""
Integration test for gumbel/topk checkpoint restore.

Run with:
    torchrun --nproc_per_node=1 scripts/test_checkpoint_restore.py

Verifies that gumbel mask logits + optimizer state survive a save/restore cycle.
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torchmetrics import MeanMetric

from olmo.config import (
    CheckpointType, DataConfig, HMatConfig, ModelConfig, OptimizerConfig,
    PaddingDirection, SchedulerConfig, TokenizerConfig, TrainConfig,
)
from olmo.data import build_train_dataloader
from olmo.model import Olmo
from olmo.optim import build_optimizer, build_scheduler
from olmo.train import Trainer
from olmo.util import barrier, get_global_rank, get_local_rank, seed_all


def make_synthetic_data(path: str, n_tokens: int = 50000):
    """Generate synthetic memmap data."""
    rng = np.random.default_rng(42)
    tokens = rng.integers(0, 50257, size=n_tokens, dtype=np.uint16)
    np.save(path, tokens)
    print(f"Generated {n_tokens} tokens at {path}")


def make_config(tmpdir: str, data_path: str, method: str, save_suffix: str = "") -> TrainConfig:
    return TrainConfig(
        run_name=f"test-ckpt-restore{save_suffix}",
        precision="amp_bf16",
        model=ModelConfig(
            d_model=128,
            n_heads=2,
            n_layers=4,
            mlp_ratio=4,
            vocab_size=50257,
            eos_token_id=50256,
            pad_token_id=50256,
            max_sequence_length=256,
            init_device=None,
            init_std=0.02,
        ),
        optimizer=OptimizerConfig(learning_rate=1e-3, betas=[0.9, 0.95]),
        scheduler=SchedulerConfig(name="cosine_with_warmup", t_warmup=5),
        data=DataConfig(
            paths=[data_path],
            pad_direction=PaddingDirection.right,
            persistent_workers=False,
            num_workers=0,
            prefetch_factor=None,
        ),
        tokenizer=TokenizerConfig(identifier="gpt2"),
        save_folder=os.path.join(tmpdir, f"checkpoints{save_suffix}"),
        save_overwrite=True,
        max_duration=10,
        global_train_batch_size=4,
        device_train_microbatch_size=4,
        matformer_factor=4,
        hmat=HMatConfig(
            enabled=True,
            method=method,
            gumbel_init_scale=1.1,
            gumbel_tau_start=1.0,
            gumbel_tau_end=0.5,
            budget_penalty_lambda=0.001,
            budget_penalty_target=0.5,
        ),
    )


def build_trainer(cfg: TrainConfig, device: torch.device) -> Trainer:
    """Build all components and return a Trainer."""
    cfg.device_train_batch_size = cfg.global_train_batch_size
    cfg.device_train_grad_accum = 1

    olmo_model = Olmo(cfg.model)
    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        mixed_precision=MixedPrecision(
            param_dtype=cfg.autocast_precision,
            reduce_dtype=cfg.autocast_precision,
            buffer_dtype=cfg.autocast_precision,
        ),
        auto_wrap_policy=olmo_model.fsdp_wrap_fn,
        use_orig_params=cfg.fsdp.use_orig_params,
        limit_all_gathers=True,
        device_id=get_local_rank(),
    )
    optim = build_optimizer(cfg, fsdp_model)
    scheduler = build_scheduler(cfg, optim)
    train_loader = build_train_dataloader(cfg)

    return Trainer(
        cfg=cfg,
        model=olmo_model,
        fsdp_model=fsdp_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        ce_train_loss_metric=MeanMetric(nan_strategy="error").to(device),
        z_train_loss_metric=None,
        evaluators=[],
        indices_file=None,
    )


def get_logits(trainer, method):
    if method in ("gumbel", "gumbel_topk", "fisher_gumbel"):
        return [m.logits.detach().clone() for m in trainer.gumbel_manager.masks]
    else:
        return [m.logits.detach().clone() for m in trainer.topk_manager.masks]


def get_optim(trainer, method):
    if method in ("gumbel", "gumbel_topk", "fisher_gumbel"):
        return trainer.gumbel_optim
    else:
        return trainer.topk_optim


def test_gumbel_restore(method: str = "gumbel"):
    """Test that gumbel/topk state survives checkpoint save+restore."""
    dist.init_process_group(backend="nccl")
    barrier()
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")
    seed_all(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.npy")
        make_synthetic_data(data_path)

        # --- Phase 1: Train for 10 steps and save ---
        print(f"\n=== Phase 1: Training 10 steps with {method} masks ===")
        cfg = make_config(tmpdir, data_path, method)
        trainer1 = build_trainer(cfg, device)
        trainer1.init_masks()

        # Save initial checkpoint (pre-train sanity check)
        ckpt_type = CheckpointType.unsharded
        ckpt_path = trainer1.save_checkpoint(checkpoint_type=ckpt_type)
        print(f"Pre-train checkpoint: {ckpt_path}")
        trainer1.restore_checkpoint(ckpt_path, checkpoint_type=ckpt_type)

        # Run training
        trainer1.fit()

        # Grab the learned logits BEFORE saving
        pre_save_logits = get_logits(trainer1, method)
        mask_optim = get_optim(trainer1, method)
        has_optim = mask_optim is not None

        print(f"Pre-save logits (layer 0, first 10): {pre_save_logits[0][:10]}")

        # Save checkpoint after training
        ckpt_path = trainer1.save_checkpoint(checkpoint_type=ckpt_type)
        print(f"Saved checkpoint to: {ckpt_path}")

        # Verify expected files
        expected_files = ["model.pt", "optim.pt", "other.pt", "config.yaml"]
        if method in ("gumbel", "gumbel_topk", "fisher_gumbel"):
            expected_files += ["gumbel.pt", "gumbel_optim.pt"]
        elif method == "topk":
            expected_files += ["topk.pt", "topk_optim.pt"]
        for f in expected_files:
            fpath = os.path.join(str(ckpt_path), f)
            assert os.path.exists(fpath), f"Missing checkpoint file: {fpath}"
        print("All expected checkpoint files present.")

        del trainer1
        torch.cuda.empty_cache()

        # --- Phase 2: New trainer, restore, verify ---
        print(f"\n=== Phase 2: Restoring checkpoint and verifying {method} state ===")
        cfg2 = make_config(tmpdir, data_path, method, save_suffix="-2")
        cfg2.load_path = str(ckpt_path)

        trainer2 = build_trainer(cfg2, device)

        # KEY: init_masks BEFORE restore (this is the fix)
        trainer2.init_masks()

        fresh_logits = get_logits(trainer2, method)
        print(f"Fresh init logits (layer 0, first 10): {fresh_logits[0][:10]}")

        # Restore
        trainer2.restore_checkpoint(str(ckpt_path))
        print("Checkpoint restored successfully.")

        restored_logits = get_logits(trainer2, method)
        print(f"Restored logits (layer 0, first 10): {restored_logits[0][:10]}")

        # Check: restored == pre-save
        all_match = True
        for i, (pre, restored) in enumerate(zip(pre_save_logits, restored_logits)):
            if not torch.allclose(pre.cpu(), restored.cpu(), atol=1e-6):
                print(f"FAIL: Layer {i} logits don't match! Max diff: {(pre.cpu() - restored.cpu()).abs().max():.6e}")
                all_match = False
            else:
                print(f"  Layer {i} logits: MATCH")

        # Check: restored != fresh init
        any_changed = False
        for fresh, restored in zip(fresh_logits, restored_logits):
            if not torch.allclose(fresh.cpu(), restored.cpu(), atol=1e-6):
                any_changed = True
                break
        if not any_changed:
            print("FAIL: Restored logits identical to fresh init — restore had no effect!")
            all_match = False
        else:
            print("  Restored logits differ from fresh init: GOOD")

        # Check optimizer state populated
        if has_optim:
            restored_optim = get_optim(trainer2, method)
            if len(restored_optim.state_dict()["state"]) > 0:
                print("  Optimizer state: POPULATED")
            else:
                print("FAIL: Optimizer state empty after restore!")
                all_match = False

        del trainer2
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    if all_match:
        print(f"PASS: {method} checkpoint restore — all checks passed!")
    else:
        print(f"FAIL: {method} checkpoint restore — some checks failed!")
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "gumbel"
    test_gumbel_restore(method=method)
