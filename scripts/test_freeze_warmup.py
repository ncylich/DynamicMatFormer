"""
Integration test for Fix 1 (gumbel freeze flag) and Fix 2 (_in_vanilla_warmup).

Run with:
    torchrun --nproc_per_node=1 scripts/test_freeze_warmup.py

Tests:
  1. Vanilla warmup: masks are disabled during warmup, then enabled.
  2. Freeze: gumbel_optim stays alive (not set to None) after freeze, but stops updating.
  3. Checkpoint after freeze: gumbel_optim state is still saveable.
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
from olmo.model import Olmo, MatformerManager
from olmo.optim import build_optimizer, build_scheduler
from olmo.train import Trainer
from olmo.util import barrier, get_global_rank, get_local_rank, seed_all


def make_synthetic_data(path: str, n_tokens: int = 50000):
    rng = np.random.default_rng(42)
    tokens = rng.integers(0, 50257, size=n_tokens, dtype=np.uint16)
    np.save(path, tokens)


def build_trainer(cfg: TrainConfig, device: torch.device) -> Trainer:
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


def test_warmup_and_freeze():
    dist.init_process_group(backend="nccl")
    barrier()
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")
    seed_all(42)

    all_pass = True

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.npy")
        make_synthetic_data(data_path)

        # ============================================================
        # Test 1: Vanilla warmup — masks disabled then enabled
        # ============================================================
        print("\n=== Test 1: Vanilla warmup (50% warmup, 20 steps) ===")
        cfg = TrainConfig(
            run_name="test-warmup",
            precision="amp_bf16",
            model=ModelConfig(
                d_model=128, n_heads=2, n_layers=4, mlp_ratio=4,
                vocab_size=50257, eos_token_id=50256, pad_token_id=50256,
                max_sequence_length=256, init_device=None, init_std=0.02,
            ),
            optimizer=OptimizerConfig(learning_rate=1e-3, betas=[0.9, 0.95]),
            scheduler=SchedulerConfig(name="cosine_with_warmup", t_warmup=5),
            data=DataConfig(
                paths=[data_path], pad_direction=PaddingDirection.right,
                persistent_workers=False, num_workers=0, prefetch_factor=None,
            ),
            tokenizer=TokenizerConfig(identifier="gpt2"),
            save_folder=os.path.join(tmpdir, "ckpt-warmup"),
            save_overwrite=True,
            max_duration=20,
            global_train_batch_size=4,
            device_train_microbatch_size=4,
            matformer_factor=4,
            hmat=HMatConfig(
                enabled=True,
                method="gumbel",
                gumbel_init_scale=1.1,
                gumbel_tau_start=1.0,
                gumbel_tau_end=0.5,
                budget_penalty_lambda=0.001,
                budget_penalty_target=0.5,
                vanilla_warmup_frac=0.5,  # First 50% = 10 steps with masks disabled
            ),
        )

        trainer = build_trainer(cfg, device)
        trainer.init_masks()

        # Verify _in_vanilla_warmup starts as False (Fix 2: proper init)
        assert hasattr(trainer, '_in_vanilla_warmup'), "Fix 2 FAIL: _in_vanilla_warmup not a dataclass field"
        assert trainer._in_vanilla_warmup == False, "Fix 2 FAIL: _in_vanilla_warmup should start False"
        print("  _in_vanilla_warmup properly initialized as False: PASS")

        # Verify MatformerManager mode is "uniform" during warmup
        matmng = MatformerManager.get_instance()
        assert matmng.mode == "uniform", f"Expected mode='uniform' during warmup, got '{matmng.mode}'"
        print("  MatformerManager mode='uniform' during warmup: PASS")

        # Grab initial logits
        init_logits = [m.logits.detach().clone() for m in trainer.gumbel_manager.masks]

        # Run training (warmup 10 steps + active 10 steps)
        trainer.fit()

        # After training, logits should have changed (gumbel was active for last 10 steps)
        final_logits = [m.logits.detach().clone() for m in trainer.gumbel_manager.masks]
        changed = any(
            not torch.allclose(init.cpu(), final.cpu(), atol=1e-6)
            for init, final in zip(init_logits, final_logits)
        )
        if changed:
            print("  Logits changed after warmup+training: PASS")
        else:
            print("  FAIL: Logits unchanged — gumbel may not have been active after warmup")
            all_pass = False

        del trainer
        torch.cuda.empty_cache()

        # ============================================================
        # Test 2: Freeze — optimizer stays alive, stops updating
        # ============================================================
        print("\n=== Test 2: Gumbel freeze (last 40% of 20 steps) ===")
        cfg2 = TrainConfig(
            run_name="test-freeze",
            precision="amp_bf16",
            model=ModelConfig(
                d_model=128, n_heads=2, n_layers=4, mlp_ratio=4,
                vocab_size=50257, eos_token_id=50256, pad_token_id=50256,
                max_sequence_length=256, init_device=None, init_std=0.02,
            ),
            optimizer=OptimizerConfig(learning_rate=1e-3, betas=[0.9, 0.95]),
            scheduler=SchedulerConfig(name="cosine_with_warmup", t_warmup=5),
            data=DataConfig(
                paths=[data_path], pad_direction=PaddingDirection.right,
                persistent_workers=False, num_workers=0, prefetch_factor=None,
            ),
            tokenizer=TokenizerConfig(identifier="gpt2"),
            save_folder=os.path.join(tmpdir, "ckpt-freeze"),
            save_overwrite=True,
            max_duration=20,
            global_train_batch_size=4,
            device_train_microbatch_size=4,
            matformer_factor=4,
            hmat=HMatConfig(
                enabled=True,
                method="gumbel",
                gumbel_init_scale=1.1,
                gumbel_tau_start=1.0,
                gumbel_tau_end=0.5,
                budget_penalty_lambda=0.001,
                budget_penalty_target=0.5,
                gumbel_freeze_fraction=0.4,  # Freeze last 40% = last 8 steps
            ),
        )

        trainer2 = build_trainer(cfg2, device)
        trainer2.init_masks()

        # Run training
        trainer2.fit()

        # Fix 1 check: gumbel_optim should NOT be None (old bug set it to None)
        assert trainer2.gumbel_optim is not None, "Fix 1 FAIL: gumbel_optim is None after freeze!"
        print("  gumbel_optim still alive after freeze: PASS")

        # Fix 1 check: _gumbel_frozen flag should be True
        assert trainer2._gumbel_frozen == True, "Fix 1 FAIL: _gumbel_frozen should be True after freeze"
        print("  _gumbel_frozen flag is True: PASS")

        # Fix 1 check: tau should be very low (frozen = hard masks)
        matmng2 = MatformerManager.get_instance()
        assert matmng2.gumbel_tau <= 0.01, f"Fix 1 FAIL: tau should be ~0.001 after freeze, got {matmng2.gumbel_tau}"
        print(f"  gumbel_tau={matmng2.gumbel_tau} (frozen, ~0.001): PASS")

        # Fix 1 check: can still save checkpoint (optim state preserved)
        ckpt_path = trainer2.save_checkpoint(checkpoint_type=CheckpointType.unsharded)
        gumbel_optim_path = os.path.join(str(ckpt_path), "gumbel_optim.pt")
        assert os.path.exists(gumbel_optim_path), "Fix 1 FAIL: gumbel_optim.pt not saved after freeze"
        optim_state = torch.load(gumbel_optim_path, weights_only=False)
        assert len(optim_state["state"]) > 0, "Fix 1 FAIL: gumbel_optim state empty in checkpoint"
        print(f"  Checkpoint saved with gumbel_optim state after freeze: PASS")

        # Fix 1 check: restore into new trainer, verify optim state survives
        cfg3 = TrainConfig(
            run_name="test-freeze-restore",
            precision="amp_bf16",
            model=ModelConfig(
                d_model=128, n_heads=2, n_layers=4, mlp_ratio=4,
                vocab_size=50257, eos_token_id=50256, pad_token_id=50256,
                max_sequence_length=256, init_device=None, init_std=0.02,
            ),
            optimizer=OptimizerConfig(learning_rate=1e-3, betas=[0.9, 0.95]),
            scheduler=SchedulerConfig(name="cosine_with_warmup", t_warmup=5),
            data=DataConfig(
                paths=[data_path], pad_direction=PaddingDirection.right,
                persistent_workers=False, num_workers=0, prefetch_factor=None,
            ),
            tokenizer=TokenizerConfig(identifier="gpt2"),
            save_folder=os.path.join(tmpdir, "ckpt-freeze-restore"),
            save_overwrite=True,
            max_duration=20,
            global_train_batch_size=4,
            device_train_microbatch_size=4,
            matformer_factor=4,
            hmat=HMatConfig(
                enabled=True,
                method="gumbel",
                gumbel_init_scale=1.1,
                gumbel_tau_start=1.0,
                gumbel_tau_end=0.5,
                budget_penalty_lambda=0.001,
                budget_penalty_target=0.5,
                gumbel_freeze_fraction=0.4,
            ),
            load_path=str(ckpt_path),
        )

        trainer3 = build_trainer(cfg3, device)
        trainer3.init_masks()
        trainer3.restore_checkpoint(str(ckpt_path))

        assert trainer3.gumbel_optim is not None, "Fix 1 FAIL: gumbel_optim None after restore"
        restored_state = trainer3.gumbel_optim.state_dict()["state"]
        assert len(restored_state) > 0, "Fix 1 FAIL: gumbel_optim state empty after restore"
        print("  gumbel_optim state restored after freeze+checkpoint: PASS")

        # Verify logits match
        pre_logits = [m.logits.detach().clone() for m in trainer2.gumbel_manager.masks]
        post_logits = [m.logits.detach().clone() for m in trainer3.gumbel_manager.masks]
        for i, (pre, post) in enumerate(zip(pre_logits, post_logits)):
            if not torch.allclose(pre.cpu(), post.cpu(), atol=1e-6):
                print(f"  FAIL: Layer {i} logits don't match after freeze+restore")
                all_pass = False
            else:
                print(f"  Layer {i} logits after freeze+restore: MATCH")

        del trainer2, trainer3
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    if all_pass:
        print("PASS: All warmup + freeze checks passed!")
    else:
        print("FAIL: Some checks failed!")
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    test_warmup_and_freeze()
