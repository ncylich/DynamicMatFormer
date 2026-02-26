"""
Integration test for the three performance optimizations.

Run with:
    torchrun --nproc_per_node=1 scripts/test_optimizations.py

Tests:
  1. bf16 mask cast: mask * h stays in bf16, no promotion to f32
  2. Vectorized budget_loss: matches old loop-based version exactly
  3. Weight view cache: forward output matches at multiple factors, cache reuses views
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
from olmo.hmat.gumbel import GumbelMaskLayer, GumbelMaskManager
from olmo.model import Olmo, MatformerManager
from olmo.optim import build_optimizer, build_scheduler
from olmo.train import Trainer
from olmo.util import barrier, get_global_rank, get_local_rank, seed_all


all_pass = True


def check(name: str, condition: bool, detail: str = ""):
    global all_pass
    if condition:
        print(f"  {name}: PASS" + (f" ({detail})" if detail else ""))
    else:
        print(f"  {name}: FAIL" + (f" ({detail})" if detail else ""))
        all_pass = False


def test_bf16_mask_cast():
    """Opt 1: Verify mask is cast to h.dtype before multiplication."""
    print("\n=== Optimization 1: bf16 mask cast ===")
    device = torch.device("cuda")

    # Simulate what happens in forward pass
    mask_layer = GumbelMaskLayer(256, init_scale=1.1).to(device)
    mask_layer.eval()

    # Get mask (float32)
    mask = mask_layer(tau=1.0, hard=True)
    check("Mask is float32", mask.dtype == torch.float32, str(mask.dtype))

    # Create bf16 hidden states (as they'd be under autocast)
    h = torch.randn(2, 16, 256, device=device, dtype=torch.bfloat16)

    # Old way: no cast — PyTorch promotes to float32
    result_old = h * mask.unsqueeze(0).unsqueeze(0)
    check("Without cast: result promoted to float32", result_old.dtype == torch.float32, str(result_old.dtype))

    # New way: cast first — stays in bf16
    result_new = h * mask.to(dtype=h.dtype).unsqueeze(0).unsqueeze(0)
    check("With cast: result stays bf16", result_new.dtype == torch.bfloat16, str(result_new.dtype))

    # Verify numerical equivalence (within bf16 tolerance)
    result_old_bf16 = result_old.to(torch.bfloat16)
    max_diff = (result_new - result_old_bf16).abs().max().item()
    check("Numerical equivalence", max_diff < 1e-3, f"max_diff={max_diff:.6e}")


def test_vectorized_budget_loss():
    """Opt 2: Verify vectorized budget_loss matches loop-based version."""
    print("\n=== Optimization 2: Vectorized budget_loss ===")
    device = torch.device("cuda")

    manager = GumbelMaskManager(n_layers=8, mlp_dim=512, init_scale=1.1).to(device)

    # Compute with vectorized method (current code)
    target = 0.5
    vectorized_loss = manager.budget_loss(target)

    # Compute with old loop-based method
    total_active = torch.tensor(0.0, device=device)
    for mask_layer in manager.masks:
        total_active = total_active + torch.sigmoid(mask_layer.logits).sum()
    mean_fraction = total_active / (manager.n_layers * manager.mlp_dim)
    loop_loss = torch.abs(mean_fraction - target)

    diff = (vectorized_loss - loop_loss).abs().item()
    check("Exact match with loop version", diff < 1e-7, f"diff={diff:.2e}")

    # Verify gradient flows (use target != 0.5 to avoid abs(0) zero-gradient)
    manager.zero_grad()
    loss2 = manager.budget_loss(0.3)
    loss2.backward()
    grad_sums = [(m.logits.grad.abs().sum().item() if m.logits.grad is not None else -1) for m in manager.masks]
    has_grad = all(g > 0 for g in grad_sums)
    check("Gradient flows through vectorized version", has_grad, f"grad_sums={grad_sums[:3]}...")

    # Test at various targets
    for t in [0.1, 0.3, 0.7, 0.9]:
        v = manager.budget_loss(t)

        total_active = torch.tensor(0.0, device=device)
        for ml in manager.masks:
            total_active = total_active + torch.sigmoid(ml.logits).sum()
        mf = total_active / (manager.n_layers * manager.mlp_dim)
        loop = torch.abs(mf - t)

        d = (v - loop).abs().item()
        check(f"  target={t}", d < 1e-7, f"diff={d:.2e}")


def test_weight_view_cache():
    """Opt 3: Verify weight view cache correctness at multiple factors."""
    print("\n=== Optimization 3: Weight view cache ===")
    device = torch.device("cuda")
    seed_all(42)

    config = ModelConfig(
        d_model=128, n_heads=2, n_layers=4, mlp_ratio=4,
        vocab_size=50257, eos_token_id=50256, pad_token_id=50256,
        max_sequence_length=256, init_device=None, init_std=0.02,
    )
    model = Olmo(config).to(device).eval()

    input_ids = torch.randint(0, 50257, (2, 16), device=device)

    matmng = MatformerManager.get_instance()
    matmng.matformer_factor = 4  # factors: 1, 2, 4

    # Run forward at each factor and verify the cache works
    for factor in [1, 2, 4]:
        matmng.current_factor = factor
        with torch.no_grad():
            out1 = model(input_ids).logits

        # Run again — should use cached views for factor > 1
        with torch.no_grad():
            out2 = model(input_ids).logits

        match = torch.allclose(out1, out2, atol=1e-6)
        check(f"factor={factor}: repeated forward matches", match)

    # Verify cache invalidation: change factor, output should differ from previous
    matmng.current_factor = 2
    with torch.no_grad():
        out_f2 = model(input_ids).logits

    matmng.current_factor = 4
    with torch.no_grad():
        out_f4 = model(input_ids).logits

    differs = not torch.allclose(out_f2, out_f4, atol=1e-6)
    check("Different factors produce different outputs", differs)

    # Verify the block's cache attribute is being used
    block = model.transformer.blocks[0]
    matmng.current_factor = 2
    with torch.no_grad():
        model(input_ids)
    cached_factor = getattr(block, '_cached_factor', None)
    check("Block has _cached_factor after forward", cached_factor == 2, f"cached={cached_factor}")


def test_training_e2e():
    """End-to-end: 10-step training with all optimizations active."""
    print("\n=== End-to-end: 10 steps with gumbel + all optimizations ===")
    device = torch.device("cuda")
    seed_all(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.npy")
        rng = np.random.default_rng(42)
        np.save(data_path, rng.integers(0, 50257, size=50000, dtype=np.uint16))

        cfg = TrainConfig(
            run_name="test-optimizations-e2e",
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
            save_folder=os.path.join(tmpdir, "checkpoints"),
            save_overwrite=True,
            max_duration=10,
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
            ),
        )
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

        with Trainer(
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
        ) as trainer:
            trainer.init_masks()
            init_logits = [m.logits.detach().clone() for m in trainer.gumbel_manager.masks]

            trainer.fit()

            final_logits = [m.logits.detach().clone() for m in trainer.gumbel_manager.masks]
            changed = any(
                not torch.allclose(i.cpu(), f.cpu(), atol=1e-6)
                for i, f in zip(init_logits, final_logits)
            )
            check("Logits changed after training", changed)
            check("Global step = 10", trainer.global_step == 10, f"step={trainer.global_step}")
            check("No NaN in final logits", all(not t.isnan().any() for t in final_logits))


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    barrier()
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    seed_all(42)

    test_bf16_mask_cast()
    test_vectorized_budget_loss()
    test_weight_view_cache()
    test_training_e2e()

    print("\n" + "=" * 60)
    if all_pass:
        print("PASS: All optimization tests passed!")
    else:
        print("FAIL: Some optimization tests failed!")
        sys.exit(1)

    dist.destroy_process_group()
