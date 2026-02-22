"""
Single-device training script for MatFormer-OLMo (no FSDP, no torchrun).

Supports CPU, MPS (Apple Silicon), and single-GPU CUDA training.
Supports all H-Mat modes: uniform, gumbel, fisher_gumbel, topk.

Usage:
    python scripts/train_local.py configs/pile-tiny-hmat-topk.yaml [--overrides]

    # Override any config value with dot notation:
    python scripts/train_local.py configs/pile-tiny-hmat-topk.yaml \
        --max_duration=100 \
        --data.paths=[data/sample.npy] \
        --hmat.method=topk
"""

import logging
import math
import os
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from olmo.config import TrainConfig
from olmo.data import DataCollator, MemMapDataset
from olmo.data.iterable_dataset import IterableDataset
from olmo.hmat.gumbel import GumbelMaskManager
from olmo.hmat.topk import TopKMaskManager
from olmo.model import Activation, MatformerManager, Olmo
from olmo.optim import build_optimizer, build_scheduler
from olmo.util import move_to_device, seed_all

log = logging.getLogger("train_local")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloader(cfg: TrainConfig, device: torch.device) -> torch.utils.data.DataLoader:
    """Build a single-device dataloader (no DistributedSampler)."""
    paths = cfg.data.paths
    if not paths:
        raise ValueError("data.paths is required")

    dataset = MemMapDataset(*paths, chunk_size=cfg.model.max_sequence_length)
    collator = DataCollator(pad_direction=cfg.data.pad_direction, pad_token_id=cfg.model.pad_token_id)

    work_dir = Path(cfg.save_folder) / "train_data"
    work_dir.mkdir(parents=True, exist_ok=True)

    iterable = IterableDataset(
        dataset,
        seed=cfg.seed,
        shuffle=True,
        drop_last=cfg.data.drop_last,
        max_examples=cfg.device_train_batch_size * cfg.max_duration,
        world_size=1,
        rank=0,
        work_dir=work_dir,
    )
    return torch.utils.data.DataLoader(
        iterable,
        batch_size=cfg.device_train_batch_size,
        drop_last=cfg.data.drop_last,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and device.type == "cuda",
        prefetch_factor=None if cfg.data.num_workers == 0 else cfg.data.prefetch_factor,
        persistent_workers=False if cfg.data.num_workers == 0 else cfg.data.persistent_workers,
    )


def build_eval_dataloader(
    cfg: TrainConfig, data_paths: List[str], batch_size: int,
) -> torch.utils.data.DataLoader:
    dataset = MemMapDataset(*data_paths, chunk_size=cfg.model.max_sequence_length)
    collator = DataCollator(pad_direction=cfg.data.pad_direction, pad_token_id=cfg.model.pad_token_id)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )


def evaluate(
    model: Olmo,
    dataloader: torch.utils.data.DataLoader,
    num_batches: int,
    device: torch.device,
    factor: int,
    label: str,
) -> Dict[str, float]:
    model.eval()
    matmng = MatformerManager.get_instance()
    matmng.current_factor = factor

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            batch = move_to_device(batch, device)
            input_ids = batch["input_ids"]
            output = model(input_ids, attention_mask=batch.get("attention_mask"))
            logits = output.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    model.train()
    return {f"{label}/CrossEntropyLoss 1/{factor}": avg_loss, f"{label}/Perplexity 1/{factor}": ppl}


class LocalTrainer:
    def __init__(self, cfg: TrainConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.global_step = 0

        # Model.
        cfg.model.precision = cfg.precision
        cfg.model.init_device = "cpu"
        log.info("Initializing model...")
        self.model = Olmo(cfg.model)
        log.info(f"Total parameters: {self.model.num_params():,d}")
        self.model.to(device)

        # Optimizer and scheduler.
        self.optim = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optim)

        # H-Mat mask managers.
        self.gumbel_manager: Optional[GumbelMaskManager] = None
        self.gumbel_optim: Optional[torch.optim.Optimizer] = None
        self.topk_manager: Optional[TopKMaskManager] = None
        self.topk_optim: Optional[torch.optim.Optimizer] = None

        self._init_hmat()

    def _init_hmat(self):
        if not self.cfg.hmat.enabled:
            return

        act = Activation.build(self.cfg.model)
        mlp_dim = int(self.cfg.model.mlp_ratio * self.cfg.model.d_model * act.output_multiplier)
        matmng = MatformerManager.get_instance()

        if self.cfg.hmat.method == "gumbel":
            self.gumbel_manager = GumbelMaskManager(
                n_layers=self.cfg.model.n_layers,
                mlp_dim=mlp_dim,
                init_scale=self.cfg.hmat.gumbel_init_scale,
                learnable=True,
            ).to(self.device)
            matmng.mode = "gumbel"
            matmng.gumbel_masks = self.gumbel_manager
            matmng.gumbel_tau = self.cfg.hmat.gumbel_tau_start
            self.gumbel_optim = torch.optim.AdamW(
                self.gumbel_manager.parameters(),
                lr=self.cfg.optimizer.learning_rate,
                weight_decay=0.0,
                betas=tuple(self.cfg.optimizer.betas),
            )
            log.info(f"Gumbel mode: mlp_dim={mlp_dim}, lambda={self.cfg.hmat.budget_penalty_lambda}")

        elif self.cfg.hmat.method == "topk":
            self.topk_manager = TopKMaskManager(
                n_layers=self.cfg.model.n_layers,
                mlp_dim=mlp_dim,
                init_scale=self.cfg.hmat.gumbel_init_scale,
            ).to(self.device)
            matmng.mode = "topk"
            matmng.topk_masks = self.topk_manager
            matmng.gumbel_tau = self.cfg.hmat.gumbel_tau_start
            self.topk_optim = torch.optim.AdamW(
                self.topk_manager.parameters(),
                lr=self.cfg.optimizer.learning_rate,
                weight_decay=0.0,
                betas=tuple(self.cfg.optimizer.betas),
            )
            log.info(f"TopK mode: mlp_dim={mlp_dim}, init_scale={self.cfg.hmat.gumbel_init_scale}")

    def _anneal_tau(self):
        """Update temperature for gumbel/topk modes."""
        if not self.cfg.hmat.enabled:
            return
        if self.cfg.hmat.method not in ("gumbel", "topk"):
            return
        matmng = MatformerManager.get_instance()
        anneal_steps = self.cfg.hmat.gumbel_tau_anneal_steps or self.cfg.max_duration
        progress = min(1.0, self.global_step / anneal_steps)
        tau_start = self.cfg.hmat.gumbel_tau_start
        tau_end = self.cfg.hmat.gumbel_tau_end
        tau = tau_start * (tau_end / tau_start) ** progress
        matmng.gumbel_tau = tau

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        self.optim.zero_grad(set_to_none=True)
        if self.gumbel_optim is not None:
            self.gumbel_optim.zero_grad(set_to_none=True)
        if self.topk_optim is not None:
            self.topk_optim.zero_grad(set_to_none=True)

        batch = move_to_device(batch, self.device)
        self._anneal_tau()

        matmng = MatformerManager.get_instance()
        losses = []

        if self.cfg.matformer_factor > 1:
            matmng.current_factor = 1
            iters = int(math.log2(self.cfg.matformer_factor)) + 1
            for i in range(iters):
                loss, ce_loss_val = self._forward_backward(batch)
                losses.append((matmng.current_factor, ce_loss_val))
                matmng.current_factor *= 2
        else:
            loss, ce_loss_val = self._forward_backward(batch)
            losses.append((1, ce_loss_val))

        # Clip gradients.
        if self.cfg.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

        # Optimizer steps.
        self.optim.step()
        self.scheduler.step()
        if self.gumbel_optim is not None:
            self.gumbel_optim.step()
        if self.topk_optim is not None:
            self.topk_optim.step()

        # Build metrics.
        metrics = {}
        for factor, ce_val in losses:
            metrics[f"train/CrossEntropyLoss 1/{factor}"] = ce_val
            metrics[f"train/Perplexity 1/{factor}"] = math.exp(ce_val) if ce_val < 20 else float("inf")

        if self.topk_manager is not None:
            metrics["topk/tau"] = matmng.gumbel_tau
        if self.gumbel_manager is not None:
            metrics["gumbel/tau"] = matmng.gumbel_tau
            if self.cfg.hmat.method == "gumbel":
                bp = self.gumbel_manager.budget_loss(self.cfg.hmat.budget_penalty_target)
                metrics["gumbel/budget_penalty"] = bp.item()

        return metrics

    def _forward_backward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """Single forward-backward pass at the current factor."""
        with torch.autocast(
            self.device.type,
            enabled=self.cfg.precision is not None and self.cfg.precision != "fp32",
            dtype=self.cfg.autocast_precision if self.cfg.precision and self.cfg.precision != "fp32" else torch.float32,
        ):
            logits = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            ).logits
            logits_for_loss = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["input_ids"][..., 1:].contiguous().view(-1)
            ce_loss = F.cross_entropy(logits_for_loss, labels, ignore_index=-100)

            loss = ce_loss

            # Gumbel budget penalty.
            if (self.gumbel_manager is not None and self.cfg.hmat.method == "gumbel"):
                bp = self.gumbel_manager.budget_loss(self.cfg.hmat.budget_penalty_target)
                loss = loss + self.cfg.hmat.budget_penalty_lambda * bp

            # Z-loss.
            if self.cfg.softmax_auxiliary_loss:
                z_sq = logits.logsumexp(-1).pow(2).mean()
                loss = loss + 1e-4 * z_sq

        loss.backward()
        return loss, ce_loss.detach().item()

    def save_checkpoint(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        self.cfg.save(path / "config.yaml")
        torch.save({"global_step": self.global_step}, path / "other.pt")
        if self.gumbel_manager is not None:
            torch.save(self.gumbel_manager.state_dict(), path / "gumbel.pt")
        if self.topk_manager is not None:
            torch.save(self.topk_manager.state_dict(), path / "topk.pt")
        log.info(f"Checkpoint saved to {path}")


def main(cfg: TrainConfig) -> None:
    device = pick_device()
    log.info(f"Using device: {device}")

    # Fill batch size fields (normally done by distributed train.py).
    cfg.device_train_batch_size = cfg.global_train_batch_size
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size

    seed_all(cfg.seed)

    # Build dataloader.
    log.info("Building dataloader...")
    train_loader = build_dataloader(cfg, device)

    # Build eval dataloaders.
    eval_loaders = {}
    for eval_cfg in cfg.evaluators:
        if eval_cfg.data.paths:
            eval_loaders[eval_cfg.label] = build_eval_dataloader(
                cfg, eval_cfg.data.paths,
                batch_size=eval_cfg.device_eval_batch_size or cfg.device_eval_batch_size,
            )

    # Create trainer.
    trainer = LocalTrainer(cfg, device)

    # Save folder.
    save_dir = Path(cfg.save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(save_dir / "config.yaml")

    # Training loop.
    log.info(f"Starting training for {cfg.max_duration} steps...")
    start_time = time.time()

    for batch in train_loader:
        trainer.global_step += 1

        metrics = trainer.train_step(batch)

        # Console logging.
        if trainer.global_step % cfg.console_log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = trainer.global_step / elapsed
            summary = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if "Perplexity" in k)
            log.info(f"[step {trainer.global_step}/{cfg.max_duration}] {steps_per_sec:.2f} steps/s | {summary}")

        # Evaluation.
        if trainer.global_step % cfg.eval_interval == 0:
            matmng = MatformerManager.get_instance()
            for label, loader in eval_loaders.items():
                num_batches = cfg.eval_subset_num_batches
                if cfg.matformer_factor > 1:
                    for i in range(int(math.log2(cfg.matformer_factor)) + 1):
                        factor = 2 ** i
                        eval_metrics = evaluate(
                            trainer.model, loader, num_batches, device, factor, label,
                        )
                        ppl_key = f"{label}/Perplexity 1/{factor}"
                        log.info(f"  [eval] {ppl_key} = {eval_metrics.get(ppl_key, 0):.2f}")
                else:
                    eval_metrics = evaluate(trainer.model, loader, num_batches, device, 1, label)
                    ppl_key = f"{label}/Perplexity 1/1"
                    log.info(f"  [eval] {ppl_key} = {eval_metrics.get(ppl_key, 0):.2f}")
            trainer.model.train()

        # Checkpoint.
        if (
            cfg.save_interval_unsharded is not None
            and trainer.global_step % cfg.save_interval_unsharded == 0
        ):
            ckpt_path = save_dir / f"step{trainer.global_step}-unsharded"
            trainer.save_checkpoint(ckpt_path)

        if trainer.global_step >= cfg.max_duration:
            break

    # Final checkpoint.
    ckpt_path = save_dir / f"step{trainer.global_step}-unsharded"
    trainer.save_checkpoint(ckpt_path)

    elapsed = time.time() - start_time
    log.info(f"Training complete: {trainer.global_step} steps in {elapsed:.1f}s ({trainer.global_step/elapsed:.2f} steps/s)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from olmo.util import clean_opt

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        print(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")
        sys.exit(1)

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
