"""
Fisher saliency computation for per-dimension importance scoring.

Computes squared-gradient saliency for each MLP hidden dimension across all layers.
Used by F-Mat (Method A) to find optimal heterogeneous per-layer width allocations.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model import Olmo
from ..util import move_to_device

log = logging.getLogger(__name__)


def compute_fisher_saliency(
    model: Olmo,
    dataloader: DataLoader,
    num_batches: int = 128,
    device: torch.device = torch.device("cpu"),
) -> Dict[int, torch.Tensor]:
    """
    Compute per-dimension Fisher saliency scores for each MLP layer.

    For each layer l and hidden dimension d, accumulates:
        saliency[l][d] = sum over batches of (
            ||grad of loss w.r.t. ff_proj.weight[d, :]||^2 +
            ||grad of loss w.r.t. ff_out.weight[:, d]||^2
        )

    Then normalizes per-layer so scores sum to 1.0.

    Args:
        model: A pre-trained Olmo model.
        dataloader: DataLoader yielding batches with "input_ids" key.
        num_batches: Number of calibration batches to accumulate over.
        device: Device to run computation on.

    Returns:
        Dict mapping layer_idx -> Tensor of shape (mlp_hidden_dim,)
        with per-layer normalized saliency scores.
    """
    model.eval()
    # Enable gradients even in eval mode (we need them for saliency)
    for param in model.parameters():
        param.requires_grad_(True)

    n_layers = len(model.transformer.blocks)
    # Use ff_out columns as the saliency dimension — this is the post-activation
    # hidden size and is correct for both standard (mlp_dim) and SwiGLU (mlp_dim/2).
    mlp_dim = model.transformer.blocks[0].ff_out.weight.shape[1]

    saliency = {l: torch.zeros(mlp_dim, device=device) for l in range(n_layers)}

    batches_processed = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        batch = move_to_device(batch, device)
        model.zero_grad()

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # Forward pass
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits

        # Compute cross-entropy loss (next-token prediction)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss.backward()

        # Accumulate per-dimension saliency
        for l, block in enumerate(model.transformer.blocks):
            # ff_proj.weight: (ff_proj_dim, d_model) — for SwiGLU ff_proj_dim = 2*mlp_dim
            grad_proj = block.ff_proj.weight.grad
            # ff_out.weight: (d_model, mlp_dim) — column d is dimension d's output weights
            grad_out = block.ff_out.weight.grad

            if grad_proj is not None and grad_out is not None:
                proj_saliency = (grad_proj ** 2).sum(dim=1)  # (ff_proj_dim,)
                # For SwiGLU: ff_proj_dim = 2*mlp_dim; sum gate + value contributions per dim.
                if proj_saliency.shape[0] != mlp_dim:
                    proj_saliency = proj_saliency.view(-1, mlp_dim).sum(dim=0)
                saliency[l] += proj_saliency + (grad_out ** 2).sum(dim=0)

        batches_processed += 1

    if batches_processed == 0:
        log.warning("No batches processed for Fisher saliency computation")
        return saliency

    log.info(f"Fisher saliency computed over {batches_processed} batches")

    # Normalize per-layer: scores sum to 1.0
    for l in range(n_layers):
        trace = saliency[l].sum()
        if trace > 0:
            saliency[l] = saliency[l] / trace

    return saliency
