# Implementation Plan: Heterogeneous Matryoshka Networks (H-Mat) in MatFormer-OLMo

## Overview

This plan transforms the existing MatFormer-OLMo codebase from **uniform** MLP slicing (same factor at every layer) to **heterogeneous** layer-wise width allocation. We implement both methods from the research proposal:

- **Method A (F-Mat):** Post-training Fisher-based analysis to find optimal per-layer widths for a pre-trained uniform MatFormer model. This is primarily an **analytical and diagnostic** phase -- it validates the U-shape sensitivity hypothesis and motivates Method B, but is not expected to produce large performance gains because the model was trained under uniform slicing assumptions.
- **Method B (Learnable H-Mat):** In-training Gumbel-Softmax gates that learn per-layer dimension importance from scratch. This is where we expect **real performance improvements** over uniform MatFormer, since the model can organize its representations around heterogeneous widths from the start of training.

All changes build on the existing `MatformerManager` / `OlmoSequentialBlock` architecture. The goal is minimal, surgical modifications to the codebase.

---

## Current Architecture Summary

| Component | File | Key Lines | Role |
|-----------|------|-----------|------|
| `MatformerManager` | `olmo/model.py:43-57` | Singleton tracking `current_factor` | Global state for uniform slicing |
| `OlmoSequentialBlock.forward` | `olmo/model.py:429-439` | `k = n / factor` weight slicing | Applies uniform MLP truncation |
| `Trainer.train_step` | `olmo/train.py:604-618` | Loops over `{1, 2, 4, 8}` factors | Multiple forward-backward per batch |
| `Trainer.eval` | `olmo/train.py:757-791` | Same factor loop for eval | Per-granularity evaluation |
| `TrainConfig.matformer_factor` | `olmo/config.py:550` | Single int config | Controls uniform factor |

**Current MLP slicing logic** (uniform):
```python
# olmo/model.py:429-439
n = self.ff_proj.weight.shape[0]        # full MLP hidden dim
k = int(n / self.matmng.current_factor) # SAME k for ALL layers
w_proj = self.ff_proj.weight[:k]
```

**What changes:** Instead of a single global `current_factor` producing the same `k` for every layer, we introduce per-layer width allocations `k_l` for each layer `l`.

---

## Phase 0: Preparatory Refactoring ✅ COMPLETED

### 0.1 Extend `MatformerManager` to Support Per-Layer Widths

**File:** `olmo/model.py:43-57`

Replace the scalar `current_factor` with a richer state that supports both uniform and heterogeneous modes.

```python
class MatformerManager:
    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.current_factor = 1            # Backward-compat: uniform factor
        self.layer_factors = None          # Dict[int, int] or None: per-layer factors
        self.mode = "uniform"              # "uniform" | "heterogeneous"
        self.gumbel_masks = None           # For Method B: per-layer soft masks

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def get_factor_for_layer(self, layer_idx: int) -> int:
        """Return the slicing factor for a specific layer."""
        if self.mode == "uniform" or self.layer_factors is None:
            return self.current_factor
        return self.layer_factors.get(layer_idx, self.current_factor)
```

### 0.2 Thread `layer_idx` Through `OlmoSequentialBlock`

**File:** `olmo/model.py`

Each block needs to know its layer index. Currently blocks are built in `Olmo.__init__` (line ~620) via a ModuleList. We add `layer_idx` as a constructor argument:

```python
# In OlmoSequentialBlock.__init__:
def __init__(self, config: ModelConfig, layer_idx: int = 0):
    super().__init__(config)
    self.layer_idx = layer_idx
    self.matmng = MatformerManager.get_instance()
    # ... rest unchanged
```

Update `Olmo.__init__` where blocks are created:
```python
# In Olmo.__init__:
blocks = nn.ModuleList([OlmoBlock.build(config, layer_idx=i) for i in range(config.n_layers)])
```

### 0.3 Update Forward Pass to Use Per-Layer Factor

**File:** `olmo/model.py:429-439`

```python
# In OlmoSequentialBlock.forward:
factor = self.matmng.get_factor_for_layer(self.layer_idx)
if factor == 1:
    x = x + self.dropout(self.ff_out(self.act(self.ff_proj(self.ff_norm(x)))))
else:
    n = self.ff_proj.weight.shape[0]
    k = int(n / factor)
    w_proj = self.ff_proj.weight[:k]
    b_proj = self.ff_proj.bias[:k]
    w_out = self.ff_out.weight[:, :k]
    b_out = self.ff_out.bias
    x = x + self.dropout(F.linear(self.act(F.linear(self.ff_norm(x), w_proj, b_proj)), w_out, b_out))
```

### 0.4 Extend Config

**File:** `olmo/config.py`

```python
@dataclass
class HMatConfig(BaseConfig):
    """Configuration for Heterogeneous Matryoshka (H-Mat)."""
    enabled: bool = False
    method: str = "fisher"           # "fisher" | "gumbel"

    # Method A (Fisher) settings
    calibration_batches: int = 128   # Number of batches for Fisher estimation
    budget_ratio: float = 0.25       # Target parameter budget as fraction of full model

    # Method B (Gumbel) settings
    gumbel_tau_start: float = 2.0    # Initial Gumbel temperature
    gumbel_tau_end: float = 0.1      # Final Gumbel temperature (annealed)
    gumbel_tau_anneal_steps: int = 0 # Steps to anneal (0 = use max_duration)
    budget_penalty_lambda: float = 0.01  # L1 penalty coefficient on mask sum
    budget_penalty_target: float = 0.5   # Target fraction of dimensions active

@dataclass
class TrainConfig(BaseConfig):
    # ... existing fields ...
    matformer_factor: int = 1
    hmat: HMatConfig = field(default_factory=HMatConfig)
```

### Tests for Phase 0

**File:** `tests/test_hmat_basic.py`

1. **test_matformer_manager_uniform_backward_compat:** Verify that with `mode="uniform"`, `get_factor_for_layer(i)` returns `current_factor` for all `i`.
2. **test_matformer_manager_heterogeneous:** Set `layer_factors = {0: 1, 1: 4, 2: 2}` and verify correct per-layer factor retrieval.
3. **test_layer_idx_assignment:** Build a small `Olmo` model and verify each block has the correct `layer_idx`.
4. **test_forward_with_layer_factors:** Run a forward pass with heterogeneous factors; verify output shape is correct and no errors.
5. **test_backward_with_layer_factors:** Run forward+backward with heterogeneous factors; verify all parameters get gradients.
6. **test_uniform_unchanged:** Ensure existing uniform MatFormer behavior is completely unchanged when `hmat.enabled = False`.

---

## Phase 1: Method A -- Post-Training Fisher Saliency (F-Mat) ✅ COMPLETED

### Goals

Phase 1 is an **analysis and motivation** phase, not a performance phase. Because F-Mat operates on a model that was *trained with uniform slicing*, its weights were optimized under the assumption that every layer gets the same width at each granularity. Applying a heterogeneous allocation at eval time introduces a train-inference mismatch, which fundamentally limits the gains.

**What we expect from Phase 1:**
- **Empirical validation of the U-shape hypothesis:** Fisher saliency scores should reveal that early and late layers are significantly more sensitive to width reduction than intermediate layers. This is the core scientific claim that justifies the entire H-Mat approach.
- **Diagnostic tooling:** Saliency heatmaps and per-layer sensitivity curves that inform which layers need protection and which are compressible.
- **Analytical baseline:** A concrete comparison point for Phase 2. If F-Mat (with its train-inference mismatch) can show *any* improvement over uniform slicing, it strongly motivates the Gumbel approach.
- **Marginal or no perplexity improvement** over uniform slicing at the same budget. The model cannot adapt its weights to the new allocation, so gains are limited to "smarter selection over fixed representations."

### Results (Tiny 17.7M model, 2700 steps on Pile, matformer_factor=8)

**Fisher Saliency (100 calibration batches):**
```
Per-Layer Sensitivity (top-1/8 concentration):
  Layer  0: 0.37  ← early: saliency spread across all dims (needs full width)
  Layer  1: 0.35  ← early: saliency spread across all dims (needs full width)
  Layer  2: 0.91  ← late: saliency concentrated in top dims (compressible)
  Layer  3: 0.85  ← late: saliency concentrated in top dims (compressible)
```
- Not a clean U-shape (likely due to only 4 layers — no "middle" to compress).
- Core finding confirmed: **layers are not equally sensitive to width reduction**.

**F-Mat Budget Allocation (25% budget):**
| Layer | Uniform 1/4 | F-Mat 25% |
|-------|-------------|-----------|
| 0 | factor=4 | factor=4 |
| 1 | factor=4 | **factor=2** (2x wider) |
| 2 | factor=4 | **factor=8** (2x narrower) |
| 3 | factor=4 | **factor=8** (2x narrower) |

**Perplexity Comparison (100 eval batches on Pile validation):**
| Config | MLP Dims | Perplexity |
|--------|----------|------------|
| Full model (1/1) | 8192 | 381.1 |
| Uniform 1/2 | 4096 | 410.5 |
| F-Mat 50% budget | 4096 | 417.5 (+7.0 worse) |
| Uniform 1/4 | 2048 | 445.7 |
| F-Mat 25% budget | 2048 | 453.7 (+8.0 worse) |
| Uniform 1/8 | 1024 | 489.1 |

- F-Mat slightly worse than uniform at matching budgets — **confirms train-inference mismatch prediction**.
- Motivates Phase 2 (Gumbel): the model must learn heterogeneous allocation from scratch to benefit.

### 1.1 Fisher Score Computation

**New file:** `olmo/hmat/fisher.py`

This module computes per-dimension saliency scores across all layers of a pre-trained model.

**Algorithm:**
1. Load a pre-trained uniform MatFormer checkpoint.
2. Run forward+backward on a calibration dataset (e.g., 128 batches from Pile validation).
3. For each MLP layer `l`, for each hidden dimension `d`:
   - Accumulate the squared gradient of the loss w.r.t. the `d`-th row of `ff_proj.weight` and the `d`-th column of `ff_out.weight`.
   - `saliency[l][d] = (grad_ff_proj_row_d ** 2 + grad_ff_out_col_d ** 2).sum()`
4. Normalize per-layer: `saliency_norm[l][d] = saliency[l][d] / sum(saliency[l])`
5. Return the full `saliency_norm` matrix of shape `(n_layers, mlp_hidden_dim)`.

```python
def compute_fisher_saliency(
    model: Olmo,
    dataloader: DataLoader,
    num_batches: int = 128,
    device: torch.device = torch.device("cuda"),
) -> Dict[int, torch.Tensor]:
    """
    Compute per-dimension Fisher saliency scores for each MLP layer.

    Returns:
        Dict mapping layer_idx -> Tensor of shape (mlp_hidden_dim,)
        with globally normalized saliency scores.
    """
    model.eval()
    n_layers = len(model.transformer.blocks)
    mlp_dim = model.transformer.blocks[0].ff_proj.weight.shape[0]

    # Accumulate squared gradients
    saliency = {l: torch.zeros(mlp_dim, device=device) for l in range(n_layers)}

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        batch = move_to_device(batch, device)
        model.zero_grad()

        # Forward pass
        logits = model(batch["input_ids"], attention_mask=batch.get("attention_mask")).logits
        loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)),
                               batch["input_ids"][..., 1:].reshape(-1))
        loss.backward()

        # Accumulate per-dimension saliency
        for l, block in enumerate(model.transformer.blocks):
            # ff_proj: (mlp_dim, d_model) -- each row is one hidden dimension
            grad_proj = block.ff_proj.weight.grad  # (mlp_dim, d_model)
            # ff_out: (d_model, mlp_dim) -- each column is one hidden dimension
            grad_out = block.ff_out.weight.grad    # (d_model, mlp_dim)

            saliency[l] += (grad_proj ** 2).sum(dim=1) + (grad_out ** 2).sum(dim=0)

    # Normalize: divide each layer's scores by the trace (sum of all scores in that layer)
    for l in range(n_layers):
        trace = saliency[l].sum()
        if trace > 0:
            saliency[l] = saliency[l] / trace

    return saliency
```

### 1.2 Budget Knapsack Solver

**New file:** `olmo/hmat/knapsack.py`

Given saliency scores and a target parameter budget, find the optimal per-layer widths.

**Constraints:**
- Layer widths must be powers-of-2 fractions of the full width (to maintain MatFormer nesting compatibility): `{n, n/2, n/4, n/8, ...}`
- Total parameters across all layers must not exceed the budget.
- Each layer's width is chosen from the set of allowed widths.

```python
def solve_budget_allocation(
    saliency: Dict[int, torch.Tensor],
    budget_ratio: float,
    allowed_factors: List[int],  # e.g., [1, 2, 4, 8]
) -> Dict[int, int]:
    """
    Solve the multi-choice knapsack problem for per-layer width allocation.

    Args:
        saliency: Per-layer saliency scores from compute_fisher_saliency.
        budget_ratio: Target fraction of total parameters to use.
        allowed_factors: List of allowed slicing factors (must be powers of 2).

    Returns:
        Dict mapping layer_idx -> chosen factor for that layer.
    """
    n_layers = len(saliency)
    mlp_dim = len(next(iter(saliency.values())))

    # Total budget in "MLP dimension units"
    total_budget = int(budget_ratio * n_layers * mlp_dim)

    # For each layer and each allowed factor, compute:
    #   - cost: number of dimensions used = mlp_dim / factor
    #   - value: sum of saliency scores for the retained dimensions
    choices = {}
    for l in range(n_layers):
        layer_choices = []
        for factor in allowed_factors:
            k = mlp_dim // factor
            value = saliency[l][:k].sum().item()  # Top-k dims (already ordered by index)
            cost = k
            layer_choices.append((factor, cost, value))
        choices[l] = layer_choices

    # Dynamic programming knapsack
    # dp[l][b] = max saliency achievable using layers 0..l with budget b
    dp = [[0.0] * (total_budget + 1) for _ in range(n_layers + 1)]
    choice_trace = [[0] * (total_budget + 1) for _ in range(n_layers)]

    for l in range(n_layers):
        for b in range(total_budget + 1):
            best_val = -1.0
            best_factor = allowed_factors[0]
            for factor, cost, value in choices[l]:
                if cost <= b and dp[l][b - cost] + value > best_val:
                    best_val = dp[l][b - cost] + value
                    best_factor = factor
            dp[l + 1][b] = best_val
            choice_trace[l][b] = best_factor

    # Traceback
    result = {}
    remaining = total_budget
    for l in range(n_layers - 1, -1, -1):
        factor = choice_trace[l][remaining]
        result[l] = factor
        remaining -= mlp_dim // factor

    return result
```

### 1.3 F-Mat Analysis Script

**New file:** `scripts/compute_fmat.py`

End-to-end script that:
1. Loads a pre-trained uniform MatFormer checkpoint.
2. Computes Fisher saliency scores on calibration data.
3. Solves the knapsack problem for multiple budget ratios.
4. Outputs a JSON file mapping `{budget_ratio: {layer_idx: factor}}`.
5. Prints a summary table and the U-shaped sensitivity visualization.

```bash
# Usage:
python scripts/compute_fmat.py \
    --checkpoint /path/to/uniform-matformer-checkpoint \
    --calibration_data /path/to/pile-validation/*.npy \
    --budget_ratios 0.125,0.25,0.5 \
    --output fmat_allocations.json
```

### 1.4 Heterogeneous Evaluation Script

**New file:** `scripts/eval_hmat.py`

Loads a pre-trained model, applies the F-Mat allocations to `MatformerManager.layer_factors`, and runs standard evaluation (perplexity + downstream tasks) to compare:
- Uniform 1/4 slice vs. Heterogeneous 1/4-budget slice
- Uniform 1/8 slice vs. Heterogeneous 1/8-budget slice

### Tests for Phase 1

**File:** `tests/test_fmat.py`

1. **test_fisher_saliency_shape:** Verify output has correct shape `(n_layers, mlp_dim)`.
2. **test_fisher_saliency_normalized:** Verify each layer's scores sum to 1.0.
3. **test_fisher_saliency_nonzero:** Verify scores are non-negative and at least some are nonzero.
4. **test_knapsack_budget_respected:** Verify the total parameters of the allocation <= budget.
5. **test_knapsack_trivial:** With budget_ratio=1.0, all layers should get factor=1.
6. **test_knapsack_minimal:** With budget_ratio close to 0, all layers should get the maximum factor.
7. **test_knapsack_heterogeneous:** With a known synthetic saliency distribution (high at ends, low in middle), verify that end layers get lower factors (more width) and middle layers get higher factors (less width) -- i.e., the U-shape hypothesis.
8. **test_fmat_end_to_end:** Load the test fixture model, run Fisher computation with 1 calibration batch, solve knapsack, apply to manager, run forward pass -- verify no errors and output shape is correct.

---

## Phase 2: Method B -- In-Training Gumbel-Softmax Masking (Learnable H-Mat)

### Goals

Phase 2 is the **primary performance phase** of this research. Unlike F-Mat (which retrofits heterogeneous allocation onto a uniformly-trained model), the Gumbel approach trains the model from scratch with learnable per-layer gates. The model can organize its representations around non-uniform widths from the very beginning of training -- there is no train-inference mismatch.

**What we expect from Phase 2:**
- **Measurable perplexity improvement** over uniform MatFormer at the same parameter budget. The proposal hypothesizes recovering 30-40% of the perplexity gap between the uniform sub-model and the full model. Even a more conservative 10-20% recovery would be a meaningful result.
- **Learned heterogeneous allocations that reflect the U-shape:** The Gumbel logits should converge to wider widths at early/late layers and narrower widths at intermediate layers, independently confirming the sensitivity pattern observed in Phase 1's Fisher analysis.
- **A strictly superior Pareto frontier:** For any given parameter budget, the Gumbel-learned allocation should match or beat the uniform allocation on perplexity and downstream task accuracy.
- **Downstream task improvements:** Better perplexity should translate to improved accuracy on zero-shot benchmarks (PIQA, HellaSwag, etc.), particularly for aggressive compression ratios (1/4, 1/8 budgets) where uniform slicing degrades the most.

### 2.1 Gumbel Mask Module

**New file:** `olmo/hmat/gumbel.py`

A learnable module that produces differentiable per-dimension masks for each layer's MLP.

```python
class GumbelMaskLayer(nn.Module):
    """
    Learnable per-dimension importance gate for a single MLP layer.
    Uses Gumbel-Softmax to produce differentiable binary masks.
    """

    def __init__(self, mlp_dim: int):
        super().__init__()
        # Learnable logits: one per MLP hidden dimension
        self.logits = nn.Parameter(torch.zeros(mlp_dim))

    def forward(self, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Returns a soft mask of shape (mlp_dim,) with values in [0, 1].

        Args:
            tau: Gumbel-Softmax temperature (lower = more discrete).
            hard: If True, use straight-through estimator for hard masks.
        """
        if self.training:
            # Sample Gumbel noise
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(self.logits).clamp(1e-8, 1.0)
            ))
            # Soft sigmoid with temperature
            noisy_logits = (self.logits + gumbel_noise) / tau
            mask = torch.sigmoid(noisy_logits)

            if hard:
                # Straight-through estimator
                hard_mask = (mask > 0.5).float()
                mask = hard_mask - mask.detach() + mask
        else:
            # Deterministic at eval time
            mask = (self.logits > 0).float()

        return mask


class GumbelMaskManager(nn.Module):
    """
    Manages Gumbel masks for all layers in the model.
    """

    def __init__(self, n_layers: int, mlp_dim: int):
        super().__init__()
        self.masks = nn.ModuleList([GumbelMaskLayer(mlp_dim) for _ in range(n_layers)])
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim

    def forward(self, layer_idx: int, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        return self.masks[layer_idx](tau=tau, hard=hard)

    def budget_penalty(self) -> torch.Tensor:
        """
        L1 penalty: sum of all mask activations across all layers.
        Drives redundant dimensions toward 0.
        """
        total = sum(mask.logits.sigmoid().sum() for mask in self.masks)
        return total / (self.n_layers * self.mlp_dim)

    def get_layer_widths(self, allowed_factors: List[int] = None) -> Dict[int, int]:
        """
        At eval time, convert learned logits into discrete per-layer widths.
        Count active dimensions (logits > 0) per layer and snap to nearest
        allowed MatFormer factor.
        """
        widths = {}
        for l, mask_layer in enumerate(self.masks):
            active = (mask_layer.logits > 0).sum().item()
            ratio = active / self.mlp_dim
            # Snap to nearest allowed factor
            if allowed_factors:
                best_factor = min(allowed_factors,
                                  key=lambda f: abs(1.0/f - ratio))
            else:
                best_factor = max(1, round(1.0 / ratio)) if ratio > 0 else self.mlp_dim
            widths[l] = best_factor
        return widths
```

### 2.2 Integrate Gumbel Masks into Forward Pass

**File:** `olmo/model.py` -- `OlmoSequentialBlock.forward`

When Gumbel mode is active, the forward pass multiplies MLP activations by the soft mask instead of hard slicing:

```python
# In OlmoSequentialBlock.forward:
if self.matmng.mode == "gumbel" and self.matmng.gumbel_masks is not None:
    # Method B: Soft masking
    tau = self.matmng.gumbel_tau
    mask = self.matmng.gumbel_masks(self.layer_idx, tau=tau, hard=not self.training)
    h = self.ff_proj(self.ff_norm(x))
    h = self.act(h)
    h = h * mask.unsqueeze(0).unsqueeze(0)  # Broadcast: (1, 1, mlp_dim)
    x = x + self.dropout(self.ff_out(h))
elif factor == 1:
    x = x + self.dropout(self.ff_out(self.act(self.ff_proj(self.ff_norm(x)))))
else:
    # Existing hard-slicing logic (uniform or heterogeneous)
    ...
```

### 2.3 Integrate Gumbel Training into `Trainer`

**File:** `olmo/train.py`

Changes to the training loop:

1. **Initialization:** Create `GumbelMaskManager` and register its parameters with the optimizer.
2. **Temperature Annealing:** Compute current `tau` based on global step and anneal schedule.
3. **Budget Penalty:** Add the L1 budget penalty to the loss at each training step.
4. **Logging:** Log mask statistics (mean activation per layer, effective width, budget penalty).

```python
# In Trainer.__init__ or Trainer.fit:
if self.cfg.hmat.enabled and self.cfg.hmat.method == "gumbel":
    self.gumbel_manager = GumbelMaskManager(
        n_layers=self.cfg.model.n_layers,
        mlp_dim=self.cfg.model.mlp_ratio * self.cfg.model.d_model,
    ).to(self.device)

    # Register with optimizer
    self.optim.add_param_group({
        "params": self.gumbel_manager.parameters(),
        "lr": self.cfg.optimizer.learning_rate,  # Same LR or separate
        "weight_decay": 0.0,  # No weight decay on mask logits
    })

    # Attach to MatformerManager
    matmng = MatformerManager.get_instance()
    matmng.mode = "gumbel"
    matmng.gumbel_masks = self.gumbel_manager

# In train_batch, after computing ce_loss:
if self.cfg.hmat.enabled and self.cfg.hmat.method == "gumbel":
    budget_penalty = self.gumbel_manager.budget_penalty()
    target = self.cfg.hmat.budget_penalty_target
    # Penalize deviation from target budget
    loss = ce_loss + self.cfg.hmat.budget_penalty_lambda * (budget_penalty - target).abs()
```

**Temperature annealing** (compute at each step):
```python
# In train_step, before calling train_batch:
if self.cfg.hmat.enabled and self.cfg.hmat.method == "gumbel":
    anneal_steps = self.cfg.hmat.gumbel_tau_anneal_steps or self.cfg.max_duration
    progress = min(1.0, self.global_step / anneal_steps)
    tau = self.cfg.hmat.gumbel_tau_start * (
        self.cfg.hmat.gumbel_tau_end / self.cfg.hmat.gumbel_tau_start
    ) ** progress
    matmng = MatformerManager.get_instance()
    matmng.gumbel_tau = tau
```

### 2.4 Checkpoint Gumbel State

**File:** `olmo/train.py` -- checkpoint save/load

Add `gumbel_manager.state_dict()` to the checkpoint so that mask logits are preserved across restarts.

### Tests for Phase 2

**File:** `tests/test_gumbel.py`

1. **test_gumbel_mask_shape:** Verify mask output has shape `(mlp_dim,)`.
2. **test_gumbel_mask_range:** Verify mask values are in `[0, 1]` during training.
3. **test_gumbel_mask_hard_eval:** Verify mask is binary `{0, 1}` during eval.
4. **test_gumbel_temperature_effect:** High tau -> uniform ~0.5 mask; low tau -> near-binary mask.
5. **test_gumbel_budget_penalty:** Verify penalty is a scalar and changes with logits.
6. **test_gumbel_forward_pass:** Run forward pass with Gumbel masking; verify output shape.
7. **test_gumbel_backward_pass:** Run forward+backward; verify gradients flow to both model params AND mask logits.
8. **test_gumbel_gradient_to_logits:** Specifically verify `gumbel_manager.masks[l].logits.grad` is not None after backward.
9. **test_gumbel_get_layer_widths:** Set known logits (half positive, half negative) and verify `get_layer_widths()` returns expected factors.
10. **test_gumbel_tau_annealing:** Verify temperature schedule computes correct tau at start, middle, and end of training.
11. **test_gumbel_training_step:** Integration test: run a single training step with Gumbel mode enabled on the test fixture model; verify metrics contain budget penalty.

---

## Phase 3: Integration, Evaluation & Comparison

### 3.1 Unified Evaluation Framework

**New file:** `scripts/eval_comparison.py`

A single evaluation script that runs all three configurations head-to-head for a given parameter budget:
1. **Full model** (baseline)
2. **Uniform MatFormer** at the target budget
3. **F-Mat heterogeneous** at the target budget
4. **Gumbel-learned heterogeneous** at the target budget (if trained)

Outputs a comparison table with:
- Perplexity (Pile validation, C4)
- Per-layer width allocation map
- Total parameter count verification
- Downstream task accuracy (when evaluators are configured)

### 3.2 Visualization Utilities

**New file:** `olmo/hmat/viz.py`

Utilities to visualize:
- **Saliency heatmap:** Layer x Dimension Fisher saliency scores
- **U-shape curve:** Per-layer total saliency (validates the hypothesis)
- **Allocation comparison:** Side-by-side uniform vs. heterogeneous width allocation bars
- **Gumbel mask evolution:** How learned logits change over training steps
- **Pareto frontier:** Accuracy vs. parameter budget for uniform vs. H-Mat

### 3.3 Training Configs

**New file:** `configs/pile-tiny-hmat-gumbel.yaml`

```yaml
# Extends pile-tiny.yaml with Gumbel H-Mat
matformer_factor: 8
hmat:
  enabled: true
  method: gumbel
  gumbel_tau_start: 2.0
  gumbel_tau_end: 0.1
  gumbel_tau_anneal_steps: 0    # Use max_duration
  budget_penalty_lambda: 0.01
  budget_penalty_target: 0.5
```

**New file:** `configs/pile-tiny-hmat-fisher.yaml`

```yaml
# Standard uniform MatFormer training, then post-hoc Fisher analysis
matformer_factor: 8
hmat:
  enabled: false  # Fisher is post-training only
```

### Tests for Phase 3

**File:** `tests/test_hmat_integration.py`

1. **test_uniform_vs_heterogeneous_same_budget:** Given a tiny model, verify that uniform and heterogeneous allocations with the same budget produce the same total parameter count.
2. **test_fmat_then_eval:** Train a tiny model for a few steps, compute F-Mat allocations, evaluate with heterogeneous factors -- verify the pipeline runs end-to-end.
3. **test_gumbel_then_extract_widths:** Train a tiny model with Gumbel for a few steps, extract learned widths, verify they're valid MatFormer factors.
4. **test_config_backward_compat:** Verify that loading an old config without `hmat` section works correctly (defaults to disabled).
5. **test_checkpoint_gumbel_roundtrip:** Save and load a checkpoint with Gumbel mask state; verify logits are preserved.

---

## File Change Summary

### Modified Files

| File | Changes |
|------|---------|
| `olmo/model.py` | Extend `MatformerManager` with per-layer factors, `mode`, Gumbel mask support. Add `layer_idx` to `OlmoBlock.__init__`. Update `OlmoSequentialBlock.forward` for heterogeneous slicing + Gumbel masking. Update `OlmoParallelBlock` similarly. Update `Olmo.__init__` to pass `layer_idx`. |
| `olmo/config.py` | Add `HMatConfig` dataclass. Add `hmat` field to `TrainConfig`. |
| `olmo/train.py` | Initialize `GumbelMaskManager` when enabled. Add budget penalty to loss. Add tau annealing logic. Save/load Gumbel state in checkpoints. Log mask statistics. |
| `configs/pile-tiny.yaml` | Add `hmat:` section (disabled by default). |

### New Files

| File | Purpose |
|------|---------|
| `olmo/hmat/__init__.py` | Package init |
| `olmo/hmat/fisher.py` | Fisher saliency computation |
| `olmo/hmat/knapsack.py` | Budget allocation solver |
| `olmo/hmat/gumbel.py` | Gumbel-Softmax mask modules |
| `olmo/hmat/viz.py` | Visualization utilities |
| `scripts/compute_fmat.py` | Post-training Fisher analysis script |
| `scripts/eval_hmat.py` | Heterogeneous evaluation script |
| `scripts/eval_comparison.py` | Head-to-head comparison script |
| `configs/pile-tiny-hmat-gumbel.yaml` | Training config for Gumbel H-Mat |
| `configs/pile-tiny-hmat-fisher.yaml` | Training config for Fisher analysis |
| `tests/test_hmat_basic.py` | Phase 0 tests |
| `tests/test_fmat.py` | Phase 1 tests (Fisher + knapsack) |
| `tests/test_gumbel.py` | Phase 2 tests (Gumbel masking) |
| `tests/test_hmat_integration.py` | Phase 3 integration tests |

---

## Implementation Order & Dependencies

```
Phase 0 (Foundation)
  ├── 0.1 Extend MatformerManager         ← No dependencies
  ├── 0.2 Thread layer_idx                ← Depends on 0.1
  ├── 0.3 Update forward pass             ← Depends on 0.1, 0.2
  ├── 0.4 Extend config                   ← No dependencies
  └── Tests                               ← Depends on 0.1-0.4

Phase 1 (Fisher / F-Mat)                  ← Depends on Phase 0
  ├── 1.1 Fisher score computation         ← No dependencies within phase
  ├── 1.2 Knapsack solver                  ← No dependencies within phase
  ├── 1.3 F-Mat script                     ← Depends on 1.1, 1.2
  ├── 1.4 Eval script                      ← Depends on 1.3
  └── Tests                                ← Depends on 1.1-1.4

Phase 2 (Gumbel / Learnable H-Mat)        ← Depends on Phase 0
  ├── 2.1 Gumbel mask module               ← No dependencies within phase
  ├── 2.2 Integrate into forward pass      ← Depends on 2.1
  ├── 2.3 Integrate into Trainer           ← Depends on 2.1, 2.2
  ├── 2.4 Checkpoint support               ← Depends on 2.3
  └── Tests                                ← Depends on 2.1-2.4

Phase 3 (Integration)                     ← Depends on Phase 1 & 2
  ├── 3.1 Comparison evaluation framework
  ├── 3.2 Visualization utilities
  ├── 3.3 Training configs
  └── Tests
```

**Note:** Phases 1 and 2 are independent of each other and can be developed in parallel after Phase 0 is complete.

---

## Key Design Decisions

1. **Backward compatibility:** When `hmat.enabled = False` (default), the entire system behaves identically to the current codebase. Zero risk of regressions.

2. **Powers-of-2 factors only:** Per-layer factors are constrained to `{1, 2, 4, 8, ...}` to maintain MatFormer's nested submodel extraction property. A model trained with heterogeneous factors `{layer_0: 2, layer_1: 4, ...}` still extracts valid submodels.

3. **Gumbel masks are a separate nn.Module:** They have their own parameters and optimizer group. This keeps the core model weights clean and allows easy ablation (remove masks, keep model).

4. **Fisher computation uses existing model:** F-Mat operates on a fully trained uniform MatFormer checkpoint. No changes to training are needed -- it's a pure post-processing step.

5. **Soft masking during training, hard slicing at inference:** Gumbel masks multiply activations by continuous [0,1] values during training (for gradient flow), but snap to hard 0/1 at eval time. The final extracted submodel uses the same weight-slicing mechanism as uniform MatFormer.

6. **Temperature annealing follows exponential schedule:** `tau = tau_start * (tau_end / tau_start) ^ progress` provides smooth transition from exploration to exploitation.

---

## Phase 2 Results: Gumbel-Softmax Learnable Masking (COMPLETED)

### Implementation Summary

**New files created:**
- `olmo/hmat/gumbel.py` — `GumbelMaskLayer` (per-layer learnable gate via Gumbel-Sigmoid) and `GumbelMaskManager` (manages masks for all layers, budget penalty, width extraction)
- `tests/test_gumbel.py` — 28 tests covering mask shape/range, temperature effects, gradients, forward/backward integration, checkpointing, config
- `configs/pile-tiny-hmat-gumbel.yaml` — Short training config (540 steps, pile data)
- `configs/pile-tiny-hmat-gumbel-long.yaml` — Long training config (2700 steps, pile-700M data)
- `scripts/eval_gumbel_comparison.py` — Eval script comparing gumbel vs baseline at all sub-model sizes
- `scripts/run_hparam_search.sh` — Parallel hyperparameter search (6 trials, 1 GPU each)

**Files modified:**
- `olmo/model.py` — `OlmoSequentialBlock.forward()` combines gumbel masks with MatFormer factor slicing: at each sub-model width, the mask is sliced to `mask[:k_out]` preserving nested structure
- `olmo/train.py` — `init_gumbel()` creates GumbelMaskManager + separate AdamW optimizer; budget penalty added to loss; tau annealing per step; gumbel state saved as `gumbel.pt` in checkpoints
- `olmo/hmat/__init__.py` — Exports `GumbelMaskLayer`, `GumbelMaskManager`

### Training Results (17M model, 4 layers, pile-700M)

**540-step comparison (directly comparable, same data/steps/seed):**

| Config | Eval PPL (full) |
|--------|----------------|
| Baseline MatFormer (uniform, factor=8) | **155.9** |
| Gumbel H-Mat (best hparams: tau=0.5, lam=0.001) | 174.0 |
| Gap | +18.1 (+11.6%) |

**2700-step comparison:**

| Config | Full (1/1) | 1/2 | 1/4 | 1/8 |
|--------|-----------|-----|-----|-----|
| Baseline (uniform MatFormer) | **381.1** | **410.5** | **445.7** | **489.1** |
| Gumbel H-Mat (tau=2.0, lam=0.01) | 454.2 | 475.4 | 501.1 | 530.8 |

**Learned mask allocations (2700 steps, tau=2.0, lam=0.01):**
- Layer 0: 66.6% active (early — widest)
- Layer 1: 56.9% active
- Layer 2: 39.9% active
- Layer 3: 36.6% active (late — narrowest)
- Mean: 50.0% (exactly at target)

The masks learned a **monotonically decreasing** allocation — early layers need more width, later layers are more compressible.

### Hyperparameter Search (6 trials, 540 steps each)

| Trial | tau_start | lambda | Eval PPL |
|-------|-----------|--------|----------|
| **T1** | **0.5** | **0.001** | **174.0** |
| T4 | 2.0 | 0.001 | 174.6 |
| T2 | 0.5 | 0.005 | 175.0 |
| T3 | 0.5 | 0.01 | 175.6 |
| T6 | 1.0 | 0.005 | 176.1 |
| T5 | 1.0 | 0.001 | 176.4 |

**Key finding:** `budget_penalty_lambda` matters more than `gumbel_tau_start`. Lower lambda (0.001) consistently outperforms higher values. Tau has minimal effect.

### Analysis

The gumbel mechanism successfully **learns heterogeneous per-layer allocation** from scratch during training, confirming that different layers have different width sensitivities. However, at this tiny model scale (17M params, 4 layers), the masking overhead hurts overall perplexity by ~12% compared to the simpler uniform baseline.

**Why the gap exists at small scale:**
1. Gumbel noise during training acts as regularization that a 4-layer model can't absorb
2. Soft masking (multiplying by ~0.5) effectively halves capacity during early training
3. With only 4 layers, there's limited room for heterogeneous allocation to help — the allocation differences are subtle (67% vs 37%)
4. The budget penalty, even at lambda=0.001, adds a competing objective

**Expected improvements at larger scale:** The gap should narrow or reverse because (a) more layers means more allocation flexibility, (b) the per-layer sensitivity U-shape is more pronounced, (c) the relative overhead of mask parameters becomes negligible, and (d) the model has more capacity to absorb the regularization effect.

### Critical Fix: Linear Decay Initialization

**Root cause of the original gap:** Zero-initialized logits produce `sigmoid(0)=0.5` masks, which **halves effective capacity at every sub-model width from step 1**. Combined with MatFormer's factor loop, the gumbel model trains at 2x more compression than baseline at every factor.

**Fix:** Initialize logits with `torch.linspace(+2.2, -2.2, mlp_dim)` so initial masks approximate baseline MatFormer's hard slicing (early dims ~0.9, late dims ~0.1, mean ~0.5).

**540-step results with linear decay init (tau=0.5, lambda=0.001):**

| Sub-model | Baseline | Gumbel (zero init) | Gumbel (linear decay) |
|-----------|---------|--------------------|-----------------------|
| Full (1/1) | **155.9** | 174.0 (+11.6%) | **160.0** (+2.6%) |
| 1/2 | 161.8 | — | **160.2** (-1.0%) |
| 1/4 | 167.2 | — | **164.8** (-1.4%) |
| 1/8 | 172.9 | — | **170.6** (-1.3%) |

Linear decay closes the gap from 11.6% to 2.6% at full width, and **outperforms baseline at all compressed sub-models**.

**2700-step results with linear decay init:**

| Sub-model | Baseline | Gumbel (linear decay) | Delta |
|-----------|---------|----------------------|-------|
| Full (1/1) | **381.1** | 415.5 | +9.0% |
| 1/2 | **410.5** | 428.4 | +4.4% |
| 1/4 | **445.7** | 460.7 | +3.4% |
| 1/8 | **489.1** | 500.5 | +2.3% |

The gap **narrows monotonically** at higher compression (9% → 4.4% → 3.4% → 2.3%), confirming that heterogeneous allocation helps most where it matters: under aggressive compression.

**Learned mask analysis (2700 steps):**
- Layer 0: 1292/2048 active (63%), logit mean +0.48 — widest
- Layer 1: 1163/2048 (57%), logit mean +0.26
- Layer 2: 980/2048 (48%), logit mean -0.27
- Layer 3: 840/2048 (41%), logit mean -0.65 — narrowest
- **97-99% of top dimensions preserved their initial ordering** (layers 0-2)
- Layer 3 showed more reordering (86% top-256 overlap), learning layer-specific importance
- The masks primarily learn **per-layer width thresholds**, not per-dimension reordering — the MatFormer nesting already provides a good dimension ordering

---

## Phase 2.0 FIX: Decouple Mask from Prefix Slicing

### Problem

Both Phase 2 (Gumbel) and Phase 2.2 (Top-K) share a fundamental structural flaw: the mask is learned over all `mlp_dim` dimensions globally, but at factor > 1 only `mask[:k_out]` is applied because the weights are prefix-sliced.

**Current forward pass at factor=2 (k_out = mlp_dim/2):**
```python
# Weights are prefix-sliced — h has shape (B, T, k_out)
h = self.act(F.linear(self.ff_norm(x), w_proj[:k], b_proj[:k]))
# Mask is global but only prefix is used
h = h * mask[:k_out]
```

This creates a semantic mismatch: the mask may rank neurons outside the prefix as important, but the sub-model can never access them. If training pushes high-importance logits to positions beyond k_out, the sliced mask `mask[:k_out]` has fewer active entries than intended — potentially far fewer.

**This is why the decreasing `linspace(+scale, -scale)` initialization was mandatory.** It wasn't just a good init — it was a necessary constraint to keep high-importance logits aligned with prefix positions. Zero init and random init both failed precisely because they broke this implicit alignment:

- **Zero init:** `sigmoid(0) = 0.5` everywhere — no preference for prefix positions, every neuron runs at half capacity
- **Random init:** High logits scattered across all positions — `mask[:k_out]` captures only a random subset of the "important" neurons
- **linspace init:** High logits at low positions by construction — `mask[:k_out]` always gets the highest-ranked neurons

The linspace init is a band-aid. Gradient dynamics mostly preserve the ordering (neurons outside the prefix get no gradient from sub-model passes), but the constraint is implicit and fragile.

### Fix: Full-Width Compute + Mask Zeroing

Instead of prefix-slicing weights and then masking the prefix, compute the full MLP hidden state and let the mask do all the work. Zeroed-out dimensions contribute nothing through `ff_out`, so the output is mathematically correct.

**Fixed forward pass at factor > 1:**
```python
# In OlmoSequentialBlock.forward, when use_gumbel or use_topk:

if factor == 1:
    h = self.act(self.ff_proj(self.ff_norm(x)))
    # No mask at full width (gumbel: mask applied but ~1.0; topk: all-ones)
    if use_gumbel:
        h = h * mask.unsqueeze(0).unsqueeze(0)
    x = x + self.dropout(self.ff_out(h))
else:
    # FULL-WIDTH compute — no weight slicing
    h = self.act(self.ff_proj(self.ff_norm(x)))  # shape: (B, T, mlp_dim)

    if use_gumbel:
        h = h * mask.unsqueeze(0).unsqueeze(0)   # mask zeros out unimportant dims
    if use_topk:
        tau = self.matmng.gumbel_tau
        topk_mask = self.matmng.topk_masks.get_mask(
            self.layer_idx, k=k_out, tau=tau, hard=not self.training
        )
        h = h * topk_mask.unsqueeze(0).unsqueeze(0)  # full mask, no [:k_out] slice

    x = x + self.dropout(self.ff_out(h))  # full-width ff_out; zeroed dims = zero contribution
```

**What changes:**
1. Sub-model passes no longer slice `ff_proj` / `ff_out` weights when masks are active — the mask replaces prefix slicing as the sparsity mechanism
2. The mask operates on the full `mlp_dim` vector — any neuron at any position can be selected
3. Nesting is preserved automatically: for top-k, top-128 ⊆ top-256 ⊆ top-512; for gumbel, higher-logit neurons are always included at wider sub-models
4. The `factor` variable still determines the target count k (how many neurons the sub-model should have), but not which positions they must occupy

**Backward-compatible behavior:** When `mode == "uniform"` (no masks), the existing prefix-slicing code path is unchanged. The fix only affects the `use_gumbel` and `use_topk` branches.

### Post-Training Neuron Reordering

At inference, MatFormer sub-models are extracted via prefix slicing (`weight[:k]`). For this to work with position-independent masks, neurons must be reordered after training so that the most important neurons come first.

**Reordering procedure (run once after training, before deployment):**

```python
def reorder_neurons_by_importance(model: Olmo, mask_manager) -> Olmo:
    """
    Reorder MLP neurons in each layer so that prefix slicing at inference
    recovers the mask-learned allocation. Modifies weights in-place.

    For each layer:
    1. Rank neurons by mask importance (logit value)
    2. Permute ff_proj rows and ff_out columns to match this ranking
    3. After reordering, weight[:k] contains the top-k neurons by importance
    """
    for layer_idx, block in enumerate(model.transformer.blocks):
        # Get importance ordering from mask logits
        logits = mask_manager.masks[layer_idx].logits.detach()

        # Permutation: sort by descending importance
        perm = logits.argsort(descending=True)

        # ff_proj: (mlp_dim, d_model) — permute rows
        block.ff_proj.weight.data = block.ff_proj.weight.data[perm]
        block.ff_proj.bias.data = block.ff_proj.bias.data[perm]

        # Account for activation output_multiplier (e.g. SwiGLU)
        # SwiGLU: ff_proj has 2*mlp_dim_out rows, ff_out has mlp_dim_out columns
        # The mask applies post-activation, so we need the output-side permutation
        act = Activation.build(model.config)
        if act.output_multiplier != 1.0:
            # For SwiGLU: ff_proj has shape (2*mlp_dim_out, d_model)
            # First half = gate, second half = value — permute both halves
            mlp_dim_out = int(block.ff_proj.weight.shape[0] * act.output_multiplier)
            gate_perm = perm  # permutation for the gate half
            val_perm = perm + mlp_dim_out  # same permutation offset for value half
            full_perm = torch.cat([gate_perm, val_perm])
            block.ff_proj.weight.data = block.ff_proj.weight.data[full_perm]
            block.ff_proj.bias.data = block.ff_proj.bias.data[full_perm]

        # ff_out: (d_model, mlp_dim_out) — permute columns
        block.ff_out.weight.data = block.ff_out.weight.data[:, perm]

    return model
```

After reordering, the model can be deployed as a standard MatFormer: `weight[:k]` gives the top-k neurons at each layer, and the mask is no longer needed at inference.

### Reorder Script

**New file:** `scripts/reorder_neurons.py`

Loads a trained checkpoint + mask state, applies `reorder_neurons_by_importance`, saves a new checkpoint. The new checkpoint is a standard MatFormer model — no mask overhead at inference.

```bash
python scripts/reorder_neurons.py \
    --checkpoint /path/to/trained-checkpoint \
    --mask_state /path/to/gumbel.pt   # or topk.pt \
    --output /path/to/reordered-checkpoint
```

### Training Compute Tradeoff

Sub-model passes now compute the full MLP width instead of the prefix:

| | Factor=1 | Factor=2 | Factor=4 | Factor=8 |
|--|---------|----------|----------|----------|
| **Before (prefix slicing)** | Full | 1/2 FLOPs | 1/4 FLOPs | 1/8 FLOPs |
| **After (full + mask)** | Full | Full | Full | Full |

Every sub-model pass costs the same as a full-model pass. For the 17M model this is negligible. For larger models, sub-model passes were already a small fraction of training time (the factor=1 pass dominates), and the benefit of correct mask semantics outweighs the cost.

### Impact on Initialization

With the fix, the decreasing linspace init is no longer structurally necessary — any init works because the mask can select neurons at any position. However, linspace may still be useful as a soft prior that speeds convergence (starts near the baseline MatFormer behavior). Zero init and random init become viable alternatives worth re-evaluating.

### Changes Required

| File | Change |
|------|--------|
| `olmo/model.py` | `OlmoSequentialBlock.forward`: when `use_gumbel` or `use_topk`, compute full-width MLP and apply full mask instead of prefix-slicing + `mask[:k_out]` |
| `olmo/hmat/topk.py` | `TopKMaskLayer.forward`: remove the `[:k_out]` slice (caller no longer needs it) — mask is always full-width |
| `scripts/reorder_neurons.py` | New script: post-training neuron reordering for inference-time prefix slicing |
| `tests/test_gumbel.py` | Update integration tests to verify full-width mask application |
| `tests/test_topk.py` | Update integration tests; add test for arbitrary-position neuron selection |
| `tests/test_reorder.py` | New: verify neuron reordering produces equivalent outputs, verify prefix slicing after reorder matches mask-selected output |

### Tests

1. **test_full_width_mask_equivalence:** With linspace init (where mask ordering matches prefix ordering), verify that full-width-mask output equals prefix-sliced output. This confirms backward compatibility at init.
2. **test_arbitrary_position_selection:** Set logits so that the top-k neurons are at non-prefix positions (e.g., positions 200-256 in a 512-dim layer). Verify the mask correctly activates those positions and the output is nonzero.
3. **test_zero_init_viable:** With zero-init logits and full-width masking, verify the model trains without collapse (loss decreases over a few steps). This was previously catastrophic with prefix slicing.
4. **test_reorder_preserves_output:** Run forward pass with mask, reorder neurons, run forward pass with prefix slicing (no mask). Verify outputs match.
5. **test_reorder_prefix_nesting:** After reordering, verify that `weight[:k]` at factor=2 is a strict prefix of `weight[:2k]` at factor=1 — the MatFormer nesting property holds.
6. **test_gumbel_full_width_gradient:** Verify gradients flow to all logits (not just prefix positions) through full-width mask application.
7. **test_topk_full_width_gradient:** Same for top-k masks — verify boundary neurons at any position receive gradient.

---

## Phase 2.0.1 Fix: Gumbel Factor Independence

### Problem

The Phase 2.0 FIX removes prefix slicing when masks are active. For **Top-K** this is clean: at each factor, k is derived from the factor and top-k selects exactly k neurons at arbitrary positions.

For **Gumbel-Sigmoid**, there is a problem: the gumbel mask `sigmoid(logits / tau)` produces the same values regardless of the current factor. With prefix slicing, different factors naturally saw different mask regions (`mask[:k_out]` at factor=2, `mask[:k_out/2]` at factor=4). With full-width compute, factors 2, 4, and 8 all apply the identical mask to the identical full-width hidden state — the factor loop produces identical outputs for all sub-model passes.

**This means vanilla Gumbel-Sigmoid is not compatible with full-width compute in the multi-factor training loop.** The sub-model losses at factor=2, 4, 8 would be identical, wasting 3/4 of the training compute.

### Solution: Gumbel-Top-K Hybrid

Combine Gumbel noise (stochastic exploration) with Top-K thresholding (factor-dependent width):

```python
# In GumbelMaskLayer.forward, when k is not None and k < mlp_dim:
def forward(self, tau=1.0, hard=False, k=None):
    if k is not None and k >= self.mlp_dim:
        return torch.ones_like(self.logits)

    if self.training and not hard:
        u = torch.rand_like(self.logits).clamp(1e-6, 1.0 - 1e-6)
        gumbel_noise = -torch.log(-torch.log(u))
        noisy_logits = self.logits + gumbel_noise
    else:
        noisy_logits = self.logits

    if k is not None and k < self.mlp_dim:
        # Top-K threshold on noisy logits — factor-dependent width
        topk_vals, _ = noisy_logits.topk(k, sorted=False)
        threshold = topk_vals.min()
        y_soft = torch.sigmoid((noisy_logits - threshold) / tau)
    else:
        # Original gumbel-sigmoid (no factor truncation)
        y_soft = torch.sigmoid(noisy_logits / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft
```

Gumbel-Top-K is the natural synthesis: it uses the structural top-k constraint to guarantee factor-dependent widths (eliminating the budget penalty), while Gumbel noise makes the threshold stochastic — different neurons are near the boundary each step, providing broader gradient coverage than deterministic top-k.

### Method Comparison

| Method | Threshold | Noise | Factor-dependent | Budget penalty |
|--------|-----------|-------|------------------|----------------|
| **Top-K** (Phase 2.2) | k-th largest logit | None | Yes (k from factor) | No |
| **Gumbel-Top-K** (new) | k-th largest noisy logit | Gumbel | Yes (k from factor) | No |
| **Gumbel-Sigmoid** (Phase 2) | None (per-element sigmoid) | Gumbel | Only at factor=1 (all-ones) | Required |

### Changes Required

| File | Change |
|------|--------|
| `olmo/hmat/gumbel.py` | `GumbelMaskLayer.forward`: when `k is not None and k < mlp_dim`, apply top-k threshold on noisy logits instead of plain sigmoid |
| `olmo/config.py` | Add `method: "gumbel_topk"` option to `HMatConfig` |
| `olmo/train.py` | Handle `"gumbel_topk"` method: same init as gumbel but no budget penalty, pass k to mask |
| `olmo/model.py` | When `use_gumbel` and factor > 1, pass `k=k_out` to `get_mask()` |

### Tests

1. **test_gumbel_topk_factor_dependent:** Verify that at different k values, the mask produces different active counts (unlike vanilla gumbel which is factor-independent).
2. **test_gumbel_topk_noise_varies_boundary:** Run forward twice with same logits — Gumbel noise should produce different boundary selections (different neurons near threshold).
3. **test_gumbel_topk_deterministic_eval:** At eval time (no noise), Gumbel-Top-K should produce identical masks to deterministic Top-K for the same logits and k.
4. **test_gumbel_topk_no_budget_penalty:** Verify no budget penalty is computed — sparsity is structural via top-k.
5. **test_gumbel_topk_gradient_to_all_logits:** Verify gradient flows to logits at all positions (not just top-k), since Gumbel noise can shift the boundary.

---

## Phase 2.1: Fisher-Guided Gumbel Masking (Saliency-Driven Allocation)

### Motivation

Phase 2's core weakness: Gumbel logits are learned via backprop through a budget penalty — an indirect, competing objective that degrades the primary language modeling loss, especially over longer training. The logits converge slowly and the soft masks reduce effective capacity throughout training.

**Key insight:** We can compute Fisher saliency scores from the gradients we already have (zero extra cost), and use them to *directly set* the Gumbel logits. This replaces the learned-logit + budget-penalty mechanism with a direct importance signal, while retaining Gumbel noise for exploration.

### How It Differs from Phase 2

| Aspect | Phase 2 (Learned Gumbel) | Phase 2.1 (Fisher-Guided Gumbel) |
|--------|--------------------------|----------------------------------|
| Logit source | Learned via backprop + budget penalty | Set from Fisher saliency EMA |
| Mask parameters | Learnable `nn.Parameter` | Non-learnable, externally updated |
| Budget penalty | Required (competing objective) | **Eliminated** |
| Second optimizer | Required for mask params | **Eliminated** |
| Exploration | Gumbel noise + temperature | Gumbel noise + temperature (same) |
| Logit polarization | Gradual, learned over training | Strong from start (saliency is informative) |

### Algorithm

```
Phase 1: Warmup (steps 0..W)
  - Train standard uniform MatFormer (no masks applied)
  - Each step after backward: accumulate per-dimension Fisher scores into EMA
    fisher_ema[l][d] = beta * fisher_ema[l][d] + (1 - beta) * (grad[l][d] ** 2).sum()
  - Purpose: build stable saliency estimates before masking begins

Phase 2: Fisher-guided masking (steps W..end)
  - Every K steps: convert Fisher EMA to Gumbel logits
    1. For each layer l, rank dimensions by fisher_ema[l]
    2. Map ranks to logits: logits[l] = linspace(+scale, -scale, mlp_dim)[rank_order]
       (top-saliency dims get positive logits, bottom get negative)
    3. Update the GumbelMaskLayer logits (non-gradient, direct assignment)
  - Each forward pass: apply Gumbel-Sigmoid masks with temperature annealing
    mask = sigmoid((logits + gumbel_noise) / tau)
  - Continue accumulating Fisher EMA (masked-out dims still occasionally
    activate via Gumbel noise, keeping their saliency scores calibrated)
  - Temperature annealing: tau high early (exploration) → low late (exploitation)

Phase 3: Freeze (final)
  - Convert soft masks to hard {0,1} based on final logits
  - Extract per-layer widths, snap to nearest MatFormer factors
```

### Why Gumbel Noise Still Matters (Exploration)

Without noise, Fisher-guided hard masking creates a rich-get-richer problem: dimensions that score high early monopolize gradient signal, reinforcing their scores even if other dimensions would ultimately be more useful. Gumbel noise + temperature provides exploration:

- **Early training (high tau):** Masks are stochastic — dimensions with low saliency still activate frequently, receive gradient signal, and their Fisher scores stay calibrated. If a masked-out dimension becomes important later, its saliency rises and the periodic logit update catches it.
- **Late training (low tau):** Masks converge toward the saliency-derived allocation. The model has had time to explore and the Fisher EMA is stable.

This is the same Gumbel-Sigmoid + temperature annealing infrastructure from Phase 2, but with externally-set logits instead of learned ones.

### Design Decisions

1. **Warmup period (W):** ~5-10% of total training steps. Needs to be long enough for Fisher EMA to stabilize, short enough that masking has time to take effect. Start with W = max_duration * 0.05.

2. **EMA decay (beta):** 0.99 — slow enough for stability, fast enough to track changing saliency as the model trains. Fisher scores from early training (random weights) are very different from mid-training scores.

3. **Logit update frequency (K):** Every 50-100 steps. Too frequent = noisy allocation changes that destabilize training. Too infrequent = slow adaptation. The knapsack solver runs in microseconds so compute cost is zero.

4. **Logit scale:** Reuse init_scale from Phase 2 (default 2.2). `linspace(+2.2, -2.2)` maps top-saliency dims to sigmoid ≈ 0.9, bottom to ≈ 0.1. The rank-based mapping ensures consistent polarization regardless of raw saliency magnitude.

5. **Fisher score computation:** Per-dimension, summing squared gradients from both `ff_proj` rows and `ff_out` columns (same formula as Phase 1's `compute_fisher_saliency`). Computed from gradients that already exist after backward — zero extra forward/backward passes.

### 2.1.1 Fisher EMA Accumulator

**File:** `olmo/hmat/fisher_ema.py` (new)

```python
class FisherEMA:
    """
    Exponential moving average of per-dimension Fisher saliency scores.
    Updated each training step from existing gradients (zero extra compute).
    """

    def __init__(self, n_layers: int, mlp_dim: int, beta: float = 0.99,
                 device: torch.device = None):
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim
        self.beta = beta
        # EMA accumulators: one vector per layer
        self.scores = [torch.zeros(mlp_dim, device=device) for _ in range(n_layers)]
        self.steps = 0

    def update(self, model: nn.Module):
        """
        Accumulate Fisher scores from current gradients (call after backward).
        """
        self.steps += 1
        for l, block in enumerate(model.transformer.blocks):
            grad_proj = block.ff_proj.weight.grad  # (mlp_dim, d_model)
            grad_out = block.ff_out.weight.grad     # (d_model, mlp_dim)
            if grad_proj is None or grad_out is None:
                continue
            fisher = (grad_proj ** 2).sum(dim=1) + (grad_out ** 2).sum(dim=0)
            self.scores[l] = self.beta * self.scores[l] + (1 - self.beta) * fisher.detach()

    def get_logits(self, scale: float = 2.2) -> List[torch.Tensor]:
        """
        Convert Fisher EMA scores to Gumbel logits via rank-based mapping.

        Dimensions are ranked by saliency within each layer. Top-ranked dims
        get positive logits (+scale), bottom-ranked get negative (-scale).
        """
        logits = []
        for l in range(self.n_layers):
            # Rank dimensions by saliency (highest = rank 0)
            ranks = self.scores[l].argsort(descending=True).argsort()
            # Map ranks to logits: rank 0 → +scale, rank mlp_dim-1 → -scale
            layer_logits = scale - (2 * scale * ranks.float() / (self.mlp_dim - 1))
            logits.append(layer_logits)
        return logits
```

### 2.1.2 Modify GumbelMaskLayer for External Logit Updates

**File:** `olmo/hmat/gumbel.py` (modify)

The existing `GumbelMaskLayer` stores logits as `nn.Parameter`. For Phase 2.1, logits are set externally and should **not** be learnable:

```python
class GumbelMaskLayer(nn.Module):
    def __init__(self, mlp_dim: int, init_scale: float = 2.2, learnable: bool = True):
        super().__init__()
        self.mlp_dim = mlp_dim
        init_logits = torch.linspace(init_scale, -init_scale, mlp_dim)
        if learnable:
            self.logits = nn.Parameter(init_logits)
        else:
            # Register as buffer — not a parameter, not in optimizer
            self.register_buffer("logits", init_logits)

    def set_logits(self, new_logits: torch.Tensor):
        """Update logits from external source (e.g., Fisher EMA)."""
        with torch.no_grad():
            self.logits.copy_(new_logits)
```

The forward pass (Gumbel-Sigmoid + temperature) remains identical.

### 2.1.3 Integrate into Training Loop

**File:** `olmo/train.py` (modify)

```python
# In Trainer.__init__ or init_gumbel():
if self.cfg.hmat.method == "fisher_gumbel":
    mlp_dim = self.cfg.model.mlp_ratio * self.cfg.model.d_model
    n_layers = self.cfg.model.n_layers

    # Fisher EMA accumulator
    self.fisher_ema = FisherEMA(
        n_layers=n_layers,
        mlp_dim=int(mlp_dim * output_multiplier),
        beta=self.cfg.hmat.fisher_ema_beta,
        device=self.device,
    )

    # Gumbel masks with non-learnable logits
    self.gumbel_manager = GumbelMaskManager(
        n_layers=n_layers,
        mlp_dim=int(mlp_dim * output_multiplier),
        init_scale=self.cfg.hmat.gumbel_init_scale,
        learnable=False,  # NEW: logits are buffers, not parameters
    )
    # No second optimizer needed!

    matmng.mode = "gumbel"
    matmng.gumbel_masks = self.gumbel_manager
    matmng.gumbel_tau = self.cfg.hmat.gumbel_tau_start

# In train_step(), after backward but before optimizer.step():
if self.cfg.hmat.method == "fisher_gumbel":
    # 1. Accumulate Fisher EMA (uses existing gradients — free)
    self.fisher_ema.update(self.fsdp_model.module)

    # 2. During warmup: no masks applied (handled by checking step < warmup)
    warmup_steps = int(self.cfg.hmat.fisher_warmup_frac * self.cfg.max_duration)

    if self.global_step == warmup_steps:
        # First time: enable masking
        matmng.mode = "gumbel"

    # 3. Every K steps after warmup: update logits from Fisher EMA
    if (self.global_step >= warmup_steps and
            self.global_step % self.cfg.hmat.fisher_update_interval == 0):
        new_logits = self.fisher_ema.get_logits(scale=self.cfg.hmat.gumbel_init_scale)
        for i, mask_layer in enumerate(self.gumbel_manager.masks):
            mask_layer.set_logits(new_logits[i])

    # 4. Temperature annealing (same as Phase 2)
    # ...
```

### 2.1.4 Config Extensions

**File:** `olmo/config.py` (modify `HMatConfig`)

```python
@dataclass
class HMatConfig(BaseConfig):
    # ... existing fields ...
    method: str = "fisher"  # "fisher" | "gumbel" | "fisher_gumbel"

    # Phase 2.1 (Fisher-Guided Gumbel) settings
    fisher_ema_beta: float = 0.99           # EMA decay for Fisher scores
    fisher_warmup_frac: float = 0.05        # Fraction of training for warmup (no masks)
    fisher_update_interval: int = 50        # Update logits from Fisher EMA every K steps
```

### 2.1.5 Forward Pass

**No changes needed.** The existing `OlmoSequentialBlock.forward` already handles gumbel masks via `matmng.mode == "gumbel"`. During warmup, `matmng.mode` stays `"uniform"` so masks aren't applied. After warmup, it switches to `"gumbel"` and the same mask application logic runs.

### Tests for Phase 2.1

**File:** `tests/test_fisher_gumbel.py` (new)

1. **test_fisher_ema_accumulation:** Run 10 forward-backward passes, verify EMA scores are non-zero and shaped correctly.
2. **test_fisher_ema_decay:** Verify that old scores decay with beta and new scores are incorporated.
3. **test_fisher_ema_to_logits:** Set known saliency scores, verify rank-based logit mapping produces correct ordering (+scale for top dims, -scale for bottom).
4. **test_logit_update_preserves_mask_shape:** After `set_logits()`, verify forward pass still produces correct mask shape.
5. **test_warmup_no_masks:** During warmup steps, verify model forward pass is identical to uniform MatFormer.
6. **test_exploration_via_gumbel_noise:** With high tau, verify that low-saliency dimensions still activate with meaningful probability (> 5%).
7. **test_convergence_at_low_tau:** With low tau, verify masks closely match the saliency ranking.
8. **test_fisher_gumbel_end_to_end:** Run a short training loop (warmup + masked steps), verify no errors and that logits change across Fisher updates.
9. **test_no_learnable_mask_params:** Verify `gumbel_manager.parameters()` is empty when `learnable=False`.
10. **test_fisher_ema_with_masking:** Verify that Fisher scores still accumulate correctly even when masks are active (masked-out dims get nonzero scores via Gumbel exploration).

### Expected Advantages Over Phase 2

1. **No competing objective:** Budget penalty eliminated — the only loss is language modeling.
2. **No second optimizer:** Mask logits are buffers, not parameters.
3. **Stronger logit polarization:** Fisher saliency provides a direct importance signal; masks are near-binary from the moment masking begins (after warmup).
4. **Adaptive allocation:** Fisher EMA tracks changing importance over training — allocation isn't frozen at initialization.
5. **Warmup eliminates early-training capacity loss:** The model trains at full capacity during warmup, building both good weights and stable saliency estimates before any masking.
6. **Exploration prevents lock-in:** Gumbel noise ensures masked-out dimensions still occasionally activate, keeping Fisher scores calibrated and allowing allocation to shift if importance changes.

---

## Experimental Results: Phase 2.1 + Freeze Ablations (540 steps, pile-700M)

### Freeze Experiment

| Config | Full (1/1) | 1/2 | 1/4 | 1/8 | Notes |
|--------|-----------|-----|-----|-----|-------|
| **Baseline** | **155.9** | 161.8 | 167.2 | 172.9 | — |
| **Phase 2 learned (best)** | 157.7 | **158.2** | **162.2** | **167.1** | scale=1.1, tau=0.5, lam=0.001 |
| Freeze 10% (broken schedule) | 167.3 | 167.5 | 171.1 | 175.8 | Tau didn't finish annealing before freeze |
| Freeze 10% (fixed schedule) | 163.3 | 163.5 | 167.7 | 172.7 | Tau completes annealing in first 90% |

**Freeze conclusion:** The fixed schedule improved over the broken one (163.3 vs 167.3), but both are worse than no-freeze (157.7). At 540 steps, masks haven't differentiated enough from init for freeze to add value. Freeze may help at longer training (2700+ steps) where masks diverge significantly.

### Fisher-Gumbel (Phase 2.1) Ablation Study

All ablations use fisher_gumbel method with 15% warmup (81 steps) as the base config.

| Ablation | Full (1/1) | 1/2 | 1/4 | 1/8 | What was tested |
|----------|-----------|-----|-----|-----|----------------|
| Fisher base (15% warmup) | 166.7 | 166.7 | 170.7 | 175.9 | Baseline fisher-gumbel |
| **A: Smooth blend** (every step, blend=0.05) | **164.8** | **164.8** | **168.9** | **173.7** | Smooth per-step logit blending vs periodic hard replace |
| B: 30% warmup | 165.0 | 165.0 | 169.0 | 173.6 | More warmup steps |
| C: Log-scaled Fisher | 300.4 | 300.4 | 300.4 | 300.4 | Log-scale magnitude preservation vs rank-based |
| D: Factor-1 only Fisher | 166.7 | 166.7 | 170.7 | 175.9 | Only accumulate from factor=1 passes |

### Ablation Analysis

1. **Smooth blend (A) helps modestly** — 164.8 vs 166.7 base. Confirms hypothesis #1: periodic hard logit replacement is disruptive; smooth per-step blending is better. But still 7 PPL behind learned gumbel.

2. **More warmup (B) barely helps** — 165.0 vs 166.7. 30% warmup only marginally better than 15%. The issue isn't warmup duration.

3. **Log-scaled mapping (C) catastrophically fails** — 300.4. The log-scale normalization concentrates most dimensions near the minimum logit value, effectively creating a near-uniform mask that destroys the ordering structure. The rank-based mapping is far superior.

4. **Factor-1 only (D) identical to base** — 166.7 = 166.7. Factor bias in Fisher accumulation is NOT an issue. Gradients from sub-model iterations don't meaningfully distort the saliency estimates.

### Root Cause Conclusion

The fundamental limitation of Fisher-gumbel is **hypothesis #6: no gradient signal on masks**. In learned gumbel (Phase 2), the CE loss gradient flows directly through the mask values, telling each dimension exactly how much it should be on/off. Fisher-gumbel only has an indirect heuristic (squared gradients → rank → logits) with no closed-loop feedback. The 7+ PPL gap (157.7 vs 164.8 best) reflects this structural disadvantage.

**Phase 2 with learned gumbel masks (scale=1.1, tau=0.5, lambda=0.001) remains the best approach**, outperforming baseline MatFormer at all compressed sub-model sizes while staying within 1.2% at full width.

---

## Phase 2.2: Soft Top-K Structural Sparsity

### Motivation

Phase 2's Gumbel-Sigmoid masks have a fundamental design problem: each neuron independently decides on/off via `sigmoid(logit)`, and a budget penalty (`budget_penalty_lambda * |mean_active - target|`) is the only thing controlling total sparsity. This creates two issues:

1. **Competing objectives.** The budget penalty fights the CE loss. At `lambda=0.001` (best found), it's weak enough to avoid hurting training — but also weak enough that active fractions drift. The model must waste gradient signal balancing two losses instead of focusing on language modeling.

2. **Soft scaling, not selection.** During training, every neuron is multiplied by a value in [0, 1], not selected or dropped. Even "fully on" neurons get scaled by ~0.75 (`sigmoid(1.1) ≈ 0.75`). This persistent capacity drag is the root cause of the full-model PPL degradation that worsens from +1.2% at 540 steps to +9% at 2700 steps.

**Key insight:** We don't need each neuron to independently decide its importance. We need to pick the top-K most important neurons per layer — a *structural* constraint, not a *regularized* one. A differentiable top-k operation guarantees exactly K active dimensions by construction, eliminating the budget penalty entirely and producing masks that are closer to binary even during training.

### How It Differs from Phase 2

| Aspect | Phase 2 (Gumbel-Sigmoid) | Phase 2.2 (Soft Top-K) |
|--------|--------------------------|------------------------|
| Mask mechanism | Per-element `sigmoid(logit + noise)` | Differentiable top-k with threshold |
| Sparsity enforcement | Budget penalty (competing loss) | **Structural** (exactly K dims active) |
| Budget penalty | Required (`lambda` hyperparameter) | **Eliminated** |
| Mask values during training | Smooth [0, 1] — never truly binary | Near-binary — sharp transition at threshold |
| Full-model behavior | All neurons scaled by ~0.75 | Top-K neurons ≈ 1.0, rest ≈ 0.0 |
| Hyperparameters | tau, lambda, target | tau only (K is derived from factor) |
| Per-layer budget | Soft (penalty pushes toward target) | **Hard** (exactly K per layer, or learnable K) |

### Algorithm

The core operation: given a logit vector of length `mlp_dim` and a target count `k`, produce a differentiable mask that is ~1 for the top-k logits and ~0 for the rest.

```
Forward pass:
  1. Compute threshold = k-th largest logit (differentiable via soft sorting)
  2. mask = sigmoid((logits - threshold) / tau)
  3. If hard mode: apply straight-through estimator

Backward pass:
  - Gradients flow through sigmoid and through the threshold computation
  - Logits above threshold get gradient pushing them higher (stay selected)
  - Logits below threshold get gradient pushing them higher (compete for selection)
  - The threshold itself shifts based on the aggregate gradient signal
```

**How K is determined per layer:** When the MatFormer training loop iterates over factors {1, 2, 4, 8}, at factor=f the sub-model uses `mlp_dim / f` output dimensions (after SwiGLU). The soft top-k mask at factor=f selects exactly `k = mlp_dim_out / f` dimensions. At factor=1 (full model), `k = mlp_dim_out` and the mask is all-ones — **the full model is never degraded**.

This naturally gives us the asymmetric property: factor=1 has no mask overhead, and sub-models get structurally enforced allocation.

### 2.2.1 Soft Top-K Mask Module

**New file:** `olmo/hmat/topk.py`

```python
class TopKMaskLayer(nn.Module):
    """
    Per-layer learnable importance gate using differentiable top-k.

    Instead of per-element sigmoid (Gumbel), this computes a soft threshold
    at the k-th largest logit value and applies sigmoid relative to that
    threshold. Guarantees exactly k dimensions are active at convergence.
    """

    def __init__(self, mlp_dim: int, init_scale: float = 1.1):
        super().__init__()
        self.mlp_dim = mlp_dim
        # Same linear decay init as Phase 2 — first dims important, last dims not
        self.logits = nn.Parameter(torch.linspace(init_scale, -init_scale, mlp_dim))

    def forward(self, k: int, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Produce a mask with ~k active dimensions.

        Args:
            k: Target number of active dimensions for this sub-model width.
            tau: Temperature. Lower = sharper transition at threshold.
            hard: If True, use straight-through estimator for {0, 1} masks.

        Returns:
            Tensor of shape (mlp_dim,) with ~k values near 1 and the rest near 0.
        """
        if k >= self.mlp_dim:
            # Full model — return all-ones, no mask overhead
            return torch.ones_like(self.logits)

        # Find the soft threshold: the k-th largest logit
        # topk returns (values, indices), we want the smallest of the top-k
        topk_vals, _ = self.logits.topk(k, sorted=False)
        threshold = topk_vals.min()  # k-th largest logit

        # Soft mask: sigmoid of (logit - threshold) / tau
        # Dims above threshold → sigmoid > 0.5 → "on"
        # Dims below threshold → sigmoid < 0.5 → "off"
        y_soft = torch.sigmoid((self.logits - threshold) / tau)

        if hard:
            y_hard = (y_soft > 0.5).float()
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    def get_active_fraction(self, k: int) -> float:
        """Return fraction of dims that would be selected for top-k."""
        with torch.no_grad():
            mask = self.forward(k, tau=0.01, hard=True)
            return mask.mean().item()


class TopKMaskManager(nn.Module):
    """Manages top-k masks for all layers in the model."""

    def __init__(self, n_layers: int, mlp_dim: int, init_scale: float = 1.1):
        super().__init__()
        self.n_layers = n_layers
        self.mlp_dim = mlp_dim
        self.masks = nn.ModuleList([
            TopKMaskLayer(mlp_dim, init_scale=init_scale)
            for _ in range(n_layers)
        ])

    def get_mask(self, layer_idx: int, k: int, tau: float = 1.0,
                 hard: bool = False) -> torch.Tensor:
        """Get the mask for a specific layer at a specific sub-model width."""
        return self.masks[layer_idx](k=k, tau=tau, hard=hard)

    def get_layer_widths(self, k: int) -> Dict[int, int]:
        """For a given target k, return actual active dim count per layer."""
        widths = {}
        with torch.no_grad():
            for i, mask_layer in enumerate(self.masks):
                hard_mask = mask_layer.forward(k, tau=0.01, hard=True)
                widths[i] = int(hard_mask.sum().item())
        return widths

    def log_summary(self, k: int) -> Dict[str, float]:
        """Return summary metrics for logging at a given sub-model width."""
        metrics = {}
        with torch.no_grad():
            for i, mask_layer in enumerate(self.masks):
                frac = mask_layer.get_active_fraction(k)
                metrics[f"topk/layer_{i}_active_frac_at_{k}"] = frac
        return metrics
```

### 2.2.2 Integrate into Forward Pass

**File:** `olmo/model.py` — `OlmoSequentialBlock.forward`

The key difference from Phase 2: the mask takes `k` as input (derived from the current factor), and at factor=1 it returns all-ones with no computation.

```python
# In OlmoSequentialBlock.forward:
factor = self.matmng.get_factor_for_layer(self.layer_idx)
use_topk = self.matmng.mode == "topk" and self.matmng.topk_masks is not None

if factor == 1:
    h = self.act(self.ff_proj(self.ff_norm(x)))
    if use_topk:
        # k = full mlp_dim_out → returns all-ones, zero overhead
        pass  # No mask at full width
    x = x + self.dropout(self.ff_out(h))
else:
    n = self.ff_proj.weight.shape[0]
    k = int(n / factor)
    w_proj = self.ff_proj.weight[:k]
    b_proj = self.ff_proj.bias[:k]
    k_out = int(k * self.act.output_multiplier)
    w_out = self.ff_out.weight[:, :k_out]
    b_out = self.ff_out.bias

    h = self.act(F.linear(self.ff_norm(x), w_proj, b_proj))
    if use_topk:
        tau = self.matmng.gumbel_tau  # Reuse tau infrastructure
        mask = self.matmng.topk_masks.get_mask(
            self.layer_idx, k=k_out, tau=tau, hard=not self.training
        )
        h = h * mask[:k_out].unsqueeze(0).unsqueeze(0)
    x = x + self.dropout(F.linear(h, w_out, b_out))
```

**Critical property:** At factor=1, no mask is applied. The full model trains at 100% capacity with zero interference — the asymmetric property comes for free from the `k >= mlp_dim` check in `TopKMaskLayer.forward`.

### 2.2.3 Integrate into Training Loop

**File:** `olmo/train.py`

Changes are smaller than Phase 2 because there is no budget penalty to compute:

```python
# In init_gumbel() or new init_topk():
if self.cfg.hmat.method == "topk":
    mlp_dim = self.cfg.model.mlp_ratio * self.cfg.model.d_model
    output_multiplier = Activation.build(self.cfg.model).output_multiplier
    mlp_dim_out = int(mlp_dim * output_multiplier)

    self.topk_manager = TopKMaskManager(
        n_layers=self.cfg.model.n_layers,
        mlp_dim=mlp_dim_out,
        init_scale=self.cfg.hmat.gumbel_init_scale,
    )

    # Separate optimizer (same pattern as Phase 2)
    self.topk_optim = torch.optim.AdamW(
        self.topk_manager.parameters(),
        lr=self.cfg.optimizer.learning_rate,
        weight_decay=0.0,
    )

    matmng = MatformerManager.get_instance()
    matmng.mode = "topk"
    matmng.topk_masks = self.topk_manager
    matmng.gumbel_tau = self.cfg.hmat.gumbel_tau_start

# In train_step — temperature annealing:
if self.cfg.hmat.method == "topk":
    # Same exponential tau annealing as Phase 2
    progress = min(1.0, self.global_step / anneal_steps)
    tau = tau_start * (tau_end / tau_start) ** progress
    matmng.gumbel_tau = tau

# In train_step — after loss.backward():
if self.topk_optim is not None:
    self.topk_optim.step()
    self.topk_optim.zero_grad()

# In train_step — loss computation:
# NO budget penalty! The loss is just CE + Z-loss (same as baseline).
# Sparsity is structural — exactly k dims active by construction.
```

**What's removed vs Phase 2:**
- No `budget_loss()` call
- No `budget_penalty_lambda` in loss
- No `budget_penalty_target` config

### 2.2.4 Gradient Flow Analysis

The soft top-k threshold creates a gradient path that differs from Gumbel-Sigmoid:

**Gumbel-Sigmoid (Phase 2):** Each logit gets gradient independently via `sigmoid'(logit / tau)`. Logits near 0 get the strongest gradient (sigmoid is steepest there). Logits far from 0 get vanishing gradient (sigmoid saturates). The budget penalty adds a uniform push toward/away from 0.

**Soft Top-K (Phase 2.2):** Each logit gets gradient via `sigmoid'((logit - threshold) / tau)`. The key difference:
- **Logits near the threshold** get the strongest gradient — these are the "boundary" neurons competing for the k-th slot
- **Logits far above threshold** get near-zero gradient (already confidently selected)
- **Logits far below threshold** get near-zero gradient (already confidently rejected)
- The **threshold itself** shifts based on the CE loss gradient: if the loss wants more capacity, the threshold drops (admitting more neurons); if it can tolerate less, the threshold rises

This concentrates gradient signal on the *decision boundary* — the neurons that are actually contested. Phase 2's Gumbel-Sigmoid wastes gradient on neurons that are clearly on or clearly off.

**Potential issue:** The `topk` + `min` operations involve a sort, which has zero gradient for tied values. In practice, logits are initialized with `linspace` so ties don't occur at init, and gradient updates push them apart over training. If ties become a problem, adding small uniform noise to logits before the topk resolves it.

### 2.2.5 Config Extensions

**File:** `olmo/config.py` — `HMatConfig`

```python
@dataclass
class HMatConfig(BaseConfig):
    # ... existing fields ...
    method: str = "fisher"  # "fisher" | "gumbel" | "fisher_gumbel" | "topk"

    # Phase 2.2 (Soft Top-K) — reuses these existing fields:
    #   gumbel_tau_start, gumbel_tau_end, gumbel_tau_anneal_steps
    #   gumbel_init_scale
    # No new fields needed — budget_penalty_lambda and budget_penalty_target
    # are simply unused when method="topk"
```

No new hyperparameters. The method reuses `gumbel_init_scale` for logit initialization and `gumbel_tau_*` for temperature annealing. The `budget_penalty_*` fields are ignored.

### 2.2.6 Training Config

**New file:** `configs/pile-tiny-hmat-topk.yaml`

```yaml
# Extends pile-tiny.yaml with Soft Top-K H-Mat
matformer_factor: 8
hmat:
  enabled: true
  method: topk
  gumbel_tau_start: 0.5        # Same best tau from Phase 2
  gumbel_tau_end: 0.1
  gumbel_init_scale: 1.1       # Same best scale from Phase 2
  # No budget_penalty_lambda or budget_penalty_target needed
```

### Tests for Phase 2.2

**File:** `tests/test_topk.py`

1. **test_topk_mask_shape:** Verify mask output has shape `(mlp_dim,)` for various k values.
2. **test_topk_mask_count:** At low tau with hard=True, verify exactly k dims are 1 and the rest are 0.
3. **test_topk_full_width_identity:** When `k >= mlp_dim`, verify mask is all-ones (no degradation).
4. **test_topk_gradient_flows_to_logits:** Run forward + CE loss + backward, verify `logits.grad` is not None.
5. **test_topk_gradient_concentrated_at_boundary:** With known logits, verify gradient magnitude is highest for logits near the threshold, near-zero for logits far from threshold.
6. **test_topk_temperature_effect:** High tau → soft transition; low tau → sharp near-binary transition.
7. **test_topk_different_k_per_factor:** In a single training step with factors {1,2,4,8}, verify each factor gets the correct k and the correct mask behavior.
8. **test_topk_no_budget_penalty:** Verify that training with method="topk" does NOT add any budget penalty to the loss.
9. **test_topk_ordering_preserved:** After several optimizer steps, verify that the relative ordering of logits changes (the model learns which dims matter), but the total active count at each k stays at k.
10. **test_topk_checkpoint_roundtrip:** Save and load topk state, verify logits are preserved.
11. **test_topk_forward_backward_integration:** Full integration test: build model, attach TopKMaskManager, run forward+backward at multiple factors, verify shapes and gradients.
12. **test_topk_vs_gumbel_at_factor1:** Verify that at factor=1, top-k produces identical output to an unmasked baseline (unlike Phase 2's Gumbel which scales by ~0.75).

### Expected Advantages Over Phase 2

1. **No full-model degradation.** At factor=1, `k = mlp_dim` → all-ones mask. The full model trains identically to baseline MatFormer. This directly addresses the +1.2% → +9% degradation pattern.

2. **No budget penalty hyperparameter.** The `lambda` and `target` parameters are eliminated. One fewer competing objective, one fewer hyperparameter to tune.

3. **Near-binary masks during training.** At reasonable tau (0.5), the sigmoid transition around the threshold is sharp. Neurons well above threshold get mask ≈ 1.0, not ≈ 0.75. This preserves more effective capacity during training.

4. **Gradient efficiency.** Gradient signal is concentrated on contested boundary neurons rather than spread across all neurons. The optimization problem is simpler: decide the ordering, not the absolute scale.

5. **Exact sparsity at inference.** Hard top-k gives exactly k active dims — no need to threshold and hope the count is close to target. Deployment is deterministic.

### Risks and Mitigations

1. **Threshold discontinuity.** The k-th largest logit shifts discretely when two logits swap rank. This could cause oscillation at the boundary.
   - *Mitigation:* Temperature smoothing (tau > 0) creates a soft transition zone. At tau=0.5, logits within ~1.0 of the threshold contribute partially, so small rank swaps don't cause sharp mask changes.

2. **`topk` gradient.** PyTorch's `topk` does not propagate gradients through the selection. But we don't need it to — gradients flow through the `sigmoid((logits - threshold) / tau)` computation. The threshold is computed in the forward pass as `topk(...).min()`, and `min` does propagate gradients (to the single element that achieved the minimum).
   - *Caveat:* Only the logit at exactly the threshold gets gradient from the threshold path. All other logits get gradient only from the `sigmoid(logit - threshold.detach())` term if we detach, or from both terms if we don't. Empirically, not detaching should work because the threshold shifts slowly.

3. **No inter-layer allocation flexibility.** Every layer gets exactly k active dims at factor=f. Unlike Phase 2 where one layer could have 60% active and another 40% at the same factor, here they all have the same count.
   - *Mitigation:* The *ordering* of neurons within each layer is still learned. Layer 0 might prioritize dims 0-50 while layer 3 prioritizes dims 200-250. The mask is per-layer, so each layer picks its own top-k from a different learned importance ordering. The reallocation happens through *which* dims are selected, not *how many*.
   - *Future extension:* Learnable per-layer k values (see "Per-Layer Budget Learning" below).

### Future Extension: Per-Layer Budget Learning

The simplest form of Phase 2.2 uses the same k at every layer for a given factor. A natural extension is to learn per-layer k values subject to a total budget constraint:

```
Constraint: sum(k_l for all layers l) = total_budget
```

This could be implemented as a higher-level Gumbel-Softmax over a small discrete set of per-layer budgets (e.g., each layer chooses from {k/4, k/2, k, 2k}), with the constraint enforced via a Lagrangian. This recovers the heterogeneous allocation from Phase 2 but with structural sparsity at each layer. This is left as a follow-up if Phase 2.2 baseline results are promising.

---

## Phase 2.3: Experimental Validation of Full-Width Fix — COMPLETE

### Phase 2.3 Results (6-run screening, 540 steps each, 1× A100)

| Run | Method | Init | tau | PPL 1/1 | PPL 1/2 | PPL 1/4 | PPL 1/8 | Layer fracs |
|-----|--------|------|-----|---------|---------|---------|---------|-------------|
| V1 | baseline | — | — | **141.6** | **147.2** | **153.0** | **159.6** | — |
| V2 | topk | linspace(1.1) | 0.5 | 146.0 (+3.1%) | 151.9 (+3.2%) | 158.2 (+3.4%) | 165.2 (+3.5%) | 0.50 uniform |
| V3 | gumbel_topk | linspace(1.1) | 0.5 | 147.1 (+3.9%) | 153.4 (+4.2%) | 159.8 (+4.4%) | 168.1 (+5.3%) | 0.50-0.51 |
| G1 | gumbel_topk | zeros | 0.5 | 149.8 (+5.8%) | 157.9 (+7.3%) | 164.3 (+7.4%) | 168.9 (+5.8%) | 0.50-0.51 |
| T1 | topk | zeros | 0.5 | 146.7 (+3.6%) | 156.6 (+6.4%) | 168.7 (+10.3%) | **236.0 (+47.9%)** | 0.50 uniform |
| A3 | gumbel_topk | linspace(1.1) | 1.0 | 148.3 (+4.7%) | 154.8 (+5.2%) | 161.6 (+5.6%) | 172.2 (+7.9%) | 0.50-0.51 |

### Phase 2.3 Analysis

**Key findings:**

1. **Baseline wins everywhere.** All mask methods add 3-8% PPL overhead at every sub-model width. The masks are a net negative — they consume training capacity without benefit.

2. **Root cause: soft mask noise.** During training, each step does 4 forward-backward passes (factors 1, 2, 4, 8). At factors > 1, soft masks multiply activations by values in [0,1], injecting noise into representations. These noisy gradients from sub-model passes pollute the shared model weights. Baseline does the same 4 passes but with clean prefix slicing — no soft noise.

3. **Masks barely learn.** Layer active fractions stayed at ~0.50 across all layers for all methods, meaning logits didn't move significantly from initialization. The gradient signal from the CE loss backpropagating through the soft mask is too weak for meaningful logit differentiation in 540 steps.

4. **TopK can't break zero-init symmetry.** T1 (topk zeros) catastrophically failed at 1/8 (PPL 236 vs baseline 160). Without Gumbel noise, deterministic topk has no mechanism to break all-tied logits. Gumbel noise helps (G1: 169 at 1/8), confirming noise is essential for symmetry breaking.

5. **Higher tau is worse.** A3 (tau=1.0) performed worse than V3 (tau=0.5) at all widths — more noise = more degradation.

6. **Contrast with Phase 2 (prefix slicing).** Phase 2 at 540 steps beat baseline at sub-model widths: PPL 158.2 (1/2, -2.2%), 162.2 (1/4, -3.0%), 167.1 (1/8, -3.4%). The full-width approach is strictly worse. The prefix ordering in Phase 2 was actually a useful inductive bias, not a limitation — it constrained the mask to a smaller space where gradients were stronger.

### Diagnosis: Why Masks Fail with Full-Width Compute

The core problem is that soft masks with values near 0.5 effectively halve neuron activations at sub-model widths. With linspace(+1.1, -1.1), sigmoid range is [0.25, 0.75] — every neuron is in an ambiguous "maybe" zone. This means:
- At factor=2 (k=1024): the top-k threshold selects ~1024 neurons, but their masks are soft (0.5-0.75), not crisp (near 1.0)
- The bottom neurons also have non-negligible mask values (0.25-0.5), so they "leak" into the output
- Gradients w.r.t. logits are weak because sigmoid'(x) is small when |x| is moderate

**Three complementary fixes identified:**
1. **Higher init scale** (e.g., 3.0 instead of 1.1) → sigmoid range [0.05, 0.95], neurons start clearly on/off
2. **Spread aux loss** → gradient pressure on logits to separate, sharpening the top-k boundary
3. **Tau → 0 at end** → masks become hard {0,1} at convergence, eliminating soft noise

---

## Phase 2.3b: Mask Sharpening Experiments

### Motivation

Phase 2.3 showed that soft masks inject noise that degrades training. The masks stayed at ~0.50 active fraction — logits didn't separate enough for crisp on/off decisions. Three complementary fixes address this from different angles.

### Changes

**1. Higher init scale (`init_scale=3.0`)**
- Config-only change: `--hmat.gumbel_init_scale=3.0`
- sigmoid(+3.0) = 0.95, sigmoid(-3.0) = 0.05 → neurons start clearly on or off
- Much less soft noise from step 1; top-k threshold is sharp from the start

**2. Spread aux loss (`spread_penalty_lambda`)**
- New config field: `HMatConfig.spread_penalty_lambda: float = 0.0`
- Loss: `-var(logits)` per layer, averaged across layers → maximizes logit separation
- Applied to both gumbel and topk methods (skip during vanilla warmup)
- Gradient pushes high logits higher and low logits lower, sharpening the top-k boundary

**3. Hard freeze at end (`gumbel_freeze_fraction=0.2`)**
- Existing config field, was set to 0.0
- `gumbel_freeze_fraction=0.2` → last 20% of training uses hard {0,1} masks
- Eliminates soft noise at convergence; final eval sees the actual discrete masks

### Experiment Design (3 runs, 540 steps each, ~4.5 hours on 1 GPU)

All three fixes applied simultaneously to the best method from Phase 2.3 (topk with linspace init, since it had the lowest PPL overhead). V1 baseline result (PPL 141.6 / 147.2 / 153.0 / 159.6) is reused as control.

| Run | Method | Init scale | Spread λ | Freeze frac | tau | Rationale |
|-----|--------|-----------|----------|-------------|-----|-----------|
| S1 | topk | 3.0 | 0.01 | 0.2 | 0.5 | All three fixes combined |
| S2 | gumbel_topk | 3.0 | 0.01 | 0.2 | 0.5 | Same fixes with Gumbel noise |
| S3 | topk | 3.0 | 0.0 | 0.0 | 0.5 | Scale-only control (isolate scale effect) |

**What this answers:**
- Does higher init scale alone fix the soft noise problem? (S3 vs V2 from Phase 2.3)
- Does the full fix package (scale + spread + freeze) close the gap to baseline? (S1 vs V1)
- Does Gumbel noise help or hurt with sharp masks? (S1 vs S2)

**Success criteria:** S1 or S2 should match or beat V1 baseline at sub-model widths (1/2, 1/4, 1/8) while staying within 2% at full width (1/1).

### Phase 2.3b Results

| Run | Config | PPL 1/1 | PPL 1/2 | PPL 1/4 | PPL 1/8 |
|-----|--------|---------|---------|---------|---------|
| V1 | baseline (no masks) | 141.6 | 147.2 | 153.0 | 159.6 |
| V2 | topk scale=1.1 (Phase 2.3) | 146.0 (+3.1%) | 151.9 (+3.2%) | 158.2 (+3.4%) | 165.2 (+3.5%) |
| **S3** | **topk scale=3.0 only** | **141.0 (-0.4%)** | **146.3 (-0.6%)** | **152.1 (-0.6%)** | **159.0 (-0.4%)** |
| **S1** | **topk scale=3.0 + spread(0.01) + freeze(0.2)** | **140.3 (-0.9%)** | **145.6 (-1.1%)** | **151.2 (-1.2%)** | **158.0 (-1.0%)** |
| S2 | gumbel_topk scale=3.0 + spread + freeze | 148.0 (+4.5%) | 154.1 (+4.7%) | 159.9 (+4.5%) | 169.5 (+6.2%) |

**Key findings:**

1. **Init scale is the critical fix.** Scale 3.0 alone (S3) turns a +3.4% regression into a -0.6% improvement. Sigmoid [0.05, 0.95] eliminates the soft noise that was degrading training with scale 1.1.

2. **Full fix package beats baseline by ~1% everywhere.** S1 (scale=3.0 + spread λ=0.01 + freeze=0.2) is the best config: -0.9% at full width and -1.0% at 1/8. The spread penalty and freeze provide incremental benefit on top of scale alone.

3. **Gumbel noise is catastrophically bad with sharp masks.** S2 (gumbel_topk) is 4-6% worse than S1 (topk) with identical settings. Deterministic topk is strictly better when masks are already decisive — Gumbel noise just adds harmful variance to an already-sharp threshold.

4. **The winning recipe:** `method=topk, init_scale=3.0, spread_penalty_lambda=0.01, gumbel_freeze_fraction=0.2`. This is ready for scale-up to Phase 2.4.

---

### Phase 2.3 Original Experimental Setup (archived)

### Experimental Setup

All experiments use the same infrastructure as prior phases:

- **Model:** 17M params, 4 layers, d_model=256, mlp_ratio=8, GELU, mlp_dim=2048
- **Data:** Pile-700M (train) / Pile (val)
- **Training:** matformer_factor=8, batch_size=128, sequence_length=2048
- **Short runs:** 540 steps (for screening). **Long runs:** 2700 steps (for winners)
- **Hardware:** 8x A100-40GB
- **Seed:** 6198 (same as all prior experiments for comparability)
- **Eval:** PPL at all sub-model widths {1/1, 1/2, 1/4, 1/8} using eval_gumbel_comparison.py

**Metrics collected per run:**
- Eval PPL at factors {1, 2, 4, 8} — the primary metric
- Per-layer active fraction at each factor
- Neuron reordering divergence: Kendall's tau between final logit ranking and initial logit ranking, per layer
- Training loss curves (CE only, no budget penalty for top-k/gumbel-top-k)

### Group 1: Fix Validation (3 runs, 540 steps)

Verify the full-width fix works by comparing to pre-fix baselines with identical configs.

| Run | Method | Init | Config | Baseline comparison |
|-----|--------|------|--------|---------------------|
| **V1** | Baseline | — | Uniform MatFormer, no masks | Control (expect ~155.9) |
| **V2** | Top-K + fix | linspace(1.1) | tau=0.5 | Compare to pre-fix top-k |
| **V3** | Gumbel-Top-K + fix | linspace(1.1) | tau=0.5 | Compare to pre-fix gumbel (157.7) |

**What this answers:** Does the fix help, hurt, or have no effect when using the same linspace init that was previously required?

**Expected:** At linspace init, the logit ordering already matches prefix ordering (97-99% overlap in Phase 2 results). So the fix should produce similar or slightly better results — the mask is doing the same thing but without the prefix-alignment fragility.

### Group 2: Unbiased Initialization (8 runs, 540 steps)

The central experiment. With prefix-alignment removed, test initialization strategies that were previously impossible.

**Gumbel-Top-K runs:**

| Run | Init | Rationale |
|-----|------|-----------|
| **G1** | `zeros` (all logits = 0) | Previously catastrophic (sigmoid(0)=0.5). Now: all logits tied → Gumbel noise alone breaks symmetry. Top-k threshold is random each step. Tests pure exploration. |
| **G2** | `N(0, 0.3)` | Small random perturbation. Breaks symmetry immediately but with no positional bias. Gumbel noise adds further exploration. |
| **G3** | `constant(+1.5)` | All logits equally positive (sigmoid ≈ 0.82). All neurons start "mostly on." The model starts at near-full capacity and learns what to prune. |
| **G4** | `linspace(1.1, -1.1)` | Control. Same as V3. Included for direct comparison within the group. |

**Top-K runs:**

| Run | Init | Rationale |
|-----|------|-----------|
| **T1** | `zeros` (all logits = 0) | All ties → threshold at 0 for any k → sigmoid(0/tau) = 0.5 for all. First gradient step breaks symmetry. Tests whether top-k can bootstrap from scratch. |
| **T2** | `N(0, 0.3)` | Random ranking from step 0. Top-k selects k random positions each step initially, gradient signal quickly reinforces useful neurons. |
| **T3** | `constant(+1.5)` | All tied like zeros but at a higher value. Same dynamics for top-k (threshold = 1.5, sigmoid(0/tau)=0.5). Gradient breaks symmetry. |
| **T4** | `linspace(1.1, -1.1)` | Control. Same as V2. |

**What this answers:**
- Can the methods learn from an unbiased starting point? (zeros, random)
- Is the linspace bias actually helpful or just a crutch for the broken prefix-slicing design?
- Does Gumbel noise help with symmetry breaking at zero/tied inits? (Compare G1 vs T1)

**Key predictions:**
- `zeros` should work for Gumbel-Top-K (noise breaks symmetry) but may struggle for Top-K (first gradient step must break all ties simultaneously)
- `N(0, 0.3)` should work for both — random ranking is a good starting point when positions don't matter
- `constant(+1.5)` is interesting: it starts at high capacity (like linspace) but without positional bias. The model must learn which neurons to turn off rather than which to turn on. May converge differently from linspace
- `linspace` controls should match Group 1 results

### Group 3: Gumbel-Top-K vs Top-K Ablation (4 runs, 540 steps)

Isolate the effect of Gumbel noise on top-k training, using the best init from Group 2.

| Run | Method | Init | Tau | Notes |
|-----|--------|------|-----|-------|
| **A1** | Top-K | best init | 0.5 | Deterministic threshold |
| **A2** | Gumbel-Top-K | best init | 0.5 | Stochastic threshold via Gumbel noise |
| **A3** | Gumbel-Top-K | best init | 1.0 | Higher tau = more exploration |
| **A4** | Gumbel-Top-K | best init | 0.2 | Lower tau = sharper, less exploration |

**What this answers:**
- Does stochastic exploration (Gumbel noise) help or hurt compared to deterministic top-k?
- How does tau interact with the noise? High tau + noise = very soft; low tau + noise = mostly deterministic with occasional boundary jitter

**Hypothesis:** Gumbel noise helps at zero/random init (breaks symmetry faster) but adds unnecessary variance at linspace init (ordering is already good). At low tau the noise effect vanishes; at high tau the noise dominates and prevents convergence. Tau=0.5 should be the sweet spot.

### Group 4: Neuron Reordering Analysis (post-hoc, no extra runs)

Run on all completed experiments from Groups 1-3. This is analysis only, no additional training.

For each run, after training:
1. Extract final logit ranking per layer (argsort descending)
2. Compare to initial logit ranking:
   - **Kendall's tau** (rank correlation, -1 to +1): 1.0 = identical ordering, 0 = random
   - **Top-k overlap**: fraction of top-k neurons at init that remain in top-k after training, for k ∈ {256, 512, 1024}
   - **Per-layer reordering magnitude**: average absolute rank change per neuron

**What this answers:**
- In Phase 2 (prefix-sliced), 97-99% of ordering was preserved — the mask mostly learned per-layer width thresholds, not per-neuron importance. With full-width compute, is there more reordering?
- Do layers reorder differently? (Phase 2 found layer 3 reordered most: 86% overlap vs 97-99% for layers 0-2)
- Does unbiased init (zero/random) lead to more diverse per-layer orderings than linspace?

**Expected:** Significantly more reordering than Phase 2 because neurons are no longer constrained to prefix positions. Random init should show the most reordering (no initial bias to preserve). Linspace init may still show less reordering because the initial ordering is a reasonable prior.

### Group 5: Long-Run Validation (2-3 runs, 2700 steps)

Promote the top 2-3 configs from Groups 1-3 to 2700-step runs. The critical question is whether the Phase 2 degradation pattern (+2.6% at 540 steps → +9% at 2700 steps for full model) is eliminated by the fix.

| Run | Method | Init | Steps |
|-----|--------|------|-------|
| **L1** | Baseline (uniform MatFormer) | — | 2700 |
| **L2** | Best from Groups 1-3 | — | 2700 |
| **L3** | Second best from Groups 1-3 | — | 2700 |

**What this answers:**
- Does the fix prevent the full-model degradation that worsened over training? (The +9% gap at 2700 steps was the main weakness of Phase 2)
- Do the sub-model improvements hold at longer training? (Phase 2 showed gains at 1/2, 1/4, 1/8 at 540 steps — do they persist?)
- Does neuron reordering increase with more training time?

**Success criteria:**
- Full model (1/1): within 1% of baseline (vs +9% in Phase 2 at 2700 steps)
- Sub-models (1/2, 1/4, 1/8): beat baseline (vs +4.4%, +3.4%, +2.3% worse in Phase 2)
- Neuron reordering: measurably different from Phase 2's 97-99% preservation

### Run Schedule

Total: 15-18 short runs + 2-3 long runs = 17-21 runs.

With 8 GPUs and 1 GPU per 540-step run (~15 min each), Groups 1-3 can be run in 3 batches:

```
Batch 1 (8 GPUs): V1, V2, V3, G1, G2, G3, G4, T1     ~15 min
Batch 2 (8 GPUs): T2, T3, T4, A1, A2, A3, A4, [spare]  ~15 min
Batch 3 (2-3 GPUs): L1, L2, L3                           ~75 min (2700 steps)
```

Analysis (Group 4) runs post-hoc on CPU using saved checkpoints.

### Summary of Key Questions

| # | Question | Experiments | Expected outcome |
|---|----------|-------------|------------------|
| 1 | Does the fix work? | V1-V3 | Similar or better than pre-fix at linspace init |
| 2 | Can methods learn from unbiased init? | G1-G4, T1-T4 | Yes — zero/random should work now |
| 3 | Is linspace still best, or just a crutch? | G1-G4, T1-T4 | Random may match or beat linspace |
| 4 | Does Gumbel noise help top-k? | A1-A4 | Helps at unbiased init, neutral at linspace |
| 5 | Does the fix eliminate long-run degradation? | L1-L3 | Full-model gap drops from +9% to <1% |
| 6 | Do neurons genuinely reorder now? | Group 4 analysis | Significantly more reordering than Phase 2's 97-99% preservation |

---

## Phase 2.3c: Fisher Saliency Re-Evaluation

### Motivation

The original Fisher saliency experiments (Phase 1 and Phase 2.1) ran **before** the masking fix in commit `e9b5ba3`. That fix changed `gumbel.py`, `model.py`, and `train.py` — correcting how masks are applied in the forward pass. This means all prior Fisher results are invalid: it's unknown whether Fisher underperformed due to the approach itself or the broken masking logic.

Additionally, the current winning config (topk, `init_scale=3.0`) uses a **position-based linear decay** for initial logit ranking. Fisher saliency offers a **data-driven** alternative: initialize logit rankings from actual neuron importance scores. With the corrected masking and the sharp-init + topk recipe now established, Fisher can be tested as a drop-in replacement for the initialization strategy.

### What to Test

Three uses of Fisher saliency, in increasing complexity:

**1. Fisher-informed initialization (simplest)**

Use the vanilla warmup phase (first 15% of training) to accumulate Fisher EMA scores. At the transition to masked training, initialize topk logits from saliency rankings instead of linear decay:

```
# Instead of: logits = linspace(+scale, -scale, mlp_dim)
# Use: logits[rank_by_fisher] = linspace(+scale, -scale, mlp_dim)
```

After initialization, training proceeds identically to the current topk recipe — logits are learned via backprop, no ongoing Fisher involvement.

**Why this might help:** The current linspace init assigns importance by position (neuron 0 > neuron 1 > ... > neuron N). This is arbitrary. Fisher-informed init assigns importance by measured gradient magnitude, giving the optimizer a better starting point. With `init_scale=3.0` the initial ranking is highly persistent (Phase 2.3 showed ~97% overlap between init and final rankings), so a better initial ranking directly translates to a better final ranking.

**2. Periodic Fisher re-ranking**

Same as (1), but periodically (every K steps) re-rank logits using updated Fisher EMA scores. This was the Phase 2.1 design. Key difference from the original attempt: masks are now applied correctly, and we use deterministic topk instead of Gumbel noise.

**Why this might help or hurt:** On one hand, ongoing re-ranking adapts to changing saliency as the model trains. On the other hand, it fights with gradient-based logit learning — the optimizer moves logits one direction, then Fisher resets them. The Phase 2.3b results showed that stable, sharp masks work best, so periodic re-ranking may introduce harmful instability.

**3. Fisher-guided spread loss (lightest touch)**

Instead of re-ranking, use Fisher scores to **weight** the spread penalty. Currently spread loss is `-var(logits)` uniformly. A Fisher-weighted version would penalize ambiguity more for high-saliency neurons (where the keep/prune decision matters most):

```python
weights = fisher_ema[layer] / fisher_ema[layer].sum()
weighted_var = (weights * (logits - logits.mean()) ** 2).sum()
loss = -weighted_var
```

This is the lightest integration — Fisher provides a signal but doesn't override the learned logits.

### Experiment Design

All runs use the Phase 2.3b winning config as baseline: `method=topk, init_scale=3.0, spread_penalty_lambda=0.01, gumbel_freeze_fraction=0.2, vanilla_warmup_frac=0.15`.

| Run | Description | Change from baseline | Key question |
|-----|-------------|---------------------|--------------|
| F0 | Baseline (reuse S1 from 2.3b) | None — control | — |
| F1 | Fisher-informed init, 15% warmup | `--hmat.fisher_init=true` | Does data-driven ordering beat positional? |
| F2 | Fisher-informed init, 30% warmup | `--hmat.fisher_init=true --hmat.vanilla_warmup_frac=0.3` | Does more Fisher data improve the ranking? |
| F3 | Fisher-weighted spread, best warmup from F1/F2 | `--hmat.fisher_init=true --hmat.fisher_weighted_spread=true` + best warmup | Should spread focus on important neurons? |

540-step runs, ~92 min each × 3 = ~4.5 hours on 1 GPU.

**Why periodic re-ranking was dropped:** Phase 2.3b showed that stability is critical with sharp masks — Gumbel noise (S2) was catastrophically bad. Periodic re-ranking is the same class of instability: it disrupts the learned logit ordering every K steps, even with optimizer reset. Predicted to hurt based on prior evidence. If Fisher init proves valuable, periodic re-ranking can be tested later at larger scale where the tradeoff may differ.

**Why F2 (longer warmup) was added:** With 15% warmup (81 steps), Fisher EMA accumulates scores from a small window. The EMA β=0.99 has an effective window of ~100 steps, so 81 steps may produce noisy rankings. With 30% warmup (162 steps), Fisher gets 2× more gradient data for a more stable ranking. The tradeoff is fewer masked training steps (378 vs 459), but if the ranking quality is the bottleneck, it's worth testing.

**Why F3 uses the best warmup from F1/F2:** F3 tests Fisher-weighted spread as an orthogonal improvement. To avoid confounding, it should use whichever warmup length won in the F1 vs F2 comparison. The script automatically picks the winner.

### Implementation

**New config fields** (`HMatConfig`):
```python
fisher_init: bool = False          # Accumulate Fisher during warmup, re-init logits at transition
fisher_rerank_interval: int = 0    # Re-rank logits from Fisher every K steps (0 = init only)
fisher_weighted_spread: bool = False  # Weight spread loss by Fisher saliency
```

**Code changes:**
- `olmo/config.py`: Add three fields above
- `olmo/train.py`:
  - `init_topk()`: When `fisher_init=True`, create FisherEMA alongside TopKMaskManager
  - `train_step()` warmup phase: Accumulate Fisher EMA from gradients during vanilla warmup
  - `train_step()` warmup→mask transition: If `fisher_init`, call `fisher_ema.get_logits(scale=3.0)` and copy into topk logits via `mask.logits.copy_(fisher_logits)`
  - `train_step()` post-warmup: If `fisher_rerank_interval > 0`, periodically re-rank
- `olmo/hmat/topk.py`: Add `spread_loss(fisher_weights=None)` variant

**CLI overrides for experiments:**
- F1: `--hmat.fisher_init=true`
- F2: `--hmat.fisher_init=true --hmat.fisher_rerank_interval=50`
- F3: `--hmat.fisher_init=true --hmat.fisher_weighted_spread=true`

### Success Criteria

- F1 should show whether data-driven init beats positional init — even a small improvement validates the approach since it's zero-cost at training time (Fisher accumulates during warmup for free)
- F2 is expected to hurt (based on the stability findings from 2.3b) but worth confirming
- F3 is the speculative bet — if Fisher-weighted spread helps, it suggests the uniform spread penalty is suboptimal

---

## Phase 2.4: Scale-Up & Publication Experiments

Phase 2.3 validates the full-width fix and method selection at the current tiny scale. This phase takes the winning configuration and scales it up to produce publishable results.

**Prerequisite:** Critical checkpoint-restore bug fixes must land first (gumbel/topk state is saved but never loaded back; see `plan-critical-bugfixes.md`).

### 2.4.1 Checkpoint Resume Validation

Before any long run, confirm that checkpoint resume correctly preserves mask state.

- Train 500 steps with best config, save checkpoint
- Resume from checkpoint, train 500 more steps
- Compare against uninterrupted 1000-step run
- Assert: mask logits are identical at step 500, final PPL matches within noise

### 2.4.2 Long Training on 17M Model

Take the Phase 2.3 winner and train for 10K–50K steps on pile-700M.

- **Goal:** Determine whether the Gumbel/TopK advantage grows, holds, or vanishes with longer training
- Use `init_scale=1.1` (best from prior sweeps) as default
- Log per-layer active fractions every 100 steps for allocation heatmaps

### 2.4.3 Scale to 180M Model

Same best config, 180M model (more layers → real "middle" for U-shape to appear).

- **Goal:** Confirm U-shaped allocation is more pronounced with deeper models (4-layer model has no true intermediate layers)
- Compare uniform MatFormer vs H-Mat at each sub-model width

### 2.4.4 Downstream Task Evaluation

Eval at multiple sub-model sizes ({1/1, 1/2, 1/4, 1/8}) using existing eval infrastructure:

- PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge
- Compare uniform MatFormer vs H-Mat at each size
- Run on both 17M long-run and 180M checkpoints

### 2.4.5 Pareto Frontier

Plot accuracy (or PPL) vs FLOPs for uniform MatFormer and H-Mat across budgets {1/1, 1/2, 1/4, 1/8}. This is the key paper figure — H-Mat should show a strictly better frontier at compressed sizes.

### 2.4.6 Allocation Visualization

Using per-layer active fractions logged during training (2.4.2), produce a heatmap of layer × training step showing U-shape emergence over time.

### 2.4.7 Reordering Sanity Check

After training, run `scripts/reorder_neurons.py`, then eval the reordered checkpoint at each sub-model width. PPL should match the mask-based eval — if it doesn't, the reordering is lossy.

---

## Phase 2.0.2 Fix — Critical Bug Fixes and Performance Optimizations

This phase addressed three critical bugs that blocked checkpoint resumption and two bugs affecting training correctness, plus three performance optimizations to reduce overhead in the Gumbel masking system. All changes were verified end-to-end on A100 GPU.

### Bug Fix 1: Checkpoint Restore for Gumbel/TopK State (Critical)

**Problem**: Gumbel and TopK mask state (learned logits + optimizer momentum) was saved to both sharded and unsharded checkpoints, but never loaded back. Any training run resumed from a checkpoint silently reinitialized masks from scratch, destroying all learned allocation patterns.

**Root cause**: `init_gumbel()` and `init_topk()` were called inside `fit()`, which runs *after* `restore_checkpoint()` in `scripts/train.py`. Even if restore loaded the state, `init_gumbel()` would overwrite it with fresh defaults.

**Fix** (4 parts):

| Change | File | Description |
|--------|------|-------------|
| `init_masks()` method | `olmo/train.py` | New wrapper calling `init_gumbel()` + `init_topk()`, called before checkpoint restore |
| Early init in entry point | `scripts/train.py` | `trainer.init_masks()` called immediately after Trainer creation, before any checkpoint ops |
| Guarded init in `fit()` | `olmo/train.py` | Changed from unconditional to `if self.gumbel_manager is None: self.init_gumbel()` |
| Sharded restore | `olmo/train.py` | Loads `gumbel_manager`, `gumbel_optim`, `topk_manager`, `topk_optim` from `state_dict` if present |
| Unsharded restore | `olmo/train.py` | Loads `gumbel.pt`, `gumbel_optim.pt`, `topk.pt`, `topk_optim.pt` with `FileNotFoundError` fallback for older checkpoints |
| Remote upload | `olmo/train.py` | Extended unsharded upload loop to include gumbel/topk files with `exists()` guard |

**Verification**: `scripts/test_checkpoint_restore.py` — trains 10 steps, saves checkpoint, restores into fresh Trainer, asserts logits match pre-save values and differ from fresh init. Tested for both `gumbel` and `topk` methods on A100.

### Bug Fix 2: Gumbel Freeze Destroys Optimizer State

**Problem**: When the freeze phase activated (last X% of training), the code set `self.gumbel_optim = None`, destroying the optimizer and its momentum buffers. Any checkpoint saved after freeze had no gumbel optimizer state, breaking resumption.

**Fix**: Added `_gumbel_frozen: bool = False` to the Trainer dataclass. Replaced `self.gumbel_optim = None` with `self._gumbel_frozen = True`. Gated `zero_grad()` and `optim.step()` on the flag instead of a None check. The optimizer stays alive for checkpointing while updates are stopped.

**Verification**: `scripts/test_freeze_warmup.py` — trains with `gumbel_freeze_fraction=0.4`, verifies `gumbel_optim is not None` after freeze, `_gumbel_frozen == True`, `gumbel_optim.pt` saved with populated state, and state survives a checkpoint round-trip.

### Bug Fix 3: `_in_vanilla_warmup` Not a Dataclass Field

**Problem**: `_in_vanilla_warmup` was accessed via `getattr(self, '_in_vanilla_warmup', False)`, relying on it being set dynamically during `train_step`. If accessed before the first step (e.g., during checkpoint restore), it would silently fall back to `False` with no indication of the fragile pattern.

**Fix**: Added `_in_vanilla_warmup: bool = False` as a proper Trainer dataclass field. Replaced all `getattr` calls with direct `self._in_vanilla_warmup` access.

**Verification**: Same `test_freeze_warmup.py` — confirms `hasattr(trainer, '_in_vanilla_warmup')` is True immediately after construction, MatformerManager mode is `"uniform"` during warmup, and logits change after warmup ends.

### Optimization 1: Cast Gumbel Masks to bf16 Before Multiplication

**Problem**: Gumbel masks are float32. Under AMP, hidden states `h` are bf16. Multiplying `h * mask` promoted to float32, creating unnecessary precision bounce.

**Fix** (`olmo/model.py`, 3 sites): Added `.to(dtype=h.dtype)` before `unsqueeze` in all mask application lines. No-op when already matching dtype.

**Verification**: Confirmed result stays bf16 (was being promoted to f32). Numerically equivalent within bf16 tolerance.

### Optimization 2: Vectorize Budget Loss Computation

**Problem**: `budget_loss()` looped over `self.masks` accumulating `sigmoid().sum()` per layer — N intermediate scalar tensors and Python loop overhead.

**Fix** (`olmo/hmat/gumbel.py`): Replaced with `torch.cat([m.logits for m in self.masks])` → `sigmoid` → `mean` — single fused operation.

**Verification**: Exact match (diff=0.00e+00) with old loop version at targets {0.1, 0.3, 0.5, 0.7, 0.9}. Gradients flow correctly.

### Optimization 3: Cache Sliced Weight Views Per Factor

**Problem**: Every forward pass through `OlmoSequentialBlock` when `factor > 1` recomputed `weight[:k]` slice views and `int(n / factor)`. While views are cheap, this happens for every micro-batch at every sub-model width.

**Fix** (`olmo/model.py`): Added `_get_ff_params(factor)` method that caches `(w_proj, b_proj, k_out, w_out, b_out)` and invalidates when factor changes. FSDP safety: cache invalidated when `_full_param_padded` attribute is detected on weights (indicates resharding).

**Verification**: Repeated forwards at same factor produce bit-identical output. Different factors produce different output. `_cached_factor` correctly tracks state. End-to-end 10-step training with all optimizations active produces no NaNs.

### Files Modified

| File | Changes |
|------|---------|
| `olmo/train.py` | `init_masks()`, guarded `fit()`, restore gumbel/topk in both checkpoint paths, `_gumbel_frozen` flag, `_in_vanilla_warmup` field, remote upload for mask files |
| `scripts/train.py` | `trainer.init_masks()` before checkpoint restore |
| `olmo/model.py` | `.to(dtype=h.dtype)` on 3 mask sites, `_get_ff_params()` weight view cache with FSDP invalidation |
| `olmo/hmat/gumbel.py` | Vectorized `budget_loss()` |

### Test Scripts Added

| Script | Purpose |
|--------|---------|
| `scripts/test_checkpoint_restore.py` | Gumbel + TopK checkpoint save/restore round-trip (A100) |
| `scripts/test_freeze_warmup.py` | Vanilla warmup + freeze flag + post-freeze checkpoint (A100) |
| `scripts/test_optimizations.py` | bf16 cast, vectorized budget loss, weight view cache, e2e training (A100) |
