"""Phase 1 tests: Fisher saliency computation and knapsack solver."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from olmo import Olmo
from olmo.config import ModelConfig
from olmo.hmat.fisher import compute_fisher_saliency
from olmo.hmat.knapsack import solve_budget_allocation
from olmo.model import MatformerManager


@pytest.fixture(autouse=True)
def reset_matformer_manager():
    """Reset the MatformerManager singleton before each test."""
    MatformerManager._instance = None
    yield
    MatformerManager._instance = None


@pytest.fixture
def small_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50257,
        eos_token_id=50256,
        pad_token_id=50256,
        d_model=128,
        n_heads=2,
        n_layers=3,
        mlp_ratio=4,
        activation_type="gelu",
        max_sequence_length=64,
        init_device="cpu",
    )


@pytest.fixture
def small_model(small_model_config) -> Olmo:
    return Olmo(small_model_config)


def make_dummy_dataloader(vocab_size: int, seq_len: int, batch_size: int = 4, num_batches: int = 2):
    """Create a simple DataLoader of random token IDs for testing."""
    total = batch_size * num_batches
    input_ids = torch.randint(0, vocab_size, (total, seq_len))
    dataset = TensorDataset(input_ids)

    def collate_fn(batch):
        ids = torch.stack([b[0] for b in batch])
        return {"input_ids": ids}

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


# ============================================================
# Fisher Saliency Tests
# ============================================================

class TestFisherSaliencyShape:
    def test_output_has_correct_keys(self, small_model):
        loader = make_dummy_dataloader(50257, 64, batch_size=2, num_batches=1)
        saliency = compute_fisher_saliency(small_model, loader, num_batches=1)
        assert set(saliency.keys()) == {0, 1, 2}

    def test_output_has_correct_shape(self, small_model):
        loader = make_dummy_dataloader(50257, 64, batch_size=2, num_batches=1)
        saliency = compute_fisher_saliency(small_model, loader, num_batches=1)
        mlp_dim = 4 * 128  # mlp_ratio * d_model
        for l in range(3):
            assert saliency[l].shape == (mlp_dim,)


class TestFisherSaliencyNormalized:
    def test_each_layer_sums_to_one(self, small_model):
        loader = make_dummy_dataloader(50257, 64, batch_size=2, num_batches=2)
        saliency = compute_fisher_saliency(small_model, loader, num_batches=2)
        for l in range(3):
            total = saliency[l].sum().item()
            assert abs(total - 1.0) < 1e-5, f"Layer {l} saliency sums to {total}, expected 1.0"


class TestFisherSaliencyNonzero:
    def test_scores_are_nonnegative(self, small_model):
        loader = make_dummy_dataloader(50257, 64, batch_size=2, num_batches=1)
        saliency = compute_fisher_saliency(small_model, loader, num_batches=1)
        for l in range(3):
            assert (saliency[l] >= 0).all(), f"Layer {l} has negative saliency scores"

    def test_at_least_some_nonzero(self, small_model):
        loader = make_dummy_dataloader(50257, 64, batch_size=2, num_batches=1)
        saliency = compute_fisher_saliency(small_model, loader, num_batches=1)
        for l in range(3):
            assert saliency[l].sum().item() > 0, f"Layer {l} has all-zero saliency"


# ============================================================
# Knapsack Solver Tests
# ============================================================

class TestKnapsackBudgetRespected:
    def test_total_dims_within_budget(self):
        """Verify the allocation respects the budget constraint."""
        mlp_dim = 512
        n_layers = 8
        # Create uniform saliency
        saliency = {l: torch.ones(mlp_dim) / mlp_dim for l in range(n_layers)}

        for ratio in [0.125, 0.25, 0.5]:
            result = solve_budget_allocation(saliency, ratio, [1, 2, 4, 8])
            total_dims = sum(mlp_dim // result[l] for l in range(n_layers))
            budget = int(ratio * n_layers * mlp_dim)
            assert total_dims <= budget, (
                f"ratio={ratio}: total_dims={total_dims} > budget={budget}"
            )


class TestKnapsackTrivial:
    def test_full_budget_gives_factor_1(self):
        """With budget_ratio=1.0, all layers should get factor=1 (full width)."""
        mlp_dim = 512
        n_layers = 4
        saliency = {l: torch.ones(mlp_dim) / mlp_dim for l in range(n_layers)}

        result = solve_budget_allocation(saliency, 1.0, [1, 2, 4, 8])
        for l in range(n_layers):
            assert result[l] == 1, f"Layer {l} got factor={result[l]}, expected 1"


class TestKnapsackMinimal:
    def test_tiny_budget_gives_max_factor(self):
        """With very small budget, all layers should get the maximum factor."""
        mlp_dim = 512
        n_layers = 4
        saliency = {l: torch.ones(mlp_dim) / mlp_dim for l in range(n_layers)}

        # Budget for 1/8 of one layer = very tight
        ratio = 1.0 / (n_layers * 8) * 0.5  # less than minimum possible
        result = solve_budget_allocation(saliency, ratio, [1, 2, 4, 8])
        for l in range(n_layers):
            assert result[l] == 8, f"Layer {l} got factor={result[l]}, expected 8"


class TestKnapsackHeterogeneous:
    def test_ushape_saliency_gives_heterogeneous_allocation(self):
        """With U-shaped saliency (high at ends, low in middle), end layers should
        get lower factors (more width) and middle layers higher factors (less width)."""
        mlp_dim = 512
        n_layers = 8
        saliency = {}
        for l in range(n_layers):
            scores = torch.ones(mlp_dim)
            # U-shape: layers 0,1 and 6,7 get 10x higher saliency
            distance_from_edge = min(l, n_layers - 1 - l)
            if distance_from_edge <= 1:
                scores *= 10.0
            scores = scores / scores.sum()
            saliency[l] = scores

        # Budget for ~25%: should assign more to edges, less to middle
        result = solve_budget_allocation(saliency, 0.25, [1, 2, 4, 8])

        # Edge layers (0,1,6,7) should have lower factor (wider) than middle
        edge_factors = [result[l] for l in [0, 1, 6, 7]]
        middle_factors = [result[l] for l in [2, 3, 4, 5]]

        avg_edge = sum(edge_factors) / len(edge_factors)
        avg_middle = sum(middle_factors) / len(middle_factors)

        assert avg_edge <= avg_middle, (
            f"Edge layers avg factor ({avg_edge}) should be <= "
            f"middle layers avg factor ({avg_middle})"
        )


# ============================================================
# End-to-End Test
# ============================================================

class TestFmatEndToEnd:
    def test_fisher_then_knapsack_then_forward(self, small_model):
        """Full pipeline: compute Fisher, solve knapsack, apply to manager, forward pass."""
        loader = make_dummy_dataloader(50257, 64, batch_size=2, num_batches=1)

        # Step 1: Compute saliency
        saliency = compute_fisher_saliency(small_model, loader, num_batches=1)

        # Step 2: Solve knapsack
        allocation = solve_budget_allocation(saliency, 0.5, [1, 2, 4])

        # Step 3: Apply to MatformerManager
        mgr = MatformerManager.get_instance()
        mgr.mode = "heterogeneous"
        mgr.layer_factors = allocation

        # Step 4: Forward pass
        small_model.eval()
        input_ids = torch.randint(0, 50257, (1, 64))
        with torch.no_grad():
            output = small_model(input_ids)

        # Verify output shape
        assert output.logits.shape == (1, 64, small_model.config.embedding_size or small_model.config.vocab_size)

        # Reset
        mgr.mode = "uniform"
        mgr.layer_factors = None
