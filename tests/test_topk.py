"""Phase 2.2 tests: Soft Top-K structural sparsity for H-Mat."""

import math

import pytest
import torch
from torch.nn import CrossEntropyLoss

from olmo import Olmo
from olmo.config import HMatConfig, ModelConfig
from olmo.hmat.topk import TopKMaskLayer, TopKMaskManager
from olmo.model import Activation, MatformerManager


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
        max_sequence_length=64,
        init_device="cpu",
    )


def _post_act_mlp_dim(config: ModelConfig) -> int:
    """Compute the post-activation MLP hidden dim (accounts for SwiGLU halving)."""
    act = Activation.build(config)
    return int(config.mlp_ratio * config.d_model * act.output_multiplier)


# ============================================================
# TopKMaskLayer Tests
# ============================================================


class TestTopKMaskLayerShape:
    def test_output_shape(self):
        mask_layer = TopKMaskLayer(512)
        mask = mask_layer(k=256, tau=1.0)
        assert mask.shape == (512,)

    def test_output_shape_various_k(self):
        mask_layer = TopKMaskLayer(512)
        for k in [1, 64, 128, 256, 511]:
            mask = mask_layer(k=k, tau=1.0)
            assert mask.shape == (512,)


class TestTopKMaskCount:
    def test_hard_mask_exactly_k(self):
        """At low tau with hard=True, exactly k dims should be 1."""
        mask_layer = TopKMaskLayer(512, init_scale=2.0)
        mask_layer.eval()
        for k in [64, 128, 256]:
            mask = mask_layer(k=k, tau=0.01, hard=True)
            active = mask.sum().item()
            assert active == k, f"Expected {k} active dims, got {active}"

    def test_hard_mask_binary(self):
        """Hard mask should only contain 0s and 1s."""
        mask_layer = TopKMaskLayer(256)
        mask_layer.eval()
        mask = mask_layer(k=128, tau=0.01, hard=True)
        unique_vals = mask.unique()
        for v in unique_vals:
            assert v.item() in (0.0, 1.0), f"Hard mask contains non-binary value {v}"


class TestTopKFullWidthIdentity:
    def test_full_width_returns_ones(self):
        """When k >= mlp_dim, mask should be all-ones."""
        mask_layer = TopKMaskLayer(256)
        mask = mask_layer(k=256, tau=1.0)
        assert torch.allclose(mask, torch.ones(256))

    def test_full_width_larger_k(self):
        """k > mlp_dim should also return all-ones."""
        mask_layer = TopKMaskLayer(256)
        mask = mask_layer(k=1024, tau=1.0)
        assert torch.allclose(mask, torch.ones(256))


class TestTopKGradients:
    def test_gradient_flows_to_logits(self):
        """Forward + CE loss + backward should produce gradients on logits."""
        mask_layer = TopKMaskLayer(128)
        mask_layer.train()
        mask = mask_layer(k=64, tau=1.0)
        loss = mask.sum()
        loss.backward()
        assert mask_layer.logits.grad is not None
        assert mask_layer.logits.grad.abs().sum() > 0

    def test_gradient_concentrated_at_boundary(self):
        """Gradient magnitude should be highest for logits near the threshold."""
        mask_layer = TopKMaskLayer(256, init_scale=3.0)
        mask_layer.train()
        mask = mask_layer(k=128, tau=0.5)
        loss = mask.sum()
        loss.backward()

        grad = mask_layer.logits.grad.abs()
        logits = mask_layer.logits.detach()

        # Find the threshold (128th largest logit)
        topk_vals, _ = logits.topk(128)
        threshold = topk_vals.min()

        # Logits near threshold should have larger gradient than those far away
        near_threshold = (logits - threshold).abs() < 0.5
        far_from_threshold = (logits - threshold).abs() > 2.0

        if near_threshold.any() and far_from_threshold.any():
            mean_near = grad[near_threshold].mean()
            mean_far = grad[far_from_threshold].mean()
            assert mean_near > mean_far, (
                f"Gradient near threshold ({mean_near:.4f}) should be > "
                f"gradient far from threshold ({mean_far:.4f})"
            )

    def test_no_gradient_at_full_width(self):
        """At k >= mlp_dim, mask is all-ones constant — no gradient path to logits."""
        mask_layer = TopKMaskLayer(128)
        mask_layer.train()
        mask = mask_layer(k=128, tau=1.0)
        # All-ones mask should not depend on logits (no grad_fn)
        assert not mask.requires_grad, "Full-width mask should be a constant (no grad_fn)"
        assert torch.allclose(mask, torch.ones(128))


class TestTopKTemperature:
    def test_high_tau_soft_transition(self):
        """High temperature → soft, gradual transition."""
        mask_layer = TopKMaskLayer(256, init_scale=2.0)
        mask_layer.eval()
        mask = mask_layer(k=128, tau=10.0)
        # With high tau, values should be spread, not near-binary
        mid_range = ((mask > 0.2) & (mask < 0.8)).float().sum()
        assert mid_range > 50, f"Expected many mid-range values at high tau, got {mid_range}"

    def test_low_tau_sharp_transition(self):
        """Low temperature → sharp, near-binary transition."""
        mask_layer = TopKMaskLayer(256, init_scale=2.0)
        mask_layer.eval()
        mask = mask_layer(k=128, tau=0.01)
        # With low tau, values should be near 0 or 1
        dist_from_binary = torch.min(mask, 1.0 - mask)
        assert dist_from_binary.mean() < 0.01


class TestTopKDifferentFactors:
    def test_different_k_per_factor(self):
        """Different factors should produce different k values."""
        mask_layer = TopKMaskLayer(512, init_scale=2.0)
        mask_layer.eval()

        masks = {}
        for factor in [2, 4, 8]:
            k = 512 // factor
            mask = mask_layer(k=k, tau=0.01, hard=True)
            active = mask.sum().item()
            masks[factor] = active
            assert active == k, f"Factor {factor}: expected {k} active, got {active}"

        # Active counts should be strictly decreasing
        assert masks[2] > masks[4] > masks[8]


# ============================================================
# TopKMaskManager Tests
# ============================================================


class TestTopKMaskManager:
    def test_creation(self):
        mgr = TopKMaskManager(n_layers=4, mlp_dim=512)
        assert len(mgr.masks) == 4
        assert mgr.n_layers == 4
        assert mgr.mlp_dim == 512

    def test_get_mask_shape(self):
        mgr = TopKMaskManager(n_layers=4, mlp_dim=512)
        for i in range(4):
            mask = mgr.get_mask(i, k=256, tau=1.0)
            assert mask.shape == (512,)

    def test_get_layer_widths(self):
        mgr = TopKMaskManager(n_layers=3, mlp_dim=256)
        widths = mgr.get_layer_widths(k=128)
        assert len(widths) == 3
        for i in range(3):
            assert widths[i] == 128

    def test_log_summary(self):
        mgr = TopKMaskManager(n_layers=3, mlp_dim=256)
        summary = mgr.log_summary(k=128)
        assert f"topk/layer_0_active_frac_at_128" in summary
        assert f"topk/layer_1_active_frac_at_128" in summary
        assert f"topk/layer_2_active_frac_at_128" in summary


class TestTopKCheckpoint:
    def test_state_dict_roundtrip(self):
        """Save and load TopKMaskManager state dict."""
        mgr = TopKMaskManager(n_layers=3, mlp_dim=128)
        # Modify logits
        mgr.masks[0].logits.data = torch.randn(128)
        mgr.masks[1].logits.data = torch.randn(128) * 2.0
        mgr.masks[2].logits.data = torch.randn(128) * 0.5

        state = mgr.state_dict()

        # Create new manager and load
        mgr2 = TopKMaskManager(n_layers=3, mlp_dim=128)
        mgr2.load_state_dict(state)

        for i in range(3):
            torch.testing.assert_close(mgr.masks[i].logits, mgr2.masks[i].logits)


# ============================================================
# Integration Tests: TopK + Model Forward/Backward
# ============================================================


class TestTopKNoBudgetPenalty:
    def test_no_budget_penalty_in_loss(self):
        """TopK mode should NOT have any budget penalty — sparsity is structural."""
        mgr = TopKMaskManager(n_layers=3, mlp_dim=256)
        # Verify there is no budget_loss method
        assert not hasattr(mgr, "budget_loss")


class TestTopKForwardPass:
    def test_forward_with_topk_masks(self, small_model_config):
        """Forward pass with topk mode should produce correct output shape."""
        small_model_config.activation_type = "gelu"  # Use GELU for simpler dim calculation
        model = Olmo(small_model_config).eval()
        mlp_dim = _post_act_mlp_dim(small_model_config)
        topk_mgr = TopKMaskManager(
            n_layers=small_model_config.n_layers, mlp_dim=mlp_dim
        )

        matmng = MatformerManager.get_instance()
        matmng.mode = "topk"
        matmng.topk_masks = topk_mgr
        matmng.gumbel_tau = 1.0
        matmng.current_factor = 2  # Need factor > 1 for topk to engage

        input_ids = torch.randint(0, small_model_config.vocab_size, (2, 16))
        with torch.no_grad():
            output = model(input_ids)

        expected_vocab = small_model_config.embedding_size or small_model_config.vocab_size
        assert output.logits.shape == (2, 16, expected_vocab)

    def test_forward_topk_at_factor1_matches_baseline(self, small_model_config):
        """At factor=1, topk produces all-ones mask — output should match baseline."""
        small_model_config.activation_type = "gelu"
        model = Olmo(small_model_config).eval()
        mlp_dim = _post_act_mlp_dim(small_model_config)
        topk_mgr = TopKMaskManager(
            n_layers=small_model_config.n_layers, mlp_dim=mlp_dim
        )

        input_ids = torch.randint(0, small_model_config.vocab_size, (1, 16))

        # Baseline: uniform mode, factor=1
        matmng = MatformerManager.get_instance()
        matmng.mode = "uniform"
        matmng.current_factor = 1
        with torch.no_grad():
            out_uniform = model(input_ids)

        # TopK mode, factor=1 — should be identical (no mask applied)
        matmng.mode = "topk"
        matmng.topk_masks = topk_mgr
        matmng.gumbel_tau = 1.0
        matmng.current_factor = 1
        topk_mgr.eval()
        with torch.no_grad():
            out_topk = model(input_ids)

        torch.testing.assert_close(out_uniform.logits, out_topk.logits, atol=1e-5, rtol=1e-5)


class TestTopKBackward:
    def test_backward_with_topk(self, small_model_config):
        """Backward pass with topk should produce gradients on model AND mask logits."""
        small_model_config.activation_type = "gelu"
        model = Olmo(small_model_config).train()
        mlp_dim = _post_act_mlp_dim(small_model_config)
        topk_mgr = TopKMaskManager(
            n_layers=small_model_config.n_layers, mlp_dim=mlp_dim
        )
        topk_mgr.train()

        matmng = MatformerManager.get_instance()
        matmng.mode = "topk"
        matmng.topk_masks = topk_mgr
        matmng.gumbel_tau = 1.0
        matmng.current_factor = 2  # Factor > 1 so masks engage

        input_ids = torch.randint(0, small_model_config.vocab_size, (2, 16))
        output = model(input_ids)
        logits = output.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        loss.backward()

        # Model params should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Model param {name} has no gradient"

        # TopK mask logits should have gradients
        for i, mask_layer in enumerate(topk_mgr.masks):
            assert mask_layer.logits.grad is not None, f"Mask layer {i} logits have no gradient"
            assert mask_layer.logits.grad.abs().sum() > 0, f"Mask layer {i} has zero gradients"


class TestTopKOrderingEvolution:
    def test_logits_change_after_optimization(self):
        """After optimizer steps, logits should change (model learns which dims matter)."""
        torch.manual_seed(42)
        mgr = TopKMaskManager(n_layers=2, mlp_dim=64)
        optim = torch.optim.Adam(mgr.parameters(), lr=0.1)
        mgr.train()

        initial_logits = [m.logits.data.clone() for m in mgr.masks]

        # Simulate optimization: try to maximize mask at k=32 for specific dims
        for _ in range(10):
            optim.zero_grad()
            # Create a target that prefers specific dimensions
            target = torch.zeros(64)
            target[32:] = 1.0  # Want the second half to be "on"
            mask = mgr.get_mask(0, k=32, tau=0.5)
            loss = ((mask - target) ** 2).sum()
            loss.backward()
            optim.step()

        # Logits should have changed
        for i, m in enumerate(mgr.masks):
            if i == 0:  # Only layer 0 was optimized
                diff = (m.logits.data - initial_logits[i]).abs().sum()
                assert diff > 0.1, f"Layer {i} logits didn't change enough"

    def test_active_count_preserved(self):
        """After optimization, hard mask should still select exactly k dims."""
        torch.manual_seed(42)
        mgr = TopKMaskManager(n_layers=2, mlp_dim=64)
        optim = torch.optim.Adam(mgr.parameters(), lr=0.1)
        mgr.train()

        for _ in range(20):
            optim.zero_grad()
            mask = mgr.get_mask(0, k=32, tau=0.5)
            loss = -mask[:32].sum() + mask[32:].sum()  # Prefer first 32
            loss.backward()
            optim.step()

        # Hard mask at k=32 should still have exactly 32 active
        mgr.eval()
        hard_mask = mgr.get_mask(0, k=32, tau=0.01, hard=True)
        assert hard_mask.sum().item() == 32


class TestTopKMultiFactor:
    def test_forward_backward_multiple_factors(self, small_model_config):
        """Run forward+backward at multiple factors like real training."""
        small_model_config.activation_type = "gelu"
        model = Olmo(small_model_config).train()
        mlp_dim = _post_act_mlp_dim(small_model_config)
        topk_mgr = TopKMaskManager(
            n_layers=small_model_config.n_layers, mlp_dim=mlp_dim
        )
        topk_mgr.train()

        matmng = MatformerManager.get_instance()
        matmng.mode = "topk"
        matmng.topk_masks = topk_mgr
        matmng.gumbel_tau = 1.0

        input_ids = torch.randint(0, small_model_config.vocab_size, (2, 16))

        # Simulate training with factor loop: 1, 2, 4
        for factor in [1, 2, 4]:
            matmng.current_factor = factor
            output = model(input_ids)
            logits = output.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss.backward()

        expected_vocab = small_model_config.embedding_size or small_model_config.vocab_size
        assert output.logits.shape == (2, 16, expected_vocab)


# ============================================================
# Config Tests
# ============================================================


class TestHMatConfigTopK:
    def test_topk_config(self):
        cfg = HMatConfig(
            enabled=True,
            method="topk",
            gumbel_tau_start=0.5,
            gumbel_tau_end=0.1,
            gumbel_init_scale=1.1,
        )
        assert cfg.enabled is True
        assert cfg.method == "topk"
        assert cfg.gumbel_tau_start == 0.5
        assert cfg.gumbel_init_scale == 1.1
