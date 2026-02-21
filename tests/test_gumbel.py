"""Phase 2 tests: Gumbel-Softmax learnable masking for H-Mat."""

import math

import pytest
import torch
from torch.nn import CrossEntropyLoss

from olmo import Olmo
from olmo.config import HMatConfig, ModelConfig
from olmo.hmat.gumbel import GumbelMaskLayer, GumbelMaskManager
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


# ============================================================
# GumbelMaskLayer Tests
# ============================================================


class TestGumbelMaskLayerShape:
    def test_output_shape(self):
        mask_layer = GumbelMaskLayer(512)
        mask = mask_layer(tau=1.0)
        assert mask.shape == (512,)

    def test_output_range_soft(self):
        mask_layer = GumbelMaskLayer(512)
        mask_layer.train()
        mask = mask_layer(tau=1.0)
        assert (mask >= 0).all() and (mask <= 1).all()

    def test_output_range_hard(self):
        mask_layer = GumbelMaskLayer(512)
        mask_layer.eval()
        mask = mask_layer(tau=1.0, hard=True)
        unique_vals = mask.unique()
        for v in unique_vals:
            assert v.item() in (0.0, 1.0), f"Hard mask contains non-binary value {v}"


class TestGumbelMaskLayerTemperature:
    def test_high_tau_produces_soft_masks(self):
        """High temperature should produce masks near 0.5."""
        mask_layer = GumbelMaskLayer(1024)
        mask_layer.eval()  # No Gumbel noise for deterministic test
        mask = mask_layer(tau=100.0)
        # With logits=0, sigmoid(0/100) = 0.5
        assert torch.allclose(mask, torch.full_like(mask, 0.5), atol=0.01)

    def test_low_tau_produces_hard_masks(self):
        """Low temperature should produce masks near 0 or 1."""
        torch.manual_seed(42)
        mask_layer = GumbelMaskLayer(1024)
        # Set some logits positive, some negative
        mask_layer.logits.data = torch.randn(1024) * 5.0
        mask_layer.eval()
        mask = mask_layer(tau=0.01)
        # Should be very close to 0 or 1
        dist_from_binary = torch.min(mask, 1.0 - mask)
        assert dist_from_binary.mean() < 0.01


class TestGumbelMaskLayerGradients:
    def test_soft_mask_has_gradients(self):
        mask_layer = GumbelMaskLayer(128)
        mask_layer.train()
        mask = mask_layer(tau=1.0)
        loss = mask.sum()
        loss.backward()
        assert mask_layer.logits.grad is not None
        assert mask_layer.logits.grad.abs().sum() > 0

    def test_hard_mask_has_gradients_via_ste(self):
        """Hard mask uses straight-through estimator, so gradients should flow."""
        mask_layer = GumbelMaskLayer(128)
        mask_layer.train()
        mask = mask_layer(tau=1.0, hard=True)
        loss = mask.sum()
        loss.backward()
        assert mask_layer.logits.grad is not None
        assert mask_layer.logits.grad.abs().sum() > 0


class TestGumbelMaskLayerActiveFraction:
    def test_active_fraction_with_positive_logits(self):
        mask_layer = GumbelMaskLayer(100)
        mask_layer.logits.data = torch.ones(100)  # All positive
        assert mask_layer.get_active_fraction() == 1.0

    def test_active_fraction_with_negative_logits(self):
        mask_layer = GumbelMaskLayer(100)
        mask_layer.logits.data = -torch.ones(100)  # All negative
        assert mask_layer.get_active_fraction() == 0.0

    def test_active_fraction_mixed(self):
        mask_layer = GumbelMaskLayer(100)
        logits = torch.zeros(100)
        logits[:60] = 1.0
        logits[60:] = -1.0
        mask_layer.logits.data = logits
        assert abs(mask_layer.get_active_fraction() - 0.6) < 1e-5


# ============================================================
# GumbelMaskManager Tests
# ============================================================


class TestGumbelMaskManager:
    def test_creation(self):
        mgr = GumbelMaskManager(n_layers=4, mlp_dim=512)
        assert len(mgr.masks) == 4
        assert mgr.n_layers == 4
        assert mgr.mlp_dim == 512

    def test_get_mask_shape(self):
        mgr = GumbelMaskManager(n_layers=4, mlp_dim=512)
        for i in range(4):
            mask = mgr.get_mask(i, tau=1.0)
            assert mask.shape == (512,)

    def test_budget_loss_at_target(self):
        """Budget loss should be ~0 when active fraction matches target."""
        mgr = GumbelMaskManager(n_layers=4, mlp_dim=128)
        # Set all logits to large positive → sigmoid ≈ 1.0 → fraction ≈ 1.0
        for mask_layer in mgr.masks:
            mask_layer.logits.data = torch.ones(128) * 10.0
        loss = mgr.budget_loss(target=1.0)
        assert loss.item() < 0.01

    def test_budget_loss_away_from_target(self):
        """Budget loss should be large when active fraction differs from target."""
        mgr = GumbelMaskManager(n_layers=4, mlp_dim=128)
        # All logits large positive → fraction ≈ 1.0, target = 0.25
        for mask_layer in mgr.masks:
            mask_layer.logits.data = torch.ones(128) * 10.0
        loss = mgr.budget_loss(target=0.25)
        assert loss.item() > 0.5

    def test_budget_loss_gradient(self):
        """Budget loss should produce gradients on logits."""
        mgr = GumbelMaskManager(n_layers=4, mlp_dim=128)
        loss = mgr.budget_loss(target=0.5)
        loss.backward()
        for mask_layer in mgr.masks:
            assert mask_layer.logits.grad is not None


class TestGumbelMaskManagerWidths:
    def test_get_layer_widths(self):
        mgr = GumbelMaskManager(n_layers=3, mlp_dim=128)
        # Set all logits positive → all dimensions active
        for mask_layer in mgr.masks:
            mask_layer.logits.data = torch.ones(128) * 10.0
        widths = mgr.get_layer_widths()
        assert len(widths) == 3
        for i in range(3):
            assert widths[i] == 128

    def test_get_layer_widths_partial(self):
        mgr = GumbelMaskManager(n_layers=2, mlp_dim=128)
        # Layer 0: first 64 positive, rest negative
        logits0 = torch.ones(128) * -10.0
        logits0[:64] = 10.0
        mgr.masks[0].logits.data = logits0
        # Layer 1: all positive
        mgr.masks[1].logits.data = torch.ones(128) * 10.0

        widths = mgr.get_layer_widths()
        assert widths[0] == 64
        assert widths[1] == 128

    def test_get_layer_factors(self):
        mgr = GumbelMaskManager(n_layers=2, mlp_dim=512)
        # Layer 0: ~256 active → factor 2
        logits0 = torch.ones(512) * -10.0
        logits0[:256] = 10.0
        mgr.masks[0].logits.data = logits0
        # Layer 1: all active → factor 1
        mgr.masks[1].logits.data = torch.ones(512) * 10.0

        factors = mgr.get_layer_factors([1, 2, 4, 8])
        assert factors[0] == 2
        assert factors[1] == 1


class TestGumbelMaskManagerSummary:
    def test_log_summary_keys(self):
        mgr = GumbelMaskManager(n_layers=3, mlp_dim=128)
        summary = mgr.log_summary()
        assert "gumbel/layer_0_active_frac" in summary
        assert "gumbel/layer_1_active_frac" in summary
        assert "gumbel/layer_2_active_frac" in summary
        assert "gumbel/mean_active_frac" in summary


# ============================================================
# Integration Tests: Gumbel + Model Forward/Backward
# ============================================================


def _post_act_mlp_dim(config: ModelConfig) -> int:
    """Compute the post-activation MLP hidden dim (accounts for SwiGLU halving)."""
    act = Activation.build(config)
    return int(config.mlp_ratio * config.d_model * act.output_multiplier)


class TestGumbelForwardPass:
    def test_forward_with_gumbel_masks(self, small_model_config):
        """Forward pass with gumbel mode should produce correct output shape."""
        model = Olmo(small_model_config).eval()
        mlp_dim = _post_act_mlp_dim(small_model_config)
        gumbel_mgr = GumbelMaskManager(
            n_layers=small_model_config.n_layers, mlp_dim=mlp_dim
        )

        matmng = MatformerManager.get_instance()
        matmng.mode = "gumbel"
        matmng.gumbel_masks = gumbel_mgr
        matmng.gumbel_tau = 1.0

        input_ids = torch.randint(0, small_model_config.vocab_size, (2, 16))
        with torch.no_grad():
            output = model(input_ids)

        expected_vocab = small_model_config.embedding_size or small_model_config.vocab_size
        assert output.logits.shape == (2, 16, expected_vocab)

    def test_forward_gumbel_vs_uniform_full_mask(self, small_model_config):
        """Gumbel with all-ones mask should produce same output as uniform factor=1."""
        model = Olmo(small_model_config).eval()
        mlp_dim = _post_act_mlp_dim(small_model_config)
        gumbel_mgr = GumbelMaskManager(
            n_layers=small_model_config.n_layers, mlp_dim=mlp_dim
        )
        # Set all logits to large positive → mask ≈ 1.0
        for mask_layer in gumbel_mgr.masks:
            mask_layer.logits.data = torch.ones(mlp_dim) * 100.0

        input_ids = torch.randint(0, small_model_config.vocab_size, (1, 16))

        # Uniform mode, factor=1
        matmng = MatformerManager.get_instance()
        matmng.mode = "uniform"
        matmng.current_factor = 1
        with torch.no_grad():
            out_uniform = model(input_ids)

        # Gumbel mode with all-ones hard mask
        matmng.mode = "gumbel"
        matmng.gumbel_masks = gumbel_mgr
        matmng.gumbel_tau = 0.01  # Very low tau + eval → hard mask ≈ 1.0
        gumbel_mgr.eval()
        with torch.no_grad():
            out_gumbel = model(input_ids)

        torch.testing.assert_close(out_uniform.logits, out_gumbel.logits, atol=1e-4, rtol=1e-4)


class TestGumbelBackward:
    def test_backward_with_gumbel(self, small_model_config):
        """Backward pass with gumbel mode should produce gradients on model and mask params."""
        model = Olmo(small_model_config).train()
        mlp_dim = _post_act_mlp_dim(small_model_config)
        gumbel_mgr = GumbelMaskManager(
            n_layers=small_model_config.n_layers, mlp_dim=mlp_dim
        )
        gumbel_mgr.train()

        matmng = MatformerManager.get_instance()
        matmng.mode = "gumbel"
        matmng.gumbel_masks = gumbel_mgr
        matmng.gumbel_tau = 1.0

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

        # Gumbel mask logits should have gradients
        for i, mask_layer in enumerate(gumbel_mgr.masks):
            assert mask_layer.logits.grad is not None, f"Mask layer {i} logits have no gradient"
            assert mask_layer.logits.grad.abs().sum() > 0, f"Mask layer {i} has zero gradients"

    def test_backward_with_budget_penalty(self, small_model_config):
        """Budget penalty should produce gradients that push masks toward target."""
        model = Olmo(small_model_config).train()
        mlp_dim = _post_act_mlp_dim(small_model_config)
        gumbel_mgr = GumbelMaskManager(
            n_layers=small_model_config.n_layers, mlp_dim=mlp_dim
        )
        gumbel_mgr.train()

        matmng = MatformerManager.get_instance()
        matmng.mode = "gumbel"
        matmng.gumbel_masks = gumbel_mgr
        matmng.gumbel_tau = 1.0

        input_ids = torch.randint(0, small_model_config.vocab_size, (2, 16))
        output = model(input_ids)
        logits = output.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        ce_loss = CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        budget_penalty = gumbel_mgr.budget_loss(target=0.25)
        loss = ce_loss + 0.01 * budget_penalty
        loss.backward()

        # Mask logits should have gradients from both CE loss and budget penalty
        for i, mask_layer in enumerate(gumbel_mgr.masks):
            assert mask_layer.logits.grad is not None


# ============================================================
# Temperature Annealing Tests
# ============================================================


class TestTemperatureAnnealing:
    def test_exponential_schedule(self):
        """Verify the exponential temperature annealing formula."""
        tau_start = 2.0
        tau_end = 0.1
        anneal_steps = 1000

        # At step 0
        progress = 0.0
        tau = tau_start * (tau_end / tau_start) ** progress
        assert abs(tau - tau_start) < 1e-6

        # At final step
        progress = 1.0
        tau = tau_start * (tau_end / tau_start) ** progress
        assert abs(tau - tau_end) < 1e-6

        # At midpoint
        progress = 0.5
        tau = tau_start * (tau_end / tau_start) ** progress
        expected_mid = math.sqrt(tau_start * tau_end)
        assert abs(tau - expected_mid) < 1e-5

    def test_tau_monotonically_decreases(self):
        """Temperature should decrease monotonically."""
        tau_start = 2.0
        tau_end = 0.1
        anneal_steps = 100
        prev_tau = tau_start
        for step in range(1, anneal_steps + 1):
            progress = step / anneal_steps
            tau = tau_start * (tau_end / tau_start) ** progress
            assert tau < prev_tau
            prev_tau = tau


# ============================================================
# State Dict / Checkpoint Tests
# ============================================================


class TestGumbelCheckpoint:
    def test_state_dict_roundtrip(self):
        """Save and load GumbelMaskManager state dict."""
        mgr = GumbelMaskManager(n_layers=3, mlp_dim=128)
        # Modify logits
        mgr.masks[0].logits.data = torch.randn(128)
        mgr.masks[1].logits.data = torch.randn(128) * 2.0
        mgr.masks[2].logits.data = torch.randn(128) * 0.5

        state = mgr.state_dict()

        # Create new manager and load
        mgr2 = GumbelMaskManager(n_layers=3, mlp_dim=128)
        mgr2.load_state_dict(state)

        for i in range(3):
            torch.testing.assert_close(mgr.masks[i].logits, mgr2.masks[i].logits)


# ============================================================
# HMatConfig Tests
# ============================================================


class TestHMatConfigGumbel:
    def test_gumbel_config_defaults(self):
        cfg = HMatConfig()
        assert cfg.gumbel_tau_start == 2.0
        assert cfg.gumbel_tau_end == 0.1
        assert cfg.gumbel_tau_anneal_steps == 0
        assert cfg.budget_penalty_lambda == 0.01
        assert cfg.budget_penalty_target == 0.5

    def test_gumbel_config_custom(self):
        cfg = HMatConfig(
            enabled=True,
            method="gumbel",
            gumbel_tau_start=5.0,
            gumbel_tau_end=0.05,
            gumbel_tau_anneal_steps=500,
            budget_penalty_lambda=0.1,
            budget_penalty_target=0.25,
        )
        assert cfg.enabled is True
        assert cfg.method == "gumbel"
        assert cfg.gumbel_tau_start == 5.0
        assert cfg.budget_penalty_target == 0.25
