"""Phase 0 tests for Heterogeneous Matryoshka (H-Mat) foundation."""

import pytest
import torch
from torch.nn import CrossEntropyLoss

from olmo import Olmo, TrainConfig
from olmo.config import HMatConfig, ModelConfig
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
        max_sequence_length=64,
        init_device="cpu",
    )


class TestMatformerManagerUniform:
    """Test that uniform mode preserves backward compatibility."""

    def test_default_mode_is_uniform(self):
        mgr = MatformerManager.get_instance()
        assert mgr.mode == "uniform"
        assert mgr.current_factor == 1
        assert mgr.layer_factors is None
        assert mgr.gumbel_masks is None

    def test_uniform_get_factor_for_layer_returns_current_factor(self):
        mgr = MatformerManager.get_instance()
        mgr.current_factor = 4
        for i in range(12):
            assert mgr.get_factor_for_layer(i) == 4

    def test_uniform_factor_1_all_layers(self):
        mgr = MatformerManager.get_instance()
        mgr.current_factor = 1
        for i in range(12):
            assert mgr.get_factor_for_layer(i) == 1


class TestMatformerManagerHeterogeneous:
    """Test heterogeneous per-layer factor retrieval."""

    def test_heterogeneous_per_layer_factors(self):
        mgr = MatformerManager.get_instance()
        mgr.mode = "heterogeneous"
        mgr.layer_factors = {0: 1, 1: 4, 2: 2}
        assert mgr.get_factor_for_layer(0) == 1
        assert mgr.get_factor_for_layer(1) == 4
        assert mgr.get_factor_for_layer(2) == 2

    def test_heterogeneous_missing_layer_falls_back_to_current_factor(self):
        mgr = MatformerManager.get_instance()
        mgr.mode = "heterogeneous"
        mgr.current_factor = 8
        mgr.layer_factors = {0: 1, 1: 4}
        # Layer 2 is not in layer_factors, should fall back to current_factor
        assert mgr.get_factor_for_layer(2) == 8

    def test_uniform_mode_ignores_layer_factors(self):
        mgr = MatformerManager.get_instance()
        mgr.mode = "uniform"
        mgr.current_factor = 2
        mgr.layer_factors = {0: 8, 1: 4}
        # In uniform mode, layer_factors should be ignored
        assert mgr.get_factor_for_layer(0) == 2
        assert mgr.get_factor_for_layer(1) == 2


class TestLayerIdxAssignment:
    """Test that blocks receive correct layer_idx."""

    def test_layer_idx_on_blocks(self, small_model_config):
        model = Olmo(small_model_config)
        for i, block in enumerate(model.transformer.blocks):
            assert hasattr(block, "layer_idx"), f"Block {i} missing layer_idx"
            assert block.layer_idx == i, f"Block {i} has layer_idx={block.layer_idx}, expected {i}"


class TestForwardWithLayerFactors:
    """Test forward passes with heterogeneous per-layer factors."""

    def test_forward_heterogeneous_factors(self, small_model_config):
        """Run forward pass with different factors per layer; verify output shape."""
        model = Olmo(small_model_config).eval()
        mgr = MatformerManager.get_instance()
        mgr.mode = "heterogeneous"
        mgr.layer_factors = {0: 1, 1: 2, 2: 4}

        input_ids = torch.randint(0, small_model_config.vocab_size, (2, 16))
        with torch.no_grad():
            output = model(input_ids)

        assert output.logits.shape == (2, 16, small_model_config.embedding_size or small_model_config.vocab_size)

    def test_forward_all_full_width(self, small_model_config):
        """All layers at factor=1 should produce same output as uniform factor=1."""
        model = Olmo(small_model_config).eval()
        input_ids = torch.randint(0, small_model_config.vocab_size, (1, 16))

        # Uniform mode, factor=1
        mgr = MatformerManager.get_instance()
        mgr.mode = "uniform"
        mgr.current_factor = 1
        with torch.no_grad():
            out_uniform = model(input_ids)

        # Heterogeneous mode, all factor=1
        mgr.mode = "heterogeneous"
        mgr.layer_factors = {i: 1 for i in range(small_model_config.n_layers)}
        with torch.no_grad():
            out_hetero = model(input_ids)

        torch.testing.assert_close(out_uniform.logits, out_hetero.logits)


class TestBackwardWithLayerFactors:
    """Test that gradients flow correctly with heterogeneous factors."""

    def test_backward_heterogeneous(self, small_model_config):
        """Forward+backward with heterogeneous factors; verify all params get gradients."""
        model = Olmo(small_model_config).train()
        mgr = MatformerManager.get_instance()
        mgr.mode = "heterogeneous"
        mgr.layer_factors = {0: 1, 1: 2, 2: 4}

        input_ids = torch.randint(0, small_model_config.vocab_size, (2, 16))
        output = model(input_ids)
        logits = output.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name} has no gradient"


class TestUniformUnchanged:
    """Ensure existing uniform MatFormer behavior is completely unchanged when hmat.enabled=False."""

    def test_uniform_matformer_factor_2(self, small_model_config):
        """Uniform factor=2 should work the same via get_factor_for_layer."""
        model = Olmo(small_model_config).eval()
        mgr = MatformerManager.get_instance()

        input_ids = torch.randint(0, small_model_config.vocab_size, (1, 16))

        # Set uniform factor=2
        mgr.mode = "uniform"
        mgr.current_factor = 2
        with torch.no_grad():
            output = model(input_ids)

        # Output shape should be correct
        assert output.logits.shape == (1, 16, small_model_config.embedding_size or small_model_config.vocab_size)

    def test_hmat_config_defaults(self):
        """Verify HMatConfig defaults are sane."""
        cfg = HMatConfig()
        assert cfg.enabled is False
        assert cfg.method == "fisher"
        assert cfg.calibration_batches == 128
        assert cfg.budget_ratio == 0.25
        assert cfg.gumbel_tau_start == 2.0
        assert cfg.gumbel_tau_end == 0.1

    def test_train_config_has_hmat_field(self):
        """TrainConfig should have an hmat field defaulting to disabled."""
        cfg = TrainConfig()
        assert hasattr(cfg, "hmat")
        assert isinstance(cfg.hmat, HMatConfig)
        assert cfg.hmat.enabled is False
