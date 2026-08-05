"""Tests for the exponential moving average (EMA) of model weights."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ema import ModelEMA
from models import MDNRNN


@pytest.fixture
def model():
    return MDNRNN(input_dim=3, hidden_dim=16, num_layers=1, num_mixtures=4, dropout=0.0)


class TestModelEMA:
    def test_invalid_decay_raises(self, model):
        with pytest.raises(ValueError):
            ModelEMA(model, decay=0.0)
        with pytest.raises(ValueError):
            ModelEMA(model, decay=1.0)
        with pytest.raises(ValueError):
            ModelEMA(model, decay=-0.1)

    def test_update_blends_weights(self, model):
        torch.manual_seed(0)
        ema = ModelEMA(model, decay=0.9)
        shadow = ema.ema_model()

        first_key = next(iter(model.state_dict()))
        before = model.state_dict()[first_key].clone()

        # Perturb the model weights, then blend once: shadow = 0.9*w0 + 0.1*w1
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        ema.update(model)
        expected = 0.9 * before + 0.1 * (before + 1.0)
        torch.testing.assert_close(shadow.state_dict()[first_key], expected)

    def test_converges_to_model_after_many_updates(self, model):
        torch.manual_seed(0)
        ema = ModelEMA(model, decay=0.5)
        shadow = ema.ema_model()

        # Fixed model weights; EMA must converge to them exponentially.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)
        for _ in range(60):
            ema.update(model)

        for (mk, mv), (sk, sv) in zip(model.state_dict().items(), shadow.state_dict().items()):
            assert mk == sk
            torch.testing.assert_close(mv, sv, atol=1e-5, rtol=1e-5)

    def test_shadow_is_eval_and_gradient_free(self, model):
        ema = ModelEMA(model)
        shadow = ema.ema_model()
        assert not shadow.training
        assert all(not p.requires_grad for p in shadow.parameters())

    def test_state_dict_roundtrip(self, model):
        ema = ModelEMA(model, decay=0.99)
        for _ in range(3):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.05)
            ema.update(model)

        saved = ema.state_dict()
        ema2 = ModelEMA(model)
        ema2.load_state_dict(saved)
        for k, v in ema2.ema_model().state_dict().items():
            torch.testing.assert_close(v, saved[k])

    def test_update_does_not_require_grad(self, model):
        ema = ModelEMA(model, decay=0.9)
        ema.update(model)  # must run under no_grad without error
        shadow = ema.ema_model()
        assert all(p.grad is None for p in shadow.parameters())


class TestEMATrainingIntegration:
    def test_ema_updates_within_train_step(self, tmp_path):
        import numpy as np
        from torch.utils.data import DataLoader

        from data import collate_fn
        from losses import MDNLoss

        sys.path.insert(0, str(Path(__file__).parent.parent))
        seqs = [np.random.randn(24, 3) for _ in range(6)]
        loader = DataLoader(seqs, batch_size=3, collate_fn=collate_fn)

        model = MDNRNN(input_dim=3, hidden_dim=16, num_mixtures=4)
        ema = ModelEMA(model, decay=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MDNLoss()

        from train import train_one_epoch_uncond

        metrics = train_one_epoch_uncond(
            model, loader, loss_fn, None, optimizer, None,
            torch.device("cpu"), grad_accum_steps=1, ema=ema,
        )

        assert metrics["mdn_loss"] > 0
        # The EMA shadow is refreshed during the epoch but distinct from the live model.
        shadow = ema.ema_model()
        first_key = next(iter(model.state_dict()))
        assert not torch.equal(shadow.state_dict()[first_key], model.state_dict()[first_key])
