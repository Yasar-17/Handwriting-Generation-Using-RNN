"""Tests for the adversarial gradient penalty."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader

from data import collate_fn
from losses import MDNLoss, gradient_penalty
from models import MDNRNN, SequenceDiscriminator
from train import train_one_epoch_uncond


@pytest.fixture
def discriminator():
    return SequenceDiscriminator(input_dim=3, hidden_dim=16, num_layers=2, dropout=0.0)


@pytest.fixture
def batch():
    torch.manual_seed(0)
    real = torch.randn(4, 32, 3)
    fake = torch.randn(4, 32, 3) * 0.5
    return real, fake


class TestGradientPenalty:
    def test_returns_scalar(self, discriminator, batch):
        real, fake = batch
        penalty = gradient_penalty(discriminator, real, fake)
        assert penalty.ndim == 0
        assert torch.isfinite(penalty)
        assert penalty.item() >= 0.0

    def test_positive_for_untrained_discriminator(self, discriminator, batch):
        real, fake = batch
        penalty = gradient_penalty(discriminator, real, fake)
        # An untrained discriminator generally has non-unit gradients
        assert penalty.item() > 0.0

    def test_creates_gradients(self, discriminator, batch):
        real, fake = batch
        penalty = gradient_penalty(discriminator, real, fake)
        grads = torch.autograd.grad(
            penalty, discriminator.parameters(), retain_graph=True, allow_unused=True
        )
        assert any(g is not None and g.abs().sum() > 0 for g in grads)

    def test_lambda_scales_penalty(self, discriminator, batch):
        real, fake = batch
        torch.manual_seed(123)
        p1 = gradient_penalty(discriminator, real, fake, lambda_=1.0)
        torch.manual_seed(123)
        p10 = gradient_penalty(discriminator, real, fake, lambda_=10.0)
        assert p10.item() == pytest.approx(10.0 * p1.item(), rel=1e-3)

    def test_empty_batch(self, discriminator):
        real = torch.empty(0, 16, 3)
        fake = torch.empty(0, 16, 3)
        penalty = gradient_penalty(discriminator, real, fake)
        assert penalty.item() == 0.0


class TestGradientPenaltyTrainingIntegration:
    """End-to-end: the penalty is wired into the adversarial training loop."""

    def test_train_step_with_gradient_penalty(self):
        torch.manual_seed(0)
        np.random.seed(0)
        seqs = [np.random.randn(24, 3) for _ in range(6)]
        loader = DataLoader(seqs, batch_size=3, collate_fn=collate_fn)

        model = MDNRNN(input_dim=3, hidden_dim=16, num_mixtures=5)
        disc = SequenceDiscriminator(input_dim=3, hidden_dim=8, num_layers=2, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-3)
        loss_fn = MDNLoss()

        metrics = train_one_epoch_uncond(
            model, loader, loss_fn, disc, optimizer, disc_optimizer,
            torch.device("cpu"), adv_weight=0.1, grad_penalty_weight=10.0,
        )

        assert all(np.isfinite(v) for v in metrics.values())
        assert metrics["disc_loss"] > 0
        assert metrics["mdn_loss"] > 0

    def test_gradient_penalty_flows_to_discriminator(self, discriminator, batch):
        real, fake = batch
        penalty = gradient_penalty(discriminator, real, fake)
        grads = torch.autograd.grad(
            penalty, discriminator.parameters(), retain_graph=True, allow_unused=True
        )
        grad_norms = [g.abs().sum().item() for g in grads if g is not None]
        assert any(n > 0 for n in grad_norms)
