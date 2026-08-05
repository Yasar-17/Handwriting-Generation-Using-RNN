"""
Mixture Density Network loss for handwriting generation.

Implements the negative log-likelihood of a mixture of bivariate Gaussians
for (delta_x, delta_y) plus a Bernoulli term for pen_up, following
Graves' "Generating Sequences With Recurrent Neural Networks".

All computations use numerically stable log-sum-exp to avoid underflow
when the mixture has many components.
"""

import math

import torch
import torch.nn as nn


def mdn_loss(
    mu_x: torch.Tensor,
    mu_y: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    rho: torch.Tensor,
    pi: torch.Tensor,
    pen_up: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the MDN negative log-likelihood per sample.

    Args:
        mu_x, mu_y: (B, T, M)  Gaussian means
        sigma_x, sigma_y: (B, T, M)  Gaussian std devs (already exp-activated)
        rho: (B, T, M)  correlation coefficients (already tanh-activated)
        pi: (B, T, M)  mixture weights (already softmax-activated)
        pen_up: (B, T)  Bernoulli probability for pen_up
        target: (B, T, 3)  ground-truth (delta_x, delta_y, pen_up)
        mask: (B, T)  boolean mask; True = valid timestep
        eps: small constant for numerical stability

    Returns:
        loss: scalar tensor, mean NLL over all valid timesteps
    """
    target_dx = target[:, :, 0]  # (B, T)
    target_dy = target[:, :, 1]  # (B, T)
    target_pen = target[:, :, 2]  # (B, T)

    _B, _T, _M = mu_x.shape

    # -----------------------------------------------------------------------
    # Bivariate Gaussian log-probability for each component
    # -----------------------------------------------------------------------
    # Z = ((x - mu_x)/sigma_x)^2 + ((y - mu_y)/sigma_y)^2
    #     - 2*rho*((x - mu_x)/sigma_x)*((y - mu_y)/sigma_y)
    #
    # log N(x,y) = -log(2*pi*sigma_x*sigma_y*sqrt(1-rho^2)) - Z / (2*(1-rho^2))

    dx = target_dx.unsqueeze(-1) - mu_x  # (B, T, M)
    dy = target_dy.unsqueeze(-1) - mu_y  # (B, T, M)

    sigma_x**2
    sigma_y**2

    norm_x = dx / (sigma_x + eps)
    norm_y = dy / (sigma_y + eps)

    rho_sq = rho**2
    one_minus_rho_sq = 1.0 - rho_sq.clamp(max=1.0 - eps)

    z = norm_x**2 + norm_y**2 - 2.0 * rho * norm_x * norm_y

    log_norm = -0.5 * (
        torch.log(2.0 * math.pi * sigma_x * sigma_y * torch.sqrt(one_minus_rho_sq)) + z / one_minus_rho_sq
    )  # (B, T, M)

    # -----------------------------------------------------------------------
    # Log-sum-exp over mixture components:
    #   log(sum_k pi_k * N_k) = logsumexp(log(pi_k) + log(N_k))
    # -----------------------------------------------------------------------
    log_pi = torch.log(pi + eps)  # (B, T, M)
    log_mdn = torch.logsumexp(log_pi + log_norm, dim=-1)  # (B, T)

    # -----------------------------------------------------------------------
    # Bernoulli NLL for pen_up
    # -----------------------------------------------------------------------
    pen_up_clamped = pen_up.clamp(min=eps, max=1.0 - eps)
    log_pen = target_pen * torch.log(pen_up_clamped) + (1.0 - target_pen) * torch.log(1.0 - pen_up_clamped)

    # -----------------------------------------------------------------------
    # Combine and mask
    # -----------------------------------------------------------------------
    nll = -(log_mdn + log_pen)  # (B, T)

    if mask is not None:
        nll = nll * mask.float()
        valid = mask.sum()
    else:
        valid = torch.tensor(nll.numel(), dtype=nll.dtype, device=nll.device)

    return nll.sum() / valid.clamp(min=1.0)


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_: float = 1.0,
) -> torch.Tensor:
    """WGAN-GP-style gradient penalty to stabilize adversarial training.

    Penalizes the discriminator when the gradient of its output with respect
    to interpolations between real and fake sequences deviates from unit
    norm. Enforcing a Lipschitz bound on the discriminator prevents it from
    overpowering the generator, which commonly destabilizes GAN training.

    Args:
        discriminator: the sequence discriminator, called as
            ``discriminator(x) -> (B,)`` probabilities.
        real: (B, T, 3) real stroke sequences.
        fake: (B, T, 3) generated (fake) stroke sequences.
        lambda_: weight of the penalty term (returned scalar is already
            multiplied by this).

    Returns:
        penalty: scalar tensor (lambda * E[(||grad||_2 - 1)^2]).
    """
    B = real.size(0)
    if B == 0:
        return torch.zeros((), device=real.device)

    alpha = torch.rand(B, 1, 1, device=real.device, dtype=real.dtype)
    interp = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    d_interp = discriminator(interp)

    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_norm = grads.reshape(B, -1).norm(2, dim=1)
    penalty = lambda_ * ((grad_norm - 1.0) ** 2).mean()
    return penalty


class MDNLoss(nn.Module):
    """Module wrapper for mdn_loss so it works cleanly with target tensors."""

    def forward(
        self,
        params: dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return mdn_loss(
            mu_x=params["mu_x"],
            mu_y=params["mu_y"],
            sigma_x=params["sigma_x"],
            sigma_y=params["sigma_y"],
            rho=params["rho"],
            pi=params["pi"],
            pen_up=params["pen_up"],
            target=target,
            mask=mask,
        )


def mdn_mixture_mean(
    mu_x: torch.Tensor,
    mu_y: torch.Tensor,
    pi: torch.Tensor,
    pen_up: torch.Tensor,
) -> torch.Tensor:
    """Extract the continuous (differentiable) expected stroke from MDN params.

    Computes the pi-weighted mean of the mixture for (dx, dy), keeping the
    pen_up probability as-is. This avoids non-differentiable sampling and
    can be used as the "fake" sequence for adversarial training.

    Args:
        mu_x, mu_y: (B, T, M) Gaussian means
        pi: (B, T, M) mixture weights
        pen_up: (B, T) pen_up probability

    Returns:
        fake_seq: (B, T, 3) expected (dx, dy, pen_up) from the mixture
    """
    expected_dx = (pi * mu_x).sum(dim=-1)  # (B, T)
    expected_dy = (pi * mu_y).sum(dim=-1)  # (B, T)
    fake_seq = torch.stack([expected_dx, expected_dy, pen_up], dim=-1)  # (B, T, 3)
    return fake_seq


def adversarial_loss(
    disc_real: torch.Tensor,
    disc_fake: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BCE adversarial losses for discriminator and generator.

    The discriminator is trained to output 1 for real sequences and 0 for fake.
    The generator is trained to make the discriminator output 1 for fake sequences.

    Args:
        disc_real: (B,) discriminator output for real sequences
        disc_fake: (B,) discriminator output for fake sequences
        mask: (B,) optional per-sample validity mask
        eps: small constant for numerical stability

    Returns:
        disc_loss: BCE loss for the discriminator
        gen_adv_loss: adversarial loss for the generator (to fool discriminator)
    """
    disc_real_clamped = disc_real.clamp(min=eps, max=1.0 - eps)
    disc_fake_clamped = disc_fake.clamp(min=eps, max=1.0 - eps)

    # Discriminator: want disc_real -> 1, disc_fake -> 0
    disc_loss_real = -torch.log(disc_real_clamped)
    disc_loss_fake = -torch.log(1.0 - disc_fake_clamped)
    disc_loss = disc_loss_real + disc_loss_fake

    # Generator: want disc_fake -> 1
    gen_adv_loss = -torch.log(disc_fake_clamped)

    if mask is not None:
        mask_float = mask.float()
        disc_loss = (disc_loss * mask_float).sum() / mask_float.sum().clamp(min=1.0)
        gen_adv_loss = (gen_adv_loss * mask_float).sum() / mask_float.sum().clamp(min=1.0)
    else:
        disc_loss = disc_loss.mean()
        gen_adv_loss = gen_adv_loss.mean()

    return disc_loss, gen_adv_loss
