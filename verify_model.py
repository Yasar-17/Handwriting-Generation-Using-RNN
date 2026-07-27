"""Quick verification of MDN-RNN model + loss shapes and numerical stability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from models import MDNRNN, SequenceDiscriminator
from losses import MDNLoss, mdn_mixture_mean, adversarial_loss


def main():
    B, T, M = 4, 50, 20
    input_dim = 3
    hidden_dim = 128

    model = MDNRNN(input_dim=input_dim, hidden_dim=hidden_dim, num_mixtures=M)
    loss_fn = MDNLoss()
    discriminator = SequenceDiscriminator(input_dim=input_dim, hidden_dim=64, num_layers=3)

    x = torch.randn(B, T, input_dim)
    target = torch.randn(B, T, input_dim)
    target[:, :, 2] = torch.sigmoid(target[:, :, 2])
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[2, 40:] = False

    params, hidden = model(x)

    print("=== Output shapes ===")
    for k, v in params.items():
        print(f"  {k}: {v.shape}")
    print(f"  hidden[0]: {hidden[0].shape}, hidden[1]: {hidden[1].shape}")

    assert params["mu_x"].shape == (B, T, M)
    assert params["sigma_x"].min() > 0
    assert params["sigma_y"].min() > 0
    assert (-1 <= params["rho"]).all() and (params["rho"] <= 1).all()
    assert torch.allclose(params["pi"].sum(-1), torch.ones(B, T), atol=1e-5)
    assert (0 <= params["pen_up"]).all() and (params["pen_up"] <= 1).all()

    loss = loss_fn(params, target, mask)
    print(f"\n  MDN Loss: {loss.item():.4f}")
    assert loss.isfinite(), "Loss is not finite!"
    assert loss > 0, "Loss should be positive"

    loss.backward()
    print("  MDN gradient check: OK (backward succeeded)")

    # --- Discriminator verification ---
    print("\n=== Discriminator ===")
    fake_seq = mdn_mixture_mean(params["mu_x"], params["mu_y"], params["pi"], params["pen_up"])
    print(f"  Fake sequence shape: {fake_seq.shape}")
    assert fake_seq.shape == (B, T, 3)

    disc_real = discriminator(x)
    disc_fake = discriminator(fake_seq.detach())
    print(f"  disc_real shape: {disc_real.shape}, range: [{disc_real.min():.4f}, {disc_real.max():.4f}]")
    print(f"  disc_fake shape: {disc_fake.shape}, range: [{disc_fake.min():.4f}, {disc_fake.max():.4f}]")
    assert disc_real.shape == (B,)
    assert disc_fake.shape == (B,)
    assert (0 <= disc_real).all() and (disc_real <= 1).all()
    assert (0 <= disc_fake).all() and (disc_fake <= 1).all()

    disc_loss, gen_adv_loss = adversarial_loss(disc_real, disc_fake, mask[:, 0])
    print(f"  Discriminator loss: {disc_loss.item():.4f}")
    print(f"  Generator adversarial loss: {gen_adv_loss.item():.4f}")
    assert disc_loss.isfinite() and disc_loss > 0
    assert gen_adv_loss.isfinite() and gen_adv_loss > 0

    disc_loss.backward(retain_graph=True)
    print("  Discriminator gradient check: OK")

    gen_adv_loss = adversarial_loss(disc_real.detach(), disc_fake, mask[:, 0])[1]
    gen_adv_loss.backward()
    print("  Generator adversarial gradient check: OK")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
