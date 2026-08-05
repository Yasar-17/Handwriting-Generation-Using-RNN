"""
Exponential Moving Average (EMA) of model weights.

EMA keeps a shadow copy of the model parameters updated as a running
average during training:

    shadow <- decay * shadow + (1 - decay) * weights

EMA-averaged weights are smoother than the raw training weights because
they average out the high-frequency noise of SGD-style updates, and are a
standard technique for producing higher-quality samples from generative
models (including GANs).

The shadow model lives in eval mode and never receives gradients, so using
it for sampling or evaluation is cheap and safe.
"""

import copy

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain an exponential moving average of a model's parameters.

    Args:
        model: the model whose parameters are averaged.
        decay: EMA decay factor in (0, 1); larger = slower adaptation.
        device: optional device for the shadow copy (defaults to the model's).

    Example:
        ema = ModelEMA(model, decay=0.999)
        for epoch in ...:
            train_step(model)
            ema.update(model)
        # sample with smoothed weights
        ema_model = ema.ema_model()
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: torch.device | None = None,
    ):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")

        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        if device is not None:
            self.shadow = self.shadow.to(device)
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the current model weights into the EMA shadow copy."""
        model_state = model.state_dict()
        shadow_state = self.shadow.state_dict()
        for key, value in model_state.items():
            if value.is_floating_point():
                shadow_state[key].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                shadow_state[key].copy_(value)

    def ema_model(self) -> nn.Module:
        """Return the EMA-averaged shadow model (eval mode, no gradients)."""
        return self.shadow

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a copy of the shadow weights for checkpointing."""
        return {k: v.clone() for k, v in self.shadow.state_dict().items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Restore shadow weights from a checkpoint dict."""
        self.shadow.load_state_dict(state_dict)
