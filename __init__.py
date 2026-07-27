"""Handwriting generation with a Mixture-Density RNN and optional GAN refinement.

This package implements Alex Graves' "Generating Sequences With Recurrent
Neural Networks" approach: an LSTM outputs the parameters of a mixture of
bivariate Gaussions over pen displacements plus a Bernoulli pen-lift term,
and (optionally) attends over a character sequence via soft monotonic
windowed attention for text-conditioned synthesis. A 1D-CNN sequence
discriminator can be trained adversarially to sharpen the strokes.
"""

from .data import (
    CharVocab,
    IAMStrokeDataset,
    IAMConditionedDataset,
    build_dataloader,
    build_conditioned_dataloader,
    collect_xml_files,
    compute_dataset_stats,
    denormalize_deltas,
    normalize_deltas,
    prepare_splits,
    render_strokes,
)
from .losses import MDNLoss, mdn_loss, mdn_mixture_mean, adversarial_loss
from .models import MDNRNN, MDNRNNConditioned, SequenceDiscriminator, WindowedAttention

__all__ = [
    "CharVocab",
    "IAMStrokeDataset",
    "IAMConditionedDataset",
    "build_dataloader",
    "build_conditioned_dataloader",
    "collect_xml_files",
    "compute_dataset_stats",
    "denormalize_deltas",
    "normalize_deltas",
    "prepare_splits",
    "render_strokes",
    "MDNLoss",
    "mdn_loss",
    "mdn_mixture_mean",
    "adversarial_loss",
    "MDNRNN",
    "MDNRNNConditioned",
    "SequenceDiscriminator",
    "WindowedAttention",
]