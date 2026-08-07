"""Handwriting generation with a Mixture-Density RNN and optional GAN refinement."""

from data import (
    CharVocab,
    IAMConditionedDataset,
    IAMStrokeDataset,
    build_conditioned_dataloader,
    build_dataloader,
    collect_xml_files,
    compute_dataset_stats,
    denormalize_deltas,
    normalize_deltas,
    prepare_splits,
    render_strokes,
)
from losses import MDNLoss, adversarial_loss, mdn_loss, mdn_mixture_mean
from models import MDNRNN, MDNRNNConditioned, SequenceDiscriminator, WindowedAttention
from render import (
    THEMES,
    RenderTheme,
    render_comparison_grid,
    render_multi_sample,
    render_strokes_svg,
    render_strokes_themed,
    save_svg,
)

__all__ = [
    "MDNRNN",
    "THEMES",
    "CharVocab",
    "IAMConditionedDataset",
    "IAMStrokeDataset",
    "MDNLoss",
    "MDNRNNConditioned",
    "RenderTheme",
    "SequenceDiscriminator",
    "WindowedAttention",
    "adversarial_loss",
    "build_conditioned_dataloader",
    "build_dataloader",
    "collect_xml_files",
    "compute_dataset_stats",
    "denormalize_deltas",
    "mdn_loss",
    "mdn_mixture_mean",
    "normalize_deltas",
    "prepare_splits",
    "render_comparison_grid",
    "render_multi_sample",
    "render_strokes",
    "render_strokes_svg",
    "render_strokes_themed",
    "save_svg",
]
