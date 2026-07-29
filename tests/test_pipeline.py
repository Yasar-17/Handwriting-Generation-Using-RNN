"""Comprehensive unit tests for the handwriting generation pipeline."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import (
    CharVocab,
    IAMConditionedDataset,
    IAMStrokeDataset,
    absolute_to_relative,
    build_conditioned_dataloader,
    build_dataloader,
    collate_fn,
    collate_fn_conditioned,
    compute_dataset_stats,
    denormalize_deltas,
    normalize_deltas,
    parse_iam_xml,
    prepare_splits,
    relative_to_absolute,
    render_strokes,
)
from losses import MDNLoss, adversarial_loss, mdn_loss, mdn_mixture_mean
from models import MDNRNN, MDNRNNConditioned, SequenceDiscriminator, WindowedAttention


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<StrokeSet>
  <Line text="hello world"/>
  <Stroke>
    <Point x="100" y="200" time="0"/>
    <Point x="110" y="195" time="10"/>
    <Point x="125" y="190" time="20"/>
    <Point x="140" y="192" time="30"/>
    <Point x="155" y="200" time="40"/>
    <Point x="160" y="210" time="50"/>
  </Stroke>
  <Stroke>
    <Point x="180" y="195" time="100"/>
    <Point x="190" y="190" time="110"/>
    <Point x="200" y="188" time="120"/>
    <Point x="210" y="192" time="130"/>
    <Point x="215" y="200" time="140"/>
  </Stroke>
</StrokeSet>
"""


@pytest.fixture
def temp_xml_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            path = Path(tmpdir) / f"sample_{i:03d}.xml"
            path.write_text(SAMPLE_XML)
        yield tmpdir


@pytest.fixture
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data Pipeline Tests
# ---------------------------------------------------------------------------

class TestCharVocab:
    def test_encode_decode(self):
        vocab = CharVocab()
        text = "hello world"
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)
        assert decoded == text

    def test_padding_index(self):
        vocab = CharVocab()
        assert vocab.char_to_idx.get("", 0) == 0

    def test_unknown_character(self):
        vocab = CharVocab()
        encoded = vocab.encode("hello\x00world")
        assert 0 in encoded

    def test_vocab_size(self):
        vocab = CharVocab()
        assert vocab.vocab_size == len(vocab.charset) + 1

    def test_custom_charset(self):
        charset = "abc"
        vocab = CharVocab(charset=charset)
        assert vocab.vocab_size == 4
        assert vocab.encode("abc") == [1, 2, 3]


class TestXMLParsing:
    def test_parse_xml(self, temp_xml_dir):
        xml_path = Path(temp_xml_dir) / "sample_000.xml"
        points, text = parse_iam_xml(xml_path)
        assert len(points) == 11
        assert text == "hello world"
        assert points.shape == (11, 3)

    def test_absolute_to_relative(self):
        points = np.array([[0, 0, 0], [10, 5, 0], [20, 15, 1]], dtype=np.float32)
        deltas = absolute_to_relative(points)
        assert deltas[0, 0] == 0
        assert deltas[0, 1] == 0
        assert deltas[1, 0] == 10
        assert deltas[1, 1] == 5
        assert deltas[2, 2] == 1

    def test_relative_to_absolute_roundtrip(self):
        points = np.array([[0, 0, 0], [10, 5, 0], [20, 15, 1]], dtype=np.float32)
        deltas = absolute_to_relative(points)
        recovered = relative_to_absolute(deltas)
        np.testing.assert_array_almost_equal(points, recovered)

    def test_empty_points(self):
        points = np.empty((0, 3), dtype=np.float32)
        deltas = absolute_to_relative(points)
        assert len(deltas) == 0


class TestNormalization:
    def test_normalize_denormalize_roundtrip(self):
        deltas = np.array([[0, 0, 0], [1.5, -2.0, 0], [3.0, 1.0, 1]], dtype=np.float32)
        mean_x, std_x, mean_y, std_y = 1.0, 2.0, 0.5, 1.5
        normed = normalize_deltas(deltas, mean_x, std_x, mean_y, std_y)
        denormed = denormalize_deltas(normed, mean_x, std_x, mean_y, std_y)
        np.testing.assert_array_almost_equal(deltas, denormed)

    def test_compute_stats(self, temp_xml_dir):
        xml_files = list(Path(temp_xml_dir).glob("*.xml"))
        mean_x, std_x, mean_y, std_y = compute_dataset_stats(xml_files)
        assert std_x > 0
        assert std_y > 0


class TestCollation:
    def test_collate_fn(self):
        seq1 = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float32)
        seq2 = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 1]], dtype=np.float32)
        batch = collate_fn([seq1, seq2])
        assert batch["data"].shape == (2, 3, 3)
        assert batch["mask"].shape == (2, 3)
        assert batch["mask"][0, 2] == False
        assert batch["mask"][1, 2] == True
        assert batch["lengths"].tolist() == [2, 3]

    def test_collate_fn_conditioned(self):
        char_ids1 = [1, 2, 3]
        seq1 = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float32)
        char_ids2 = [4, 5]
        seq2 = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 1]], dtype=np.float32)
        batch = collate_fn_conditioned([(char_ids1, seq1), (char_ids2, seq2)])
        assert batch["data"].shape == (2, 3, 3)
        assert batch["char_ids"].shape == (2, 3)
        assert batch["char_mask"][0, 2] == True
        assert batch["char_mask"][1, 2] == False


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

class TestMDNRNN:
    def test_forward_shape(self):
        B, T, M = 4, 50, 20
        model = MDNRNN(input_dim=3, hidden_dim=128, num_mixtures=M)
        x = torch.randn(B, T, 3)
        params, hidden = model(x)

        assert params["mu_x"].shape == (B, T, M)
        assert params["mu_y"].shape == (B, T, M)
        assert params["sigma_x"].shape == (B, T, M)
        assert params["sigma_y"].shape == (B, T, M)
        assert params["rho"].shape == (B, T, M)
        assert params["pi"].shape == (B, T, M)
        assert params["pen_up"].shape == (B, T)

    def test_output_constraints(self):
        B, T, M = 2, 30, 10
        model = MDNRNN(input_dim=3, hidden_dim=64, num_mixtures=M)
        x = torch.randn(B, T, 3)
        params, _ = model(x)

        assert (params["sigma_x"] > 0).all()
        assert (params["sigma_y"] > 0).all()
        assert (-1 <= params["rho"]).all() and (params["rho"] <= 1).all()
        assert torch.allclose(params["pi"].sum(-1), torch.ones(B, T), atol=1e-5)
        assert (0 <= params["pen_up"]).all() and (params["pen_up"] <= 1).all()

    def test_hidden_state_passing(self):
        model = MDNRNN(input_dim=3, hidden_dim=64, num_mixtures=10)
        x = torch.randn(2, 10, 3)
        params1, hidden1 = model(x)
        params2, hidden2 = model(x, hidden1)
        assert hidden2[0].shape == hidden1[0].shape

    def test_init_hidden(self):
        model = MDNRNN(input_dim=3, hidden_dim=64, num_layers=3, num_mixtures=10)
        h, c = model.init_hidden(5)
        assert h.shape == (3, 5, 64)
        assert c.shape == (3, 5, 64)

    def test_gradient_flow(self):
        model = MDNRNN(input_dim=3, hidden_dim=64, num_mixtures=10)
        x = torch.randn(2, 20, 3)
        target = torch.randn(2, 20, 3)
        target[:, :, 2] = torch.sigmoid(target[:, :, 2])
        mask = torch.ones(2, 20, dtype=torch.bool)

        params, _ = model(x)
        loss_fn = MDNLoss()
        loss = loss_fn(params, target, mask)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestMDNRNNConditioned:
    def test_forward_shape(self):
        B, T, C, M = 2, 40, 15, 10
        model = MDNRNNConditioned(
            input_dim=3, hidden_dim=64, num_mixtures=M,
            num_windows=5, char_vocab_size=80, char_embed_dim=16,
        )
        x = torch.randn(B, T, 3)
        char_ids = torch.randint(0, 80, (B, C))
        char_mask = torch.ones(B, C, dtype=torch.bool)

        params, hidden, phi = model(x, char_ids, char_mask)

        assert params["mu_x"].shape == (B, T, M)
        assert phi.shape == (B, T, C)

    def test_chunked_forward(self):
        B, T, C, M = 2, 40, 15, 10
        model = MDNRNNConditioned(
            input_dim=3, hidden_dim=64, num_mixtures=M,
            num_windows=5, char_vocab_size=80, char_embed_dim=16,
            chunk_size=8,
        )
        x = torch.randn(B, T, 3)
        char_ids = torch.randint(0, 80, (B, C))
        char_mask = torch.ones(B, C, dtype=torch.bool)

        params, hidden, phi = model(x, char_ids, char_mask, chunk_size=8)
        assert params["mu_x"].shape == (B, T, M)


class TestWindowedAttention:
    def test_forward_shape(self):
        B, T, C, K, E = 2, 30, 20, 5, 16
        attn = WindowedAttention(hidden_dim=64, char_embed_dim=E, num_windows=K)
        lstm_out = torch.randn(B, T, 64)
        char_embeddings = torch.randn(B, C, E)
        char_mask = torch.ones(B, C, dtype=torch.bool)

        context, phi = attn(lstm_out, char_embeddings, char_mask)
        assert context.shape == (B, T, E)
        assert phi.shape == (B, T, C)

    def test_monotonic_kappa(self):
        B, T, K = 2, 50, 5
        attn = WindowedAttention(hidden_dim=64, char_embed_dim=16, num_windows=K)
        lstm_out = torch.randn(B, T, 64)
        char_embeddings = torch.randn(B, 20, 16)
        char_mask = torch.ones(B, 20, dtype=torch.bool)

        context, phi, final_kappa = attn(lstm_out, char_embeddings, char_mask, return_kappa=True)
        assert final_kappa.shape == (B, K)


class TestSequenceDiscriminator:
    def test_forward_shape(self):
        B, T = 4, 100
        disc = SequenceDiscriminator(input_dim=3, hidden_dim=64, num_layers=3)
        x = torch.randn(B, T, 3)
        output = disc(x)
        assert output.shape == (B,)
        assert (0 <= output).all() and (output <= 1).all()

    def test_gradient_flow(self):
        disc = SequenceDiscriminator(input_dim=3, hidden_dim=64, num_layers=3)
        x = torch.randn(2, 50, 3, requires_grad=True)
        output = disc(x)
        loss = output.mean()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Loss Tests
# ---------------------------------------------------------------------------

class TestMDNLoss:
    def test_loss_is_positive(self):
        B, T, M = 4, 50, 20
        params = {
            "mu_x": torch.randn(B, T, M),
            "mu_y": torch.randn(B, T, M),
            "sigma_x": torch.exp(torch.randn(B, T, M)),
            "sigma_y": torch.exp(torch.randn(B, T, M)),
            "rho": torch.tanh(torch.randn(B, T, M)),
            "pi": torch.softmax(torch.randn(B, T, M), dim=-1),
            "pen_up": torch.sigmoid(torch.randn(B, T)),
        }
        target = torch.randn(B, T, 3)
        target[:, :, 2] = torch.sigmoid(target[:, :, 2])
        mask = torch.ones(B, T, dtype=torch.bool)

        loss_fn = MDNLoss()
        loss = loss_fn(params, target, mask)
        assert loss > 0
        assert loss.isfinite()

    def test_masking(self):
        B, T, M = 2, 30, 10
        params = {
            "mu_x": torch.randn(B, T, M),
            "mu_y": torch.randn(B, T, M),
            "sigma_x": torch.exp(torch.randn(B, T, M)),
            "sigma_y": torch.exp(torch.randn(B, T, M)),
            "rho": torch.tanh(torch.randn(B, T, M)),
            "pi": torch.softmax(torch.randn(B, T, M), dim=-1),
            "pen_up": torch.sigmoid(torch.randn(B, T)),
        }
        target = torch.randn(B, T, 3)
        target[:, :, 2] = torch.sigmoid(target[:, :, 2])
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 20:] = False

        loss_fn = MDNLoss()
        loss_masked = loss_fn(params, target, mask)
        loss_unmasked = loss_fn(params, target, None)
        assert loss_masked != loss_unmasked


class TestMixtureMean:
    def test_shape(self):
        B, T, M = 4, 50, 20
        mu_x = torch.randn(B, T, M)
        mu_y = torch.randn(B, T, M)
        pi = torch.softmax(torch.randn(B, T, M), dim=-1)
        pen_up = torch.sigmoid(torch.randn(B, T))

        fake_seq = mdn_mixture_mean(mu_x, mu_y, pi, pen_up)
        assert fake_seq.shape == (B, T, 3)


class TestAdversarialLoss:
    def test_shape_and_finite(self):
        B = 4
        disc_real = torch.sigmoid(torch.randn(B))
        disc_fake = torch.sigmoid(torch.randn(B))
        mask = torch.ones(B, dtype=torch.bool)

        disc_loss, gen_adv = adversarial_loss(disc_real, disc_fake, mask)
        assert disc_loss.isfinite()
        assert gen_adv.isfinite()
        assert disc_loss > 0
        assert gen_adv > 0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_unconditional_train_step(self, temp_xml_dir):
        xml_files = list(Path(temp_xml_dir).glob("*.xml"))
        mean_x, std_x, mean_y, std_y = compute_dataset_stats(xml_files)
        loader = build_dataloader(
            xml_files, batch_size=2, shuffle=True,
            mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y,
        )
        model = MDNRNN(input_dim=3, hidden_dim=64, num_mixtures=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MDNLoss()

        model.train()
        batch = next(iter(loader))
        data = batch["data"]
        mask = batch["mask"]

        optimizer.zero_grad()
        params, _ = model(data)
        loss = loss_fn(params, data, mask)
        loss.backward()
        optimizer.step()

        assert loss.isfinite()

    def test_conditioned_train_step(self, temp_xml_dir):
        xml_files = list(Path(temp_xml_dir).glob("*.xml"))
        vocab = CharVocab()
        mean_x, std_x, mean_y, std_y = compute_dataset_stats(xml_files)
        loader = build_conditioned_dataloader(
            xml_files, vocab=vocab, batch_size=2, shuffle=True,
            mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y,
        )
        model = MDNRNNConditioned(
            input_dim=3, hidden_dim=64, num_mixtures=10,
            num_windows=5, char_vocab_size=len(vocab), char_embed_dim=16,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = MDNLoss()

        model.train()
        batch = next(iter(loader))
        data = batch["data"]
        mask = batch["mask"]
        char_ids = batch["char_ids"]
        char_mask = batch["char_mask"]

        optimizer.zero_grad()
        params, _, _ = model(data, char_ids, char_mask)
        loss = loss_fn(params, data, mask)
        loss.backward()
        optimizer.step()

        assert loss.isfinite()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
