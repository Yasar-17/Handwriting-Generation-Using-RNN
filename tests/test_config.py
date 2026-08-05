"""Tests for YAML config file support in the training CLI."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import apply_config, flatten_config, load_yaml_config

SAMPLE_CONFIG = """
data:
  data_dir: "./synthetic_data"
  batch_size: 16
  num_workers: 2
  unknown_section_key: true
model:
  hidden_dim: 128
training:
  epochs: 10
  lr: 0.0005
  use_gan: true
sampling:
  temperature: 0.7
output:
  output_dir: "./my_output"
"""


def write_config(tmp_path: Path, text: str = SAMPLE_CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoadYamlConfig:
    def test_loads_nested_mapping(self, tmp_path):
        path = write_config(tmp_path)
        config = load_yaml_config(path)
        assert config["data"]["batch_size"] == 16
        assert config["training"]["lr"] == 0.0005

    def test_non_mapping_raises(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("- just\n- a\n- list", encoding="utf-8")
        with pytest.raises(SystemExit):
            load_yaml_config(path)


class TestFlattenConfig:
    def test_flattens_nested_dicts(self):
        config = {"a": {"b": 1, "c": 2}, "d": 3}
        flat = flatten_config(config)
        assert flat == {"a.b": 1, "a.c": 2, "d": 3}


class TestApplyConfig:
    def test_sets_known_defaults(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--temperature", type=float, default=0.5)
        parser.add_argument("--data_dir", type=str, default=None)

        apply_config(
            parser,
            {"data": {"batch_size": 16}, "training": {"epochs": 10}, "sampling": {"temperature": 0.7}},
        )
        args = parser.parse_args([])
        assert args.batch_size == 16
        assert args.epochs == 10
        assert args.temperature == 0.7

    def test_ignores_unknown_and_none_values(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=64)
        apply_config(parser, {"data": {"should_be_ignored": 3}, "other": {"batch_size": None}})
        args = parser.parse_args([])
        assert args.batch_size == 64

    def test_none_values_not_applied(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lr", type=float, default=1e-3)
        apply_config(parser, {"training": {"lr": None}})
        assert parser.parse_args([]).lr == 1e-3
