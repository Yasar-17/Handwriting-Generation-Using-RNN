# Contributing

Thank you for considering contributing to the Handwriting Generation project.

## Development Setup

```bash
git clone https://github.com/Yasar-17/Handwriting-Generation-Using-RNN
cd Handwriting-Generation-Using-RNN
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS
pip install -e ".[dev]"
pre-commit install
```

## Code Quality

```bash
make lint        # Run ruff linter
make typecheck   # Run mypy type checker
make test        # Run pytest suite
```

## Project Structure

| Module | Purpose |
|---|---|
| `data.py` | XML parsing, normalization, datasets, rendering |
| `models.py` | MDNRNN, MDNRNNConditioned, WindowedAttention, SequenceDiscriminator |
| `losses.py` | MDN NLL, mixture-mean, adversarial BCE losses |
| `train.py` | Training loop (unconditional & conditioned) |
| `eval.py` | NLL evaluation, comparison images, demo API |
| `inference.py` | Production CLI for batch generation |
| `render.py` | SVG export and themed rendering |
| `benchmark.py` | Inference latency and throughput benchmarks |
| `augmentations.py` | Stroke-level data augmentation transforms |
| `api.py` | FastAPI REST API |
| `app.py` | Gradio web demo |

## Pull Request Guidelines

1. Fork the repository and create a feature branch from `main`.
2. Add tests for any new functionality under `tests/`.
3. Ensure `make lint`, `make typecheck`, and `make test` all pass.
4. Update documentation (README, docstrings) as needed.
5. Keep PRs focused on a single concern.

## Commit Message Format

```
type: short description

feat:     new feature
fix:      bug fix
docs:     documentation changes
test:     test additions or fixes
refactor: code refactoring
chore:    maintenance tasks
```

## Reporting Issues

Open an issue at https://github.com/Yasar-17/Handwriting-Generation-Using-RNN/issues with:
- Python version and OS
- Steps to reproduce
- Expected vs. actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
