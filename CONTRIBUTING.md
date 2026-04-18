# Contributing to WhiteMatter

## Setup

```bash
git clone https://github.com/hwang2409/whitematter.git
cd whitematter
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## Pull request process

1. Fork and create a branch from `main` (`feat/description` or `fix/description`)
2. Make focused changes and add tests
3. Run `pytest tests/ -v` — all tests must pass
4. Push and open a PR against `main`

## Conventions

- **Commit messages**: imperative mood ("Add LSTM layer", not "Added LSTM layer")
- **One concern per PR**: keep bug fixes and features in separate PRs
- **Tests required**: all changes need passing tests
- **Code style**: follow existing patterns in the codebase

## Prerequisites

- Python 3.9+
- NumPy (installed automatically)
- pytest (installed with `.[dev]`)
