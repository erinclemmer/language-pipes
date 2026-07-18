# Contributing to Language Pipes

Thanks for your interest in Language Pipes! Contributions of all kinds are
welcome. Bug reports, documentation fixes, new model architectures, and larger
features. This guide covers how to get set up and what to expect when you open a
pull request.

If you just want to ask a question or float an idea, start a
[GitHub Discussion](https://github.com/erinclemmer/language-pipes/discussions).
For bugs and concrete feature requests, open an
[issue](https://github.com/erinclemmer/language-pipes/issues).

## Development setup

Language Pipes targets **Python 3.10+**.

```bash
# 1. Clone your fork
git clone https://github.com/<you>/language-pipes.git
cd language-pipes

# 2. Create and activate a virtual environment
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

# 3. Install the in-repo sub-packages (editable), then language-pipes itself.
# language-pipes pins exact versions of these, so installing them editable first
# satisfies those pins with your local checkout instead of pulling from PyPI.
pip install -e packages/llm-layer-collector -e packages/distributed-state-network
pip install -e .
pip install pytest coverage

# (Optional) For GPU support, install the PyTorch build that matches your CUDA
# version from https://pytorch.org/get-started/locally/ before/after step 3.
```

Once installed, the CLI is available as `language-pipes`:

```bash
language-pipes --help          # top-level help
language-pipes -c config.toml  # launch the TUI with a config preloaded
language-pipes -c config.toml run   # run a node headless
```

See the [CLI reference](./documentation/cli.md) and
[Configuration reference](./documentation/configuration.md) for details.

## Running the tests

The suite lives under `tests/` and mirrors `src/language_pipes/`.

```bash
# Run the fast unit tests (no model downloads required)
pytest tests/language_pipes/unit

# Run the whole suite
pytest
```

> **Note:** The integration tests (`tests/language_pipes/integration`) and the
> "real model" tests download model weights from HuggingFace and start real
> nodes, for the tests to pass the models must be downloaded. It is not recommended
> to run them unless you are adding support for a new model. The unit tests under
> `tests/language_pipes/unit` do not require model weights, and are what continuous integration runs.

Continuous integration runs the unit tests on every pull request — see
[`.github/workflows/tests.yml`](./.github/workflows/tests.yml).

## Pull request expectations

- **Branch from `main`** and open your PR against `main`.
- **Keep PRs focused.** One logical change per PR is much easier to review.
- **Add or update tests** when you change runtime behavior. New architectures and
  bug fixes should come with a test that would have caught the problem.
- **Update the docs** in `documentation/` when you change behavior:
  - CLI changes → `documentation/cli.md`
  - Configuration fields → `documentation/configuration.md`
  - New supported model → `documentation/model_support.md`
- **Make sure `pytest tests/language_pipes/unit` passes** locally before pushing.
- Describe **what** changed and **why** in the PR description. Link any related
  issue or discussion.

## Code style

- Target Python 3.10+ syntax and features.
- Match the style of the surrounding code: type hints on public functions,
  descriptive names, and small, single-purpose functions.
- Keep changes aligned with the documented architecture and configuration
  behavior in `documentation/`. If your change makes a doc inaccurate, update the
  doc in the same PR.
- Node configuration is entirely TOML-based (`-c`/`--config`); there are no
  per-flag config overrides. Don't add hidden environment-variable overrides for
  config values.

## Adding a new model architecture

Adding support for a model family is one of the most valuable contributions you can make, and makes a great first PR. The model-loading code lives under
`packages/llm-layer-collector/src/llm_layer_collector/modeling/`, alongside the existing
`Qwen3`, `Qwen3Moe`, `Phi3`, `Llama`, `Gemma3`, `Gemma4`, and `Ministral3`
implementations. Use one of those as a template, wire it into
`llm_layer_collector/helpers.py`, add a tested checkpoint to
`documentation/model_support.md`, and add a test.
