# AGENTS.md

## Project overview
Language Pipes is a Python 3.10 application that distributes LLM inference across multiple machines. The CLI entrypoint is `language_pipes.cli:main`, and the core modules live under `src/language_pipes/`. See the docs in `documentation/` for architecture, configuration, CLI usage, and the OpenAI-compatible API.

## Repository layout
- `src/language_pipes/`: application code
  - `tui/`: interactive terminal UI for creating, editing, and running configs
  - `pipes/`, `jobs/`, `modeling/`: pipe construction and the job-processing FSM
  - `content_provider/`: bridges the TUI to runtime/config state
  - `distributed_state_network/`: default peer-to-peer router implementation
  - `llm_layer_collector/`: layer-level HuggingFace model loading helper
  - `util/`: shared utilities, including the OpenAI-compatible server helpers
  - `cli.py`, `runner.py`, `oai_server.py`, `config.py`: CLI entrypoint, node runner, OAI HTTP server, TOML config model
- `documentation/`: product and operator docs (source of truth; also published to the website)
- `website/`: Astro + Starlight site (landing page + docs). Consumes `documentation/` in place at build time; deployed to GitHub Pages via `.github/workflows/deploy-website.yml`
- `tests/`: pytest suite, mirroring `src/language_pipes` (`tests/language_pipes`, `tests/distributed_state_network`, `tests/llm_layer_collector`)
- `pyproject.toml`: project metadata and dependencies

## Development guidelines
- Target Python 3.10+ (per `pyproject.toml`).
- Keep changes aligned with the documented architecture and configuration behaviors in `documentation/`.
- Prefer updating or adding tests when modifying runtime behavior.
- If you touch CLI behavior, update `documentation/cli.md`.
- If you touch configuration fields, update `documentation/configuration.md`.
- Node configuration is entirely TOML-based (`-c`/`--config`); there are no per-flag config overrides (see `documentation/cli.md`).

## Common commands
- Install (editable): `pip install -e .`
- Run CLI: `language-pipes --help`
- Launch TUI with a config: `language-pipes -c config.toml`
- Run headless: `language-pipes -c config.toml run`

## Tests
- Run the full suite: `pytest`
