# AGENTS.md

## Project overview
Language Pipes is a Python 3.10 application that distributes LLM inference across multiple machines. The CLI entrypoint is `language_pipes.cli:main`, and the core modules live under `src/language_pipes/`. See the docs in `documentation/` for architecture, configuration, CLI usage, and the OpenAI-compatible API.

## Repository layout
- `src/language_pipes/`: application code (networking, model management, job execution, API server)
- `documentation/`: product and operator docs
- `tests/`: automated tests (if any)
- `pyproject.toml`: project metadata and dependencies

## Development guidelines
- Target Python 3.10 (per `pyproject.toml`).
- Keep changes aligned with the documented architecture and configuration behaviors in `documentation/`.
- Prefer updating or adding tests when modifying runtime behavior.
- If you touch CLI behavior, update `documentation/cli.md`.
- If you touch configuration fields, update `documentation/configuration.md`.

## Common commands
- Install (editable): `pip install -e .`
- Run CLI: `language-pipes --help`
- Example server start: `language-pipes serve -c config.toml`

## Tests
No standard test command is documented. If you run tests, mention the command and result in your summary.
