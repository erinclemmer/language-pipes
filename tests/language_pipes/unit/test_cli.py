"""Behavioural tests for the command line interface.

These lock in the contract documented in ``documentation/cli.md`` so the docs
and the parser cannot silently drift apart. Every assertion here corresponds to
a statement in that document.
"""

import io
import os
import re
import sys
import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from language_pipes.cli import main, VERSION


class _NoExit:
    """Sentinel: ``main`` returned without raising SystemExit."""


def run_cli(argv):
    """Invoke ``main`` with ``argv`` and capture stdout + any SystemExit.

    Returns ``(stdout, exit_code)`` where ``exit_code`` is ``_NoExit`` when the
    command returned normally, otherwise the SystemExit code (which may be
    ``None`` for a bare ``exit()``).
    """
    out = io.StringIO()
    exit_code = _NoExit
    try:
        with contextlib.redirect_stdout(out):
            main(argv)
    except SystemExit as exc:
        exit_code = exc.code
    return out.getvalue(), exit_code


def write_config(directory, **values):
    """Write a minimal TOML config file and return its path."""
    path = Path(directory) / "node.toml"
    lines = []
    for key, value in values.items():
        if isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f'{key} = {value}')
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


class GlobalOptionTests(unittest.TestCase):
    def test_version_prints_version_and_exits(self):
        # Docs: `-v`, `--version` prints the version and exits.
        out, code = run_cli(["-v"])
        self.assertEqual(code, 0)
        self.assertIn(VERSION, out)

    def test_invalid_config_exits_with_message(self):
        # Docs: a non-existent `--config` exits with a specific error.
        out, code = run_cli(["-c", "/definitely/missing.toml"])
        self.assertIsNot(code, _NoExit)
        self.assertIn("is not a valid path or saved configuration", out)

    def test_options_must_precede_subcommand(self):
        # Docs: `-c` after the subcommand fails (it is a top-level option).
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, job_port=8000)
            _, code = run_cli(["run", "-c", cfg])
        # argparse reports an error and exits non-zero.
        self.assertEqual(code, 2)


class TuiCommandTests(unittest.TestCase):
    def test_no_subcommand_launches_tui_at_main_menu(self):
        # Docs: no subcommand opens the TUI; without `-c` it shows the main menu.
        with mock.patch("language_pipes.tui.initialize_tui") as init:
            run_cli([])
        init.assert_called_once_with(None, False)

    def test_config_preloads_tui(self):
        # Docs: `-c FILE` preloads a configuration into the TUI.
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, job_port=8000)
            with mock.patch("language_pipes.tui.initialize_tui") as init:
                run_cli(["-c", cfg])
            init.assert_called_once_with(cfg, False)

    def test_start_flag_autostarts_tui(self):
        # Docs: `--start` begins serving immediately (auto_start=True).
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, job_port=8000)
            with mock.patch("language_pipes.tui.initialize_tui") as init:
                run_cli(["-c", cfg, "--start"])
            init.assert_called_once_with(cfg, True)


class RunCommandTests(unittest.TestCase):
    def test_run_requires_config(self):
        # Docs: `run` without `--config` exits with a specific error.
        with mock.patch("language_pipes.runner.LpRunner") as runner:
            out, _ = run_cli(["run"])
        self.assertIn("ERROR: --config param required", out)
        runner.assert_not_called()

    def test_run_constructs_runner_from_config(self):
        # Docs: `-c FILE run` starts a headless node from the config file.
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, job_port=8000)
            with mock.patch("language_pipes.runner.LpRunner") as runner:
                run_cli(["-c", cfg, "run"])
            runner.assert_called_once_with(Path(cfg), None)

    def test_run_passes_token_argument_to_runner(self):
        # Docs: `-t/--token` forwards a HuggingFace token to the runner.
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, job_port=8000)
            with mock.patch("language_pipes.runner.LpRunner") as runner:
                run_cli(["-c", cfg, "run", "--token", "hf_secret"])
            runner.assert_called_once_with(Path(cfg), "hf_secret")

    def test_run_short_token_flag(self):
        # Docs: `-t` is the short form of `--token`.
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, job_port=8000)
            with mock.patch("language_pipes.runner.LpRunner") as runner:
                run_cli(["-c", cfg, "run", "-t", "hf_secret"])
            runner.assert_called_once_with(Path(cfg), "hf_secret")


class ConfigCommandTests(unittest.TestCase):
    def test_config_takes_no_positional_arguments(self):
        # Docs: `config` accepts no override arguments; extras are rejected.
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, job_port=8000)
            _, code = run_cli(["-c", cfg, "config", "job_port=9999"])
        self.assertEqual(code, 2)

    def test_config_prints_human_readable_report(self):
        # Docs: `config` prints a human-readable report, not valid TOML.
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, job_port=8000, node_id="node-4")
            out, _ = run_cli(["-c", cfg, "config"])
        self.assertIn("Configuration Settings", out)
        self.assertIn("Job Port: 8000", out)
        # The report is decorative, not TOML: it contains a banner rule.
        self.assertIn("====", out)


class KeygenCommandTests(unittest.TestCase):
    def test_keygen_prints_generated_key(self):
        # Docs: `keygen` prints a newly generated hex-encoded AES key.
        out, code = run_cli(["keygen"])
        self.assertEqual(code, _NoExit)
        match = re.search(r"Network key generated: ([0-9a-f]{32})", out)
        self.assertIsNotNone(match)

    def test_keygen_takes_no_positional_arguments(self):
        # Docs: `keygen` takes no arguments; extras are rejected.
        _, code = run_cli(["keygen", "somefile"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
