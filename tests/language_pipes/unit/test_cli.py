"""Behavioural tests for the command line interface.

These lock in the contract documented in ``documentation/cli.md`` so the docs
and the parser cannot silently drift apart. Every assertion here corresponds to
a statement in that document.
"""

import io
import os
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
            cfg = write_config(d, oai_port=8000)
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
            cfg = write_config(d, oai_port=8000)
            with mock.patch("language_pipes.tui.initialize_tui") as init:
                run_cli(["-c", cfg])
            init.assert_called_once_with(cfg, False)

    def test_start_flag_autostarts_tui(self):
        # Docs: `--start` begins serving immediately (auto_start=True).
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, oai_port=8000)
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
            cfg = write_config(d, oai_port=8000)
            with mock.patch("language_pipes.runner.LpRunner") as runner:
                run_cli(["-c", cfg, "run"])
            runner.assert_called_once_with(Path(cfg), {})


class ConfigCommandTests(unittest.TestCase):
    def test_config_requires_a_positional_token(self):
        # Docs: the parser requires at least one positional KEY=VALUE token.
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, oai_port=8000)
            _, code = run_cli(["-c", cfg, "config"])
        self.assertEqual(code, 2)

    def test_config_prints_human_readable_report(self):
        # Docs: `config` prints a human-readable report, not valid TOML.
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, oai_port=8000, node_id="node-4")
            out, _ = run_cli(["-c", cfg, "config", "node_id=node-4"])
        self.assertIn("Configuration Settings", out)
        self.assertIn("Job Port: 8000", out)
        # The report is decorative, not TOML: it contains a banner rule.
        self.assertIn("====", out)

    def test_config_positional_overrides_are_not_applied(self):
        # Docs (Known limitations): override positionals are parsed but ignored.
        with tempfile.TemporaryDirectory() as d:
            cfg = write_config(d, oai_port=8000)
            out, _ = run_cli(["-c", cfg, "config", "oai_port=9999"])
        self.assertIn("Job Port: 8000", out)
        self.assertNotIn("9999", out)


class KeygenCommandTests(unittest.TestCase):
    def test_keygen_writes_to_named_output(self):
        # Docs: `keygen [output]` writes the key to the given path.
        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "my.key")
            out, _ = run_cli(["keygen", out_path])
            self.assertTrue(os.path.exists(out_path))
            self.assertIn("Network key saved", out)
            self.assertIn(out_path, out)

    def test_keygen_defaults_to_network_key(self):
        # Docs: the default output path is `network.key`.
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                run_cli(["keygen"])
                self.assertTrue(os.path.exists(os.path.join(d, "network.key")))
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
