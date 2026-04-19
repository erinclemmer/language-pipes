import os
import sys
import unittest
from typing import cast
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from language_pipes.cli import VERSION
from language_pipes.tui.main_menu import main_menu


class _FakeWindow:
    last_instance = None

    def __init__(self, *args, **kwargs):
        type(self).last_instance = self
        self.added_texts = []

    def add_text(self, text, pos):
        self.added_texts.append((text.value, pos))
        return len(self.added_texts) - 1

    def paint(self):
        return None

    def remove_all(self):
        return None


class MainMenuVersionTests(unittest.TestCase):
    def test_banner_uses_version_constant(self):
        with (
            patch("language_pipes.tui.main_menu.TuiWindow", _FakeWindow),
            patch("language_pipes.tui.main_menu.load_libraries", lambda window: None),
            patch(
                "language_pipes.tui.main_menu.default_app_dir",
                return_value="/tmp/language-pipes",
            ),
            patch(
                "language_pipes.tui.main_menu.default_model_dir",
                return_value="/tmp/language-pipes-models",
            ),
            patch("language_pipes.tui.main_menu.get_config_files", return_value=[]),
            patch("language_pipes.tui.main_menu.select_option", return_value="Exit"),
            patch("language_pipes.tui.main_menu.os.path.exists", return_value=True),
            self.assertRaises(SystemExit),
        ):
            main_menu((80, 24))

        banner_window = _FakeWindow.last_instance
        self.assertIsNotNone(banner_window)
        banner_window = cast(_FakeWindow, banner_window)
        self.assertIn((f"Version {VERSION}", (0, 7)), banner_window.added_texts)
