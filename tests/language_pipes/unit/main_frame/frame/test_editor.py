import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.frame.editor import Editor

class TestEditor(unittest.TestCase):
    """Unit tests for Editor in isolation."""

    def test_initial_state(self):
        ed = Editor()
        self.assertFalse(ed.edit_mode)
        self.assertEqual(ed.edit_field_idx, 0)
        self.assertEqual(ed.edit_fields, [])
        self.assertIsNone(ed.form)

    def test_start_edit_mode(self):
        ed = Editor()
        fields = [{"name": "a", "value": "1"}, {"name": "b", "value": "2"}]
        mock_form = MagicMock()
        ed.start_edit_mode("test_form", fields, mock_form)
        self.assertTrue(ed.edit_mode)
        self.assertEqual(ed.edit_form_name, "test_form")
        self.assertEqual(len(ed.edit_fields), 2)
        self.assertIs(ed.form, mock_form)

    def test_exit_edit_mode_resets(self):
        ed = Editor()
        ed.start_edit_mode("f", [{"name": "x"}], MagicMock())
        ed.exit_edit_mode()
        self.assertFalse(ed.edit_mode)
        self.assertEqual(ed.edit_fields, [])
        self.assertEqual(ed.edit_field_idx, 0)

    def test_next_field_clamps(self):
        ed = Editor()
        ed.start_edit_mode("f", [{"name": "a"}, {"name": "b"}], MagicMock())
        ed.next_field()
        self.assertEqual(ed.edit_field_idx, 1)
        ed.next_field()
        self.assertEqual(ed.edit_field_idx, 1)  # clamped

    def test_prev_field_clamps(self):
        ed = Editor()
        ed.start_edit_mode("f", [{"name": "a"}, {"name": "b"}], MagicMock())
        ed.prev_field()
        self.assertEqual(ed.edit_field_idx, 0)  # clamped

    def test_get_current_field(self):
        ed = Editor()
        ed.start_edit_mode("f", [{"name": "host", "value": "localhost"}], MagicMock())
        result = ed.get_current_field()
        self.assertEqual(result, ("host", "localhost"))

    def test_get_current_field_empty(self):
        ed = Editor()
        self.assertIsNone(ed.get_current_field())