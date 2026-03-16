"""
Testing plan and tests for MainFrame (src/language_pipes/tui/frame/main_frame.py).

MainFrame is the top-level TUI controller. It wires together:
  - TuiWindow (terminal grid rendering)
  - Editor (form field editing state machine)
  - FrameState (running/exit flags, status messages)
  - ExitConfirm / Confirm (modal dialogs)
  - ContentLoader (provider-backed data loading with cache)
  - NavState (tab/side-nav/content cursor tracking)
  - NetworkForm (network configuration editing workflow)
  - FrameLayout (rendering, window lifecycle)
  - FrameKeyHandler (keyboard dispatch)

Testing strategy
================
The TUI writes directly to stdout via ANSI escape codes and reads raw
keypresses from stdin.  To unit-test MainFrame without a real terminal we
must **mock the I/O boundary** (screen_utils.write, screen_utils.print_pos,
read_key) and verify behaviour through the object graph.

We split the plan into four layers:

1. **Construction & wiring** – verify __init__ assembles the object graph
   correctly and leaves the frame in the expected initial state.

2. **Navigation state** – verify that simulated keypress sequences move
   focus between top-nav, side-nav, and content pane correctly.

3. **Run-loop lifecycle** – verify that `run()` starts the state machine,
   processes keys, and returns the correct exit code.

4. **Interactive overlays** – verify exit-confirm and edit-mode flows
   (NetworkForm) are triggered and resolved correctly.

Each section below contains concrete test cases.
"""

import os
import sys
from unittest.mock import patch
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.tui.util.kb_utils import PressedKey

# ---------------------------------------------------------------------------
# Helpers – suppress all real terminal I/O
# ---------------------------------------------------------------------------

# Patch targets for terminal output (print_pos, write) and input (read_key).
_WRITE_PATCH = "language_pipes.tui.util.screen_utils.write"
_PRINT_POS_PATCH = "language_pipes.tui.util.screen_utils.print_pos"
_READ_KEY_PATCH = "language_pipes.tui.frame.main_frame.read_key"


def _noop_write(s: str):
    """Swallow all terminal writes."""
    pass


def _noop_print_pos(*args, **kwargs):
    """Swallow all positioned prints."""
    pass


def _make_main_frame(providers=None, size=(80, 24), pos=(0, 0)):
    """Create a MainFrame with all terminal I/O suppressed."""
    with patch(_WRITE_PATCH, _noop_write), \
         patch(_PRINT_POS_PATCH, _noop_print_pos):
        from language_pipes.tui.frame.main_frame import MainFrame
        return MainFrame(size, pos, providers)


def _simulate_keys(frame, key_sequence: List[Tuple[PressedKey, str]]):
    """Feed a sequence of (PressedKey, ch) pairs through the key handler,
    suppressing rendering I/O after each key."""
    with patch(_WRITE_PATCH, _noop_write), \
         patch(_PRINT_POS_PATCH, _noop_print_pos):
        for key, ch in key_sequence:
            frame.key_handler.handle_key(key, ch)
            frame.layout._render_all()
