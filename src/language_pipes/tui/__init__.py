from pathlib import Path
from shutil import get_terminal_size
from typing import Optional

from language_pipes.tui.util.screen_utils import (
    enable_vt_mode,
    restore_mode,
    exit_vt_mode,
)

def initialize_tui(config_file: Optional[str], auto_start: bool):
    fd_in = None
    old_in_attrs = None
    entered_alt_screen = False

    term_size = get_terminal_size(fallback=(80, 24))
    
    try:
        fd_in, old_in_attrs = enable_vt_mode()
        entered_alt_screen = True

        from language_pipes.tui.main_menu import main_menu

        main_menu((term_size.columns, term_size.lines), config_file, auto_start)
    finally:
        if entered_alt_screen:
            exit_vt_mode()
        if fd_in is not None and old_in_attrs is not None:
            restore_mode(fd_in, old_in_attrs)
