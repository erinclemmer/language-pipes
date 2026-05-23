from typing import Optional
from shutil import get_terminal_size

from ansinout import (
    enable_vt_mode,
    exit_vt_mode,
)

def initialize_tui(config_file: Optional[str], auto_start: bool):
    term_size = get_terminal_size(fallback=(80, 24))
    
    fd_in, old_in_attrs = enable_vt_mode()
    try:
        
        from language_pipes.tui.main_menu import main_menu

        main_menu((term_size.columns, min(term_size.lines, 25)), config_file, auto_start)
    finally:
        exit_vt_mode(fd_in, old_in_attrs)
