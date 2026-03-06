from shutil import get_terminal_size

from language_pipes.tui.screen_utils import enable_vt_mode, restore_mode, exit_vt_mode
from language_pipes.tui.tui import TuiWindow
from language_pipes.tui.main_menu import main_menu

def initialize_tui():
    fd_in = None
    old_in_attrs = None
    entered_alt_screen = False

    term_size = get_terminal_size(fallback=(80, 24))
    
    try:
        fd_in, old_in_attrs = enable_vt_mode()
        entered_alt_screen = True

        main_menu((term_size.columns, term_size.lines))
    finally:
        if entered_alt_screen:
            exit_vt_mode()
        if fd_in is not None and old_in_attrs is not None:
            restore_mode(fd_in, old_in_attrs)
