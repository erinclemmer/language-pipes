
from typing import Tuple

from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.components.exit_confirm import ExitConfirm
from language_pipes.tui.components.sub_nav import SubNav
from language_pipes.tui.components.top_nav import TopNav
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.tui import TermText, TuiWindow

class NavWindow:
    top_nav: TopNav
    sub_nav: SubNav
    nav_state: NavState

    def __init__(self, window: TuiWindow, nav: NavState, size: Tuple[int, int], pos: Tuple[int, int]):
        self.window = window
        self.bar_id = self.window.add_text(TermText("_" * (size[0] - 2)), (1, 2))
        
        self.nav_state = nav
        self.top_nav = TopNav(
            window,
            self.nav_state
        )
        self.sub_nav = SubNav(
            window,
            self.top_nav,
            self.nav_state
        )

    def hide_overlay(self):
        self.window.hide_txt(self.bar_id)
    
    def show_overlay(self):
        self.window.show_txt(self.bar_id)

    def sync(self, exit_confirm: ExitConfirm, edit_confirm: Confirm):
        self.top_nav.sync_headers()

        interactive_overlay_open = exit_confirm.is_open or edit_confirm.is_open

        self.top_nav.set_focus(
            self.nav_state.focus_depth == 0 and not interactive_overlay_open
        )
        
        self.sub_nav.set_options(self.nav_state.active_sub_options())
        self.sub_nav.update_cursor()

        if self.nav_state.focus_depth == 0:
            self.top_nav.show()
            self.sub_nav.hide()
            self.show_overlay()
        
        if self.nav_state.focus_depth == 1:
            self.top_nav.show()
            self.sub_nav.show()
            self.show_overlay()
        
        if self.nav_state.focus_depth == 2:
            self.top_nav.hide()
            self.sub_nav.hide()
            self.hide_overlay()