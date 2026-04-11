
from typing import Tuple

from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.components.exit_confirm import ExitConfirm
from language_pipes.tui.components.side_nav import SideNav
from language_pipes.tui.components.top_nav import TopNav
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.tui import TermText, TuiWindow


class NavWindow:
    top_nav: TopNav
    side_nav: SideNav
    nav_state: NavState

    def __init__(self, window: TuiWindow, nav: NavState, size: Tuple[int, int], pos: Tuple[int, int]):
        self.window = window
        self.bar_id = self.window.add_text(TermText("_" * (size[0] - 2)), (1, 2))
        
        self.nav_state = nav
        self.top_nav = TopNav(
            window,
            self.nav_state.TOP_HEADERS
        )
        self.side_nav = SideNav(
            window,
            self.top_nav,
            self.nav_state.active_side_options()
        )

    def hide_overlay(self):
        self.window.hide_txt(self.bar_id)
    
    def show_overlay(self):
        self.window.show_txt(self.bar_id)

    def side_next(self):
        self.nav_state.side_next(self.side_nav)

    def side_prev(self):
        self.nav_state.side_prev(self.side_nav)

    def sync(self, exit_confirm: ExitConfirm, edit_confirm: Confirm):
        active_options = self.nav_state.active_side_options()
        self.side_nav.focused_idx = (
            min(self.nav_state.active_side_idx(), len(active_options) - 1)
            if active_options
            else 0
        )
        self.side_nav.set_options(active_options)

        self.top_nav.focused_idx = self.nav_state.active_top_idx
        interactive_overlay_open = (
            exit_confirm.is_open or edit_confirm.is_open
        )
        self.top_nav.set_focus(
            self.nav_state.focus_depth == 0 and not interactive_overlay_open
        )
        self.top_nav._update_styles()
        self.side_nav.set_focus(
            self.nav_state.focus_depth == 1 and not interactive_overlay_open
        )

        if self.nav_state.focus_depth == 0:
            self.top_nav.show()
            self.side_nav.hide()
            self.show_overlay()
        
        if self.nav_state.focus_depth == 1:
            self.top_nav.show()
            self.side_nav.show()
            self.show_overlay()
        
        if self.nav_state.focus_depth == 2:
            self.top_nav.hide()
            self.side_nav.hide()
            self.hide_overlay()