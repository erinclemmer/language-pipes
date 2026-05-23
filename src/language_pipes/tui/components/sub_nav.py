from typing import List

from language_pipes.tui.components.top_nav import TopNav
from language_pipes.tui.frame.nav_state import NavState
from ansinout import TuiWindow, TermText

class SubNav:
    window: TuiWindow
    top_nav: TopNav
    state: NavState
    options: List[str]
    option_ids: List[int]
    cursor_id: int
    
    def __init__(
            self,
            window: TuiWindow,
            top_nav: TopNav,
            state: NavState
        ):
        self.window = window
        self.top_nav = top_nav
        self.state = state
        self.options = []
        self.option_ids = []
        
        self.cursor_id = self.window.add_text(TermText(" "), (0, 0))

        self.set_options(state.active_sub_options())

    def hide(self):
        self.window.hide_txt(self.cursor_id)
        for oid in self.option_ids:
            self.window.hide_txt(oid)

    def show(self):
        self.window.show_txt(self.cursor_id)
        for oid in self.option_ids:
            self.window.show_txt(oid)

    def _cursor_y(self) -> int:
        return (self.state.sub_idx * 2) + 4

    def update_cursor(self):
        x = self.top_nav.header_positions[self.state.top_idx] - 3
        self.window.update_text(self.cursor_id, TermText("|>"), (x, self._cursor_y()))

    def set_options(self, options: List[str]):
        if options == self.options:
            return
        
        for option_id in self.option_ids:
            self.window.remove_txt(option_id)

        self.options = options
        self.option_ids = []

        x = self.top_nav.header_positions[self.state.top_idx]
        for i, opt in enumerate(options):
            self.option_ids.append(self.window.add_text(TermText(opt), (x, (i * 2) + 4)))