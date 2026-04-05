from typing import Dict, List, Tuple

from language_pipes.tui.components.top_nav import TopNav
from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.util.screen_utils import Color

class SideNav:
    window: TuiWindow
    focused_idx: int
    is_focused: bool
    options: List[str]
    option_ids: List[int]
    l_cursor_id: int
    r_cursor_id: int

    def __init__(
            self,
            window: TuiWindow,
            top_nav: TopNav,
            options: List[str]
        ):
        self.window = window
        self.top_nav = top_nav
        self.options = []
        self.option_ids = []
        self.focused_idx = 0
        self.is_focused = False

        self.l_cursor_id = self.window.add_text(TermText(" "), (0, 0))

        self.set_options(options)

    def hide(self):
        self.window.hide_txt(self.l_cursor_id)
        for oid in self.option_ids:
            self.window.hide_txt(oid)

    def show(self):
        self.window.show_txt(self.l_cursor_id)
        for oid in self.option_ids:
            self.window.show_txt(oid)

    def _cursor_y(self) -> int:
        return (self.focused_idx * 2) + 4

    def _update_cursor(self):
        if len(self.options) == 0:
            self.window.update_text(self.l_cursor_id, TermText(" "))
            return

        l_cursor = "|>" if self.is_focused else " "
        self.window.update_text(self.l_cursor_id, TermText(l_cursor), (2 + self.top_nav.focused_idx * 12, self._cursor_y()))

    def _update_option_styles(self):
        for i, option_id in enumerate(self.option_ids):
            self.window.update_text(
                option_id,
                TermText(
                    self.options[i],
                    fg=Color.Cyan if i == self.focused_idx else None,
                    bold=(i == self.focused_idx)
                ),
                (5 + self.top_nav.focused_idx * 12, (i * 2) + 4)
            )
        self._update_cursor()

    def move_next(self):
        if len(self.options) == 0:
            return
        self.focused_idx = (self.focused_idx + 1) % len(self.options)
        self._update_option_styles()

    def move_prev(self):
        if len(self.options) == 0:
            return
        self.focused_idx = (self.focused_idx - 1) % len(self.options)
        self._update_option_styles()

    def set_focus(self, is_focused: bool):
        if is_focused == self.is_focused:
            return
        
        self.is_focused = is_focused
        self._update_option_styles()

    def set_options(self, options: List[str]):
        if options == self.options:
            return
        
        for option_id in self.option_ids:
            self.window.remove_txt(option_id)

        self.options = options
        self.option_ids = []
        self.focused_idx = min(self.focused_idx, len(options) - 1) if len(options) > 0 else 0

        for i, opt in enumerate(options):
            self.option_ids.append(self.window.add_text(TermText(opt), (3, (i * 2) + 4)))

        self._update_option_styles()
