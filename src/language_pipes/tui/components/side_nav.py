from typing import Dict, List, Tuple

from language_pipes.tui.tui import TuiWindow, TermText

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
            size: Tuple[int, int],
            pos: Tuple[int, int],
            options: List[str]
        ):
        self.window = TuiWindow(size, pos)
        self.options = []
        self.option_ids = []
        self.focused_idx = 0
        self.is_focused = False

        self.l_cursor_id = self.window.add_text(TermText(" "), (0, 0))
        self.r_cursor_id = self.window.add_text(TermText(" "), (self.window.size[0] - 1, 0))

        self.set_options(options)
        self.window.paint()

    def _cursor_y(self) -> int:
        return self.focused_idx * 2

    def _update_cursor(self):
        if len(self.options) == 0:
            self.window.update_text(self.l_cursor_id, TermText(" "))
            self.window.update_text(self.r_cursor_id, TermText(" "))
            return

        cursor_char = ">" if self.is_focused else " "
        self.window.update_text(self.l_cursor_id, TermText(cursor_char), (0, self._cursor_y()))
        self.window.update_text(self.r_cursor_id, TermText("<" if self.is_focused else " "), (self.window.size[0] - 1, self._cursor_y()))

    def _update_option_styles(self):
        for i, option_id in enumerate(self.option_ids):
            self.window.update_text(
                option_id,
                TermText(
                    self.options[i],
                    fg=36 if i == self.focused_idx else None,
                    bold=(i == self.focused_idx)
                )
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
        self.is_focused = is_focused
        self._update_option_styles()

    def set_options(self, options: List[str]):
        for option_id in self.option_ids:
            self.window.remove_txt(option_id)

        self.options = options
        self.option_ids = []
        self.focused_idx = min(self.focused_idx, len(options) - 1) if len(options) > 0 else 0

        for i, opt in enumerate(options):
            self.option_ids.append(self.window.add_text(TermText(opt), (2, i * 2)))

        self._update_option_styles()
