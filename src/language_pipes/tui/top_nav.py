from typing import List, Tuple

from language_pipes.tui.tui import TuiWindow, TermText

class TopNav:
    def __init__(self, size: Tuple[int, int], pos: Tuple[int, int], headers: List[str]):
        self.window = TuiWindow(size, pos)
        self.headers = headers
        self.header_ids: List[int] = []
        self.header_positions: List[int] = []
        self.is_focused = True

        for i, h in enumerate(self.headers):
            header_x = 5 + i * 15
            self.header_positions.append(header_x)
            self.header_ids.append(self.window.add_text(TermText(h), (header_x, 0)))

        self.focused_idx = 0
        self.l_cursor_id = self.window.add_text(TermText("["), (3, 0))
        self.r_cursor_id = self.window.add_text(TermText("]"), (3, 0))

        self._update_styles()

    def _update_styles(self):
        for i, header_id in enumerate(self.header_ids):
            self.window.update_text(
                header_id,
                TermText(
                    self.headers[i],
                    fg=51 if i == self.focused_idx else None,
                    bold=(self.is_focused and i == self.focused_idx),
                ),
            )

        selected_x = self.header_positions[self.focused_idx]
        selected_header = self.headers[self.focused_idx]
        if self.is_focused:
            self.window.update_text(self.l_cursor_id, TermText("["), (selected_x - 2, 0))
            self.window.update_text(self.r_cursor_id, TermText("]"), (selected_x + len(selected_header), 0))
        else:
            self.window.update_text(self.l_cursor_id, TermText(" "), (selected_x - 2, 0))
            self.window.update_text(self.r_cursor_id, TermText(" "), (selected_x + len(selected_header), 0))

    def move_next(self):
        self.focused_idx = (self.focused_idx + 1) % len(self.headers)
        self._update_styles()

    def move_prev(self):
        self.focused_idx = (self.focused_idx - 1) % len(self.headers)
        self._update_styles()

    def set_focus(self, is_focused: bool):
        self.is_focused = is_focused
        self._update_styles()

        self.window.paint()