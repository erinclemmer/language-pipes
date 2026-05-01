from typing import List

from language_pipes.content_provider.content_provider import ProviderState
from language_pipes.tui.tui import TuiWindow, TermText

class TopNav:
    def __init__(self, window: TuiWindow, headers: List[str]):
        self.window = window
        self.headers = headers
        self.header_ids: List[int] = []
        self.header_positions: List[int] = []
        self.is_focused = True
        self._last_provider_state = None

        for i, h in enumerate(self.headers):
            header_x = 5 + i * 12
            self.header_positions.append(header_x)
            self.header_ids.append(self.window.add_text(TermText(h), (header_x, 1)))

        self.focused_idx = 0
        self.l_cursor_id = self.window.add_text(TermText("["), (3, 1))
        self.r_cursor_id = self.window.add_text(TermText("]"), (3, 1))

        self._update_styles()

    def sync_state(self, state: ProviderState):
        for hid in self.header_ids:
            txt = self.window.get_text(hid)
            if txt.text.value in state.visible_headers:
                self.window.show_txt(hid)
            else:
                self.window.hide_txt(hid)
        self._last_provider_state = state
    
    def header_visible(self, hdr: str) -> bool:
        if self._last_provider_state is None:
            return True
        return hdr in self._last_provider_state.visible_headers

    def hide(self):
        self.window.hide_txt(self.l_cursor_id)
        self.window.hide_txt(self.r_cursor_id)
        for hid in self.header_ids:
            txt = self.window.get_text(hid)
            if self.header_visible(txt.text.value):
                self.window.hide_txt(hid)

    def show(self):
        self.window.show_txt(self.l_cursor_id)
        self.window.show_txt(self.r_cursor_id)
        for hid in self.header_ids:
            txt = self.window.get_text(hid)
            if self.header_visible(txt.text.value):
                self.window.show_txt(hid)

    def _update_styles(self):
        for i, header_id in enumerate(self.header_ids):
            self.window.update_text(
                header_id,
                TermText(self.headers[i]),
            )

        selected_x = self.header_positions[self.focused_idx]
        selected_header = self.headers[self.focused_idx]
        if self.is_focused:
            self.window.update_text(self.l_cursor_id, TermText("["), (selected_x - 1, 1))
            self.window.update_text(self.r_cursor_id, TermText("]"), (selected_x + len(selected_header), 1))
        else:
            self.window.update_text(self.l_cursor_id, TermText(" "), (selected_x - 1, 1))
            self.window.update_text(self.r_cursor_id, TermText(" "), (selected_x + len(selected_header), 1))

    def set_focus(self, is_focused: bool):
        if is_focused == self.is_focused:
            return
        self.is_focused = is_focused
        self._update_styles()