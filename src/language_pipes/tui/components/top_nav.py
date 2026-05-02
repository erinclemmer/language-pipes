from typing import List

from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.tui import TuiWindow, TermText

class TopNav:
    ALL_HEADERS = ["Home", "Network", "Models", "Pipes", "Jobs"]

    def __init__(self, window: TuiWindow, state: NavState):
        self.window = window
        self.state = state
        self.header_ids: List[int] = []
        self.header_positions: List[int] = []
        self.init_headers()

        self.l_cursor_id = self.window.add_text(TermText("["), (3, 1))
        self.r_cursor_id = self.window.add_text(TermText("]"), (3, 1))

    def init_headers(self):
        total_len = 0
        for hdr in self.ALL_HEADERS:
            total_len += len(hdr)

        spacing = int((85 - total_len) / len(self.ALL_HEADERS) - 1)

        current_pos = 5
        for h in self.ALL_HEADERS:
            self.header_ids.append(self.window.add_text(TermText(h), (current_pos, 1)))
            self.header_positions.append(current_pos)
            current_pos += len(h) + spacing

    def sync_headers(self):
        for hid in self.header_ids:
            txt = self.window.get_text(hid)
            if txt.text.value in self.state.top_headers:
                self.window.show_txt(hid)
            else:
                self.window.hide_txt(hid)
    
    def header_visible(self, hdr: str) -> bool:
        return hdr in self.state.top_headers

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

    def set_focus(self, is_focused: bool):
        selected_x = self.header_positions[self.state.top_idx]
        selected_header = self.state.active_tab()
        if is_focused:
            self.window.update_text(self.l_cursor_id, TermText("["), (selected_x - 1, 1))
            self.window.update_text(self.r_cursor_id, TermText("]"), (selected_x + len(selected_header), 1))
        else:
            self.window.update_text(self.l_cursor_id, TermText(" "), (selected_x - 1, 1))
            self.window.update_text(self.r_cursor_id, TermText(" "), (selected_x + len(selected_header), 1))