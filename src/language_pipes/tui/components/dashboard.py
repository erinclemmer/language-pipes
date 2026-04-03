from typing import List, Callable

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.content_loader import ContentLoader


class Dashboard:
    OPTIONS = [
        ("Start Network", "Network", "Status"),
        ("Host Models", "Models", "Hosted"),
    ]

    def __init__(
        self,
        loader: ContentLoader,
        exit_page: Callable,
        is_focused: Callable,
        change_nav: Callable,
    ):
        self.loader = loader
        self.exit_page = exit_page
        self.is_focused = is_focused
        self.change_nav = change_nav
        self.selected_idx = 0

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self.selected_idx = (self.selected_idx - 1) % len(self.OPTIONS)
        elif key == PressedKey.ArrowDown:
            self.selected_idx = (self.selected_idx + 1) % len(self.OPTIONS)
        elif key == PressedKey.Enter:
            _, tab, section = self.OPTIONS[self.selected_idx]
            self.change_nav(tab, section)
        elif key == PressedKey.Escape:
            self.exit_page()

    def get_view(self) -> List[str]:
        focused = self.is_focused()
        lines = ["Dashboard", ""]
        for idx, (label, _, _) in enumerate(self.OPTIONS):
            selected = focused and idx == self.selected_idx
            l_cursor = "|>" if selected else "  "
            r_cursor = "<|" if selected else "  "
            lines.append(f"{l_cursor} {label} {r_cursor}")
        return lines

    def get_footer(self) -> str:
        return "Arrows U/D: Move   Enter: Select   Esc: Back"
