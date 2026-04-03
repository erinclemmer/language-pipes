from typing import Any, Callable, List, Optional

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.frame.provider_calls import ProviderCall


class Dashboard:
    OPTIONS = [
        ("Start Network", "Network", "Status"),
        ("Host Models", "Models", "Hosted"),
    ]

    def __init__(
        self,
        loader: Any,
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
            if self.selected_idx == 0:
                status = self._get_status()
                if status is None or not status.running:
                    self.loader.call_provider(ProviderCall.start_network)
            else:
                _, tab, section = self.OPTIONS[self.selected_idx]
                self.change_nav(tab, section)
        elif key == PressedKey.Escape:
            self.exit_page()

    def _get_status(self) -> Optional[Any]:
        try:
            return self.loader.call_provider(ProviderCall.get_network_status)
        except LookupError:
            return None

    def get_view(self) -> List[str]:
        status = self._get_status()
        is_running = status is not None and status.running
        focused = self.is_focused()
        lines = [f"Network Server: {'On' if is_running else 'Off'}", ""]
        for idx, (label, _, _) in enumerate(self.OPTIONS):
            selected = focused and idx == self.selected_idx
            l_cursor = "|>" if selected else "  "
            r_cursor = "<|" if selected else "  "
            lines.append(f"{l_cursor} {label} {r_cursor}")
        return lines

    def get_footer(self) -> str:
        return "Arrows U/D: Move   Enter: Select   Esc: Back"
