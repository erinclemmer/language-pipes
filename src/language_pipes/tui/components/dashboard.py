from typing import Any, Callable, List, Optional

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.frame.provider_calls import ProviderCall


class Dashboard:
    OPTIONS = [
        ("Start Network Server", "Network", "Status"),
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
                state = self._get_state(status)
                if state == "stopped":
                    self.loader.call_provider(ProviderCall.start_network)
                elif state == "running":
                    self.loader.call_provider(ProviderCall.stop_network)
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

    def _get_ram_usage(self) -> Optional[str]:
        try:
            used_ram = self.loader.call_provider(ProviderCall.get_used_system_ram)
            total_ram = self.loader.call_provider(ProviderCall.get_total_system_ram)
        except LookupError:
            return None
        return f"System RAM: {used_ram:.1f} / {total_ram:.1f} GB"

    @staticmethod
    def _get_state(status: Optional[Any]) -> str:
        if status is None:
            return "stopped"
        return getattr(status, "state", "running" if getattr(status, "running", False) else "starting")

    def get_view(self) -> List[str]:
        status = self._get_status()
        ram_usage = self._get_ram_usage()
        state = self._get_state(status)
        is_running = state == "running"
        focused = self.is_focused()
        peer_text = (
            f" ({getattr(status, 'num_peers', 0)} peer(s) connected)"
            if is_running
            else ""
        )
        lines = [f"Network Server: {state.title()}{peer_text}", ""]
        if ram_usage is not None:
            lines.extend([ram_usage, ""])
        for idx, (label, _, _) in enumerate(self.OPTIONS):
            if idx == 0:
                if state == "running":
                    label = "Stop Network Server"
                elif state == "starting":
                    label = "Starting Network Server"
                elif state == "stopping":
                    label = "Stopping Network Server"
            selected = focused and idx == self.selected_idx
            l_cursor = "|>" if selected else "  "
            r_cursor = "<|" if selected else "  "
            lines.append(f"{l_cursor} {label} {r_cursor}")
        return lines

    def get_footer(self) -> str:
        return "Arrows U/D: Move   Enter: Select   Esc: Back"
