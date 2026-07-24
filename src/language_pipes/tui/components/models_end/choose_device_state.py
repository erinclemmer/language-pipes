from typing import Dict, List

from ansinout import PressedKey
import torch

from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text, make_selectable_text, make_window_text


class ChooseDevicePageState(PageState):
    available_devices: List[str]
    select_idx: int

    def __init__(self):
        super().__init__('choose_device')
        self.available_devices = []
        self.select_idx = 0

    def on_change(self, args: Dict):
        self.available_devices = self._get_available_devices()
        device = args.get("device")
        if device is not None and device in self.available_devices:
            self.select_idx = self.available_devices.index(device)
        else:
            self.select_idx = 0

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self._on_prev()
        if key == PressedKey.ArrowDown:
            self._on_next()
        if key == PressedKey.Enter:
            self._on_enter()
        if key == PressedKey.Escape:
            self._on_escape()

    def _on_next(self):
        if len(self.available_devices) > 0:
            self.select_idx = (self.select_idx + 1) % len(self.available_devices)

    def _on_prev(self):
        if len(self.available_devices) > 0:
            self.select_idx = (self.select_idx - 1) % len(self.available_devices)

    def _on_enter(self):
        if len(self.available_devices) > 0:
            self.change_state('edit', { "device": self.available_devices[self.select_idx] })
        else:
            self.change_state('edit', { })

    def _on_escape(self):
        self.change_state('edit', { })

    def get_view(self) -> List[str]:
        lines = ["Choose Device", ""]

        if len(self.available_devices) <= 1 and not torch.cuda.is_available():
            lines.extend(["[WARNING] Could not detect cuda device", ""])

        entries: List[List[str]] = []
        for i, device in enumerate(self.available_devices):
            entries.append([make_selectable_text(device, self.select_idx == i), ""])

        lines.extend(make_window_text(entries, self.select_idx, 14))

        return lines

    def get_footer(self) -> str:
        return make_footer_text(["Arrows U/D: Move", "Enter: Select Device", "Esc: Back"])

    @staticmethod
    def _get_available_devices() -> List[str]:
        devices = ["cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        return devices
