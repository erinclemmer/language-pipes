from typing import Dict, List

from ansinout import PressedKey

from language_pipes.content_provider.model_provider import ModelProvider
from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text, make_selectable_text, make_window_text


class ChooseModelPageState(PageState):
    installed_models: List[str]
    select_idx: int

    def __init__(self):
        super().__init__('choose_model')
        self.installed_models = []
        self.select_idx = 0

    def on_change(self, args: Dict):
        self.installed_models = ModelProvider.get_installed_models()
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
        if len(self.installed_models) > 0:
            self.select_idx = (self.select_idx + 1) % len(self.installed_models)

    def _on_prev(self):
        if len(self.installed_models) > 0:
            self.select_idx = (self.select_idx - 1) % len(self.installed_models)

    def _on_enter(self):
        if len(self.installed_models) == 0:
            self.change_state('edit', { })
            return
        self.change_state('edit', { "model_id": self.installed_models[self.select_idx] })

    def _on_escape(self):
        self.change_state('edit', { })

    def get_view(self) -> List[str]:
        lines = ["Choose Model to Host", ""]

        if not self._network_running():
            lines.extend(["[WARNING] Network is not started.\nModels cannot be loaded until the network is started.", ""])

        entries: List[List[str]] = []
        for i, model in enumerate(self.installed_models):
            entries.append([make_selectable_text(model, self.select_idx == i), ""])

        lines.extend(make_window_text(entries, self.select_idx, 14))

        return lines

    def get_footer(self) -> str:
        return make_footer_text(["Arrows U/D: Move", "Enter: Select Model", "Esc: Back"])

    def _network_running(self) -> bool:
        network_status = self.provider.network_provider.get_network_status()
        return network_status is not None and network_status.running
