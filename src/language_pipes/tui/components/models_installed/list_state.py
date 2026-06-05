from typing import Dict, List

from ansinout import PressedKey

from language_pipes.content_provider.model_provider import ModelProvider
from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text, make_selectable_text, make_window_text


class ListPageState(PageState):
    def __init__(self):
        super().__init__('list')
        self.focus_idx = 0
        self.installed_models: List[str] = []

    def on_change(self, args: Dict):
        self.focus_idx = 0

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self._on_prev()
        if key == PressedKey.ArrowDown:
            self._on_next()
        if key == PressedKey.Enter:
            self._on_enter()
        if key == PressedKey.Delete:
            self._on_delete()
        if key == PressedKey.Escape:
            self._on_escape()

    def _on_escape(self):
        self.exit_page()

    def _on_enter(self):
        if self.focus_idx == len(self.installed_models):
            self.change_state('download', {'fresh': True})

    def _on_next(self):
        self.focus_idx = (self.focus_idx + 1) % (len(self.installed_models) + 1)

    def _on_prev(self):
        self.focus_idx = (self.focus_idx - 1) % (len(self.installed_models) + 1)

    def _on_delete(self):
        if self.focus_idx < 0 or self.focus_idx >= len(self.installed_models):
            return
        model_name = self.installed_models[self.focus_idx]

        def on_apply():
            model_statuses = self.provider.model_provider.get_models_status()
            if model_name in model_statuses:
                for m in model_statuses[model_name]:
                    if m.end_model:
                        self.provider.model_provider.unload_end_model(model_name)
                    else:
                        self.provider.model_provider.unload_layer_models(model_name, m.device)

            layer_models = [m for m in self.provider.model_provider.get_layer_models() if m.model_id != model_name]
            self.provider.model_provider.save_layer_models(layer_models)

            end_models = [m for m in self.provider.model_provider.get_end_models() if m != model_name]
            self.provider.model_provider.save_end_models(end_models)

            self.provider.model_provider.delete_installed_model(model_name)

        self.confirm.open(f"Delete {model_name}?", on_apply=on_apply, on_discard=lambda: None)

    def get_view(self) -> List[str]:
        self.installed_models = ModelProvider.get_installed_models()

        lines = ["Installed Models:", ""]
        entries = []
        for i, model in enumerate(self.installed_models):
            entries.append([make_selectable_text(model, self.focus_idx == i), ""])
        entries.append([make_selectable_text("Install New Model", self.focus_idx == len(self.installed_models))])

        lines.extend(make_window_text(entries, self.focus_idx, 14))
        return lines

    def get_footer(self) -> str:
        return make_footer_text(["Arrow U/D: Move", "Delete: Delete Model", "Esc: Menu"])
