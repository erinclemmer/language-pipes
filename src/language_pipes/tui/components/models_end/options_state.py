from typing import Dict, List, Optional

from ansinout import PressedKey

from language_pipes.config import EndModelConfig
from language_pipes.content_provider.model_provider import ModelStatus
from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text, make_selectable_text


class OptionsPageState(PageState):
    model: Optional[EndModelConfig]
    model_idx: Optional[int]
    option_idx: int

    def __init__(self):
        super().__init__('options')
        self.model = None
        self.model_idx = None
        self.option_idx = 0

    def on_change(self, args: Dict):
        self.model = args["model"]
        self.model_idx = args["model_idx"]
        self.option_idx = 0

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
        self.option_idx = (self.option_idx + 1) % 3

    def _on_prev(self):
        self.option_idx = (self.option_idx - 1) % 3

    def _on_escape(self):
        self.change_state('list', { })

    def _on_enter(self):
        if self.option_idx == 0:
            self.change_state('edit', { "model": self.model, "model_idx": self.model_idx })
        elif self.option_idx == 1 and not self._current_model_loading():
            model = self.model
            if model is not None:
                if self._current_model_running():
                    self.provider.model_provider.unload_end_model(model.model_id)
                else:
                    self.provider.model_provider.load_end_model(model.model_id)
            self.change_state('list', { })
        elif self.option_idx == 2 or (self.option_idx == 1 and self._current_model_loading()):
            self.change_state('list', { })

    def get_view(self) -> List[str]:
        model = self.model
        if model is None:
            return ["ERROR"]

        lines = [f"Options for {model.model_id}"]
        options = ["Edit Model"]

        if not self._current_model_loading():
            options.append("Unload Model" if self._current_model_running() else "Load Model")

        options.append("Back")

        for i, opt in enumerate(options):
            lines.append(make_selectable_text(opt, self.option_idx == i))

        lines.append("")
        if self.option_idx == 0:
            lines.append("Help: Edit model configuration")

        if self.option_idx == 1 and not self._current_model_running():
            lines.append(f"Help: Load {model.model_id} with {model.num_local_layers} local layers")

        if self.option_idx == 1 and self._current_model_running():
            lines.append("Help: Unload the model from memory")

        return lines

    def get_footer(self) -> str:
        return make_footer_text(["Arrows U/D: Move", "Enter: Select Option", "Esc: Back"])

    def _current_model_loading(self) -> bool:
        model = self.model
        assert model is not None
        statuses = self.provider.model_provider.get_models_status()
        if model.model_id not in statuses:
            return False
        for status in statuses[model.model_id]:
            if status.end_model and (status.status == ModelStatus.Starting or status.status == ModelStatus.Stopping):
                return True
        return False

    def _current_model_running(self) -> bool:
        model = self.model
        if model is None:
            return False
        return any(
            status.end_model and status.status == ModelStatus.Running
            for status in self.provider.model_provider.get_models_status().get(model.model_id, [])
        )
