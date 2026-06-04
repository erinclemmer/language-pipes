from typing import Dict, List, Optional

from ansinout import PressedKey

from language_pipes.config import ModelToLoad
from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.hosted_models_view import format_model_line
from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import make_footer_text, make_selectable_text, make_window_text

class ListPageState(PageState):
    layer_models: List[ModelToLoad]

    def __init__(self):
        super().__init__('list')
        self.layer_models = []
        self.model_idx = 0

    def on_change(self, args: Dict):
        pass
    
    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self._on_prev()
        if key == PressedKey.ArrowDown:
            self._on_next()
        if key == PressedKey.Enter:
            self._on_enter()
        if key == PressedKey.Escape:
            self._on_escape()
        if key == PressedKey.Delete:
            self._on_delete()

    def _on_delete(self):
        if self.model_idx == len(self.layer_models):
            return

        model_to_delete = self.layer_models[self.model_idx]

        running_text = ""
        if self._current_model_running():
            running_text = "\nThis will unload the model from memory"

        self.confirm.open(
            f"Remove {model_to_delete.model_id} on {model_to_delete.device}?{running_text}", 
            on_apply=self._delete_model, 
            on_discard=lambda: None
        )

    def _on_escape(self):
        self.exit_page()

    def _on_enter(self):
        args = { "model": self._get_current_model(), "model_idx": self.model_idx }

        if self.model_idx == len(self.layer_models) or not self._network_running():
            self.change_state('edit', args)
        else:
            self.change_state('options', args)
    
    def _on_next(self):
        self.model_idx = (self.model_idx + 1) % (len(self.layer_models) + 1)

    def _on_prev(self):
        self.model_idx = (self.model_idx - 1) % (len(self.layer_models) + 1)

    def get_view(self) -> List[str]:
        lines = [ContentProvider.get_ram_usage(), ""]

        lines.append("Layer Models:")

        if not self._network_running():
            lines.extend(["[WARNING] Network is not started.\nModels cannot be loaded until the network is started.", ""])

        self.layer_models = self.provider.model_provider.get_layer_models()
        models_status = self.provider.model_provider.get_models_status()

        entries: List[List[str]] = []
        for i, model in enumerate(self.layer_models):
            entry = list(format_model_line(
                model=model,
                selected=self.model_idx == i,
                running=models_status.get(model.model_id, [])
            ))
            entry.append("")
            entries.append(entry)
        entries.append([make_selectable_text(
            "Add Layer Model", self.model_idx == len(self.layer_models)
        )])

        lines.extend(make_window_text(entries, self.model_idx, 10))

        lines.extend([
            "", 
            "Tip: Layer models are segments of a model's transformer layers loaded",
            "into memory on a device. Multiple nodes can each host different layer",
            "ranges to distribute inference across machines. Layer ranges are inferred",
            "from the network state."
        ])

        return lines

    def get_footer(self):
        return make_footer_text(["Arrows U/D: Move", "Enter: Select Option", "Delete: Remove", "Esc: Menu"])

    def _current_model_running(self) -> bool:
        model = self._get_current_model()
        if model is None:
            return False
        return self._model_running(model)
    
    def _get_current_model(self) -> Optional[ModelToLoad]:
        if self.model_idx == len(self.layer_models):
            return None
        if self.model_idx < len(self.layer_models):
            return self.layer_models[self.model_idx]
        return None

    def _model_running(self, model: ModelToLoad) -> bool:
        return any(
            not status.end_model and str(status.device) == str(model.device)
            for status in self.provider.model_provider.get_models_status().get(model.model_id, [])
        )
    
    def _delete_model(self):
        model_to_delete = self.layer_models[self.model_idx]

        if self._current_model_running():
            self.provider.model_provider.unload_layer_models(model_to_delete.model_id, model_to_delete.device)
        
        models_to_load = []
        for m in self.layer_models:
            if m.model_id == model_to_delete.model_id and str(m.device) == str(model_to_delete.device):
                continue
            models_to_load.append(m)

        self.layer_models = models_to_load
        self.provider.model_provider.save_layer_models(self.layer_models)

    def _network_running(self) -> bool:
        network_status = self.provider.network_provider.get_network_status()
        return network_status is not None and network_status.running
