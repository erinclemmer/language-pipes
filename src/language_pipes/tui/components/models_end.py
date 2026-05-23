from enum import Enum
from typing import Callable, Dict

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.content_provider.model_provider import ModelProvider, ModelStatus
from language_pipes.tui.components.confirm import Confirm
from ansinout import PressedKey
from language_pipes.tui.util.text import make_footer_text, make_selectable_text, make_window_text


class EndModelsState(Enum):
    LIST = "LIST"
    CHOOSE = "CHOOSE"

class ModelsEndModels:
    provider: ContentProvider
    exit_page: Callable
    is_focused: Callable

    list_idx: int
    model_sizes: Dict[str, float]
    
    def __init__(
        self, provider: ContentProvider, confirm: Confirm, exit_page: Callable, is_focused: Callable
    ):
        self.provider = provider
        self.confirm = confirm
        self.exit_page = exit_page
        self.is_focused = is_focused
        self.state = EndModelsState.LIST
        self.end_models = []
        self.installed_models = []
        self.list_idx = 0
        self.choose_idx = 0
        self.model_sizes = { }
    
    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Escape:
            self.on_escape()
        if key == PressedKey.Enter:
            self.on_enter()
        if key == PressedKey.Delete:
            self.on_delete()
        if key == PressedKey.ArrowUp:
            self.on_prev()
        if key == PressedKey.ArrowDown:
            self.on_next()

    def on_delete(self):
        if self.state != EndModelsState.LIST or self.list_idx == len(self.end_models):
            return
        selected_model = self.end_models[self.list_idx]
        self.confirm.open(
            f"Remove {selected_model}?",
            on_apply=self.remove_selected_model,
            on_discard=lambda: None
        )
        
    def remove_selected_model(self):
        model_to_remove = self.end_models[self.list_idx]
        self.end_models = [m for m in self.end_models if m != model_to_remove]
        self.list_idx = 0
        self.provider.model_provider.save_end_models(self.end_models)

    def on_escape(self):
        if self.state == EndModelsState.LIST:
            self.exit_page()
        elif self.state == EndModelsState.CHOOSE:
            self.state = EndModelsState.LIST
            self.choose_idx = 0

    def on_enter(self):
        if self.state == EndModelsState.LIST:
            selected_model = self.end_models[self.list_idx]
            if self.list_idx == len(self.end_models):
                self.state = EndModelsState.CHOOSE
            elif self._is_loaded(selected_model) and not self._is_loading(selected_model):
                def on_apply():
                    self.provider.model_provider.unload_end_model(selected_model)
                self.confirm.open(
                    f"Unload {selected_model}",
                    on_apply=on_apply,
                    on_discard=lambda: None
                )
            elif not self._is_loaded(selected_model) and not self._is_loading(selected_model):
                def on_apply():
                    self.provider.model_provider.load_end_model(selected_model)
                self.confirm.open(
                    f"Load {selected_model}",
                    on_apply=on_apply,
                    on_discard=lambda: None
                )
        elif self.state == EndModelsState.CHOOSE and len(self.available_models()) > 0:
            self.state = EndModelsState.LIST
            selected_model = self.available_models()[self.choose_idx]
            self.confirm.open(
                f"Add {selected_model}?",
                on_apply=self.add_model,
                on_discard=lambda: None
            )

    def available_models(self):
        return [m for m in self.installed_models if m not in self.end_models]

    def add_model(self):
        available_models = self.available_models()
        self.end_models.append(available_models[self.choose_idx])
        self.provider.model_provider.save_end_models(self.end_models)

    def on_next(self):
        if self.state == EndModelsState.LIST:
            self.list_idx = (self.list_idx + 1) % (len(self.end_models) + 1)
        elif self.state == EndModelsState.CHOOSE:
            self.choose_idx = (self.choose_idx + 1) % len(self.available_models())

    def on_prev(self):
        if self.state == EndModelsState.LIST:
            self.list_idx = (self.list_idx - 1) % (len(self.end_models) + 1)
        elif self.state == EndModelsState.CHOOSE:
            self.choose_idx = (self.choose_idx - 1) % len(self.available_models())

    def _get_ram_usage(self) -> str:
        used_ram = self.provider.get_used_system_ram()
        total_ram = self.provider.get_total_system_ram()
        
        return f"System RAM: {used_ram:.1f}/{total_ram:.1f}GB"

    def get_view(self):
        if self.state == EndModelsState.LIST:
            return self.get_list_view()
        if self.state == EndModelsState.CHOOSE:
            return self.get_choose_view()
    
    def _is_loaded(self, model_id: str):
        model_statuses = self.provider.model_provider.get_models_status()
        loaded_models = []
        if model_id in model_statuses:
            loaded_models = [s for s in model_statuses[model_id] if s.end_model and s.status == ModelStatus.Running]
        
        return len(loaded_models) > 0
    
    def _is_loading(self, model_id: str):
        model_statuses = self.provider.model_provider.get_models_status()
        loading_models = []
        if model_id in model_statuses:
            loading_models = [s for s in model_statuses[model_id] if s.end_model and (s.status == ModelStatus.Starting or s.status == ModelStatus.Stopping)]
        
        return len(loading_models) > 0

    def get_list_view(self):
        lines = [self._get_ram_usage(), "", "End Models:", ""]

        self.end_models = self.provider.model_provider.get_end_models()
        entries = []
        for i, m in enumerate(self.end_models):
            loaded_text = ""
            if self._is_loaded(m):
                loaded_text = "(Loaded)"
            if self._is_loading(m):
                loaded_text = "(Loading...)"
    
            line = f"{m} ({self.get_model_size(m):.2f} GB) {loaded_text}"
            entries.append([make_selectable_text(line, self.list_idx == i), ""])

        line = make_selectable_text("Add End Model", len(self.end_models) == self.list_idx)
        entries.append([line, ""])
        
        lines.extend(make_window_text(entries, self.list_idx, 10))

        lines.extend([
            "",
            "Tip: End models hold the embedding and output head of a model and",
            "keep text data on this trusted node. Load one to serve a model's",
            "text-handling side while layer models run on peers across the network."
        ])

        return lines
    
    def get_model_size(self, model_id: str) -> float:
        if model_id in self.model_sizes:
            return self.model_sizes[model_id]
        metadata = ModelProvider.get_model_metadata(model_id)
        size = (metadata.head_size + metadata.embed_size) / 10 ** 9
        self.model_sizes[model_id] = size
        return size

    def get_choose_view(self):
        lines = ["Choose an installed model:", ""]

        self.installed_models = ModelProvider.get_installed_models()
        entries = []
        available_models = self.available_models()
        for i, model in enumerate(available_models):
            model_size = self.get_model_size(model)
            line = f"{model} ({model_size:.2f} GB)"
            entries.append([make_selectable_text(line, i == self.choose_idx), ""])

        if len(available_models) == 0:
            lines.append("No models available")

        make_window_text(entries, self.choose_idx, 17)

        if len(self.installed_models) == 0:
            lines.append("No models installed, please install from the models/installed page")

        return lines

    def get_footer(self) -> str:
        if self.state == EndModelsState.LIST and self.list_idx != len(self.end_models):
            selected_model = self.end_models[self.list_idx]
            if self._is_loading(selected_model):
                return make_footer_text(["Arrows U/D: Move", "Delete: Remove Model", "Esc: Menu"])
            else:
                return make_footer_text(["Arrows U/D: Move", "Enter: load/unload", "Delete: Remove Model", "Esc: Menu"])

        if self.state == EndModelsState.LIST and self.list_idx == len(self.end_models):
            return make_footer_text(["Arrows U/D: Move", "Enter: Add End Model", "Esc: Menu"])
        
        if self.state == EndModelsState.CHOOSE and len(self.available_models()) == 0:
            return make_footer_text(["Arrows U/D: Move", "Esc: Back"])

        if self.state == EndModelsState.CHOOSE and len(self.available_models()) > 0:
            return make_footer_text(["Arrows U/D: Move", "Enter: Choose Model", "Esc: Back"])

        return ""
