from ansinout import PressedKey

from language_pipes.config import EndModelConfig
from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.content_provider.model_provider import ModelProvider, ModelStatus
from language_pipes.tui.components.page import PageState
from language_pipes.tui.util.text import (
    make_footer_text,
    make_selectable_text,
    make_window_text,
)


class ListPageState(PageState):
    end_models: list[EndModelConfig]
    model_sizes: dict[str, float]

    def __init__(self):
        super().__init__('list')
        self.end_models = []
        self.model_idx = 0
        self.model_sizes = { }

    def on_change(self, args: dict):
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
        if self.model_idx == len(self.end_models):
            return

        model_to_delete = self.end_models[self.model_idx]

        running_text = ""
        if self._is_loaded(model_to_delete.model_id):
            running_text = "\nThis will unload the model from memory"

        self.confirm.open(
            f"Remove {model_to_delete.model_id}?{running_text}",
            on_apply=self._delete_model,
            on_discard=lambda: None
        )

    def _on_escape(self):
        self.exit_page()

    def _on_enter(self):
        if self.model_idx == len(self.end_models):
            self.change_state('edit', { "model": None, "model_idx": None })
        else:
            self.change_state('options', {
                "model": self.end_models[self.model_idx],
                "model_idx": self.model_idx,
            })

    def _on_next(self):
        self.model_idx = (self.model_idx + 1) % (len(self.end_models) + 1)

    def _on_prev(self):
        self.model_idx = (self.model_idx - 1) % (len(self.end_models) + 1)

    def get_view(self) -> list[str]:
        lines = [ContentProvider.get_ram_usage(), "", "End Models:"]

        self.end_models = self.provider.model_provider.get_end_model_configs()

        entries: list[list[str]] = []
        for i, model in enumerate(self.end_models):
            loaded_text = ""
            if self._is_loaded(model.model_id):
                loaded_text = "(Loaded)"
            if self._is_loading(model.model_id):
                loaded_text = "(Loading...)"

            size = self._get_model_size(model.model_id, model.num_local_layers)
            if size is None:
                line = f"{model.model_id} [Error getting metadata]"
            else:
                line = f"{model.model_id} ({size:.2f} GB) {loaded_text}"
            entries.append([make_selectable_text(line, self.model_idx == i), ""])

        entries.append([make_selectable_text(
            "Add End Model", self.model_idx == len(self.end_models)
        )])

        lines.extend(make_window_text(entries, self.model_idx, 10))

        lines.extend([
            "",
            "Tip: End models hold the embedding and output head of a model and",
            "keep text data on this trusted node. Load one to serve a model's",
            "text-handling side while layer models run on peers across the network."
        ])

        return lines

    def get_footer(self) -> str:
        if self.model_idx == len(self.end_models):
            return make_footer_text(["Arrows U/D: Move", "Enter: Add End Model", "Esc: Menu"])
        return make_footer_text(["Arrows U/D: Move", "Enter: Select Option", "Delete: Remove", "Esc: Menu"])

    def _delete_model(self):
        model_to_delete = self.end_models[self.model_idx]

        if self._is_loaded(model_to_delete.model_id):
            self.provider.model_provider.unload_end_model(model_to_delete.model_id)

        self.end_models = [m for m in self.end_models if m.model_id != model_to_delete.model_id]
        self.model_idx = 0
        self.provider.model_provider.save_end_model_configs(self.end_models)

    def _get_model_size(self, model_id: str, num_local_layers: int) -> float | None:
        cache_key = f"{model_id}:{num_local_layers}"
        if cache_key in self.model_sizes:
            return self.model_sizes[cache_key]
        metadata = ModelProvider.get_model_metadata(model_id)
        if not metadata.loaded:
            return None
        size = (metadata.head_size + metadata.embed_size + (metadata.avg_layer_size * num_local_layers)) / 1024**3
        self.model_sizes[cache_key] = size
        return size

    def _is_loaded(self, model_id: str) -> bool:
        model_statuses = self.provider.model_provider.get_models_status()
        loaded_models = []
        if model_id in model_statuses:
            loaded_models = [s for s in model_statuses[model_id] if s.end_model and s.status == ModelStatus.Running]

        return len(loaded_models) > 0

    def _is_loading(self, model_id: str) -> bool:
        model_statuses = self.provider.model_provider.get_models_status()
        loading_models = []
        if model_id in model_statuses:
            loading_models = [s for s in model_statuses[model_id] if s.end_model and (s.status == ModelStatus.Starting or s.status == ModelStatus.Stopping)]

        return len(loading_models) > 0
