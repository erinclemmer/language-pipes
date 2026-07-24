from typing import Dict, List, Optional

from ansinout import PressedKey

from language_pipes.config import DEFAULT_END_MODEL_DEVICE, DEFAULT_NUM_LOCAL_LAYERS, EndModelConfig
from language_pipes.content_provider.model_provider import ModelProvider, ModelStatus
from language_pipes.tui.components.page import PageState
from language_pipes.tui.frame.tips import TIPS
from language_pipes.tui.util.text import make_footer_text, make_selectable_text


class EditPageState(PageState):
    editing_model: Optional[EndModelConfig]
    editing_model_idx: Optional[int]

    model_id: str
    device_name: str
    local_layers: str

    select_idx: int

    def __init__(self):
        super().__init__('edit')
        self.editing_model = None
        self.editing_model_idx = None
        self.model_id = ""
        self.device_name = DEFAULT_END_MODEL_DEVICE
        self.local_layers = str(DEFAULT_NUM_LOCAL_LAYERS)
        self.select_idx = 0

    def on_change(self, args: Dict):
        # Returning from the model selector keeps the rest of the form intact.
        if "model_id" in args:
            self.model_id = args["model_id"]
            return
        # Returning from the device selector keeps the rest of the form intact.
        if "device" in args:
            self.device_name = args["device"]
            return
        # Cancelling out of a selector returns here with no args; keep the form.
        if "model" not in args:
            return

        # Fresh entry from the list/options page resets the form.
        self.editing_model = args["model"]
        self.editing_model_idx = args["model_idx"] if self.editing_model is not None else None
        self.model_id = self.editing_model.model_id if self.editing_model is not None else ""
        self.device_name = (
            self.editing_model.device
            if self.editing_model is not None
            else DEFAULT_END_MODEL_DEVICE
        )
        self.local_layers = (
            str(self.editing_model.num_local_layers)
            if self.editing_model is not None
            else str(DEFAULT_NUM_LOCAL_LAYERS)
        )
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
        if key in (PressedKey.Alpha, PressedKey.Paste):
            self._on_char(ch)
        if key == PressedKey.Backspace:
            self._on_backspace()

    def _on_backspace(self):
        if self.select_idx == 2:
            self.local_layers = self.local_layers[:-1]

    def _on_char(self, ch: str):
        if self.select_idx == 2:
            self.local_layers += ch

    def _on_escape(self):
        self.change_state('list', { })

    def _on_enter(self):
        if self.select_idx == 0:
            self.change_state('choose_model', { })
        elif self.select_idx == 1:
            self.change_state('choose_device', { "device": self.device_name })
        elif self.select_idx == 3:
            self._add_model()

    def _on_next(self):
        self.select_idx = (self.select_idx + 1) % (self._max_select_idx() + 1)

    def _on_prev(self):
        self.select_idx = (self.select_idx - 1) % (self._max_select_idx() + 1)

    def _model_id_lines(self) -> List[str]:
        lines = []
        model_id_label = (
            self.model_id if self._valid_model_id() else "Choose model..."
        )
        lines.append(make_selectable_text(f"Model ID: {model_id_label}", self.select_idx == 0))

        if not self._valid_model_id():
            lines.extend(["   ! Warning: Must choose model to load", ""])

        return lines

    def _device_lines(self) -> List[str]:
        device_label = (
            self.device_name if self._valid_device() else "Choose device..."
        )
        lines = [make_selectable_text(f"Device: {device_label}", self.select_idx == 1)]
        if not self._valid_device():
            lines.append("   !Warning: Must choose device")

        return lines

    def _local_layers_lines(self) -> List[str]:
        cursor = "|" if self.select_idx == 2 else ""
        line = f"   Local Layers: {self.local_layers}{cursor} "
        if not self._validate_local_layers():
            line += "(Warning: Must be a whole number >= 0)"
        else:
            line += self._get_total_layers()

        return [line]

    def _get_tip_lines(self) -> List[str]:
        tip_key = None
        if self.select_idx == 0:
            tip_key = "model_id"
        elif self.select_idx == 1:
            tip_key = "device"
        elif self.select_idx == 2:
            tip_key = "local_layers"

        if tip_key is not None:
            return ["", "Tip:", TIPS["end_models"][tip_key]]
        return []

    def get_view(self) -> List[str]:
        header = "Editing End Model" if self.editing_model is not None else "Creating End Model Configuration"
        lines = [header, ""]

        lines.extend(self._model_id_lines())

        if self._valid_model_id():
            lines.extend(self._device_lines())
            lines.extend(self._local_layers_lines())

        if self._is_adding_model() and self._has_model_already(self.model_id):
            lines.append("   !Warning: Model ID already in configuration")

        if self._can_save():
            lines.append("")
            lines.append(make_selectable_text("Save Model", self.select_idx == 3))

        lines.extend(self._get_tip_lines())

        return lines

    def get_footer(self) -> str:
        if self.select_idx == 0:
            return make_footer_text(["Arrows U/D: Move", "Enter: Change Model", "Esc: Back"])
        elif self.select_idx == 1:
            return make_footer_text(["Arrows U/D: Move", "Enter: Change Device", "Esc: Back"])
        elif self.select_idx == 2:
            return make_footer_text(["Arrows U/D: Move", "[0-9]: Type", "Esc: Back"])
        elif self.select_idx == 3:
            return make_footer_text(["Arrows U/D: Move", "Enter: Save End Model", "Esc: Back"])
        return ""

    def _max_select_idx(self) -> int:
        max_idx = 0
        if self._valid_model_id():
            # Device (1) and Local Layers (2) are always shown together.
            max_idx += 2
            if self._can_save():
                max_idx += 1
        return max_idx

    def _add_model(self) -> None:
        if not self._can_save():
            return

        model = EndModelConfig(
            model_id=self.model_id,
            num_local_layers=int(self.local_layers),
            device=self.device_name,
        )

        was_running = (
            self.editing_model is not None
            and self._current_model_running()
        )
        config_changed = (
            self.editing_model is not None
            and (
                self.editing_model.num_local_layers != model.num_local_layers
                or self.editing_model.device != model.device
            )
        )

        configs = self.provider.model_provider.get_end_model_configs()
        if self.editing_model_idx is None:
            configs.append(model)
        else:
            configs[self.editing_model_idx] = model

        self.provider.model_provider.save_end_model_configs(configs)

        if was_running and config_changed:
            def on_apply():
                self.provider.model_provider.reload_end_model(model.model_id)

            self.confirm.open(
                "Reload end model to apply changes?",
                on_apply=on_apply,
                on_discard=lambda: None
            )
        elif not was_running:
            def on_apply():
                self.provider.model_provider.load_end_model(model.model_id)

            self.confirm.open(
                "Load end model now?",
                on_apply=on_apply,
                on_discard=lambda: None
            )

        self.change_state('list', { })

    def _get_total_layers(self) -> str:
        try:
            metadata = ModelProvider.get_model_metadata(self.model_id)
        except Exception:
            return ""
        return f"(of {metadata.num_hidden_layers} layers)"

    def _has_model_already(self, model_id: str) -> bool:
        if model_id == "":
            return False

        for i, m in enumerate(self.provider.model_provider.get_end_model_configs()):
            if m.model_id == model_id and i != self.editing_model_idx:
                return True

        return False

    def _can_save(self) -> bool:
        return self._validate_local_layers()\
            and self._valid_model_id()\
            and self._valid_device()\
            and not self._has_model_already(self.model_id)

    def _validate_local_layers(self) -> bool:
        try:
            return int(self.local_layers) >= 0
        except ValueError:
            return False

    def _valid_model_id(self) -> bool:
        return self.model_id is not None and self.model_id != ""

    def _valid_device(self) -> bool:
        return ModelProvider.validate_device_name(self.device_name)

    def _current_model_running(self) -> bool:
        model = self.editing_model
        if model is None:
            return False
        return any(
            status.end_model and status.status == ModelStatus.Running
            for status in self.provider.model_provider.get_models_status().get(model.model_id, [])
        )

    def _is_adding_model(self) -> bool:
        return self.editing_model_idx is None
