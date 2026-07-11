from typing import Dict, List, Optional

from ansinout import PressedKey
import torch

from language_pipes.config import ModelToLoad
from language_pipes.content_provider.model_provider import ModelProvider, ModelStatus
from language_pipes.tui.components.page import PageState
from language_pipes.tui.frame.tips import TIPS
from language_pipes.tui.util.text import make_footer_text, make_selectable_text
from language_pipes.util.config import is_8_bit_mode


class EditPageState(PageState):
    editing_model: Optional[ModelToLoad]
    editing_model_idx: Optional[int]

    model_id: str
    device_name: str
    device_memory: str
    num_layers_cache: Dict[str, Dict[str, str]]

    select_idx: int

    def __init__(self):
        super().__init__('edit')
        self.editing_model = None
        self.editing_model_idx = None
        self.model_id = ""
        self.device_name = "cpu"
        self.device_memory = ""
        self.select_idx = 0
        self.num_layers_cache = { }

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
        self.device_name = str(self.editing_model.device) if self.editing_model is not None else "cpu"
        self.device_memory = str(self.editing_model.memory) if self.editing_model is not None else ""
        self.select_idx = 0
        self.num_layers_cache = { }

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self._on_prev()
        if key == PressedKey.ArrowDown:
            self._on_next()
        if key == PressedKey.Enter:
            self._on_enter()
        if key == PressedKey.Escape:
            self._on_escape()
        if key == PressedKey.Alpha:
            self._on_char(ch)
        if key == PressedKey.Backspace:
            self._on_backspace()

    def _on_backspace(self):
        if self.select_idx == 2:
            self.device_memory = self.device_memory[:-1]

    def _on_char(self, ch: str):
        if self.select_idx == 2:
            self.device_memory += ch

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
        lines = []
        device_label = (
            self.device_name if self._valid_device() else "Choose device..."
        )
        lines.append(make_selectable_text(f"Device: {device_label}", self.select_idx == 1))
        if not self._valid_device():
            lines.append("   !Warning: Must choose device")
        
        return lines

    def _memory_lines(self) -> List[str]:
        lines = []
        memory_cursor = "|" if self.select_idx == 2 else ""
        max_memory_line = f"   Max Memory: {self.device_memory}{memory_cursor} GB "
        if not self._validate_memory():
            max_memory_line += "(Warning: Invalid memory amount)"
        elif len(self.device_memory) > 0 and self.model_id is not None:
            max_memory_line += self._get_num_layers()
        
        lines.append(max_memory_line)
        
        return lines

    def _get_tip_lines(self) -> List[str]:
        tip_key = None
        if self.select_idx == 0:
            tip_key = "model_id"
        elif self.select_idx == 1:
            tip_key = "device"
        elif self.select_idx == 2:
            tip_key = "max_memory"

        if tip_key is not None:            
            return ["", "Tip:", TIPS["layer_models"][tip_key]]
        return []

    def get_view(self) -> List[str]:
        header = "Choosing Model" if self.editing_model is not None else "Creating Layer Model Configuration"
        lines = [header, ""]

        if not self._network_running():
            lines.extend(["[WARNING] Network is not started.\nModels cannot be loaded until the network is started.", ""])

        lines.extend(self._model_id_lines())

        if self._valid_model_id():
            lines.extend(self._device_lines())

        if self._valid_model_id() and self._valid_device():
            lines.extend(self._memory_lines())

        if self._is_adding_model() and self._has_model_already(self.model_id, self.device_name):
            lines.append("   !Warning: Model ID / Device Combination already in configuration")

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
            return make_footer_text(["Arrows U/D: Move", "[A-Z]: Type", "Esc: Back"])
        elif self.select_idx == 3:
            return make_footer_text(["Arrows U/D: Move", "Enter: Save Layer Model", "Esc: Back"])
        return ""

    def _max_select_idx(self) -> int:
        max_idx = 0
        if self._valid_model_id():
            max_idx += 1
        if self._valid_device():
            max_idx += 1
        if self._can_save():
            max_idx += 1
        return max_idx
    
    def _add_model(self) -> None:
        if not self._can_save():
            return
        
        model = ModelToLoad(
            model_id=self.model_id,
            device=torch.device(self.device_name),
            memory=float(self.device_memory),
        )

        should_restart = (
            self.editing_model is not None
            and self._model_running(self.editing_model)
            and self._model_config_changed(self.editing_model, model)
        )

        models = self.provider.model_provider.get_layer_models()
        if self.editing_model_idx is None:
            # Adding new model
            models.append(model)
        else:
            # Replacing existing model
            models[self.editing_model_idx] = model
        
        self.provider.model_provider.save_layer_models(models)

        if should_restart and self.editing_model is not None:
            self.provider.model_provider.restart_layer_model(self.editing_model, model)
            self.change_state('list', { })
            return

        if self._network_running():
            def on_apply():
                self.provider.model_provider.load_layer_model(model)

            def on_discard():
                pass

            self.confirm.open(
                "Load layer model now?", 
                on_apply=on_apply, 
                on_discard=on_discard
            )

        self.change_state('list', { })

    def _get_num_layers(self) -> str:
        if self.device_name not in self.num_layers_cache:
            self.num_layers_cache[self.device_name] = { }
        if str(self.device_memory) in self.num_layers_cache[self.device_name]:
            return self.num_layers_cache[self.device_name][str(self.device_memory)]
        
        assert self.model_id is not None
        metadata = ModelProvider.get_model_metadata(self.model_id)
        layer_size = metadata.avg_layer_size / 1024**3
        if is_8_bit_mode():
            layer_size /= 2

        num_layers = min(int(float(self.device_memory) / layer_size), metadata.num_hidden_layers)
        total_size = layer_size * metadata.num_hidden_layers
        s = f"/ {total_size:.1f}GB ({num_layers} of {metadata.num_hidden_layers} layers)"
        self.num_layers_cache[self.device_name][str(self.device_memory)] = s
        return s
    
    def _has_model_already(self, model_id: Optional[str], device: str) -> bool:
        if model_id is None:
            return False
        
        for m in self.provider.model_provider.get_layer_models():
            if m.model_id == model_id and str(m.device) == device:
                return True

        return False

    def _can_save(self) -> bool:
        return self._validate_memory()\
            and self._valid_device()\
            and self._valid_model_id()\
            and (not self._is_adding_model() or not self._has_model_already(self.model_id, self.device_name))

    def _validate_memory(self) -> bool:
        try:
            mem = float(self.device_memory)
            if mem < 0:
                return False
            return True
        except ValueError:
            return False
        
    def _valid_model_id(self) -> bool:
        return self.model_id is not None and self.model_id != ""

    def _valid_device(self) -> bool:
        return ModelProvider.validate_device_name(self.device_name)

    def _current_model_loading(self) -> bool:
        model = self.editing_model
        assert model is not None
        statuses = self.provider.model_provider.get_models_status()
        if model.model_id not in statuses:
            return False
        for status in statuses[model.model_id]:
            if status.status == ModelStatus.Starting or status.status == ModelStatus.Stopping:
                return True
        return False

    def _current_model_running(self) -> bool:
        model = self.editing_model
        if model is None:
            return False
        return self._model_running(model)

    def _model_running(self, model: ModelToLoad) -> bool:
        return any(
            not status.end_model and str(status.device) == str(model.device)
            for status in self.provider.model_provider.get_models_status().get(model.model_id, [])
        )
    
    def _network_running(self) -> bool:
        network_status = self.provider.network_provider.get_network_status()
        return network_status is not None and network_status.running
    
    def _is_adding_model(self) -> bool:
        return self.editing_model_idx is None
    
    @staticmethod
    def _model_config_changed(old_model: ModelToLoad, new_model: ModelToLoad) -> bool:
        return (
            old_model.model_id != new_model.model_id
            or str(old_model.device) != str(new_model.device)
            or old_model.memory != new_model.memory
        )
    