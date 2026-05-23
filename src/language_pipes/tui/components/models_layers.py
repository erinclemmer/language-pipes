from enum import Enum
from typing import List, Callable, Optional

import torch

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.frame.tips import TIPS
from ansinout import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.components.hosted_models_view import format_model_line
from language_pipes.content_provider.model_provider import ModelProvider, ModelToLoad
from language_pipes.tui.util.text import make_footer_text, make_selectable_text, make_window_text

class LayerModelsState(Enum):
    List = 'list'
    Options = 'options'
    Edit = 'edit'
    ChooseModel = 'choose_model'

class ModelsLayerModels:
    provider: ContentProvider
    confirm: Confirm
    exit_page: Callable
    is_focused: Callable

    state: LayerModelsState
    model_idx: int
    edit_idx: int
    option_idx: int
    choose_model_idx: int # Select the model to use in the editor
    
    models_to_load: List[ModelToLoad]
    installed_models: List[str]
    
    edit_model_id: Optional[str] # Editor: ID of model
    edit_device_name: str # Editor: name
    edit_device_memory: str # Editor: Max memory
    editing_config_idx: Optional[int]  # Config index being edited, None if adding new

    def __init__(
        self,
        provider: ContentProvider,
        confirm: Confirm,
        exit_page: Callable,
        is_focoused: Callable,
    ):
        self.provider = provider
        self.confirm = confirm
        self.exit_page = exit_page
        self.is_focused = is_focoused
        self.state = LayerModelsState.List
        self.model_idx = 0
        self.edit_idx = 0
        self.option_idx = 0
        self.choose_model_idx = 0
        self.models_to_load = []
        self.installed_models = []
        self._reset_editor()
        
    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self.on_prev()
        if key == PressedKey.ArrowDown:
            self.on_next()
        if key == PressedKey.Enter:
            self.on_enter()
        if key == PressedKey.Escape:
            self.on_escape()
        if key == PressedKey.Alpha:
            self.on_char(ch)
        if key == PressedKey.Backspace:
            self.on_backspace()
        if key == PressedKey.Delete:
            self.on_delete()

    def on_delete(self):
        if self.model_idx == len(self.models_to_load):
            return

        # Convert display index to config index
        model_to_delete = self.models_to_load[self.model_idx]

        def on_apply():
            if self._current_model_running():
                self.provider.model_provider.unload_layer_models(model_to_delete.model_id, model_to_delete.device)
            
            models_to_load = []
            for m in self.models_to_load:
                if m.model_id == model_to_delete.model_id and str(m.device) == str(model_to_delete.device):
                    continue
                models_to_load.append(m)

            self.models_to_load = models_to_load
            self.provider.model_provider.save_layer_models(self.models_to_load)

        running_text = ""
        if self._current_model_running():
            running_text = "\nThis will unload the model from memory"

        self.confirm.open(
            f"Remove {model_to_delete.model_id} on {model_to_delete.device}?{running_text}", on_apply=on_apply, on_discard=lambda: None
        )

    def on_backspace(self):
        if self.state != LayerModelsState.Edit:
            return
        if self.edit_idx == 1:
            self.edit_device_name = self.edit_device_name[:-1]
        if self.edit_idx == 2:
            self.edit_device_memory = self.edit_device_memory[:-1]

    def on_char(self, ch: str):
        if self.state != LayerModelsState.Edit:
            return
        if self.edit_idx == 1:
            self.edit_device_name += ch
        if self.edit_idx == 2:
            self.edit_device_memory += ch

    def on_escape(self):
        if self.state == LayerModelsState.ChooseModel:
            self.state = LayerModelsState.Edit
        elif self.state == LayerModelsState.Edit:
            self.state = LayerModelsState.List
            self._reset_editor()
        elif self.state == LayerModelsState.Options:
            self.state = LayerModelsState.List
        else:
            self.exit_page()

    def _reset_editor(self):
        self.edit_idx = 0
        self.edit_device_memory = ""
        self.edit_device_name = ""
        self.edit_load_ends = False
        self.edit_model_id = ""
        self.editing_config_idx = None
        self.num_layers_cache = { }

    def _start_edit(self):
        self._reset_editor()
        self.state = LayerModelsState.Edit
        model = self.get_editing_model()
        self.edit_model_id = model.model_id if model is not None else ""
        self.edit_device_name = str(model.device) if model is not None else ""
        self.edit_device_memory = str(model.memory) if model is not None else ""
        # Store config index for when we save
        if model is not None and self.model_idx < len(self.models_to_load):
            self.editing_config_idx = self.model_idx
        else:
            self.editing_config_idx = None

    def on_enter(self):
        if self.state == LayerModelsState.List:
            if self.model_idx == len(self.models_to_load) or not self._network_running():
                self._start_edit()
            else:
                self.state = LayerModelsState.Options
        elif self.state == LayerModelsState.ChooseModel:
            self.edit_model_id = self.installed_models[self.choose_model_idx]
            self.state = LayerModelsState.Edit
            self.choose_model_idx = 0
        elif self.state == LayerModelsState.Edit:
            if self.edit_idx == 0:
                self.state = LayerModelsState.ChooseModel
            elif self.edit_idx == 3:
                self.add_model()
        elif self.state == LayerModelsState.Options:
            if self.option_idx == 0:
                self._start_edit()
            elif self.option_idx == 1:
                model = self.get_editing_model()
                if model is not None:
                    if self._current_model_running():
                        self.provider.model_provider.unload_layer_models(model.model_id, model.device)
                    else:
                        self.provider.model_provider.load_layer_model(model)
                self.state = LayerModelsState.List
            elif self.option_idx == 2:
                self.state = LayerModelsState.List

    def _network_running(self) -> bool:
        network_status = self.provider.network_provider.get_network_status()
        return network_status is not None and network_status.running


    def _get_ram_usage(self) -> str:
        used_ram = self.provider.get_used_system_ram()
        total_ram = self.provider.get_total_system_ram()
        
        return f"System RAM: {used_ram:.1f}/{total_ram:.1f}GB"

    def has_model_already(self, model_id: Optional[str], device: str) -> bool:
        if model_id is None:
            return False
        
        for m in self.models_to_load:
            if m.model_id == model_id and str(m.device) == device:
                return True
        return False

    def can_save(self):
        return self.validate_memory()\
            and self.valid_device()\
            and self.valid_model_id()\
            and (not self.is_adding_model() or not self.has_model_already(self.edit_model_id, self.edit_device_name))

    def add_model(self):
        if not self.can_save() or self.edit_model_id is None:
            return

        previous_model = self.get_editing_model()
        
        model = ModelToLoad(
            model_id=self.edit_model_id,
            device=torch.device(self.edit_device_name),
            memory=float(self.edit_device_memory),
        )
        should_restart = (
            previous_model is not None
            and self._model_running(previous_model)
            and self._model_config_changed(previous_model, model)
        )
        if self.editing_config_idx is None:
            # Adding new model
            self.models_to_load.append(model)
        else:
            # Replacing existing model
            self.models_to_load[self.editing_config_idx] = model

        self.provider.model_provider.save_layer_models(self.models_to_load)

        if should_restart and previous_model is not None:
            self.provider.model_provider.restart_layer_model(previous_model, model)
            self.state = LayerModelsState.List
            self._reset_editor()
            return

        if self._network_running():

            def on_apply():
                self.provider.model_provider.load_layer_model(model)

            def on_discard():
                pass

            self.confirm.open(
                "Load layer model now?", on_apply=on_apply, on_discard=on_discard
            )
        self.state = LayerModelsState.List
        self._reset_editor()

    def max_edit_idx(self):
        max_idx = 0
        if self.valid_model_id():
            max_idx += 1
        if self.valid_device():
            max_idx += 1
        if self.can_save():
            max_idx += 1
        return max_idx
    
    def on_next(self):
        if self.state == LayerModelsState.Edit:
            self.edit_idx = (self.edit_idx + 1) % (self.max_edit_idx() + 1)
        elif self.state == LayerModelsState.List:
            self.model_idx = (self.model_idx + 1) % (len(self.models_to_load) + 1)
        elif self.state == LayerModelsState.Options:
            self.option_idx = (self.option_idx + 1) % 3
        elif self.state == LayerModelsState.ChooseModel:
            self.choose_model_idx = (self.choose_model_idx + 1) % len(self.installed_models)

    def on_prev(self):
        if self.state == LayerModelsState.Edit:
            self.edit_idx = (self.edit_idx - 1) % (self.max_edit_idx() + 1)
        elif self.state == LayerModelsState.List:
            self.model_idx = (self.model_idx - 1) % (len(self.models_to_load) + 1)
        elif self.state == LayerModelsState.Options:
            self.option_idx = (self.option_idx - 1) % 3
        elif self.state == LayerModelsState.ChooseModel:
            self.choose_model_idx = (self.choose_model_idx - 1) % len(self.installed_models)

    def get_view(self) -> List[str]:
        if self.state == LayerModelsState.ChooseModel:
            return self.get_choosing_model_view()
        elif self.state == LayerModelsState.Edit:
            return self.get_editor_view()
        elif self.state == LayerModelsState.Options:
            return self.get_options_view()
        else:
            return self.get_list_view()
        
    def _current_model_running(self) -> bool:
        model = self.get_editing_model()
        if model is None:
            return False
        return self._model_running(model)

    def _model_running(self, model: ModelToLoad) -> bool:
        return any(
            str(status.device) == str(model.device)
            for status in self.provider.model_provider.get_models_status().get(model.model_id, [])
        )

    @staticmethod
    def _model_config_changed(old_model: ModelToLoad, new_model: ModelToLoad) -> bool:
        return (
            old_model.model_id != new_model.model_id
            or str(old_model.device) != str(new_model.device)
            or old_model.memory != new_model.memory
        )
        
    def get_options_view(self) -> List[str]:
        model = self.get_editing_model()
        if model is None:
            return ["ERROR"]
        lines = [f"Options for {model.model_id} on {model.device}"]
        options = [
            "Edit Model", 
            "Unload Model" if self._current_model_running() else "Load Model", 
            "Back"
        ]
        for i, opt in enumerate(options):
            l_cursor = "|>" if i == self.option_idx else "  "
            r_cursor = "<|" if i == self.option_idx else "  "
            lines.append(f"{l_cursor} {opt} {r_cursor}")

        lines.append("")
        if self.option_idx == 0:
            lines.append("Help: Edit model configuration")

        if self.option_idx == 1 and not self._current_model_running():
            lines.append(f"Help: Load model into {model.device} memory using {model.memory}GB")
        
        if self.option_idx == 1 and self._current_model_running():
            lines.append(f"Help: Unload the model from {model.device} memory")

        return lines

    def validate_memory(self):
        try:
            mem = float(self.edit_device_memory)
            if mem < 0:
                return False
            return True
        except ValueError:
            return False
        
    def _network_not_started_warning(self):
        return ["[WARNING] Network is not started.\nModels cannot be loaded until the network is started.", ""]

    def get_choosing_model_view(self) -> List[str]:
        lines = ["Choose Model to Host", ""]

        network_status = self.provider.network_provider.get_network_status()
        if network_status is None or not network_status.running:
            lines.extend(self._network_not_started_warning())


        entries = []
        self.installed_models = self.provider.model_provider.get_installed_models()
        for i, model in enumerate(self.installed_models):
            entries.append([make_selectable_text(model, self.choose_model_idx == i), ""])

        lines.extend(make_window_text(entries, self.choose_model_idx, 14))

        return lines

    def get_editing_model(self) -> Optional[ModelToLoad]:
        if self.model_idx == len(self.models_to_load):
            return None
        if self.model_idx < len(self.models_to_load):
            return self.models_to_load[self.model_idx]
        return None

    def valid_model_id(self) -> bool:
        return self.edit_model_id is not None and self.edit_model_id != ""

    def valid_device(self) -> bool:
        return ModelProvider.validate_device_name(self.edit_device_name)

    def is_adding_model(self) -> bool:
        return self.model_idx == len(self.models_to_load)

    def get_num_layers(self) -> str:
        if self.edit_device_name not in self.num_layers_cache:
            self.num_layers_cache[self.edit_device_name] = { }
        if str(self.edit_device_memory) in self.num_layers_cache[self.edit_device_name]:
            return self.num_layers_cache[self.edit_device_name][str(self.edit_device_memory)]
        
        assert self.edit_model_id is not None
        metadata = ModelProvider.get_model_metadata(self.edit_model_id)
        layer_size = metadata.avg_layer_size / 10**9
        num_layers = min(int(float(self.edit_device_memory) / layer_size), metadata.num_hidden_layers)

        s = f"   Info: Loads {num_layers} / {metadata.num_hidden_layers} layers"
        self.num_layers_cache[self.edit_device_name][str(self.edit_device_memory)] = s
        return s

    def get_editor_view(self) -> List[str]:
        editing_model = self.get_editing_model()
        header = "Choosing Model" if editing_model is not None else "Creating Layer Model Configuration"
        lines = [header, ""]

        if not self._network_running():
            lines.extend(self._network_not_started_warning())

        model_id_label = (
            self.edit_model_id if self.valid_model_id() else "Choose model..."
        )

        lines.append(make_selectable_text(f"Model ID: {model_id_label}", self.edit_idx == 0))    
        if not self.valid_model_id():
            lines.extend(["   ! Warning: Must choose model to load", ""])

        if self.valid_model_id():
            name_cursor = "|" if self.edit_idx == 1 else " "
            lines.append(f"   Device: {self.edit_device_name}{name_cursor}")
            if not self.valid_device():
                lines.append("   !Warning: Invalid device name")

        if self.valid_model_id() and self.valid_device():
            memory_cursor = "|" if self.edit_idx == 2 else ""
            lines.append(f"   Max Memory: {self.edit_device_memory}{memory_cursor} GB")
            if not self.validate_memory():
                lines.append("   !Warning: Invalid memory amount")
            elif self.edit_model_id is not None:
                lines.append(self.get_num_layers())

        if self.is_adding_model() and self.has_model_already(self.edit_model_id, self.edit_device_name):
            lines.append("   !Warning: Model ID / Device Combination already in configuration")

        if self.can_save():
            lines.append("")
            lines.append(make_selectable_text("Save Model", self.edit_idx == 3))

        tip_key = None
        if self.edit_idx == 0:
            tip_key = "model_id"
        elif self.edit_idx == 1:
            tip_key = "device"
        elif self.edit_idx == 2:
            tip_key = "max_memory"

        if tip_key is not None:            
            lines.extend(["", "Tip:", TIPS["layer_models"][tip_key]])

        return lines

    def get_list_view(self) -> List[str]:
        lines = ["Layer Models", "", self._get_ram_usage(), ""]

        if not self._network_running():
            lines.extend(self._network_not_started_warning())

        self.models_to_load = self.provider.model_provider.get_layer_models()
        models_status = self.provider.model_provider.get_models_status()

        entries: List[List[str]] = []
        for i, model in enumerate(self.models_to_load):
            entry = list(format_model_line(
                model=model,
                selected=self.model_idx == i and self.is_focused(),
                running=models_status.get(model.model_id, [])
            ))
            entry.append("")
            entries.append(entry)
        entries.append([make_selectable_text(
            "Add Layer Model", self.model_idx == len(self.models_to_load)
        )])

        lines.extend(make_window_text(entries, self.model_idx, 10))

        lines.extend([
            "", 
            "Tip: Layer models are segments of a model's transformer layers loaded",
            "into memory on a device. Multiple nodes can each host different layer",
            "ranges to distribute inference across machines."
        ])

        return lines

    def get_footer(self) -> str:
        if self.state == LayerModelsState.List:
            return make_footer_text(["Arrows U/D: Move", "Enter: Select Option", "Delete: Remove", "Esc: Menu"])
        elif self.state == LayerModelsState.Edit:
            if self.edit_idx == 0:
                return make_footer_text(["Arrows U/D: Move", "Enter: Change Model", "Esc: Back"])
            elif self.edit_idx == 1 or self.edit_idx == 2:
                return make_footer_text(["Arrows U/D: Move", "[A-Z]: Type", "Esc: Back"])
            elif self.edit_idx == 3:
                return make_footer_text(["Arrows U/D: Move", "Enter: Save Layer Model", "Esc: Back"])
        elif self.state == LayerModelsState.ChooseModel:
            return make_footer_text(["Arrows U/D: Move", "Enter: Select Model", "Esc: Back"])
        elif self.state == LayerModelsState.Options:
            return make_footer_text(["Arrows U/D: Move", "Enter: Select Option", "Esc: Back"])

        return ""
