from enum import Enum
from typing import List, Callable, Optional

import torch

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.frame.tips import TIPS
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.components.hosted_models_view import format_model_line
from language_pipes.content_provider.model_provider import ModelProvider, ModelToLoad
from language_pipes.tui.util.text import make_footer_text

class LayerModelsState(Enum):
    List = 'list'
    Options = 'options'
    Edit = 'edit'
    ChooseModel = 'choose_model'

# TODO: Whenever we change a configuration and the model is running, restart the model
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
        config_idx = self.models_to_load[self.model_idx]

        def on_apply():
            self.models_to_load = [
                m for i, m in enumerate(self.models_to_load) if i != config_idx
            ]
            self.provider.model_provider.save_layer_models(self.models_to_load)

        self.confirm.open(
            "Remove this model?", on_apply=on_apply, on_discard=lambda: None
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
        self.editing_config_idx = 0
        self.edit_device_memory = ""
        self.edit_device_name = ""
        self.edit_load_ends = False
        self.edit_model_id = ""
        self.editing_config_idx = None

    def _start_edit(self):
        self._reset_editor()
        self.state = LayerModelsState.Edit
        model = self.get_editing_model()
        self.edit_model_id = model.model_id if model is not None else ""
        self.edit_device_name = str(model.device) if model is not None else ""
        self.edit_device_memory = str(model.memory) if model is not None else ""
        # Store config index for when we save
        if model is not None and self.model_idx < len(self.models_to_load):
            self.editing_config_idx = 0
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
                        self.provider.model_provider.unload_layer_models(model.model_id)
                    else:
                        self.provider.model_provider.load_layer_model(model)
                self.state = LayerModelsState.List
            elif self.option_idx == 2:
                self.state = LayerModelsState.List

    def _network_running(self) -> bool:
        network_status = self.provider.network_provider.get_network_status()
        return network_status is not None and network_status.running

    def can_save(self):
        return self.validate_memory()\
            and self.valid_device()\
            and self.valid_model_id()

    def add_model(self):
        if not self.can_save() or self.edit_model_id is None:
            return
        
        model = ModelToLoad(
            model_id=self.edit_model_id,
            device=torch.device(self.edit_device_name),
            memory=float(self.edit_device_memory),
        )
        if self.editing_config_idx is None:
            # Adding new model
            self.models_to_load.append(model)
        else:
            # Replacing existing model
            self.models_to_load[self.editing_config_idx] = model

        self.provider.model_provider.save_layer_models(self.models_to_load)

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
            self.edit_idx += 1
            if self.edit_idx > self.max_edit_idx():
                self.edit_idx = 0
        elif self.state == LayerModelsState.List:
            self.model_idx += 1
            # +1 for the "Host New Model" button
            if self.model_idx > len(self.models_to_load):
                self.model_idx = 0
        elif self.state == LayerModelsState.Options:
            self.option_idx += 1
            if self.option_idx > 2:
                self.option_idx = 0

    def on_prev(self):
        if self.state == LayerModelsState.Edit:
            self.edit_idx -= 1
            if self.edit_idx < 0:
                self.edit_idx = self.max_edit_idx()
        elif self.state == LayerModelsState.List:
            self.model_idx -= 1
            if self.model_idx < 0:
                # +1 for the "Host New Model" button
                self.model_idx = len(self.models_to_load)
        elif self.state == LayerModelsState.Options:
            self.option_idx -= 1
            if self.option_idx < 0:
                self.option_idx = 2

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
        return len(self.provider.model_provider.get_models_status().get(model.model_id, [])) > 0
        
    def get_options_view(self) -> List[str]:
        model = self.get_editing_model()
        if model is None:
            return ["ERROR"]
        lines = [f"Options for {model.model_id}"]
        options = [
            "Edit Model", 
            "Unload Model" if self._current_model_running() else "Load Model", 
            "Back"
        ]
        for i, opt in enumerate(options):
            l_cursor = "|>" if i == self.option_idx else "  "
            r_cursor = "<|" if i == self.option_idx else "  "
            lines.append(f"{l_cursor} {opt} {r_cursor}")
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

        self.installed_models = self.provider.model_provider.get_installed_models()
        for i, model in enumerate(self.installed_models):
            l_cursor = "|>" if self.choose_model_idx == i else "  "
            r_cursor = "<|" if self.choose_model_idx == i else "  "
            lines.append(f"{l_cursor} {model} {r_cursor}")

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

    def get_editor_view(self) -> List[str]:
        editing_model = self.get_editing_model()
        header = "Choosing Model" if editing_model is not None else "Creating Layer Model Configuration"
        lines = [header, ""]

        if not self._network_running():
            lines.extend(self._network_not_started_warning())

        model_id_label = (
            self.edit_model_id if self.valid_model_id() else "Choose model..."
        )

        l_cursor = "|>" if self.edit_idx == 0 else "  "
        r_cursor = "<|" if self.edit_idx == 0 else "  "    
        lines.append(f"{l_cursor} Model ID: {model_id_label} {r_cursor}")
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

        if self.can_save():
            l_cursor = "|>" if self.edit_idx == 3 else "  "
            r_cursor = "<|" if self.edit_idx == 3 else "  "
            lines.append("")
            lines.append(f"{l_cursor} Save Model {r_cursor}")

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
        lines = ["Layer Models", ""]

        if not self._network_running():
            lines.extend(self._network_not_started_warning())

        self.models_to_load = self.provider.model_provider.get_layer_models()
        models_status = self.provider.model_provider.get_models_status()

        for i, model in enumerate(self.models_to_load):
            lines.append(format_model_line(
                model=model,
                selected=self.model_idx == i and self.is_focused(),
                running=models_status.get(model.model_id, [])
            ))
            
        # Clamp model_idx to valid range (+1 for "Host New Model" button)
        max_idx = len(self.models_to_load)
        if self.model_idx > max_idx:
            self.model_idx = max_idx

        l_cursor = (
            "|>"
            if self.model_idx == len(self.models_to_load) and self.is_focused()
            else "  "
        )
        r_cursor = (
            "<|"
            if self.model_idx == len(self.models_to_load) and self.is_focused()
            else "  "
        )
        lines.append("")
        lines.append(f" {l_cursor} Add Layer Model {r_cursor}")

        lines.extend(["", "Tip: Layer models are segments of a model's transformer layers loaded\ninto memory on a device. Multiple nodes can each host different layer\nranges to distribute inference across machines."])

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

        return ""