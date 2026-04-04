from typing import List, Callable, Optional, Dict

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.components.hosted_models_view import format_model_line
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.tui.content_provider.model_provider import ModelToLoad, ModelStatusInfo




class ModelsHosted:
    loader: ContentLoader
    confirm: Confirm
    exit_page: Callable
    is_focused: Callable

    model_idx: int
    edit_idx: int
    choose_model_idx: int
    models_to_load: List[ModelToLoad]
    installed_models: List[str]
    editing_model: bool
    choosing_model: bool

    edit_model_id: Optional[str]
    edit_load_ends: bool
    edit_device_name: str
    edit_device_memory: str
    editing_config_idx: Optional[int]  # Config index being edited, None if adding new

    def __init__(
        self,
        loader: ContentLoader,
        confirm: Confirm,
        exit_page: Callable,
        is_focoused: Callable,
    ):
        self.loader = loader
        self.confirm = confirm
        self.exit_page = exit_page
        self.is_focused = is_focoused
        self.editing_model = False
        self.model_idx = 0
        self.edit_idx = 0
        self.choose_model_idx = 0
        self.models_to_load = []
        self.installed_models = []
        self.edit_model_id = None
        self.edit_load_ends = False
        self.choosing_model = False
        self.edit_device_name = ""
        self.edit_device_memory = ""
        self.editing_config_idx = None

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
            self.loader.call_provider(
                ProviderCall.save_models_to_load, self.models_to_load
            )

        self.confirm.open(
            "Remove this model?", on_apply=on_apply, on_discard=lambda: None
        )

    def on_backspace(self):
        if not self.editing_model:
            return
        if self.edit_idx == 2:
            self.edit_device_name = self.edit_device_name[:-1]
        if self.edit_idx == 3:
            self.edit_device_memory = self.edit_device_memory[:-1]

    def on_char(self, ch: str):
        if not self.editing_model:
            return
        if self.edit_idx == 2:
            self.edit_device_name += ch
        if self.edit_idx == 3:
            self.edit_device_memory += ch

    def on_escape(self):
        if self.choosing_model:
            self.choosing_model = False
        elif self.editing_model:
            self.editing_model = False
            self.editing_config_idx = None
        else:
            self.exit_page()

    def on_enter(self):
        if not self.editing_model:
            model = self.get_editing_model()
            self.edit_model_id = model.model_id if model is not None else ""
            self.edit_load_ends = model.load_ends if model is not None else False
            self.edit_device_name = model.device if model is not None else ""
            self.edit_device_memory = str(model.max_memory) if model is not None else ""
            self.editing_model = True
            # Store config index for when we save
            if model is not None and self.model_idx < len(self.models_to_load):
                self.editing_config_idx = 0
            else:
                self.editing_config_idx = None
        elif self.choosing_model:
            self.edit_model_id = self.installed_models[self.choose_model_idx]
            self.choosing_model = False
            self.choose_model_idx = 0
        elif self.editing_model:
            if self.edit_idx == 0:
                self.choosing_model = True
            elif self.edit_idx == 1:
                self.edit_load_ends = not self.edit_load_ends
            elif self.edit_idx == 4:
                self.add_model()

    def add_model(self):
        valid_device_name = self.loader.call_provider(
            ProviderCall.validate_device_name, self.edit_device_name
        )
        if (
            not self.validate_memory()
            or not valid_device_name
            or self.edit_model_id is None
        ):
            return
        model = ModelToLoad(
            model_id=self.edit_model_id,
            load_ends=self.edit_load_ends,
            device=self.edit_device_name,
            max_memory=float(self.edit_device_memory),
        )
        if self.editing_config_idx is None:
            # Adding new model
            self.models_to_load.append(model)
        else:
            # Replacing existing model
            self.models_to_load[self.editing_config_idx] = model

        self.loader.call_provider(ProviderCall.save_models_to_load, self.models_to_load)

        network_status = self.loader.call_provider(ProviderCall.get_network_status)
        if network_status is not None and network_status.running:

            def on_apply():
                self.loader.call_provider(ProviderCall.host_model, model)

            def on_discard():
                pass

            self.confirm.open(
                "Host model now?", on_apply=on_apply, on_discard=on_discard
            )
        self.edit_device_memory = ""
        self.edit_device_name = ""
        self.edit_load_ends = False
        self.edit_model_id = ""
        self.editing_model = False
        self.editing_config_idx = None

    def on_next(self):
        if self.editing_model:
            self.edit_idx += 1
            if self.edit_idx > 4:
                self.edit_idx = 0
        else:
            self.model_idx += 1
            # +1 for the "Host New Model" button
            if self.model_idx > len(self.models_to_load):
                self.model_idx = 0

    def on_prev(self):
        if self.editing_model:
            self.edit_idx -= 1
            if self.edit_idx < 0:
                self.edit_idx = 3
        else:
            self.model_idx -= 1
            if self.model_idx < 0:
                # +1 for the "Host New Model" button
                self.model_idx = len(self.models_to_load)

    def get_view(self) -> List[str]:
        if self.choosing_model:
            return self.get_choosing_model_view()
        elif self.editing_model:
            return self.get_editor_view()
        else:
            return self.get_list_view()

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

        network_status = self.loader.call_provider(ProviderCall.get_network_status)
        if network_status is None or not network_status.running:
            lines.extend(self._network_not_started_warning())

        self.installed_models = self.loader.call_provider(
            ProviderCall.get_installed_models
        )
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

    def get_editor_view(self) -> List[str]:
        editing_model = self.get_editing_model()
        header = "Editing Model" if editing_model is not None else "Adding new Model"
        lines = [header, ""]

        network_status = self.loader.call_provider(ProviderCall.get_network_status)
        if network_status is None or not network_status.running:
            lines.extend(self._network_not_started_warning())

        model_id_label = (
            self.edit_model_id if self.edit_model_id is not None else "Choose model..."
        )
        model_id_line = f"Model ID: {model_id_label}"

        load_ends_label = "Yes" if self.edit_load_ends else "No"
        load_ends_line = f"Load Ends: {load_ends_label}"

        for i, line in enumerate([model_id_line, load_ends_line]):
            l_cursor = "|>" if self.edit_idx == i else "  "
            r_cursor = "<|" if self.edit_idx == i else "  "
            lines.append(f"{l_cursor} {line} {r_cursor}")

        name_cursor = "|" if self.edit_idx == 2 else " "
        lines.append(f"   Device: {self.edit_device_name}{name_cursor}")
        if len(self.edit_device_name) > 0 and not self.loader.call_provider(
            ProviderCall.validate_device_name, self.edit_device_name
        ):
            lines.append("[ERROR] Invalid device name")

        memory_cursor = "|" if self.edit_idx == 3 else ""
        lines.append(f"   Max Memory: {self.edit_device_memory}{memory_cursor} GB")
        if len(self.edit_device_memory) > 0 and not self.validate_memory():
            lines.append("[ERROR] Invalid memory amount")

        l_cursor = "|>" if self.edit_idx == 4 else "  "
        r_cursor = "<|" if self.edit_idx == 4 else "  "
        lines.append("")
        lines.append(f"{l_cursor} Save Model {r_cursor}")

        return lines

    def get_list_view(self) -> List[str]:
        lines = ["Hosted Models", ""]

        network_status = self.loader.call_provider(ProviderCall.get_network_status)
        if network_status is None or not network_status.running:
            lines.extend(self._network_not_started_warning())

        self.models_to_load = self.loader.call_provider(ProviderCall.get_models_to_load)
        models_status: Dict[str, List[ModelStatusInfo]] = self.loader.call_provider(ProviderCall.get_models_status)

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
        lines.append(f" {l_cursor} Host New Model {r_cursor}")

        return lines

    def get_footer(self) -> str:
        return ""
