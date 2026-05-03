from enum import Enum
from typing import List, Callable, Optional

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.content_provider.model_provider import ModelProvider
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm

class ModelsInstalledState(Enum):
    LIST = "LIST"
    DOWNLOAD = "DOWNLOAD"
    API_KEY = "API_KEY"

class ModelsInstalled:
    provider: ContentProvider
    confirm: Confirm
    exit_page: Callable
    is_focused: Callable
    state: ModelsInstalledState
    downloading: bool
    installed_models: List[str]
    focus_idx: int
    
    new_model_id: str
    token_string: str
    download_status: Optional[str]

    def __init__(self, provider: ContentProvider, confirm: Confirm, exit_page: Callable, is_focused: Callable):
        self.provider = provider
        self.confirm = confirm
        self.installed_models = []
        self.focus_idx = 0
        self.exit_page = exit_page
        self.is_focused = is_focused
        self.state = ModelsInstalledState.LIST
        self.downloading = False
        self.new_model_id = ""
        self.token_string = ""
        self.download_status = None

    def on_key(self, key: PressedKey, ch: str):
        if key == PressedKey.ArrowUp:
            self.on_prev()
        if key == PressedKey.ArrowDown:
            self.on_next()
        if key == PressedKey.Delete:
            self.on_delete()
        if key == PressedKey.Enter:
            self.on_enter()
        if key == PressedKey.Alpha:
            self.on_char(ch)
        if key == PressedKey.Backspace:
            self.on_backspace()
        if key == PressedKey.Escape:
            self.on_escape()

    def on_escape(self):
        if self.state == ModelsInstalledState.DOWNLOAD:
            if self.downloading:
                self.confirm.open(
                    "Stop Current Download?",
                    on_apply=self.stop_download,
                    on_discard=lambda:None
                )
            else:
                self.state = ModelsInstalledState.LIST
        elif self.state == ModelsInstalledState.API_KEY:
            self.state = ModelsInstalledState.DOWNLOAD
        else:
            self.exit_page()
            
    def on_backspace(self):
        if self.downloading:
            return
        
        if self.state == ModelsInstalledState.API_KEY:
            self.token_string = self.token_string[:-1]
        elif self.state == ModelsInstalledState.DOWNLOAD:
            self.new_model_id = self.new_model_id[:-1]
        
    def on_char(self, ch: str):
        if self.downloading:
            return
        
        if self.state == ModelsInstalledState.API_KEY:
            self.token_string += ch
        elif self.state == ModelsInstalledState.DOWNLOAD:
            self.new_model_id += ch

    def on_enter(self):
        if self.focus_idx != len(self.installed_models):
            return
        
        if self.state == ModelsInstalledState.API_KEY:
            def apply_token():
                ModelProvider.save_hf_token(self.token_string)
                self.state = ModelsInstalledState.DOWNLOAD
                self.start_download(self.token_string)

            def discard_token():
                self.download_new_model()
                self.state = ModelsInstalledState.DOWNLOAD
                self.token_string = ""

            self.confirm.open(
                f"Save this token?\n{self.token_string}",
                on_apply=apply_token,
                on_discard=discard_token
            )
            return
        elif self.downloading:
            if self.download_status is not None and ("SUCCESS" in self.download_status or "ERROR" in self.download_status):
                self.downloading = False
                if "SUCCESS" in self.download_status:
                    self.state = ModelsInstalledState.LIST
            else:
                self.confirm.open(
                    "Stop Current Download?",
                    on_apply=self.stop_download,
                    on_discard=lambda:None
                )
            return
        
        if self.state == ModelsInstalledState.DOWNLOAD:
            self.confirm.open(
                f"Download {self.new_model_id}",
                on_apply=self.download_new_model,
                on_discard=lambda:None
            )
        else:
            self.state = ModelsInstalledState.DOWNLOAD
            self.new_model_id = ""

    def stop_download(self):
        self.provider.model_provider.stop_model_download()
        self.downloading = False

    def download_new_model(self, token: Optional[str] = None, check_token: bool = True):
        cfg_token = ModelProvider.get_hf_config_token()
        if token is None:
            token = cfg_token
        
        if token is None and check_token:
            def enter_key():
                self.state = ModelsInstalledState.API_KEY
            def dont_use_key():
                self.download_new_model()
            self.confirm.open(
                "Enter Huggingface API Key?\nAPI Keys allow you to download gated models and better rate limits.",
                on_apply=enter_key,
                on_discard=dont_use_key
            )
        else:
            self.provider.model_provider.start_download(self.new_model_id, token)

    def start_download(self, token: Optional[str] = None):
        self.provider.model_provider.start_download(self.new_model_id, token)
        self.downloading = True

    def on_delete(self):
        if self.focus_idx < 0 or self.focus_idx >= len(self.installed_models):
            return
        model_name = self.installed_models[self.focus_idx]
        self.confirm.open(
            f"Delete {model_name}?",
            on_apply=lambda: self.provider.model_provider.delete_installed_model(model_name),
            on_discard=lambda:None
        )

    def on_next(self):
        if self.state != ModelsInstalledState.LIST:
            return
        
        self.focus_idx += 1
        if self.focus_idx > len(self.installed_models):
            self.focus_idx = 0

    def on_prev(self):
        if self.state != ModelsInstalledState.LIST:
            return
        self.focus_idx -= 1
        if self.focus_idx < 0:
            self.focus_idx = len(self.installed_models)

    def get_view(self) -> List[str]:
        if self.state == ModelsInstalledState.API_KEY:
            return self.get_token_view()
        elif self.state == ModelsInstalledState.DOWNLOAD:
            return self.get_installing_view()
        else:
            return self.get_list_view()
    
    def get_token_view(self) -> List[str]:
        token_string = self.token_string[-40:] if len(self.token_string) > 40 else self.token_string
        lines = [
            "Type or paste to enter a huggingface API key", "",
            f"API Key |> {token_string}|",
            "",
            "Create an access token at https://huggingface.co/settings/tokens"
        ]
        return lines

    def can_download(self):
        if self.new_model_id == "":
            return False
        
        try:
            self.new_model_id.index("")
        except ValueError:
            return False
        return True

    def get_installing_view(self) -> List[str]:
        model_id = self.new_model_id[-40:] if len(self.new_model_id) > 40 else self.new_model_id
        lines = [
            "Install New Model:", "", 
            "Type the huggingface model ID from huggingface.co", ""
        ]

        cfg_key = ModelProvider.get_hf_config_token()
        if cfg_key is not None:
            lines.extend(["Global API Key loaded", ""])
        
        if self.downloading:
            lines.extend([
                f"Downloading {model_id}...", ""
            ])
            self.download_status = self.provider.model_provider.check_download_progress()
            if self.download_status is not None:
                lines.append(self.download_status)
                if "SUCCESS" in self.download_status or "ERROR" in self.download_status:
                    lines.extend(["", "Press Enter to continue..."])
        else:
            lines.extend([f"Model ID: {model_id}|", ""])
            if self.new_model_id == "":
                lines.append("!Warning: Model ID empty")
            else:
                try:
                    self.new_model_id.index("/")
                except ValueError:
                    lines.append("!Warning: model name must be in [organization]/[model] format")
                
            if cfg_key is None:
                lines.extend(["", "TIP: Create an access token at https://huggingface.co/settings/tokens"])

        return lines

    def get_list_view(self):
        self.installed_models = self.provider.model_provider.get_installed_models()
        
        lines = ["Installed Models:", ""]
        for i, model in enumerate(self.installed_models):
            l_cursor = " |>" if i == self.focus_idx and self.is_focused() else "   "
            r_cursor = "<|" if i == self.focus_idx and self.is_focused() else "  "
            lines.append(f"{l_cursor} {model} {r_cursor}")
        
        lines.append("")
        l_cursor = " |>" if self.focus_idx == len(self.installed_models) and self.is_focused() else "   "
        r_cursor = "<|" if self.focus_idx == len(self.installed_models) and self.is_focused() else "  "
        lines.append(f"{l_cursor} Install New Model {r_cursor}")

        return lines
    
    def get_footer(self) -> str:
        return "Arrow U/D: Move   Enter: Select   Delete: Delete Model   Esc: Back"