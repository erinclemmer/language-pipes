from typing import List, Callable, Optional

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.content_loader import ContentLoader
from language_pipes.content_provider.provider_calls import ProviderCall

class ModelsInstalled:
    loader: ContentLoader
    confirm: Confirm
    exit_page: Callable
    is_focused: Callable
    installed_models: List[str]
    focus_idx: int
    installing_model: bool
    downloading_model: bool
    entering_token: bool
    
    new_model_id: str
    token_string: str
    download_status: Optional[str]

    def __init__(self, loader: ContentLoader, confirm: Confirm, exit_page: Callable, is_focused: Callable):
        self.loader = loader
        self.confirm = confirm
        self.installed_models = []
        self.focus_idx = 0
        self.exit_page = exit_page
        self.is_focused = is_focused
        self.installing_model = False
        self.downloading_model = False
        self.entering_token = False
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
        if self.installing_model:
            if self.downloading_model:
                self.confirm.open(
                    "Stop Current Download?",
                    on_apply=self.stop_download,
                    on_discard=lambda:None
                )
            else:
                self.installing_model = False
        else:
            self.exit_page()
            
    def on_backspace(self):
        if self.downloading_model:
            return
        
        if self.entering_token:
            self.token_string = self.token_string[:-1]
        elif self.installing_model:
            self.new_model_id = self.new_model_id[:-1]
        
    def on_char(self, ch: str):
        if self.downloading_model:
            return
        
        if self.entering_token:
            self.token_string += ch
        elif self.installing_model:
            self.new_model_id += ch

    def on_enter(self):
        if self.focus_idx != len(self.installed_models):
            return
        
        if self.entering_token:
            def apply_token():
                self.loader.call_provider(ProviderCall.save_hf_token, self.token_string)
                self.entering_token = False
                self.token_string = ""
                self.download_new_model()

            def discard_token():
                self.entering_token = False
                self.token_string = ""

            self.confirm.open(
                f"Use this token?\n{self.token_string}",
                on_apply=apply_token,
                on_discard=discard_token
            )
            return
        elif self.downloading_model:
            if self.download_status is not None and ("SUCCESS" in self.download_status or "ERROR" in self.download_status):
                if "ERROR" in self.download_status:
                    self.downloading_model = False
                if "SUCCESS" in self.download_status:
                    self.downloading_model = False
                    self.installing_model = False
            else:
                self.confirm.open(
                    "Stop Current Download?",
                    on_apply=self.stop_download,
                    on_discard=lambda:None
                )
            return
        
        if self.installing_model:
            self.confirm.open(
                f"Download {self.new_model_id}",
                on_apply=self.download_new_model,
                on_discard=lambda:None
            )
        else:
            self.installing_model = True
            self.new_model_id = ""

    def stop_download(self):
        self.loader.call_provider(ProviderCall.stop_model_download)
        self.downloading_model = False

    def download_new_model(self, check_token: bool = True):
        token: Optional[str] = self.loader.call_provider(ProviderCall.get_hf_token)
        if token is None and check_token:
            def use_key():
                self.entering_token = True
            def dont_use_key():
                self.download_new_model(False)
            self.confirm.open(
                "Use Huggingface API Key?",
                on_apply=use_key,
                on_discard=dont_use_key
            )
        else:
            self.loader.call_provider(ProviderCall.start_download, self.new_model_id)
            self.downloading_model = True

    def on_delete(self):
        if self.focus_idx < 0 or self.focus_idx >= len(self.installed_models):
            return
        model_name = self.installed_models[self.focus_idx]
        self.confirm.open(
            f"Delete {model_name}?",
            on_apply=lambda: self.loader.call_provider(ProviderCall.delete_installed_model, model_name),
            on_discard=lambda:None
        )

    def on_next(self):
        if self.downloading_model or self.entering_token:
            return
        
        self.focus_idx += 1
        if self.focus_idx > len(self.installed_models):
            self.focus_idx = 0

    def on_prev(self):
        if self.downloading_model or self.entering_token:
            return
        self.focus_idx -= 1
        if self.focus_idx < 0:
            self.focus_idx = len(self.installed_models)

    def get_view(self) -> List[str]:
        if self.entering_token:
            return self.get_token_view()
        elif self.installing_model:
            return self.get_installing_view()
        else:
            return self.get_list_view()
    
    def get_token_view(self) -> List[str]:
        token_string = self.token_string[-40:] if len(self.token_string) > 40 else self.token_string
        lines = [
            "Type or paste to enter a huggingface API key", "",
            f"API Key |> {token_string}|"
        ]
        return lines

    def get_installing_view(self) -> List[str]:
        model_id = self.new_model_id[-40:] if len(self.new_model_id) > 40 else self.new_model_id
        lines = [
            "Install New Model:", "", 
            "Type the huggingface model ID from huggingface.co", ""
        ]

        if self.downloading_model:
            lines.extend([
                f"Downloading {model_id}...", ""
            ])
            self.download_status = self.loader.call_provider(ProviderCall.check_download_progress)
            if self.download_status is not None:
                lines.append(self.download_status)
                if "SUCCESS" in self.download_status or "ERROR" in self.download_status:
                    lines.extend(["", "Press Enter to continue..."])
        else:
            lines.extend([f"Model ID: {model_id}|", ""])

        return lines

    def get_list_view(self):
        self.installed_models = self.loader.call_provider(ProviderCall.get_installed_models)
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