from typing import List

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.frame.provider_calls import ProviderCall

class ModelsInstalled:
    loader: ContentLoader
    confirm: Confirm
    installed_models: List[str]
    focus_idx: int
    installing_model: bool

    new_model_id: str

    def __init__(self, loader: ContentLoader, confirm: Confirm):
        self.loader = loader
        self.confirm = confirm
        self.installed_models = []
        self.focus_idx = 0
        self.installing_model = False
        self.new_model_id = ""

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
            self.installing_model = False
            
    def on_backspace(self):
        if not self.installing_model:
            return
        self.new_model_id = self.new_model_id[:-1]
        
    def on_char(self, ch: str):
        if not self.installing_model:
            return
        self.new_model_id += ch

    def on_enter(self):
        if self.focus_idx != len(self.installed_models):
            return
        if not self.installing_model:
            self.installing_model = True
            return
        else:
            self.confirm.open(
                f"Download {self.new_model_id}",
                on_apply=self.download_new_model,
                on_discard=lambda:None
            )

    def download_new_model(self):
        pass

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
        self.focus_idx += 1
        if self.focus_idx > len(self.installed_models):
            self.focus_idx = 0

    def on_prev(self):
        self.focus_idx -= 1
        if self.focus_idx < 0:
            self.focus_idx = len(self.installed_models)

    def get_view(self) -> List[str]:
        if self.installing_model:
            return self.get_installing_view()
        else:
            return self.get_list_view()
    
    def get_installing_view(self) -> List[str]:
        model_id = self.new_model_id[-40:] if len(self.new_model_id) > 40 else self.new_model_id
        lines = [
            "Install New Model:", "", 
            "Type the huggingface model ID from huggingface.co", "",
            f"Model ID: {model_id}|"
        ]

        return lines

    def get_list_view(self):
        self.installed_models = self.loader.call_provider(ProviderCall.get_installed_models)
        lines = ["Installed Models:", ""]
        for i, model in enumerate(self.installed_models):
            l_cursor = " |>" if i == self.focus_idx else "   "
            r_cursor = "<|" if i == self.focus_idx else "  "
            lines.append(f"{l_cursor} {model} {r_cursor}")
        
        lines.append("")
        l_cursor = " |>" if self.focus_idx == len(self.installed_models) else "   "
        r_cursor = "<|" if self.focus_idx == len(self.installed_models) else "  "
        lines.append(f"{l_cursor} Install New Model {r_cursor}")

        return lines
    
    def get_footer(self) -> str:
        return "Arrow U/D: Move   Enter: Select   Delete: Delete Model   Esc: Back"