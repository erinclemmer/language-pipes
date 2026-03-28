from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.frame.provider_calls import ProviderCall

@dataclass
class ModelToLoad:
    model_id: str
    load_ends: bool
    devices: List[Tuple[str, float]] # (device, max_memory)    

class ModelsConfigure:
    loader: ContentLoader
    confirm: Confirm
    exit_page: Callable
    is_focused: Callable

    models_to_load: List[ModelToLoad]

    def __init__(self, loader: ContentLoader, confirm: Confirm, exit_page: Callable, is_focoused: Callable):
        self.loader = loader
        self.confirm = confirm
        self.exit_page = exit_page
        self.is_focused = is_focoused

    def on_key(self, key: PressedKey, ch: str):
        pass

    def get_view(self) -> List[str]:
        return []
    
    def get_footer(self) -> str:
        return ""