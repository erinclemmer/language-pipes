from typing import Callable

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.components.models_layers.list_state import ListPageState
from language_pipes.tui.components.page import Page


class ModelsLayers(Page):
    def __init__(
        self,
        provider: ContentProvider, 
        confirm: Confirm,
        exit_page: Callable
    ):
        super().__init__(
            [
                ListPageState()
            ],
            provider,
            confirm,
            exit_page
        )

