from typing import Callable

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.components.models_end.choose_device_state import ChooseDevicePageState
from language_pipes.tui.components.models_end.choose_model_state import ChooseModelPageState
from language_pipes.tui.components.models_end.edit_state import EditPageState
from language_pipes.tui.components.models_end.list_state import ListPageState
from language_pipes.tui.components.models_end.options_state import OptionsPageState
from language_pipes.tui.components.page import Page


class ModelsEndModels(Page):
    def __init__(
        self,
        provider: ContentProvider,
        confirm: Confirm,
        exit_page: Callable
    ):
        super().__init__(
            [
                ListPageState(),
                EditPageState(),
                OptionsPageState(),
                ChooseModelPageState(),
                ChooseDevicePageState(),
            ],
            provider,
            confirm,
            exit_page
        )
