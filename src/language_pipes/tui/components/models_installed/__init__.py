from typing import Callable

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.components.models_installed.api_key_state import ApiKeyPageState
from language_pipes.tui.components.models_installed.download_state import DownloadPageState
from language_pipes.tui.components.models_installed.list_state import ListPageState
from language_pipes.tui.components.page import Page


class ModelsInstalled(Page):
    def __init__(self, provider: ContentProvider, confirm: Confirm, exit_page: Callable):
        super().__init__(
            [
                ListPageState(),
                DownloadPageState(),
                ApiKeyPageState(),
            ],
            provider,
            confirm,
            exit_page,
        )
