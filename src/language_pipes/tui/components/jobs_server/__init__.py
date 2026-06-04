from typing import Callable

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.components.jobs_server.add_key_type_state import AddKeyTypePageState
from language_pipes.tui.components.jobs_server.key_gen_state import KeyGenPageState
from language_pipes.tui.components.jobs_server.keys_state import KeysPageState
from language_pipes.tui.components.jobs_server.top_state import TopPageState
from language_pipes.tui.components.jobs_server.type_key_state import TypeKeyPageState
from language_pipes.tui.components.page import Page


class JobsServer(Page):
    def __init__(
        self,
        provider: ContentProvider,
        confirm: Confirm,
        exit_page: Callable
    ):
        super().__init__(
            [
                TopPageState(),
                KeysPageState(),
                AddKeyTypePageState(),
                KeyGenPageState(),
                TypeKeyPageState(),
            ],
            provider,
            confirm,
            exit_page
        )
