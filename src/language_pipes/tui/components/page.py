from typing import Callable, Dict, List

from ansinout import PressedKey

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.confirm import Confirm

class PageState:
    name: str
    provider: ContentProvider
    confirm: Confirm
    exit_page: Callable
    change_state: Callable[[str, Dict], None]

    def __init__(self, name: str):
        self.name = name

    def on_change(self, args: Dict):
        pass

    def on_key(self, key: PressedKey, ch: str):
        pass

    def get_view(self) -> List[str]:
        return []
    
    def get_footer(self) -> str:
        return ""

class Page:
    current_state: str
    states: List[PageState]

    def __init__(self, 
        states: List[PageState], 
        provider: ContentProvider, 
        confirm: Confirm,
        exit_page: Callable
    ):
        self.states = states

        for state in self.states:
            state.provider = provider
            state.confirm = confirm
            state.exit_page = exit_page
            state.change_state = self._change_state

        self.current_state = states[0].name
    
    def _change_state(self, state_name: str, args: Dict):
        for state in self.states:
            if state.name == state_name:
                self.current_state = state_name
                state.on_change(args)
                return

        raise Exception(f"State change to {state_name} failed")

    def _current_state(self) -> PageState:
        for state in self.states:
            if state.name == self.current_state:
                return state
            
        raise Exception(f"Could not find current page state {self.current_state}")

    def on_key(self, key: PressedKey, ch: str):
        self._current_state().on_key(key, ch)

    def get_view(self) -> List[str]:
        return self._current_state().get_view()
        
    def get_footer(self) -> str:
        return self._current_state().get_footer()
        