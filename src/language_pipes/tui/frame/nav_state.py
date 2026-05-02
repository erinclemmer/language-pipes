from typing import Dict, List
from language_pipes.content_provider.content_provider import ProviderState

class NavState:
    top_headers: List[str]
    sub_options: Dict[str, List[str]]

    focus_depth: int
    top_idx: int
    sub_idx: int
    
    def __init__(
        self,
        top_headers: List[str],
        side_options_by_tab: Dict[str, List[str]],
    ):
        self.top_headers = top_headers
        self.sub_options = side_options_by_tab

        self.focus_depth = 0
        self.top_idx = 0
        self.sub_idx = 0

    def sync_provider_state(self, state: ProviderState):
        self.top_headers = state.visible_headers
        self.sub_options = state.visible_sub_menu

    def active_tab(self) -> str:
        return self.top_headers[self.top_idx]

    def active_sub_options(self) -> List[str]:
        return self.sub_options.get(self.active_tab(), ["Overview"])

    def active_side_option(self) -> str:
        options = self.active_sub_options()
        if not options:
            return ""
        idx = min(self.sub_idx, len(options) - 1)
        return options[idx]

    def tab_next(self):
        self.top_idx = (self.top_idx + 1) % len(self.top_headers)
        self.sub_idx = 0

    def tab_prev(self):
        self.top_idx = (self.top_idx - 1) % len(self.top_headers)
        self.sub_idx = 0

    def side_next(self) -> None:
        self.sub_idx = (self.sub_idx + 1) % len(self.active_sub_options())
        
    def side_prev(self) -> None:
        self.sub_idx = (self.sub_idx - 1) % len(self.active_sub_options())

    def focus_deeper(self) -> None:
        self.focus_depth = min(self.focus_depth + 1, 2)

    def focus_shallower(self) -> None:
        self.focus_depth = max(self.focus_depth - 1, 0)
        if self.focus_depth < 2:
            self.content_cursor_idx = 0

    def set_tab(self, tab_name: str):
        if tab_name not in self.top_headers:
            return
        self.top_idx = self.top_headers.index(tab_name)
        self.focus_depth = 1
        self.sub_idx = 0

    def set_side_nav(self, name: str):
        opts = self.active_sub_options()
        if name not in opts:
            return
        idx = opts.index(name)
        self.sub_idx = idx
        self.focus_depth = 2