from typing import Dict, List, Optional
from language_pipes.content_provider.content_provider import ProviderState

class NavState:
    top_headers: List[str]
    SIDE_OPTIONS_BY_TAB: Dict[str, List[str]]

    focus_depth: int
    active_top_idx: int
    side_idx: int
    _last_provider_state: Optional[ProviderState]

    def __init__(
        self,
        top_headers: List[str],
        side_options_by_tab: Dict[str, List[str]],
    ) -> None:
        self.top_headers = top_headers
        self.SIDE_OPTIONS_BY_TAB = side_options_by_tab

        self.focus_depth = 0
        self.active_top_idx = 0
        self.side_idx = 0
        self._last_provider_state = None

    def sync_provider_state(self, state: ProviderState):
        self._last_provider_state = state
        self.top_headers = state.visible_headers

    def active_tab(self) -> str:
        return self.top_headers[self.active_top_idx]

    def active_side_options(self) -> List[str]:
        return self.SIDE_OPTIONS_BY_TAB.get(self.active_tab(), ["Overview"])

    def active_side_option(self) -> str:
        options = self.active_side_options()
        if not options:
            return ""
        idx = min(self.side_idx, len(options) - 1)
        return options[idx]

    def tab_next(self):
        self.active_top_idx = (self.active_top_idx + 1) % len(self.top_headers)
        self.side_idx = 0

    def tab_prev(self):
        self.active_top_idx = (self.active_top_idx - 1) % len(self.top_headers)
        self.side_idx = 0

    def side_next(self) -> None:
        self.side_idx = (self.side_idx + 1) % len(self.active_side_options())
        
    def side_prev(self) -> None:
        self.side_idx = (self.side_idx - 1) % len(self.active_side_options())

    def focus_deeper(self) -> None:
        self.focus_depth = min(self.focus_depth + 1, 2)

    def focus_shallower(self) -> None:
        self.focus_depth = max(self.focus_depth - 1, 0)
        if self.focus_depth < 2:
            self.content_cursor_idx = 0

    def set_tab(self, tab_name: str):
        if tab_name not in self.top_headers:
            return
        self.active_top_idx = self.top_headers.index(tab_name)
        self.focus_depth = 1
        self.side_idx = 0

    def set_side_nav(self, name: str):
        opts = self.active_side_options()
        if name not in opts:
            return
        idx = opts.index(name)
        self.side_idx = idx
        self.focus_depth = 2