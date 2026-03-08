"""
NavState: tracks which tab and side-nav section is currently active.
"""
from typing import Dict, List
from language_pipes.tui.components.side_nav import SideNav

class NavState:
    """
    Holds the navigation cursor positions for the top-tab bar and the
    per-tab side-nav, plus the current focus depth.

    Focus depths:
      0 – top-nav bar is active
      1 – side-nav is active
      2 – content pane is active
    """

    TOP_HEADERS: List[str]
    SIDE_OPTIONS_BY_TAB: Dict[str, List[str]]

    focus_depth: int
    active_top_idx: int
    side_idx_by_tab: Dict[str, int]
    content_cursor_idx: int

    def __init__(
        self,
        top_headers: List[str],
        side_options_by_tab: Dict[str, List[str]],
    ) -> None:
        self.TOP_HEADERS = top_headers
        self.SIDE_OPTIONS_BY_TAB = side_options_by_tab

        self.focus_depth = 0
        self.active_top_idx = 0
        self.side_idx_by_tab = {tab: 0 for tab in top_headers}
        self.content_cursor_idx = 0

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def active_tab(self) -> str:
        return self.TOP_HEADERS[self.active_top_idx]

    def active_side_options(self) -> List[str]:
        return self.SIDE_OPTIONS_BY_TAB.get(self.active_tab(), ["Overview"])

    def active_side_idx(self) -> int:
        return self.side_idx_by_tab.get(self.active_tab(), 0)

    def active_side_option(self) -> str:
        options = self.active_side_options()
        if not options:
            return ""
        idx = min(self.active_side_idx(), len(options) - 1)
        return options[idx]

    def active_view_key(self):
        return self.active_tab(), self.active_side_option()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def tab_next(self) -> None:
        self.active_top_idx = (self.active_top_idx + 1) % len(self.TOP_HEADERS)
        self.content_cursor_idx = 0

    def tab_prev(self) -> None:
        self.active_top_idx = (self.active_top_idx - 1) % len(self.TOP_HEADERS)
        self.content_cursor_idx = 0

    def set_tab(self, tab_name: str):
        if tab_name not in self.TOP_HEADERS:
            return
        self.active_top_idx = self.TOP_HEADERS.index(tab_name)
        self.focus_depth = 1

    def side_next(self, side_nav: SideNav) -> None:
        side_nav.move_next()
        self.side_idx_by_tab[self.active_tab()] = side_nav.focused_idx
        self.content_cursor_idx = 0

    def side_prev(self, side_nav: SideNav) -> None:
        side_nav.move_prev()
        self.side_idx_by_tab[self.active_tab()] = side_nav.focused_idx
        self.content_cursor_idx = 0

    def set_side_nav(self, side_nav: SideNav, name: str):
        if name not in side_nav.options:
            return
        idx = side_nav.options.index(name)
        side_nav.focused_idx = idx
        self.side_idx_by_tab[self.active_tab()] = idx
        self.content_cursor_idx = 0
        self.focus_depth = 2

    def focus_deeper(self) -> None:
        self.focus_depth = min(self.focus_depth + 1, 2)

    def focus_shallower(self) -> None:
        self.focus_depth = max(self.focus_depth - 1, 0)
        if self.focus_depth < 2:
            self.content_cursor_idx = 0

    def content_cursor_down(self) -> None:
        self.content_cursor_idx += 1

    def content_cursor_up(self) -> None:
        self.content_cursor_idx = max(0, self.content_cursor_idx - 1)
