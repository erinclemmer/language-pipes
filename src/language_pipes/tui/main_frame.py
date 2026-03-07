from typing import Dict, List, Tuple

from language_pipes.tui.kb_utils import PressedKey, read_key
from language_pipes.tui.tui import TuiWindow, TermText


class SideNav:
    window: TuiWindow
    focused_idx: int
    is_focused: bool
    options: List[str]
    option_ids: List[int]
    l_cursor_id: int
    r_cursor_id: int

    def __init__(
            self,
            size: Tuple[int, int],
            pos: Tuple[int, int],
            options: List[str]
        ):
        self.window = TuiWindow(size, pos)
        self.options = []
        self.option_ids = []
        self.focused_idx = 0
        self.is_focused = False

        self.l_cursor_id = self.window.add_text(TermText(" "), (0, 0))
        self.r_cursor_id = self.window.add_text(TermText(" "), (self.window.size[0] - 1, 0))

        self.set_options(options)
        self.window.paint()

    def _cursor_y(self) -> int:
        return self.focused_idx * 2

    def _update_cursor(self):
        if len(self.options) == 0:
            self.window.update_text(self.l_cursor_id, TermText(" "))
            self.window.update_text(self.r_cursor_id, TermText(" "))
            return

        cursor_char = ">" if self.is_focused else " "
        self.window.update_text(self.l_cursor_id, TermText(cursor_char), (0, self._cursor_y()))
        self.window.update_text(self.r_cursor_id, TermText("<" if self.is_focused else " "), (self.window.size[0] - 1, self._cursor_y()))

    def _update_option_styles(self):
        for i, option_id in enumerate(self.option_ids):
            self.window.update_text(
                option_id,
                TermText(
                    self.options[i],
                    fg=51 if i == self.focused_idx else None,
                    bold=(i == self.focused_idx)
                )
            )
        self._update_cursor()

    def move_next(self):
        if len(self.options) == 0:
            return
        self.focused_idx = (self.focused_idx + 1) % len(self.options)
        self._update_option_styles()

    def move_prev(self):
        if len(self.options) == 0:
            return
        self.focused_idx = (self.focused_idx - 1) % len(self.options)
        self._update_option_styles()

    def set_focus(self, is_focused: bool):
        self.is_focused = is_focused
        self._update_option_styles()

    def set_options(self, options: List[str]):
        for option_id in self.option_ids:
            self.window.remove_txt(option_id)

        self.options = options
        self.option_ids = []
        self.focused_idx = min(self.focused_idx, len(options) - 1) if len(options) > 0 else 0

        for i, opt in enumerate(options):
            self.option_ids.append(self.window.add_text(TermText(opt), (2, i * 2)))

        self._update_option_styles()


class TopNav:
    def __init__(self, size: Tuple[int, int], pos: Tuple[int, int], headers: List[str]):
        self.window = TuiWindow(size, pos)
        self.headers = headers
        self.header_ids: List[int] = []
        self.header_positions: List[int] = []
        self.is_focused = True

        for i, h in enumerate(self.headers):
            header_x = 5 + i * 15
            self.header_positions.append(header_x)
            self.header_ids.append(self.window.add_text(TermText(h), (header_x, 0)))

        self.focused_idx = 0
        self.l_cursor_id = self.window.add_text(TermText("["), (3, 0))
        self.r_cursor_id = self.window.add_text(TermText("]"), (3, 0))

        self._update_styles()

    def _update_styles(self):
        for i, header_id in enumerate(self.header_ids):
            self.window.update_text(
                header_id,
                TermText(
                    self.headers[i],
                    fg=51 if i == self.focused_idx else None,
                    bold=(self.is_focused and i == self.focused_idx),
                ),
            )

        selected_x = self.header_positions[self.focused_idx]
        selected_header = self.headers[self.focused_idx]
        if self.is_focused:
            self.window.update_text(self.l_cursor_id, TermText("["), (selected_x - 2, 0))
            self.window.update_text(self.r_cursor_id, TermText("]"), (selected_x + len(selected_header), 0))
        else:
            self.window.update_text(self.l_cursor_id, TermText(" "), (selected_x - 2, 0))
            self.window.update_text(self.r_cursor_id, TermText(" "), (selected_x + len(selected_header), 0))

    def move_next(self):
        self.focused_idx = (self.focused_idx + 1) % len(self.headers)
        self._update_styles()

    def move_prev(self):
        self.focused_idx = (self.focused_idx - 1) % len(self.headers)
        self._update_styles()

    def set_focus(self, is_focused: bool):
        self.is_focused = is_focused
        self._update_styles()

        self.window.paint()


class MainFrame:
    TOP_HEADERS = ["Network", "Models", "Pipes", "Jobs", "Activity"]
    SIDE_OPTIONS_BY_TAB: Dict[str, List[str]] = {
        "Network": ["Status", "Peers", "Configure"],
        "Models": ["Installed", "Download", "Cache"],
        "Pipes": ["Overview", "Routes", "Configure"],
        "Jobs": ["Queue", "History", "Stats"],
        "Activity": ["Logs", "Events", "Metrics"],
    }

    top_nav: TopNav
    side_nav: SideNav
    window: TuiWindow
    focus_depth: int
    active_top_idx: int
    side_idx_by_tab: Dict[str, int]
    running: bool
    content_id: int
    footer_id: int

    def __init__(self, size: Tuple[int, int], pos: Tuple[int, int]):
        self.window = TuiWindow(size, pos)
        self.active_top_idx = 0
        self.focus_depth = 0
        self.side_idx_by_tab = {
            tab: 0 for tab in self.TOP_HEADERS
        }
        self.running = False

        self._init_layout(size, pos)
        self._render_all()

    def _init_layout(self, size: Tuple[int, int], pos: Tuple[int, int]):
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, 2))
        self.window.add_text(TermText("|\n" * (size[1] - 5)), (15, 3))
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, size[1] - 3))

        self.content_id = self.window.add_text(TermText(""), (17, 4))
        self.footer_id = self.window.add_text(TermText(""), (2, size[1] - 2))

        self.top_nav = TopNav((80, 1), (pos[0], pos[1] + 1), self.TOP_HEADERS)
        self.side_nav = SideNav(
            (13, size[1] - 5),
            (pos[0] + 1, pos[1] + 4),
            self._active_side_options(),
        )

    def _active_tab(self) -> str:
        return self.TOP_HEADERS[self.active_top_idx]

    def _active_side_options(self) -> List[str]:
        return self.SIDE_OPTIONS_BY_TAB.get(self._active_tab(), ["Overview"])

    def _active_side_idx(self) -> int:
        return self.side_idx_by_tab.get(self._active_tab(), 0)

    def _active_side_option(self) -> str:
        options = self._active_side_options()
        if len(options) == 0:
            return ""
        idx = min(self._active_side_idx(), len(options) - 1)
        return options[idx]

    def _footer_text(self) -> str:
        if self.focus_depth == 0:
            return "Arrows L/R: Switch Tab   Enter: Side Nav   Esc/q: Exit"
        if self.focus_depth == 1:
            return "Arrows U/D: Switch Section   Enter: Content   Esc: Top Tabs   q: Exit"
        return "Esc: Back to Side Nav   q: Exit"

    def _render_content(self):
        tab = self._active_tab()
        section = self._active_side_option()
        content = "\n".join([
            f"View: {tab}",
            f"Section: {section}",
            "",
            "This is a placeholder content panel.",
            "Future sessions will wire live data and editing workflows here.",
            "",
            f"Focus depth: {self.focus_depth} (0=top, 1=side, 2=content)",
        ])
        self.window.update_text(self.content_id, TermText(content))

    def _sync_navigation(self):
        active_options = self._active_side_options()
        self.side_nav.focused_idx = min(self._active_side_idx(), len(active_options) - 1) if len(active_options) > 0 else 0
        self.side_nav.set_options(active_options)

        self.top_nav.focused_idx = self.active_top_idx
        self.top_nav.set_focus(self.focus_depth == 0)
        self.side_nav.set_focus(self.focus_depth == 1)

    def _render_all(self):
        self._sync_navigation()
        self._render_content()
        self.window.update_text(self.footer_id, TermText(self._footer_text()))

        self.window.paint()
        self.top_nav.window.paint()
        self.side_nav.window.paint()

    def _handle_key(self, key: PressedKey, ch: str):
        if key == PressedKey.Alpha and ch.lower() == "q":
            self.running = False
            return

        if key == PressedKey.Escape:
            if self.focus_depth > 0:
                self.focus_depth -= 1
            else:
                self.running = False
            return

        if key == PressedKey.Enter:
            if self.focus_depth < 2:
                self.focus_depth += 1
            return

        if self.focus_depth == 0:
            if key == PressedKey.ArrowRight:
                self.active_top_idx = (self.active_top_idx + 1) % len(self.TOP_HEADERS)
            elif key == PressedKey.ArrowLeft:
                self.active_top_idx = (self.active_top_idx - 1) % len(self.TOP_HEADERS)
            return

        if self.focus_depth == 1:
            if key == PressedKey.ArrowDown:
                self.side_nav.move_next()
                self.side_idx_by_tab[self._active_tab()] = self.side_nav.focused_idx
            elif key == PressedKey.ArrowUp:
                self.side_nav.move_prev()
                self.side_idx_by_tab[self._active_tab()] = self.side_nav.focused_idx

    def run(self):
        self.running = True
        self._render_all()
        while self.running:
            key, ch = read_key()
            self._handle_key(key, ch)
            self._render_all()
