from typing import Dict, List, Tuple

from language_pipes.tui.top_nav import TopNav
from language_pipes.tui.side_nav import SideNav
from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.kb_utils import PressedKey, read_key

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
    exit_tui: bool
    content_id: int
    footer_id: int
    status_message: str
    status_level: str
    content_cursor_idx: int
    confirm_escape_open: bool
    confirm_choice_idx: int
    content_bg_id: int
    content_area_size: Tuple[int, int]

    def __init__(self, size: Tuple[int, int], pos: Tuple[int, int]):
        self.window = TuiWindow(size, pos)
        self.active_top_idx = 0
        self.focus_depth = 0
        self.side_idx_by_tab = {
            tab: 0 for tab in self.TOP_HEADERS
        }
        self.running = False
        self.exit_tui = False
        self.status_message = ""
        self.status_level = "info"
        self.content_cursor_idx = 0
        self.confirm_escape_open = False
        self.confirm_choice_idx = 2

        self._init_layout(size, pos)
        self._render_all()

    def _init_layout(self, size: Tuple[int, int], pos: Tuple[int, int]):
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, 2))
        self.window.add_text(TermText("|\n" * (size[1] - 5)), (15, 3))
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, size[1] - 3))

        self.content_area_size = (max(1, size[0] - 19), max(1, size[1] - 7))
        self.content_bg_id = self.window.add_text(
            TermText(self._content_blank_block()),
            (17, 4),
        )
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
        if self.confirm_escape_open:
            return "Confirm Exit: Arrows U/D or L/R to choose   Enter: Confirm   Esc: Cancel"
        if self.focus_depth == 0:
            return "Arrows L/R: Switch Tab   Enter: Side Nav   Esc: Back/Quit Options   q: Exit"
        if self.focus_depth == 1:
            return "Arrows U/D: Switch Section   Enter: Content   Esc: Top Tabs   q: Exit"
        return "Arrows U/D: Navigate Placeholder   Enter: Activate   r: Refresh   Esc: Back   q: Exit"

    def _status_text(self) -> str:
        if self.status_message == "":
            return ""
        return f"[{self.status_level.upper()}] {self.status_message}"

    def _set_status(self, message: str, level: str = "info"):
        self.status_message = message
        self.status_level = level

    def _clear_status(self):
        self.status_message = ""
        self.status_level = "info"

    def _confirm_options(self) -> List[str]:
        return ["Return to menu", "Exit TUI", "Cancel"]

    def _render_confirm_prompt(self) -> str:
        options = self._confirm_options()
        lines = [
            "Safe exit confirmation",
            "",
            "Use arrows to choose an option, then press Enter:",
        ]
        for i, opt in enumerate(options):
            cursor = ">" if i == self.confirm_choice_idx else " "
            lines.append(f"{cursor} {opt}")
        lines.extend([
            "",
            "Esc: cancel and continue working in MainFrame",
        ])
        return "\n".join(lines)

    def _section_placeholder(self) -> Tuple[str, str, str]:
        placeholders: Dict[str, Dict[str, Tuple[str, str, str]]] = {
            "Network": {
                "Status": (
                    "No live network provider connected yet.",
                    "Next: Open Network -> Configure to review bootstrap and identity settings.",
                    "info",
                ),
                "Peers": (
                    "Peer list unavailable in placeholder mode.",
                    "Next: Open Network -> Configure, then refresh after provider wiring in Phase 3.",
                    "warning",
                ),
                "Configure": (
                    "Network configuration editor is not wired in this phase.",
                    "Next: Keep credentials/config ready; edit workflow arrives in Phase 4.",
                    "info",
                ),
            },
            "Models": {
                "Installed": (
                    "Installed model inventory is not loaded yet.",
                    "Next: Use CLI model commands now; this panel will show provider data in Phase 3.",
                    "info",
                ),
                "Download": (
                    "Download workflow placeholder only.",
                    "Next: Pre-stage model assets, then return here for guided actions in Phase 4.",
                    "warning",
                ),
                "Cache": (
                    "Cache usage and eviction stats unavailable.",
                    "Next: Run refresh after provider wiring to inspect cache health.",
                    "info",
                ),
            },
            "Pipes": {
                "Overview": (
                    "Pipe topology summary not yet connected.",
                    "Next: Validate your routes in config; live topology appears in Phase 3.",
                    "info",
                ),
                "Routes": (
                    "Route table is currently placeholder-only.",
                    "Next: Open Pipes -> Configure for route editing in a later phase.",
                    "warning",
                ),
                "Configure": (
                    "Pipe configuration editor pending.",
                    "Next: Use current config files as source of truth until edit flows land.",
                    "info",
                ),
            },
            "Jobs": {
                "Queue": (
                    "Active job queue telemetry not connected.",
                    "Next: Trigger a workload, then refresh once job provider integration is enabled.",
                    "info",
                ),
                "History": (
                    "Historical jobs are not surfaced in this phase.",
                    "Next: Capture logs now; timeline UI arrives with backend integration.",
                    "warning",
                ),
                "Stats": (
                    "Job throughput stats unavailable.",
                    "Next: Use Activity -> Metrics placeholder for expected future summary shape.",
                    "info",
                ),
            },
            "Activity": {
                "Logs": (
                    "Live logs are not connected to the content pane yet.",
                    "Next: Tail CLI logs externally; in-TUI stream arrives in a later phase.",
                    "info",
                ),
                "Events": (
                    "Event timeline provider is not active.",
                    "Next: Refresh after Phase 3 data wiring to inspect operational events.",
                    "warning",
                ),
                "Metrics": (
                    "Metrics feed is currently placeholder-only.",
                    "Next: Use this section to validate navigation while wiring metrics provider next.",
                    "info",
                ),
            },
        }

        tab = self._active_tab()
        section = self._active_side_option()
        return placeholders.get(tab, {}).get(
            section,
            (
                "No placeholder registered for this section.",
                "Next: Return to top tabs and choose a known section.",
                "warning",
            ),
        )

    def _render_content(self):
        self.window.update_text(self.content_bg_id, TermText(self._content_blank_block()))

        if self.confirm_escape_open:
            self.window.update_text(self.content_id, TermText(self._render_confirm_prompt()))
            return

        tab = self._active_tab()
        section = self._active_side_option()
        state_summary, next_action, level = self._section_placeholder()
        selection_hint = f"Selection index: {self.content_cursor_idx + 1}"
        content = "\n".join([
            f"View: {tab}",
            f"Section: {section}",
            "",
            f"State ({level.upper()}): {state_summary}",
            f"Next Action: {next_action}",
            "",
            selection_hint,
            f"Focus depth: {self.focus_depth} (0=top, 1=side, 2=content)",
        ])
        self.window.update_text(self.content_id, TermText(content))

    def _content_blank_block(self) -> str:
        width, height = self.content_area_size
        return "\n".join([" " * width for _ in range(height)])

    def _sync_navigation(self):
        active_options = self._active_side_options()
        self.side_nav.focused_idx = min(self._active_side_idx(), len(active_options) - 1) if len(active_options) > 0 else 0
        self.side_nav.set_options(active_options)

        self.top_nav.focused_idx = self.active_top_idx
        self.top_nav.set_focus(self.focus_depth == 0 and not self.confirm_escape_open)
        self.side_nav.set_focus(self.focus_depth == 1 and not self.confirm_escape_open)

    def _render_footer(self):
        footer_base = self._footer_text()
        status = self._status_text()
        footer_text = footer_base if status == "" else f"{footer_base}   |   {status}"
        self.window.update_text(self.footer_id, TermText(footer_text))

    def _activate_selection(self):
        self._set_status(
            f"Activated {self._active_tab()} -> {self._active_side_option()} (placeholder action)",
            "info",
        )

    def _refresh_current_view(self):
        self._set_status("Refreshed (placeholder view)", "info")

    def _confirm_prev(self):
        options = self._confirm_options()
        self.confirm_choice_idx = (self.confirm_choice_idx - 1) % len(options)

    def _confirm_next(self):
        options = self._confirm_options()
        self.confirm_choice_idx = (self.confirm_choice_idx + 1) % len(options)

    def _resolve_confirm_choice(self):
        choice = self._confirm_options()[self.confirm_choice_idx]
        self.confirm_escape_open = False

        if choice == "Return to menu":
            self.exit_tui = False
            self.running = False
            return
        if choice == "Exit TUI":
            self.exit_tui = True
            self.running = False
            return

        self._set_status("Exit canceled", "info")

    def _handle_confirm_key(self, key: PressedKey):
        if key in (PressedKey.ArrowUp, PressedKey.ArrowLeft):
            self._confirm_prev()
            return
        if key in (PressedKey.ArrowDown, PressedKey.ArrowRight):
            self._confirm_next()
            return
        if key == PressedKey.Enter:
            self._resolve_confirm_choice()
            return
        if key == PressedKey.Escape:
            self.confirm_escape_open = False
            self._set_status("Exit canceled", "info")

    def _open_exit_confirm(self):
        self.confirm_escape_open = True
        self.confirm_choice_idx = 2
        self._set_status("Choose: Return to menu, Exit TUI, or Cancel", "warning")

    def _render_all(self):
        self._sync_navigation()
        self._render_content()
        self._render_footer()

        self.window.paint()
        self.top_nav.window.paint()
        self.side_nav.window.paint()

    def _teardown_windows(self):
        self.window.remove_all()
        self.top_nav.window.remove_all()
        self.side_nav.window.remove_all()

        self.window.paint()
        self.top_nav.window.paint()
        self.side_nav.window.paint()

    def _handle_key(self, key: PressedKey, ch: str):
        if self.confirm_escape_open:
            self._handle_confirm_key(key)
            return

        if key == PressedKey.Alpha and ch.lower() == "q":
            self._open_exit_confirm()
            return

        if key == PressedKey.Alpha and ch.lower() == "r":
            self._refresh_current_view()
            return

        if key == PressedKey.Escape:
            if self.focus_depth > 0:
                self.focus_depth -= 1
                if self.focus_depth < 2:
                    self.content_cursor_idx = 0
            else:
                self._open_exit_confirm()
            return

        if key == PressedKey.Enter:
            if self.focus_depth < 2:
                self.focus_depth += 1
            else:
                self._activate_selection()
            return

        if self.focus_depth == 0:
            if key == PressedKey.ArrowRight:
                self.active_top_idx = (self.active_top_idx + 1) % len(self.TOP_HEADERS)
                self.content_cursor_idx = 0
                self._clear_status()
            elif key == PressedKey.ArrowLeft:
                self.active_top_idx = (self.active_top_idx - 1) % len(self.TOP_HEADERS)
                self.content_cursor_idx = 0
                self._clear_status()
            return

        if self.focus_depth == 1:
            if key == PressedKey.ArrowDown:
                self.side_nav.move_next()
                self.side_idx_by_tab[self._active_tab()] = self.side_nav.focused_idx
                self.content_cursor_idx = 0
                self._clear_status()
            elif key == PressedKey.ArrowUp:
                self.side_nav.move_prev()
                self.side_idx_by_tab[self._active_tab()] = self.side_nav.focused_idx
                self.content_cursor_idx = 0
                self._clear_status()
            return

        if self.focus_depth == 2:
            if key == PressedKey.ArrowDown:
                self.content_cursor_idx += 1
                self._set_status(
                    "Moved selection cursor (placeholder content)",
                    "info",
                )
            elif key == PressedKey.ArrowUp:
                self.content_cursor_idx = max(0, self.content_cursor_idx - 1)
                self._set_status(
                    "Moved selection cursor (placeholder content)",
                    "info",
                )
            elif key in (PressedKey.ArrowLeft, PressedKey.ArrowRight):
                self._set_status("No horizontal action in placeholder content", "info")

    def run(self) -> str:
        self.running = True
        self.exit_tui = False
        self._render_all()
        while self.running:
            key, ch = read_key()
            self._handle_key(key, ch)
            self._render_all()

        self._teardown_windows()

        return "exit" if self.exit_tui else "menu"
