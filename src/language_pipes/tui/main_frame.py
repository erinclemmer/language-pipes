from typing import Any, Dict, List, Optional, Tuple

from language_pipes.tui.top_nav import TopNav
from language_pipes.tui.side_nav import SideNav
from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.kb_utils import PressedKey, read_key
from language_pipes.tui.nav_state import NavState
from language_pipes.tui.confirm_dialog import ConfirmDialog
from language_pipes.tui.content_loader import ContentLoader


class MainFrame:
    TOP_HEADERS = ["Network", "Models", "Pipes", "Jobs", "Activity"]
    SIDE_OPTIONS_BY_TAB: Dict[str, List[str]] = {
        "Network": ["Status", "Peers", "Configure"],
        "Models": ["Installed", "Download", "Cache"],
        "Pipes": ["Overview", "Routes", "Configure"],
        "Jobs": ["Queue", "History", "Stats"],
        "Activity": ["Logs", "Events", "Metrics"],
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        size: Tuple[int, int],
        pos: Tuple[int, int],
        providers: Optional[object] = None,
    ):
        self.window = TuiWindow(size, pos)

        self.nav = NavState(self.TOP_HEADERS, self.SIDE_OPTIONS_BY_TAB)
        self.confirm = ConfirmDialog()
        self.loader = ContentLoader(providers)

        self.running = False
        self.exit_tui = False
        self.status_message = ""
        self.status_level = "info"

        self._init_layout(size, pos)
        self._render_all()

    # ------------------------------------------------------------------
    # Compatibility shims – tests access these attributes directly
    # ------------------------------------------------------------------

    @property
    def focus_depth(self) -> int:
        return self.nav.focus_depth

    @focus_depth.setter
    def focus_depth(self, value: int) -> None:
        self.nav.focus_depth = value

    @property
    def active_top_idx(self) -> int:
        return self.nav.active_top_idx

    @active_top_idx.setter
    def active_top_idx(self, value: int) -> None:
        self.nav.active_top_idx = value

    @property
    def side_idx_by_tab(self) -> Dict[str, int]:
        return self.nav.side_idx_by_tab

    @property
    def confirm_escape_open(self) -> bool:
        return self.confirm.is_open

    @confirm_escape_open.setter
    def confirm_escape_open(self, value: bool) -> None:
        self.confirm.is_open = value

    @property
    def confirm_choice_idx(self) -> int:
        return self.confirm.choice_idx

    @confirm_choice_idx.setter
    def confirm_choice_idx(self, value: int) -> None:
        self.confirm.choice_idx = value

    @property
    def content_cursor_idx(self) -> int:
        return self.nav.content_cursor_idx

    @content_cursor_idx.setter
    def content_cursor_idx(self, value: int) -> None:
        self.nav.content_cursor_idx = value

    @property
    def view_state_by_section(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        # Expose the loader's internal cache for test assertions.
        return self.loader._cache

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

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
            self.nav.active_side_options(),
        )

    def _content_blank_block(self) -> str:
        width, height = self.content_area_size
        return "\n".join([" " * width for _ in range(height)])

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, message: str, level: str = "info"):
        self.status_message = message
        self.status_level = level

    def _clear_status(self):
        self.status_message = ""
        self.status_level = "info"

    # ------------------------------------------------------------------
    # Navigation helpers (thin wrappers kept for internal use)
    # ------------------------------------------------------------------

    def _active_tab(self) -> str:
        return self.nav.active_tab()

    def _active_side_option(self) -> str:
        return self.nav.active_side_option()

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def _footer_text(self) -> str:
        if self.confirm.is_open:
            return "Confirm Exit: Arrows U/D or L/R to choose   Enter: Confirm   Esc: Cancel"
        if self.nav.focus_depth == 0:
            return "Arrows L/R: Switch Tab   Enter: Side Nav   Esc: Back/Quit Options   q: Exit"
        if self.nav.focus_depth == 1:
            return "Arrows U/D: Switch Section   Enter: Content   Esc: Top Tabs   q: Exit"
        return "Arrows U/D: Navigate Placeholder   Enter: Activate   r: Refresh   Esc: Back   q: Exit"

    def _status_text(self) -> str:
        if self.status_message == "":
            return ""
        return f"[{self.status_level.upper()}] {self.status_message}"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_active_view_data(self, update_status: bool, force: bool) -> Dict[str, Any]:
        tab, section = self.nav.active_view_key()
        result = self.loader.load(tab, section, update_status=update_status, force=force)
        if update_status and self.loader.last_status_message:
            self._set_status(self.loader.last_status_message, self.loader.last_status_level)
        return result

    def _refresh_current_view(self):
        self._load_active_view_data(update_status=True, force=True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _sync_navigation(self):
        active_options = self.nav.active_side_options()
        self.side_nav.focused_idx = (
            min(self.nav.active_side_idx(), len(active_options) - 1)
            if active_options
            else 0
        )
        self.side_nav.set_options(active_options)

        self.top_nav.focused_idx = self.nav.active_top_idx
        self.top_nav.set_focus(self.nav.focus_depth == 0 and not self.confirm.is_open)
        self.side_nav.set_focus(self.nav.focus_depth == 1 and not self.confirm.is_open)

    def _render_content(self):
        self.window.update_text(self.content_bg_id, TermText(self._content_blank_block()))

        if self.confirm.is_open:
            self.window.update_text(self.content_id, TermText(self.confirm.render()))
            return

        tab = self._active_tab()
        section = self._active_side_option()
        view_state = self._load_active_view_data(update_status=False, force=False)

        state_summary = str(view_state.get("summary", "No summary available."))
        next_action = str(view_state.get("hint", "Next: Press r to refresh."))
        level = str(view_state.get("level", "info"))
        details = view_state.get("details", [])

        detail_lines: List[str] = []
        if isinstance(details, list) and details:
            detail_lines.extend([str(line) for line in details])

        selection_hint = f"Selection index: {self.nav.content_cursor_idx + 1}"
        state_label = str(view_state.get("state", "placeholder")).upper()

        content_parts = [
            f"View: {tab}",
            f"Section: {section}",
            "",
            f"State ({state_label}/{level.upper()}): {state_summary}",
        ]

        if detail_lines:
            content_parts.extend(["", "Details:", *detail_lines])

        content_parts.extend([
            "",
            f"Next Action: {next_action}",
            "",
            selection_hint,
            f"Focus depth: {self.nav.focus_depth} (0=top, 1=side, 2=content)",
        ])

        self.window.update_text(self.content_id, TermText("\n".join(content_parts)))

    def _render_footer(self):
        footer_base = self._footer_text()
        status = self._status_text()
        footer_text = footer_base if status == "" else f"{footer_base}   |   {status}"
        self.window.update_text(self.footer_id, TermText(footer_text))

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

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _activate_selection(self):
        self._set_status(
            f"Activated {self._active_tab()} -> {self._active_side_option()} (placeholder action)",
            "info",
        )

    def _open_exit_confirm(self):
        self.confirm.open()
        self._set_status("Choose: Return to menu, Exit TUI, or Cancel", "warning")

    def _resolve_confirm_choice(self):
        choice = self.confirm.selected_option()
        self.confirm.close()

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
        action = self.confirm.handle_key(key)
        if action == "confirm":
            self._resolve_confirm_choice()
        elif action == "cancel":
            self.confirm.close()
            self._set_status("Exit canceled", "info")

    def _handle_key(self, key: PressedKey, ch: str):
        if self.confirm.is_open:
            self._handle_confirm_key(key)
            return

        if key == PressedKey.Alpha and ch.lower() == "q":
            self._open_exit_confirm()
            return

        if key == PressedKey.Alpha and ch.lower() == "r":
            self._refresh_current_view()
            return

        if key == PressedKey.Escape:
            if self.nav.focus_depth > 0:
                self.nav.focus_shallower()
            else:
                self._open_exit_confirm()
            return

        if key == PressedKey.Enter:
            if self.nav.focus_depth < 2:
                self.nav.focus_deeper()
            else:
                self._activate_selection()
            return

        if self.nav.focus_depth == 0:
            if key == PressedKey.ArrowRight:
                self.nav.tab_next()
                self._clear_status()
            elif key == PressedKey.ArrowLeft:
                self.nav.tab_prev()
                self._clear_status()
            return

        if self.nav.focus_depth == 1:
            if key == PressedKey.ArrowDown:
                self.nav.side_next(self.side_nav)
                self._clear_status()
            elif key == PressedKey.ArrowUp:
                self.nav.side_prev(self.side_nav)
                self._clear_status()
            return

        if self.nav.focus_depth == 2:
            if key == PressedKey.ArrowDown:
                self.nav.content_cursor_down()
                self._set_status("Moved selection cursor (placeholder content)", "info")
            elif key == PressedKey.ArrowUp:
                self.nav.content_cursor_up()
                self._set_status("Moved selection cursor (placeholder content)", "info")
            elif key in (PressedKey.ArrowLeft, PressedKey.ArrowRight):
                self._set_status("No horizontal action in placeholder content", "info")

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

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
