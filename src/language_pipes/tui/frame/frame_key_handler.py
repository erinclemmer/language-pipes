from language_pipes.tui.frame.layout import FrameLayout
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.components.exit_confirm import ExitConfirm
from language_pipes.tui.frame.page_router import PageRouter


class FrameKeyHandler:
    nav: NavState
    confirm: Confirm
    state: FrameState
    layout: FrameLayout
    exit_confirm: ExitConfirm
    page_router: PageRouter

    def __init__(self, layout: FrameLayout, page_router: PageRouter):
        self.state = layout.state
        self.nav = layout.nav_state
        self.confirm = layout.edit_confirm
        self.exit_confirm = layout.exit_confirm
        self.page_router = page_router

        self.layout = layout

    def _resolve_exit_choice(self):
        choice = self.exit_confirm.selected_option()
        self.exit_confirm.close()

        if choice == "Return to menu":
            self.state.exit_tui = False
            self.state.running = False
            return
        if choice == "Exit TUI":
            self.state.exit_tui = True
            self.state.running = False
            return

    def _handle_confirm_key(self, key: PressedKey):
        action = self.confirm.handle_key(key)
        if action == "confirm":
            self.confirm.close()
            if self.confirm.on_apply is not None:
                self.confirm.on_apply()
        elif action == "discard":
            self.confirm.close()
            if self.confirm.on_discard is not None:
                self.confirm.on_discard()

    def _open_exit_confirm(self):
        self.exit_confirm.open()

    def activate_selection(self):
        page = self.page_router.get_page()
        start = getattr(page, "start", None)
        if callable(start):
            start()

    def handle_key(self, key: PressedKey, ch: str):
        if self.exit_confirm.is_open:
            res = self.exit_confirm.handle_key(key)
            if res == "confirm" or res == "cancel":
                self._resolve_exit_choice()
            return

        if self.confirm.is_open:
            if self._handle_confirm_key(key):
                self.nav.focus_shallower()
            return

        current_page = self.page_router.get_page()
        if self.nav.focus_depth == 2 and current_page is not None:
            current_page.on_key(key, ch)
            return

        if key == PressedKey.Alpha and ch.lower() == "q":
            self._open_exit_confirm()
            return

        if key == PressedKey.Escape:
            if self.nav.focus_depth > 0:
                self.nav.focus_shallower()
            else:
                self._open_exit_confirm()
            return

        if key == PressedKey.Enter:
            if self.nav.focus_depth < 1:
                self.nav.focus_deeper()
            else:
                self.nav.focus_deeper()
                self.activate_selection()
            return

        if self.nav.focus_depth == 0:
            if key == PressedKey.ArrowRight:
                self.nav.tab_next()
            elif key == PressedKey.ArrowLeft:
                self.nav.tab_prev()
            return

        if self.nav.focus_depth == 1:
            if key == PressedKey.ArrowDown:
                self.layout.nav_window.side_next()
            elif key == PressedKey.ArrowUp:
                self.layout.nav_window.side_prev()
            return