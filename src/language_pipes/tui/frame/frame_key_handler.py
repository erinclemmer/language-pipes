from language_pipes.tui.frame.editor import Editor
from language_pipes.tui.frame.layout import FrameLayout
from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.components.network_form import NetworkForm
from language_pipes.tui.components.exit_confirm import ExitConfirm

class FrameKeyHandler:
    nav: NavState
    editor: Editor
    confirm: Confirm
    state: FrameState
    layout: FrameLayout
    network_form: NetworkForm
    exit_confirm: ExitConfirm

    def __init__(self, layout: FrameLayout, network_form: NetworkForm):
        self.state = layout.state
        self.nav = layout.nav_state
        self.editor = layout.editor
        self.confirm = layout.edit_confirm
        self.exit_confirm = layout.exit_confirm

        self.layout = layout
        self.network_form = network_form

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

        self.state.set_status("Exit canceled", "info")

    def _handle_confirm_key(self, key: PressedKey):
        action = self.confirm.handle_key(key)
        if action == "confirm":
            self._resolve_exit_choice()
        elif action == "cancel":
            self.confirm.close()
            self.state.set_status("Exit canceled", "info")

    def _open_exit_confirm(self):
        self.exit_confirm.open()
        self.state.set_status("Choose: Return to menu, Exit TUI, or Cancel", "warning")

    def activate_selection(self):
        tab = self.nav.active_tab()
        section = self.nav.active_side_option()
        if tab == "Network" and section == "Configure":
            self.network_form.start()

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

        if self.editor.edit_mode:
            self._handle_edit_mode_key(key, ch)
            return

        if key == PressedKey.Alpha and ch.lower() == "q":
            self._open_exit_confirm()
            return

        if key == PressedKey.Alpha and ch.lower() == "r":
            self.layout._refresh_current_view()
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
                self.state.clear_status()
            elif key == PressedKey.ArrowLeft:
                self.nav.tab_prev()
                self.state.clear_status()
            return

        if self.nav.focus_depth == 1:
            if key == PressedKey.ArrowDown:
                self.nav.side_next(self.layout.side_nav)
                self.state.clear_status()
            elif key == PressedKey.ArrowUp:
                self.nav.side_prev(self.layout.side_nav)
                self.state.clear_status()
            return

        if self.nav.focus_depth == 2:
            if key == PressedKey.ArrowDown:
                self.nav.content_cursor_down()
                self.state.set_status("Moved selection cursor (placeholder content)", "info")
            elif key == PressedKey.ArrowUp:
                self.nav.content_cursor_up()
                self.state.set_status("Moved selection cursor (placeholder content)", "info")
            elif key in (PressedKey.ArrowLeft, PressedKey.ArrowRight):
                self.state.set_status("No horizontal action in placeholder content", "info")

    def _discard_form(self) -> None:
        self.editor.exit_edit_mode()
        self.state.set_status("Discarded edits", "info")

    def _handle_edit_mode_key(self, key: PressedKey, ch: str) -> None:
        if key == PressedKey.Escape:
            self._discard_form()
            self.nav.focus_shallower()
            return
        if key == PressedKey.ArrowUp:
            self.editor.prev_field()
            return
        if key == PressedKey.ArrowDown:
            self.editor.next_field()
            return
        if key == PressedKey.Enter:
            self.editor.change_field_editor(True)
            return
        
        if not self.editor.edit_fields:
            return

        field = self.editor.edit_fields[self.editor.edit_field_idx]
        value = str(field.get("value", ""))

        if key == PressedKey.Alpha:
            field["value"] = value + ch
            field["error"] = None
            return
        if key == PressedKey.Backspace:
            field["value"] = value[:-1]
            field["error"] = None
            return
        if key == PressedKey.Delete:
            field["value"] = ""
            field["error"] = None