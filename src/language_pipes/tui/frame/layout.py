from typing import Tuple, List, Dict, Any
from language_pipes.tui.frame import view_state as vs
from language_pipes.tui.tui import TermText, TuiWindow
from language_pipes.tui.util.screen_utils import Color
from language_pipes.tui.components.top_nav import TopNav
from language_pipes.tui.components.side_nav import SideNav
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.frame.editor import Editor
from language_pipes.tui.components.exit_confirm import ExitConfirm
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.frame.tips import TIPS

class FrameLayout:
    content_id: int
    footer_id: int
    status_text: str

    editor: Editor
    top_nav: TopNav
    side_nav: SideNav
    nav_state: NavState
    loader: ContentLoader
    exit_confirm: ExitConfirm
    edit_confirm: Confirm
    state: FrameState
    window: TuiWindow

    def __init__(
            self, 
            window: TuiWindow, 
            nav: NavState, 
            editor: Editor,
            loader: ContentLoader, 
            exit_confirm: ExitConfirm,
            edit_confirm: Confirm,
            state: FrameState
        ):
        self.nav_state = nav
        self.window = window
        self.loader = loader
        self.exit_confirm = exit_confirm
        self.edit_confirm = edit_confirm
        self.state = state
        self.editor = editor
        self.status_text = ""

    def _init_layout(self, size: Tuple[int, int], pos: Tuple[int, int]):
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, 2))
        self.window.add_text(TermText("|\n" * (size[1] - 5)), (15, 3))
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, size[1] - 3))

        self.content_area_size = (max(1, size[0] - 19), max(1, size[1] - 7))
        self.content_bg_id = self.window.add_text(
            TermText(self._content_blank_block()),
            (17, 4),
        )
        self.content_id = self.window.add_text(TermText(""), (17, 5))
        self.footer_id = self.window.add_text(TermText(""), (2, size[1] - 2))
        self.status_id = self.window.add_text(TermText(""), (17, size[1] - 4))

        self.top_nav = TopNav((80, 1), (pos[0], pos[1] + 1), self.nav_state.TOP_HEADERS)
        self.side_nav = SideNav(
            (13, size[1] - 5),
            (pos[0] + 1, pos[1] + 4),
            self.nav_state.active_side_options(),
        )
    
    def _sync_navigation(self):
        active_options = self.nav_state.active_side_options()
        self.side_nav.focused_idx = (
            min(self.nav_state.active_side_idx(), len(active_options) - 1)
            if active_options
            else 0
        )
        self.side_nav.set_options(active_options)

        self.top_nav.focused_idx = self.nav_state.active_top_idx
        interactive_overlay_open = self.exit_confirm.is_open or self.edit_confirm.is_open or self.editor.edit_mode
        self.top_nav.set_focus(self.nav_state.focus_depth == 0 and not interactive_overlay_open)
        self.top_nav._update_styles()
        self.side_nav.set_focus(self.nav_state.focus_depth == 1 and not interactive_overlay_open)

    def _content_blank_block(self) -> str:
        width, height = self.content_area_size
        return "\n".join([" " * width for _ in range(height)])

    def _load_active_view_data(self, update_status: bool) -> Dict[str, Any]:
        tab, section = self.nav_state.active_view_key()
        result = self.loader.load(tab, section, update_status=update_status)
        if update_status and self.loader.last_status_message:
            self.state.set_status(self.loader.last_status_message, self.loader.last_status_level)
        return result

    def _refresh_current_view(self):
        self._load_active_view_data(update_status=True)

    def _render_content(self):
        self.window.update_text(self.content_bg_id, TermText(self._content_blank_block()))

        if self.exit_confirm.is_open:
            self.window.update_text(self.content_id, TermText(self.exit_confirm.render()))
            return

        if self.edit_confirm.is_open:
            self.window.update_text(self.content_id, TermText(self.edit_confirm.render()))
            return

        if self.editor.edit_mode:
            self._render_form_content()
            return

        tab = self.nav_state.active_tab()
        section = self.nav_state.active_side_option()
        view_state = self._load_active_view_data(update_status=False)
        details = view_state.get("details", [])

        detail_lines: List[str] = []
        if isinstance(details, list) and details:
            detail_lines.extend([str(line) for line in details])


        content_parts = [
            f"{tab} / {section}", ""
        ]

        if detail_lines:
            content_parts.extend(detail_lines)


        self.window.update_text(self.content_id, TermText("\n".join(content_parts)))

    def _render_form_content(self) -> None:
        tab = self.nav_state.active_tab()
        section = self.nav_state.active_side_option()

        lines = [
            f"{tab} / {section}",
            "",
        ]

        if self.editor.field_editor_visible:
            if tab == "Network" and section == "Configure":
                lines.extend(self.editor.form.get_editor_lines())
        else:
            lines.append("Edit Network Configuration:")
            for idx, field in enumerate(self.editor.edit_fields):
                l_cursor = "|>" if idx == self.editor.edit_field_idx else "  "
                r_cursor = "<|" if idx == self.editor.edit_field_idx else "  "
                name = str(field.get("label", "field"))
                value = str(field.get("value", ""))
                error = field.get("error")
                lines.append(f" {l_cursor} {name}: {value} {r_cursor}")
                if error:
                    lines.append(f"    ! {error}")

        tip = ""
        res = self.editor.get_current_field()
        if not self.editor.field_editor_visible and res is not None and res[0] in TIPS["network"]["configure"]:
            tip = TIPS["network"]["configure"][res[0]]

        lines.extend([
            "",
            tip
        ])

        self.window.update_text(self.content_id, TermText("\n".join(lines)))

    def _render_footer(self):
        self.window.update_text(self.footer_id, TermText(self._footer_text()))

    def _render_status(self):
        msg = self.state.status_message
        lvl = self.state.status_level.upper()
        fg = Color.Black
        if lvl == "INFO":
            fg = Color.Black
        if lvl == "WARNING":
            fg = Color.BrightYellow
        if lvl == "ERROR":
            fg = Color.BrightRed
        status_text = f"[{lvl}] {msg}"
        if self.status_text != status_text:
            self.status_text = status_text
            self.window.update_text(self.status_id, TermText(f"[{lvl}] {msg}", fg))

    def _render_all(self):
        self._sync_navigation()
        self._render_status()
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

    def _active_form_view_state(self) -> Dict[str, Any]:
        fields: List[Dict[str, Any]] = []
        for field in self.editor.edit_fields:
            value = str(field.get("value", ""))
            if field.get("masked"):
                value = "*" * len(value) if value else ""
            fields.append(
                {
                    "name": field.get("name", ""),
                    "value": value,
                    "error": field.get("error"),
                }
            )

        hint = "Complete fields and press Enter on the last field to confirm."
        if self.editor.edit_form_name == "model_assignments":
            hint = "Format layer assignments as layer:model (comma separated), then confirm."
        return vs.form_view_state(fields, hint, "info")

    def _footer_text(self) -> str:
        if self.exit_confirm.is_open:
            return "Arrows U/D: Navigate   Enter: Select   Esc: Cancel"
        if self.edit_confirm.is_open:
            return "Arrows U/D: Change choice   Enter: Select   Esc: Cancel"
        if self.editor.edit_mode:
            if self.editor.field_editor_visible:
                return self.editor.form.get_footer()
            else:
                return "Arrows U/D: Change property to edit   Enter: Next   Esc: Back"
        if self.nav_state.focus_depth == 0:
            return "Arrows L/R: Switch Tab   Enter: Side Nav   Esc: Back/Quit Options   q: Exit"
        if self.nav_state.focus_depth == 1:
            return "Arrows U/D: Switch Section   Enter: Content   Esc: Top Tabs   q: Exit"
        return ""

