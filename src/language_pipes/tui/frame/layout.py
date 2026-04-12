from typing import Tuple

from language_pipes.tui.frame.nav_window import NavWindow
from language_pipes.tui.tui import TermText, TuiWindow
from language_pipes.tui.util.screen_utils import Color
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.components.exit_confirm import ExitConfirm
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.frame.page_router import PageRouter


class FrameLayout:
    footer_id: int
    status_id: int
    content_id: int
    content_bg_id: int
    
    status_text: str
    content_area_size: Tuple[int, int]

    nav_window: NavWindow
    nav_state: NavState
    loader: ContentLoader
    exit_confirm: ExitConfirm
    edit_confirm: Confirm
    state: FrameState
    window: TuiWindow
    page_router: PageRouter

    def __init__(
        self,
        window: TuiWindow,
        nav: NavState,
        loader: ContentLoader,
        exit_confirm: ExitConfirm,
        edit_confirm: Confirm,
        state: FrameState,
        page_router: PageRouter,
    ):
        self.nav_state = nav
        self.window = window
        self.loader = loader
        self.exit_confirm = exit_confirm
        self.edit_confirm = edit_confirm
        self.page_router = page_router
        self.state = state
        self.status_text = ""

    def _init_layout(self, size: Tuple[int, int], pos: Tuple[int, int]):
        self.content_area_size = (max(1, size[0]), max(1, size[1]))
        self.content_bg_id = self.window.add_text(
            TermText(self._content_blank_block()),
            (17, 4),
        )
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, size[1] - 3))
        self.content_id = self.window.add_text(TermText(""), (0, 0))
        self.right_panel_id = self.window.add_text(TermText(""), (40, 0))
        self.footer_id = self.window.add_text(TermText(""), (2, size[1] - 2))
        self.exit_confirm_id = self.window.add_text(TermText(""), (0, 4))
        self.seperator_column_id = self.window.add_text(TermText("|\n"*(size[1] - 3)), (38, 0))

        self.nav_window = NavWindow(self.window, self.nav_state, size, pos)

    def _sync_navigation(self):
        self.nav_window.sync(self.exit_confirm, self.edit_confirm)

    def _content_blank_block(self) -> str:
        width, height = self.content_area_size
        return "\n".join([" " * width for _ in range(height)])

    def _render_exit_confirm(self):
        self.window.update_text(
            self.exit_confirm_id, TermText(self.exit_confirm.render())
        )
    
    def _render_content(self):
        self.window.update_text(
            self.content_bg_id, TermText(self._content_blank_block())
        )

        if self.edit_confirm.is_open:
            self.window.update_text(
                self.content_id, TermText(self.edit_confirm.render())
            )
            return

        tab = self.nav_state.active_tab()
        section = self.nav_state.active_side_option()

        content_parts = [f"{tab} / {section}", ""]
        right_panel_parts = ["", ""]

        page = self.page_router.get_page()
        if page is not None:
            res = page.get_view() 
            if res is not None:
                if type(res) is type([]):
                    content_parts.extend(res) # pyright: ignore[reportArgumentType]
                elif type(res) is type(()):
                    lft, rt = res
                    content_parts.extend(lft)
                    right_panel_parts.extend(rt)

        self.window.update_text(self.content_id, TermText("\n".join(content_parts)))
        self.window.update_text(self.right_panel_id, TermText("\n".join(right_panel_parts)))
        
        if self.nav_state.focus_depth == 2 and len(right_panel_parts) > 2:
            self.window.show_txt(self.seperator_column_id)
        else:
            self.window.hide_txt(self.seperator_column_id)

        if self.nav_state.focus_depth == 2:
            self.show_content()
        else:
            self.hide_content()

    def show_content(self):
        self.window.show_txt(self.content_bg_id)
        self.window.show_txt(self.content_id)
        self.window.show_txt(self.right_panel_id)

    def hide_content(self):
        self.window.hide_txt(self.content_bg_id)
        self.window.hide_txt(self.content_id)
        self.window.hide_txt(self.right_panel_id)

    def _render_footer(self):
        self.window.update_text(self.footer_id, TermText(self._footer_text()))

    def _render_all(self):
        if self.nav_state.focus_depth == 2:
            self._sync_navigation()
            self._render_content()
        else:
            self._render_content()    
            self._sync_navigation()
        
        if self.exit_confirm.is_open:
            self.window.show_txt(self.exit_confirm_id)
            self._render_exit_confirm()
        else:
            self.window.hide_txt(self.exit_confirm_id)
        self._render_footer()

        self.window.paint()

    def _teardown_windows(self):
        self.window.remove_all()
        
        self.window.paint()
        
    def _footer_text(self) -> str:
        if self.exit_confirm.is_open:
            return "Arrows U/D: Navigate   Enter: Select   Esc: Cancel"
        if self.edit_confirm.is_open:
            return "Arrows U/D: Change choice   Enter: Select   Esc: Cancel"
        if self.nav_state.focus_depth == 0:
            return "Arrows L/R: Switch Tab   Enter: Side Nav   Esc: Back/Quit Options   q: Exit"
        if self.nav_state.focus_depth == 1:
            return (
                "Arrows U/D: Switch Section   Enter: Content   Esc: Top Tabs   q: Exit"
            )
        if self.nav_state.focus_depth == 2:
            page = self.page_router.get_page()
            if page is not None:
                return page.get_footer()
        return ""
