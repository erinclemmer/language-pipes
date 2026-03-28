from time import time, sleep
from threading import Thread
from typing import Dict, List, Optional, Tuple

from language_pipes.tui.tui import TermText, TuiWindow
from language_pipes.tui.frame.editor import Editor
from language_pipes.tui.util.kb_utils import read_key
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.frame.layout import FrameLayout
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.components.network_form.network_form import NetworkForm
from language_pipes.tui.components.exit_confirm import ExitConfirm
from language_pipes.tui.frame.frame_key_handler import FrameKeyHandler
from language_pipes.tui.frame.page_router import PageRouter

class MainFrame:
    TOP_HEADERS = ["Network", "Models", "Pipes", "Jobs", "Activity"]
    SIDE_OPTIONS_BY_TAB: Dict[str, List[str]] = {
        "Network": ["Status", "Peers", "Configure"],
        "Models": ["Status", "Configure", "Installed"],
        "Pipes": ["Overview", "Routes", "Configure"],
        "Jobs": ["Queue", "History", "Stats"],
        "Activity": ["Logs", "Events", "Metrics"],
    }

    def __init__(
        self,
        size: Tuple[int, int],
        pos: Tuple[int, int],
        providers: Optional[object] = None,
    ):
        self.window = TuiWindow(size, pos)

        self.editor = Editor()
        self.state = FrameState()
        self.exit_confirm = ExitConfirm()
        self.loader = ContentLoader(providers)
        self.confirm = Confirm()
        self.nav = NavState(self.TOP_HEADERS, self.SIDE_OPTIONS_BY_TAB)
        self.page_router = PageRouter(self.loader, self.confirm, self.nav)

        self.network_form = NetworkForm(self.loader, self.state, self.editor, self.confirm, self.change_nav)
        self.layout = FrameLayout(
            self.window, 
            self.nav, 
            self.editor, 
            self.loader, 
            self.exit_confirm, 
            self.confirm, 
            self.state,
            self.page_router
        )
        
        self.key_handler = FrameKeyHandler(self.layout, self.network_form, self.page_router)

        self.render_time_id = self.window.add_text(TermText(""), (0, 0))
        self.layout._init_layout(size, pos)
        self._init_view()
        self.layout._render_all()

    def change_nav(self, tab: str, section: str):
        self.nav.set_tab(tab)
        self.nav.set_side_nav(self.layout.side_nav, section)

    def _init_view(self):
        self.change_nav("Network", "Configure")
        self.key_handler.activate_selection()

    def frame_render_thread(self):
        while True:
            self.layout._render_all()
            sleep(1)

    def run(self) -> str:
        self.state.startup()
        self.layout._render_all()
        Thread(target=self.frame_render_thread, args=()).start()
        while self.state.running:
            key, ch = read_key()
            start_time = time()
            self.key_handler.handle_key(key, ch)
            self.layout._render_all()
            self.window.update_text(self.render_time_id, TermText(f"Render: {(time() - start_time) * 1000:.0f}ms"))
            self.window.paint()

        self.layout._teardown_windows()
        return "exit" if self.state.exit_tui else "menu"
