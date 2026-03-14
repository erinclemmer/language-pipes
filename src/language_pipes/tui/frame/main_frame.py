from typing import Dict, List, Optional, Tuple

from language_pipes.tui.tui import TuiWindow
from language_pipes.tui.frame.editor import Editor
from language_pipes.tui.util.kb_utils import read_key
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.frame.layout import FrameLayout
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.components.network_form import NetworkForm
from language_pipes.tui.components.exit_confirm import ExitConfirm
from language_pipes.tui.frame.frame_key_handler import FrameKeyHandler

class MainFrame:
    TOP_HEADERS = ["Network", "Models", "Pipes", "Jobs", "Activity"]
    SIDE_OPTIONS_BY_TAB: Dict[str, List[str]] = {
        "Network": ["Status", "Peers", "Configure"],
        "Models": ["Installed", "Assignments", "Validation"],
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
        self.network_form = NetworkForm(self.loader, self.state, self.editor, self.confirm)
        self.layout = FrameLayout(self.window, self.nav, self.editor, self.loader, self.exit_confirm, self.confirm, self.state)
        self.key_handler = FrameKeyHandler(self.layout, self.network_form)

        self.layout._init_layout(size, pos)
        self._init_view()
        self.layout._render_all()

    def _init_view(self):
        self.nav.set_tab("Network")
        self.nav.set_side_nav(self.layout.side_nav, "Configure")
        self.key_handler.activate_selection()

    def run(self) -> str:
        self.state.startup()
        self.layout._render_all()
        while self.state.running:
            key, ch = read_key()
            self.key_handler.handle_key(key, ch)
            self.layout._render_all()

        self.layout._teardown_windows()
        return "exit" if self.state.exit_tui else "menu"
