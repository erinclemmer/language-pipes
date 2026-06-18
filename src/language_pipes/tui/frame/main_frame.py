from pathlib import Path
from time import sleep
from threading import Lock, Thread
from typing import Optional, Tuple

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.alert import Alert
from ansinout import TuiWindow, read_key
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.frame.layout import FrameLayout
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.components.exit_confirm import ExitConfirm
from language_pipes.tui.frame.frame_key_handler import FrameKeyHandler
from language_pipes.tui.frame.page_router import PageRouter


class MainFrame:

    def __init__(
        self,
        size: Tuple[int, int],
        pos: Tuple[int, int],
        config_file: Path,
        auto_start: Optional[bool] = None
    ):
        self.window = TuiWindow(size, pos)
        self.shutdown = False
        self.state = FrameState()
        self.exit_confirm = ExitConfirm()
        self.alert = Alert()
        self.provider = ContentProvider(config_file, self.alert.create_alert)
        self.confirm = Confirm()
        self.nav = NavState()
        self.render_lock = Lock()
        self.page_router = PageRouter(
            self.provider, self.confirm, self.nav, self.state, self.change_nav
        )
        self.network_form = self.page_router.network_form
        self.layout = FrameLayout(
            self.window,
            self.nav,
            self.provider,
            self.exit_confirm,
            self.confirm,
            self.alert,
            self.state,
            self.page_router,
        )

        self.key_handler = FrameKeyHandler(self.layout, self.page_router)

        self.layout._init_layout(size, pos)
        self._init_view()
        self._render_all()
        if auto_start:
            self.auto_start()

    def auto_start(self):
        self.provider.network_provider.start_network()
        sleep(2)
        status = self.provider.network_provider.get_network_status()
        if status is not None and not status.running:
            return
        
        self.provider.job_provider.start_oai_server()
        for model in self.provider.model_provider.get_layer_models():
            self.provider.model_provider.load_layer_model(model)
        
        for model in self.provider.model_provider.get_end_models():
            self.provider.model_provider.load_end_model(model)

    def change_nav(self, tab: str, section: str):
        self.nav.set_tab(tab)
        self.layout._sync_navigation()
        self.nav.set_side_nav(section)

    def _init_view(self):
        self.change_nav("Home", "Dashboard")
        self.key_handler.activate_selection()

    def _render_all(self):
        with self.render_lock:
            self.layout._render_all()

    def frame_render_thread(self):
        while True:
            if self.shutdown:
                return
            self.provider.sync_provider_state()
            self._render_all()
            sleep(1)

    def shutdown_frame(self):
        self.layout._teardown_windows()
        self.shutdown = True
        self.provider.shutdown()

    def run(self) -> str:
        self.state.startup()
        self._render_all()
        Thread(target=self.frame_render_thread, args=()).start()
        while self.state.running:
            key, ch = read_key()
            self.key_handler.handle_key(key, ch)
            self._render_all()

        self.shutdown_frame()
        return "exit" if self.state.exit_tui else "menu"
