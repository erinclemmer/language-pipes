from typing import Callable

from language_pipes.tui.components.jobs_active import JobsActive
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.components.dashboard import Dashboard
from language_pipes.tui.components.jobs_server import JobsServer
from language_pipes.tui.components.models_hosted import ModelsHosted
from language_pipes.tui.components.network_peers import NetworkPeers
from language_pipes.tui.components.network_status import NetworkStatus
from language_pipes.tui.components.pipes_complete import PipesComplete
from language_pipes.tui.components.pipes_connected import PipesConnected
from language_pipes.tui.components.pipes_incomplete import PipesIncomplete
from language_pipes.tui.components.models_installed import ModelsInstalled
from language_pipes.tui.components.network_form.network_form import NetworkForm

class PageRouter:
    loader: ContentLoader
    confirm: Confirm
    nav: NavState

    network_status: NetworkStatus
    network_peers: NetworkPeers
    network_form: NetworkForm

    models_installed: ModelsInstalled
    models_hosted: ModelsHosted

    dashboard: Dashboard
    pipes_connected: PipesConnected

    def __init__(
        self,
        loader: ContentLoader,
        confirm: Confirm,
        nav: NavState,
        state: FrameState,
        change_nav: Callable,
    ):
        self.nav = nav
        self.dashboard = Dashboard(loader, self.exit_page, self.is_focused, change_nav)
        self.network_status = NetworkStatus(loader, self.exit_page, self.is_focused)
        self.network_peers = NetworkPeers(loader, self.exit_page, self.is_focused)
        self.network_form = NetworkForm(
            loader, state, confirm, change_nav, self.exit_page, self.is_focused
        )
        self.models_installed = ModelsInstalled(
            loader, confirm, self.exit_page, self.is_focused
        )
        self.models_hosted = ModelsHosted(
            loader, confirm, self.exit_page, self.is_focused
        )
        self.pipes_connected = PipesConnected(loader, self.exit_page, self.is_focused)
        self.pipes_complete = PipesComplete(loader, self.exit_page, self.is_focused)
        self.pipes_incomplete = PipesIncomplete(loader, self.exit_page, self.is_focused)

        self.jobs_server = JobsServer(loader, confirm, self.exit_page, self.is_focused)
        self.jobs_active = JobsActive(loader, self.exit_page, self.is_focused)

    def is_focused(self):
        return self.nav.focus_depth == 2

    def exit_page(self):
        self.nav.focus_shallower()

    def get_page(self):
        tab = self.nav.active_tab()
        section = self.nav.active_side_option()
        if tab == "Home" and section == "Dashboard":
            return self.dashboard
        
        
        if tab == "Network" and section == "Status":
            return self.network_status
        if tab == "Network" and section == "Peers":
            return self.network_peers
        if tab == "Network" and section == "Configure":
            return self.network_form
        
        
        if tab == "Models" and section == "Installed":
            return self.models_installed
        if tab == "Models" and section == "Hosted":
            return self.models_hosted
        
        
        if tab == "Pipes" and section == "Connected":
            return self.pipes_connected
        if tab == "Pipes" and section == "Complete":
            return self.pipes_complete
        if tab == "Pipes" and section == "Incomplete":
            return self.pipes_incomplete
        
        
        if tab == "Jobs" and section == "Server":
            return self.jobs_server
        if tab == "Jobs" and section == "Active":
            return self.jobs_active
        
        return None
