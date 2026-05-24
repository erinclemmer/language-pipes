from typing import Callable

from language_pipes.content_provider.content_provider import ContentProvider
from language_pipes.tui.components.home_activity import HomeActivity
from language_pipes.tui.components.jobs_active import JobsActive
from language_pipes.tui.components.models_end import ModelsEndModels
from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.frame.frame_state import FrameState
from language_pipes.tui.components.home_dashboard import Dashboard
from language_pipes.tui.components.jobs_server import JobsServer
from language_pipes.tui.components.models_layers import ModelsLayerModels
from language_pipes.tui.components.network_peers import NetworkPeers
from language_pipes.tui.components.network_status import NetworkStatus
from language_pipes.tui.components.pipes_complete import PipesComplete
from language_pipes.tui.components.pipes_connected import PipesConnected
from language_pipes.tui.components.pipes_incomplete import PipesIncomplete
from language_pipes.tui.components.models_installed import ModelsInstalled
from language_pipes.tui.components.network_form.network_form import NetworkForm

class PageRouter:
    provider: ContentProvider
    confirm: Confirm
    nav: NavState

    network_status: NetworkStatus
    network_peers: NetworkPeers
    network_form: NetworkForm

    models_installed: ModelsInstalled
    models_layer_models: ModelsLayerModels

    dashboard: Dashboard
    pipes_connected: PipesConnected

    def __init__(
        self,
        provider: ContentProvider,
        confirm: Confirm,
        nav: NavState,
        state: FrameState,
        change_nav: Callable,
    ):
        self.nav = nav
        self.dashboard = Dashboard(provider, self.exit_page, change_nav)
        self.home_activity = HomeActivity(provider, self.exit_page, change_nav)
        self.network_status = NetworkStatus(provider, self.exit_page)
        self.network_peers = NetworkPeers(provider, self.exit_page)
        self.network_form = NetworkForm(
            provider, state, confirm, change_nav, self.exit_page
        )
        self.models_installed = ModelsInstalled(
            provider, confirm, self.exit_page
        )
        self.models_layer_models = ModelsLayerModels(
            provider, confirm, self.exit_page
        )
        self.models_end_models = ModelsEndModels(
            provider, confirm, self.exit_page
        )
        self.pipes_connected = PipesConnected(provider, self.exit_page)
        self.pipes_complete = PipesComplete(provider, self.exit_page)
        self.pipes_incomplete = PipesIncomplete(provider, self.exit_page)

        self.jobs_server = JobsServer(provider, confirm, self.exit_page)
        self.jobs_active = JobsActive(provider, self.exit_page)

    def exit_page(self):
        self.nav.focus_shallower()

    def get_page(self):
        tab = self.nav.active_tab()
        section = self.nav.active_side_option()
        if tab == "Home" and section == "Dashboard":
            return self.dashboard
        if tab == "Home" and section == "Activity":
            return self.home_activity
        
        
        if tab == "Network" and section == "Status":
            return self.network_status
        if tab == "Network" and section == "Peers":
            return self.network_peers
        if tab == "Network" and section == "Configure":
            return self.network_form
        
        
        if tab == "Models" and section == "Installed":
            return self.models_installed
        if tab == "Models" and section == "Layer Models":
            return self.models_layer_models
        if tab == "Models" and section == "End Models":
            return self.models_end_models
        
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
