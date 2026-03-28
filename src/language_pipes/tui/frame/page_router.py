from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.components.network_status import NetworkStatus
from language_pipes.tui.components.network_peers import NetworkPeers
from language_pipes.tui.components.models_installed import ModelsInstalled

class PageRouter:
    loader: ContentLoader
    confirm: Confirm
    nav: NavState

    network_status: NetworkStatus
    network_peers: NetworkPeers

    models_installed: ModelsInstalled

    def __init__(self, loader: ContentLoader, confirm: Confirm, nav: NavState):
        self.nav = nav
        self.network_status = NetworkStatus(loader)
        self.network_peers = NetworkPeers(loader)
        self.models_installed = ModelsInstalled(loader, confirm)

    def get_page(self):
        tab = self.nav.active_tab()
        section = self.nav.active_side_option()
        if tab == "Network" and section == "Status":
            return self.network_status
        if tab == "Network" and section == "Peers":
            return self.network_peers
        if tab == "Models" and section == "Installed":
            return self.models_installed
        return None