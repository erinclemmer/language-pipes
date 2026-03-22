from language_pipes.tui.frame.nav_state import NavState
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.components.network_status import NetworkStatus


class PageRouter:
    loader: ContentLoader
    nav: NavState

    network_status: NetworkStatus

    def __init__(self, loader: ContentLoader, nav: NavState):
        self.network_status = NetworkStatus(loader)
        self.nav = nav
        self.network_status = NetworkStatus(loader)

    def get_page(self):
        tab = self.nav.active_tab()
        section = self.nav.active_side_option()
        if tab == "Network" and section == "Status":
            return self.network_status
        return None