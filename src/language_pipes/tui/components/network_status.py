from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.content_loader import ContentLoader
from language_pipes.tui.content_provider import RouterStatus
from language_pipes.tui.frame.provider_calls import ProviderCall

def network_status_on_key(key: PressedKey, loader: ContentLoader):
    if key == PressedKey.Enter:
        on_enter(loader)

def on_enter(loader: ContentLoader):
    status: RouterStatus = loader.call_provider(ProviderCall.get_network_status)
    if status is not None and status.running:
        loader.call_provider(ProviderCall.stop_network)
    else:
        loader.call_provider(ProviderCall.start_network)