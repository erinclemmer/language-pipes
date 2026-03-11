from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, List

from language_pipes.config import LayerModel
from language_pipes.distributed_state_network.objects.config import DSNodeConfig

class ProviderCall(Enum):
    get_network_config = "get_network_config"
    save_network_config = "save_network_config"
    list_models = "list_models"
    save_model_assignments = "save_model_assignments"

class ContentLoader:
    provider: object
    last_status_message: str
    last_status_level: str
    _cache: Dict[Tuple[str, str], Dict[str, Any]]

    def __init__(self, providers: Optional[object] = None) -> None:
        self.providers = providers
        self._cache = {}
        self.last_status_message = ""
        self.last_status_level = "info"

    def invalidate(self, tab: str, section: str) -> None:
        self._cache.pop((tab, section), None)

    def invalidate_all(self) -> None:
        self._cache.clear()

    def provider_available(self, name: ProviderCall) -> bool:
        return self._get_provider(name) is not None

    def call_provider(self, name: ProviderCall, **kwargs) -> Any:
        provider = self._get_provider(name)
        if provider is None:
            raise LookupError(f"Provider '{name}' unavailable")
        call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return provider(**call_kwargs)

    def get_network_config(self) -> DSNodeConfig:
        payload = self.call_provider(ProviderCall.get_network_config)
        if not isinstance(payload, DSNodeConfig):
            raise ValueError("get_network_config must return a DSNodeConfig")
        return payload

    def save_network_config(self, data: DSNodeConfig):
        self.call_provider(ProviderCall.save_network_config, data=data)
        self.invalidate("Network", "Configure")

    def list_models(self) -> List[LayerModel]:
        payload = self.call_provider(ProviderCall.list_models)
        if not isinstance(payload, List):
            raise ValueError("get_network_config must return a List of LayerModel")
        return payload

    def save_model_assignments(self, data: List[LayerModel]):
        self.call_provider(ProviderCall.save_model_assignments, data=data)
        self.invalidate("Models", "Assignments")

    def _get_provider(self, name: ProviderCall) -> Optional[Callable[..., Any]]:
        if self.providers is None:
            return None
        if isinstance(self.providers, Dict):
            provider = self.providers.get(name)
            return provider if callable(provider) else None
        provider = getattr(self.providers, name.value, None)
        return provider if callable(provider) else None
