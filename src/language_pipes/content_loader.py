from typing import Any, Callable, Dict, Optional

from language_pipes.content_provider.provider_calls import ProviderCall

class ContentLoader:
    provider: object
    last_status_message: str
    last_status_level: str

    def __init__(self, providers: Optional[object] = None) -> None:
        self.providers = providers
        self.last_status_message = ""
        self.last_status_level = "info"

    def provider_available(self, name: ProviderCall) -> bool:
        return self._get_provider(name) is not None

    def call_provider(self, name: ProviderCall, data: Any = None) -> Any:
        provider = self._get_provider(name)
        if provider is None:
            raise LookupError(f"Provider '{name}' unavailable")
        if data is None:
            return provider()
        else:
            return provider(data)

    def _get_provider(self, name: ProviderCall) -> Optional[Callable[..., Any]]:
        if self.providers is None:
            return None
        if isinstance(self.providers, Dict):
            provider = self.providers.get(name)
            return provider if callable(provider) else None
        provider = getattr(self.providers, name.value, None)
        return provider if callable(provider) else None
