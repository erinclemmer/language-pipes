from typing import Any, Callable, Dict, Optional, Tuple, List

from language_pipes.config import LayerModel
import language_pipes.tui.frame.view_state as vs
from language_pipes.tui.frame.placeholders import PLACEHOLDERS
from language_pipes.tui.frame.provider_calls import ProviderCall
from language_pipes.distributed_state_network.objects.config import DSNodeConfig

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

    def load(
        self,
        tab: str,
        section: str,
        *,
        update_status: bool,
        force: bool,
    ) -> Dict[str, Any]:
        """Return the view-state for (tab, section), using the cache unless *force* is True."""
        view_key = (tab, section)

        if not force and view_key in self._cache:
            return self._cache[view_key]

        provider_name, kwargs, formatter = vs.section_provider_spec(tab, section)

        if provider_name is None:
            result = self._placeholder_view_state(tab, section)
            self._cache[view_key] = result
            if update_status:
                self.last_status_message = (
                    f"No provider mapping for {tab} -> {section}; showing guidance"
                )
                self.last_status_level = "info"
            return result

        provider = self._get_provider(provider_name)
        if provider is None:
            result = self._placeholder_view_state(tab, section)
            self._cache[view_key] = result
            if update_status:
                self.last_status_message = (
                    f"Provider '{provider_name}' unavailable for {tab} -> {section}; showing guidance"
                )
                self.last_status_level = "info"
            return result

        try:
            provider_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            payload = provider(**provider_kwargs)
            result = formatter(tab, section, payload)
        except Exception as ex:
            result = vs.error_view_state(
                f"Provider call failed for {tab} -> {section}: {ex}",
                "Next: Verify provider connectivity/configuration, then press r.",
            )

        self._cache[view_key] = result

        if update_status:
            state = result["state"]
            if state == "ok":
                self.last_status_message = f"Refreshed {tab} -> {section}"
                self.last_status_level = "info"
            elif state == "empty":
                self.last_status_message = f"No data for {tab} -> {section} yet"
                self.last_status_level = "info"
            elif state == "error":
                self.last_status_message = f"Refresh failed for {tab} -> {section}; check provider"
                self.last_status_level = "error"
            else:
                self.last_status_message = f"Showing guidance for {tab} -> {section}"
                self.last_status_level = "info"

        return result

    def _placeholder_view_state(self, tab: str, section: str) -> Dict[str, Any]:
        placeholders: Dict = PLACEHOLDERS
        summary, hint, level = placeholders.get(tab, {}).get(
            section,
            (
                "No placeholder registered for this section.",
                "Next: Return to top tabs and choose a known section.",
                "warning",
            ),
        )
        return vs.build_view_state("placeholder", summary, [], hint, level)

    def invalidate(self, tab: str, section: str) -> None:
        self._cache.pop((tab, section), None)

    def invalidate_all(self) -> None:
        self._cache.clear()

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
