"""
ContentLoader: resolves and caches view-state data for each (tab, section) pair.
"""
from typing import Any, Callable, Dict, Optional, Tuple

from language_pipes.tui.frame.placeholders import PLACEHOLDERS
from language_pipes.tui.frame import view_state as vs


class ContentLoader:
    """
    Owns the provider registry and the per-section view-state cache.

    Responsibilities:
    - Look up the right provider callable for a given (tab, section).
    - Call the provider and format the result into a view-state dict.
    - Cache results so repeated renders don't re-call providers.
    - Expose a status message/level after each load so callers can surface it.
    """

    last_status_message: str
    last_status_level: str

    def __init__(self, providers: Optional[object] = None) -> None:
        self.providers = providers
        self._cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.last_status_message = ""
        self.last_status_level = "info"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def invalidate(self, tab: str, section: str) -> None:
        self._cache.pop((tab, section), None)

    def invalidate_all(self) -> None:
        self._cache.clear()

    def provider_available(self, name: str) -> bool:
        return self._get_provider(name) is not None

    def call_provider(self, name: str, **kwargs) -> Any:
        provider = self._get_provider(name)
        if provider is None:
            raise LookupError(f"Provider '{name}' unavailable")
        call_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return provider(**call_kwargs)

    def get_network_config(self) -> Dict[str, Any]:
        payload = self.call_provider("get_network_config")
        if not isinstance(payload, dict):
            raise ValueError("get_network_config must return a dict")
        return payload

    def save_network_config(self, data: Dict[str, Any]) -> None:
        self.call_provider("save_network_config", data=data)
        self.invalidate("Network", "Configure")

    def save_model_assignments(self, data: Dict[str, Any]) -> None:
        self.call_provider("save_model_assignments", data=data)
        self.invalidate("Models", "Assignments")

    def set_validation_mode(self, enabled: bool) -> None:
        self.call_provider("set_validation_mode", enabled=enabled)
        self.invalidate("Models", "Validation")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_provider(self, name: str) -> Optional[Callable[..., Any]]:
        if self.providers is None:
            return None
        if isinstance(self.providers, dict):
            provider = self.providers.get(name)
            return provider if callable(provider) else None
        provider = getattr(self.providers, name, None)
        return provider if callable(provider) else None

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
