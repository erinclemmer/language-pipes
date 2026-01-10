from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class NetworkConfig:
    provider: str
    settings: Dict[str, Any]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NetworkConfig":
        if "provider" in data and "settings" in data:
            return NetworkConfig(
                provider=data["provider"],
                settings=data["settings"],
            )
        return NetworkConfig(provider="dsn", settings=data)

    def get_node_id(self) -> Optional[str]:
        return self.settings.get("node_id")
