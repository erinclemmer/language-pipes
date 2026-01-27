from typing import Dict, List, Optional
from dataclasses import dataclass

from distributed_state_network.objects.endpoint import Endpoint

@dataclass(frozen=True)
class DSNodeConfig:
    node_id: str
    credential_dir: str
    port: int
    network_ip: Optional[str]
    aes_key: Optional[str]
    bootstrap_nodes: List[Endpoint]

    @staticmethod
    def from_dict(data: Dict) -> 'DSNodeConfig':
        return DSNodeConfig(
            data["node_id"], 
            data["credential_dir"] if "credential_dir" in data else "credentials",
            data["port"],
            data["network_ip"] if "network_ip" in data else None, 
            data["aes_key"] if "aes_key" in data else None, 
            [Endpoint.from_json(e) for e in data["bootstrap_nodes"]]
        )

    def to_string(self) -> str:
        lines = []
        
        lines.append("")
        lines.append("=" * 60)
        lines.append("  DSNode Configuration Details")
        lines.append("=" * 60)
        
        # Core settings
        lines.append("")
        lines.append("--- Node Settings ---")
        lines.append(f"  {'Node ID:':<18} {self.node_id}")
        lines.append(f"  {'Credential Dir:':<18} {self.credential_dir}")
        lines.append(f"  {'Port:':<18} {self.port}")
        
        # Network settings
        lines.append("")
        lines.append("--- Network Settings ---")
        if self.network_ip:
            lines.append(f"  {'Network IP:':<18} {self.network_ip}")
        else:
            lines.append("  Network IP:         Not configured")
        
        if self.aes_key:
            # Show truncated key for security
            display_key = self.aes_key[:8] + "..." + self.aes_key[-8:] if len(self.aes_key) > 20 else self.aes_key
            lines.append(f"  {'AES Key:':<18} {display_key}")
        else:
            lines.append("  Network Encryption: Disabled")
        
        # Bootstrap nodes
        lines.append("")
        lines.append(f"--- Bootstrap Nodes ({len(self.bootstrap_nodes)}) ---")
        if self.bootstrap_nodes:
            for i, endpoint in enumerate(self.bootstrap_nodes):
                lines.append(f"  Node #{i+1}:          {endpoint.address}:{endpoint.port}")
        else:
            lines.append("  No bootstrap nodes configured (standalone/first node)")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
