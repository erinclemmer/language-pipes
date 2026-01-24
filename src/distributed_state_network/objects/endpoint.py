from typing import Dict, Optional
from dataclasses import dataclass
from distributed_state_network.util.byte_helper import ByteHelper

@dataclass(frozen=True)
class Endpoint:
    address: Optional[str]
    port: int

    def to_string(self):
        return f"{self.address}:{self.port}"

    def to_bytes(self):
        bts = ByteHelper()
        if self.address is not None:
            bts.write_string(self.address)
        else:
            bts.write_string("")
        bts.write_int(self.port)

        return bts.get_bytes()

    def to_json(self):
        return {
            "address": self.address,
            "port": self.port
        }

    @staticmethod
    def from_bytes(data: bytes):
        bts = ByteHelper(data)
        address = bts.read_string()
        if address == "":
            address = None
        port = bts.read_int()

        return Endpoint(address, port)

    @staticmethod
    def from_json(data: Dict) -> 'Endpoint':
        return Endpoint(data['address'], data['port'])
