
from enum import Enum

from language_pipes.util.byte_helper import ByteHelper

class RFMPacketType(Enum):
    WHO_HAS_MODEL = 0
    I_HAVE_MODEL = 1
    READY_TO_RECEIVE = 2
    SENDING_DATA = 3
    DONE_SENDING = 4    

class RFMPacket:
    bts: ByteHelper
    req_type: RFMPacketType

    def __init__(self, data: bytes):
        self.bts = ByteHelper(data)
        protocol =  self.bts.read_int()
        assert protocol == 1
        self.req_type = RFMPacketType(self.bts.read_int())

    @staticmethod
    def create_base(t: RFMPacketType) -> ByteHelper:
        bts = ByteHelper()
        bts.write_int(1)
        bts.write_int(t.value)
        return bts

class WhoHasRFMPacket(RFMPacket):
    model_id: str

    def __init__(self, data: bytes):
        super().__init__(data)
        assert self.req_type == RFMPacketType.WHO_HAS_MODEL
        self.model_id = self.bts.read_string()

    @staticmethod
    def create(requesting_model: str) -> bytes:
        bts = RFMPacket.create_base(RFMPacketType.WHO_HAS_MODEL)
        bts.write_string(requesting_model)
        return bts.get_bytes()

class IHaveModelRFMPacket(RFMPacket):
    model_id: str

    def __init__(self, data: bytes):
        super().__init__(data)
        assert self.req_type == RFMPacketType.I_HAVE_MODEL
        self.model_id = self.bts.read_string()

    @staticmethod
    def create(model_id: str) -> bytes:
        base = RFMPacket.create_base(RFMPacketType.I_HAVE_MODEL)
        base.write_string(model_id)
        return base.get_bytes()

class ReadyToReceiveRFMPacket(RFMPacket):
    model_id: str

    def __init__(self, data: bytes):
        super().__init__(data)
        assert self.req_type == RFMPacketType.READY_TO_RECEIVE
        self.model_id = self.bts.read_string()

    @staticmethod
    def create(model_id: str) -> bytes:
        bts = RFMPacket.create_base(RFMPacketType.READY_TO_RECEIVE)
        bts.write_string(model_id)
        return bts.get_bytes()

class SendingDataRFMPacket(RFMPacket):
    model_id: str
    file_name: str
    packet_idx: int
    file_done: bool
    packet_data: bytes

    def __init__(self, data: bytes):
        super().__init__(data)
        assert self.req_type == RFMPacketType.SENDING_DATA
        self.model_id = self.bts.read_string()
        self.file_name = self.bts.read_string()
        self.packet_idx = self.bts.read_int()
        self.file_done = self.bts.read_int() == 1
        self.packet_data = self.bts.read_bytes()

    @staticmethod
    def create(model_id: str, file_name: str, packet_idx: int, file_done: bool, packet_data: bytes):
        bts = RFMPacket.create_base(RFMPacketType.SENDING_DATA)
        bts.write_string(model_id)
        bts.write_string(file_name)
        bts.write_int(packet_idx)
        bts.write_int(1 if file_done else 0)
        bts.write_bytes(packet_data)
        return bts.get_bytes()

class DoneSendingRFMPacket(RFMPacket):
    model_id: str
    
    def __init__(self, data: bytes):
        super().__init__(data)
        assert self.req_type == RFMPacketType.DONE_SENDING
        self.model_id = self.bts.read_string()

    @staticmethod
    def create(model_id: str):
        base = RFMPacket.create_base(RFMPacketType.DONE_SENDING)
        base.write_string(model_id)
        return base.get_bytes()
