from language_pipes.request_for_model.rfm_packets import DoneSendingRFMPacket, IHaveModelRFMPacket, RFMPacketType, ReadyToReceiveRFMPacket, SendingDataRFMPacket, WhoHasRFMPacket
from language_pipes.util.byte_helper import ByteHelper

def read_packet(data: bytes) -> WhoHasRFMPacket | IHaveModelRFMPacket | ReadyToReceiveRFMPacket | SendingDataRFMPacket | DoneSendingRFMPacket:
    bts = ByteHelper(data)
    assert bts.read_int() == 1
    t = RFMPacketType(bts.read_int())
    if t == RFMPacketType.WHO_HAS_MODEL:
        return WhoHasRFMPacket(data)
    if t == RFMPacketType.I_HAVE_MODEL:
        return IHaveModelRFMPacket(data)
    if t == RFMPacketType.READY_TO_RECEIVE:
        return ReadyToReceiveRFMPacket(data)
    if t == RFMPacketType.SENDING_DATA:
        return SendingDataRFMPacket(data)
    return DoneSendingRFMPacket(data)

def assert_fn(test: bool, assertion_statement: str):
    if not test:
        raise Exception(assertion_statement)
