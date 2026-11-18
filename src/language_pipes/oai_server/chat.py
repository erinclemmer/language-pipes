from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from language_pipes.oai_server.oai_cache import CacheMode

class ChatRole(Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'
    DEVELOPER = 'developer'

class ChatPhase(Enum):
    COMMENTARY = 'commentary'
    FINAL = 'final_answer'

class ChatType(Enum):
    MESSAGE = 'message'

@dataclass
class ChatBreakpoint:
    mode: CacheMode
    ttl_minutes: int | None

    def to_dict(self) -> dict[str, Any]:
        o: dict[str, Any] = {
            "mode": str(self.mode)
        }

        if self.ttl_minutes is not None:
            o["ttl_minutes"] = self.ttl_minutes

        return o

    @staticmethod
    def from_dict(data: dict) -> 'ChatBreakpoint':
        return ChatBreakpoint(
            CacheMode(data['mode']),
            int(data['ttl_minutes'])
        )

    @staticmethod
    def from_request(data: dict) -> 'ChatBreakpoint':
        data_mode = data.get('mode', None)
        data_ttl = data.get('ttl', None)

        if data_mode is None:
            data_mode = "explicit"

        try:
            mode = CacheMode(data_mode)
        except:
            raise ValueError('breakpoint: mode is not valid')

        ttl_minutes = None
        if data_ttl is not None:
            try:
                ttl_minutes = int(data_ttl[0:-1])
            except:
                raise ValueError('breakpoint: can\'t parse ttl')

        return ChatBreakpoint(mode, ttl_minutes)

class ChatContent:
    text: str
    type: str
    prompt_cache_breakpoint: ChatBreakpoint | None

    def __init__(
        self,
        text: str,
        content_type: str,
        prompt_cache_breakpoint: ChatBreakpoint | None = None
    ):
        self.text = text
        self.type = content_type
        self.prompt_cache_breakpoint = prompt_cache_breakpoint

    def to_dict(self) -> dict[str, Any]:
        o: dict[str, Any] = {
            "text": self.text,
            "type": self.type,
        }
        if self.prompt_cache_breakpoint is not None:
            o["prompt_cache_breakpoint"] = self.prompt_cache_breakpoint.to_dict()

        return o

    @staticmethod
    def from_dict(data: dict) -> 'ChatContent':
        breakpoint = None
        if "prompt_cache_breakpoint" in data:
            breakpoint = ChatBreakpoint.from_dict(data["prompt_cache_breakpoint"])
        
        return ChatContent(
            data['text'],
            data['type'],
            breakpoint
        )

    @staticmethod
    def from_request(data: dict) -> 'ChatContent':
        data_text = data.get('text', None)
        data_type = data.get('type', None)
        data_breakpoint = data.get('prompt_cache_breakpoint', None)

        if data_text is None:
            raise ValueError('content: text is required for content item')

        if data_type is None:
            data_type = "input_text"

        if type(data_text) != type(''):
            raise ValueError('content: invalid type for text')

        breakpoint = None
        if data_breakpoint is not None:
            breakpoint = ChatBreakpoint.from_request(data_breakpoint)

        return ChatContent(data_text, data_type, breakpoint)

class ChatMessage:
    role: ChatRole
    type: ChatType
    phase: ChatPhase | None
    content: list[ChatContent]

    def __init__(
        self, 
        role: ChatRole,
        chat_type: ChatType, 
        phase: ChatPhase | None,
        content: list[ChatContent]
    ):
        self.role = role
        self.type = chat_type
        self.phase = phase
        self.content = content

    def to_dict(self) -> dict[str, Any]:
        o = {
            'role': self.role.value,
            'type': self.type.value,
            'content': [c.to_dict() for c in self.content]
        }
        if self.phase is not None:
            o['phase'] = str(self.phase)

        return o
    
    @staticmethod
    def from_dict(data) -> "ChatMessage":
        data_content = data.get('content', None)
        data_role = data.get('role', None)
        data_phase = data.get('phase', None)
        data_type = data.get('type', None)

        if data_content is None:
            raise ValueError('input: content is a required property')

        if data_role is None:
            raise ValueError('input: role is a required property')

        if data_type is None:
            raise ValueError('input: type is a required property')

        if type(data_role) != type(''):
            raise ValueError('input: role must be a string')

        try:
            role = ChatRole(data_role)
        except:
            raise ValueError('input: invalid role value')

        phase = None
        if data_phase is not None:
            try:
                phase = ChatPhase(data_phase)
            except:
                raise ValueError('input: invalid phase value')

        try:
            chat_type = ChatType(data_type)
        except:
            raise ValueError('input: invalid type value')

        if type(data_content) != '' and type(data_content) != type([]):
            raise ValueError('input: content must either be string or array')

        if type(data_content) == type(''):
            return ChatMessage(role, chat_type, phase, [ChatContent(str(data_content), "input_text")])

        content = []
        if type(data_content) == type([ ]):
            for item in data_content:
                if type(item) != type({ }):
                    raise ValueError('input: content is invalid')
                content.append(ChatContent)

        return ChatMessage(
            role,
            chat_type,
            phase,
            content
        )