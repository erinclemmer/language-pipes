from enum import Enum
from typing import Optional

class ChatRole(Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'

def str_to_chat_role(role: str) -> Optional[ChatRole]:
    if role == 'system':
        return ChatRole.SYSTEM
    if role == 'assistant':
        return ChatRole.ASSISTANT
    if role == 'user':
        return ChatRole.USER
    return None

class ChatMessage:
    role: ChatRole
    content: str

    def __init__(self, role: ChatRole, content: str):
        self.role = role
        self.content = content

    def to_json(self):
        return {
            'role': self.role.value,
            'content': self.content
        }
    
    @staticmethod
    def from_dict(data) -> Optional["ChatMessage"]:
        role = str_to_chat_role(data['role'])
        if role is None:
            return None
        return ChatMessage(role, data['content'])
