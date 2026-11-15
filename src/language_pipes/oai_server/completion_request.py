from language_pipes.oai_server.chat import ChatMessage
from language_pipes.oai_server.oai_cache import CacheOptions
from language_pipes.oai_server.oai_cache import parse_cache_options

class ChatCompletionRequest:
    model: str
    stream: bool
    messages: list[ChatMessage]
    max_completion_tokens: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    presence_penalty: float
    cache_options: CacheOptions
    # `stream_options.include_usage`: the only OpenAI-sanctioned way to get a
    # usage block out of a chat-completion stream.
    include_usage: bool

    def __init__(
            self, 
            model: str, 
            stream: bool,
            max_completion_tokens: int,
            messages: list[ChatMessage],
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            presence_penalty: float = 0.0,
            cache_options: CacheOptions | None = None,
            include_usage: bool = False
        ):
        self.model = model
        self.stream = stream
        self.max_completion_tokens = max_completion_tokens
        self.messages = messages
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.cache_options = cache_options if cache_options is not None else CacheOptions()
        self.include_usage = include_usage

    def to_json(self):
        return {
            'model': self.model,
            'stream': self.stream,
            'max_completion_tokens': self.max_completion_tokens,
            'messages': [m.to_json() for m in self.messages],
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'min_p': self.min_p,
            'presence_penalty': self.presence_penalty
        }
    
    @staticmethod
    def from_dict(data):
        max_completion_tokens = 1000
        if "max_tokens" in data:
            max_completion_tokens = data['max_tokens']
        if "max_completion_tokens" in data:
            max_completion_tokens = data['max_completion_tokens']
        
        stream = data.get('stream', False)
        temperature = data.get('temperature', 1.0)
        top_k = data.get('top_k', 0)
        top_p = data.get('top_p', 1.0)
        min_p = data.get('min_p', 0.0)
        presence_penalty = data.get('presence_penalty', 0.0)
        cache_options = parse_cache_options(data)
        stream_options = data.get('stream_options')
        include_usage = bool(stream_options.get('include_usage')) if isinstance(stream_options, dict) else False
        return ChatCompletionRequest(data['model'], stream, max_completion_tokens, [ChatMessage.from_dict(m) for m in data['messages']], temperature, top_k, top_p, min_p, presence_penalty, cache_options, include_usage)
