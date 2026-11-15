from uuid import uuid4

from language_pipes.oai_server.chat import ChatMessage
from language_pipes.oai_server.oai_cache import CacheMode, CacheOptions

# Missing from official schema to implement:
# https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
# frequency_penalty
# logprobs
# logit_bias
# reasoning_effort
# response_format
# safety_identifier
# tools
# tool_choice
# top_logprobs
class ChatCompletionRequest:
    cid: str
    metadata: dict[str, str] | None
    model: str
    stream: bool
    messages: list[ChatMessage]
    max_completion_tokens: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    presence_penalty: float
    prompt_cache_key: str | None
    prompt_cache_mode: CacheMode | None
    prompt_cache_minutes: int | None
    include_usage: bool

    def __init__(
            self, 
            cid: str,
            metadata: dict[str, str] | None,
            model: str, 
            stream: bool,
            max_completion_tokens: int,
            messages: list[ChatMessage],
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            presence_penalty: float = 0.0,
            prompt_cache_key: str | None = None,
            prompt_cache_mode: CacheMode | None = None,
            prompt_cache_minutes: int | None = None,
            include_usage: bool = False
        ):
        self.id = cid
        self.model = model
        self.metadata = metadata
        self.stream = stream
        self.max_completion_tokens = max_completion_tokens
        self.messages = messages
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.prompt_cache_key = prompt_cache_key
        self.prompt_cache_mode = prompt_cache_mode
        self.prompt_cache_minutes = prompt_cache_minutes
        self.include_usage = include_usage

    def to_json(self):
        o = {
            'id': self.cid,
            'model': self.model,
            'stream': self.stream,
            'max_completion_tokens': self.max_completion_tokens,
            'messages': [m.to_json() for m in self.messages],
            'temperature': self.temperature,
            'top_k': self.top_k,
            'top_p': self.top_p,
            'min_p': self.min_p,
            'prompt_cache_key': self.prompt_cache_key,
            'presence_penalty': self.presence_penalty
        }

        if self.prompt_cache_mode is not None or self.prompt_cache_minutes is not None:
            o['prompt_cache_options'] = { }

        if self.prompt_cache_mode is not None:
            o['prompt_cache_options']['mode'] = self.prompt_cache_mode

        if self.prompt_cache_minutes is not None:
            o['prompt_cache_options']['ttl'] = str(self.prompt_cache_minutes) + "m"

        if self.include_usage is not None:
            o['stream_options'] = {
                'include_usage': self.include_usage
            }

        return o

    def cache_options(self) -> CacheOptions:
        co = CacheOptions()
        if self.prompt_cache_mode is not None:
            co.mode = self.prompt_cache_mode
        if self.prompt_cache_minutes is not None:
            co.ttl_seconds = self.prompt_cache_minutes * 60
        if self.prompt_cache_key is not None:
            co.prompt_cache_key = self.prompt_cache_key

        return co
    
    @staticmethod
    def from_dict(data):
        max_completion_tokens: int = 1000
        if "max_tokens" in data:
            max_completion_tokens = data['max_tokens']
        if "max_completion_tokens" in data:
            max_completion_tokens = data['max_completion_tokens']

        if type(max_completion_tokens) != type(0):
            raise ValueError('max_completion_tokens must be an integer')

        # TODO: Double check this with the completion endpoint schema
        # https://developers.openai.com/api/reference/resources/completions
        # https://developers.openai.com/api/docs/guides/prompt-caching
        cid = data.get('id', str(uuid4()))
        model = data.get('model', None)
        metadata = data.get('metadata', None)
        stream = data.get('stream', False)
        temperature = data.get('temperature', 1.0)
        top_k = data.get('top_k', 0)
        top_p = data.get('top_p', 1.0)
        min_p = data.get('min_p', 0.0)
        presence_penalty = data.get('presence_penalty', 0.0)
        prompt_cache_key = data.get("prompt_cache_key", None)
        prompt_cache_options = data.get('prompt_cache_options', None)
        stream_options = data.get('stream_options', None)

        if model is None:
            raise ValueError('model param must be supplied')

        if metadata is not None:
            if type(metadata) != type({ }):
                raise ValueError('metadata must be a dictionary of strings')
            
            for key in metadata.keys():
                if type(metadata[key]) != type(''):
                    raise ValueError('all metadata keys must be strings')

        prompt_cache_mode = None
        prompt_cache_minutes = None
        if prompt_cache_options is not None:
            if type(prompt_cache_options) != type({ }):
                raise ValueError('prompt_cache_options must be a dictionary of strings')

            if 'mode' in prompt_cache_options:
                mode = prompt_cache_options['mode']
                if type(mode) != type(''):
                    raise ValueError('prompt_cache_options.mode must be a string')
                
                if mode != 'implicit' and mode != 'explicit':
                    raise ValueError('prompt_cache_options.mode property must either be \'implicit\' or \'explicit\'')
                prompt_cache_mode = CacheMode(mode)

            if 'ttl' in prompt_cache_options:
                ttl: str = prompt_cache_options['ttl']
                if type('ttl') != type(''):
                    raise ValueError('prompt_cache_options.ttl must be a string like \'30m\'')
                if ttl[-1] != 'm':
                    raise ValueError('prompt_cache_options.ttl must be a string like \'30m\'')

                try:
                    prompt_cache_minutes = int(ttl[0:-1])
                except:
                    raise ValueError('prompt_cache_options.ttl must be a string like \'30m\'')

        include_usage = False
        if stream_options is not None:
            if type(stream_options) != type({ }):
                raise ValueError('stream_options must be a dictionary')

            if 'include_usage' in stream_options:
                iu = stream_options['include_usage']
                if type(iu) != type(False):
                    raise ValueError('stream_options.include_usage must be a boolean')
                include_usage = iu

        return ChatCompletionRequest(
            cid=cid,
            metadata=metadata,
            model=model, 
            stream=stream, 
            max_completion_tokens=max_completion_tokens, 
            messages=[ChatMessage.from_dict(m) for m in data['messages']], 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p, 
            min_p=min_p, 
            presence_penalty=presence_penalty,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_mode=prompt_cache_mode,
            prompt_cache_minutes=prompt_cache_minutes,
            include_usage=include_usage
        )
