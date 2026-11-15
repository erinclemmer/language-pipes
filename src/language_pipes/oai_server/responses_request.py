from language_pipes.oai_server.chat import ChatMessage
from language_pipes.oai_server.oai_tool_calls import ResponsesTool

# Missing from official schema to implement:
# https://developers.openai.com/api/reference/resources/responses/methods/create

class ChatResponsesRequest:
    conversation_id: str | None
    model: str
    stream: bool
    input: Any
    instructions: str | None
    messages: list[ChatMessage]
    max_output_tokens: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    presence_penalty: float
    tools: list[ResponsesTool]
    tool_choice: Any
    parallel_tool_calls: bool
    cache_options: CacheOptions

    def __init__(
            self,
            conversation_id: str | None,
            model: str,
            stream: bool,
            response_input: Any,
            instructions: str | None,
            max_output_tokens: int,
            messages: list[ChatMessage],
            cache_options: CacheOptions,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            min_p: float = 0.0,
            presence_penalty: float = 0.0,
            tools: list[ResponsesTool] | None = None,
            tool_choice: Any = None,
            parallel_tool_calls: bool = False
        ):
        self.conversation_id = conversation_id
        self.model = model
        self.stream = stream
        self.input = response_input
        self.instructions = instructions
        self.max_output_tokens = max_output_tokens
        self.messages = messages
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.tools = tools if tools is not None else []
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.cache_options = cache_options

    @staticmethod
    def from_dict(data: dict):
        max_output_tokens: int = 1000
        if "max_tokens" in data:
            max_output_tokens = data['max_tokens']
        if "max_completion_tokens" in data:
            max_output_tokens = data['max_completion_tokens']
        if "max_output_tokens" in data:
            max_output_tokens = data['max_output_tokens']

        if type(max_output_tokens) != type(0):
            raise ValueError('max_completion_tokens must be an integer')

        conversation = data.get('conversation', None)
        stream = data.get('stream', False)
        temperature = data.get('temperature', 1.0)
        top_k = data.get('top_k', 0)
        top_p = data.get('top_p', 1.0)
        min_p = data.get('min_p', 0.0)
        presence_penalty = data.get('presence_penalty', 0.0)
        instructions = data.get('instructions')

        conversation_id: str | None = None
        if conversation is not None:
            if type(conversation) == type(''):
                conversation_id = conversation
            if type(conversation) == type({ }) and 'id' in conversation:
                conversation_id = conversation['id'] # pyright: ignore[reportArgumentType]

        tools: list[ResponsesTool] = []
        tool_choice = data.get('tool_choice')
        parallel_tool_calls = bool(data.get('parallel_tool_calls', False))
        if data.get('tools') is not None:
            tools = parse_tool_definitions(data['tools'])
            validate_tool_choice(tool_choice, tools)

        messages, sources = _response_input_to_messages(data['input'])
        # Everything inserted here goes ahead of the input, so it shifts every
        # breakpoint index by the same amount.
        inserted = 0
        if instructions is not None:
            messages.insert(0, ChatMessage(ChatRole.SYSTEM, str(instructions)))
            inserted += 1
        if len(tools) > 0:
            # Inject tool schemas as a system-level instruction so the model
            # sees the available tools and the expected JSON output format.
            tool_message = ChatMessage(ChatRole.SYSTEM, build_tool_instructions(tools, tool_choice))
            insert_at = 1 if instructions is not None else 0
            messages.insert(insert_at, tool_message)
            inserted += 1
        if len(messages) == 0:
            raise ValueError("input must contain at least one text message")

        cache_options = parse_cache_options(data)
        cache_options.breakpoints = _breakpoints_to_message_indices(
            cache_options.breakpoints, sources, inserted
        )
        return ResponsesRequest(
            conversation_id=conversation_id,
            model=data['model'], 
            stream=stream, 
            input=data['input'], 
            instructions=instructions, 
            max_output_tokens=max_output_tokens, 
            messages=messages, 
            cache_options=cache_options, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p, 
            min_p=min_p, 
            presence_penalty=presence_penalty, 
            tools=tools, 
            tool_choice=tool_choice, 
            parallel_tool_calls=parallel_tool_calls
        )
