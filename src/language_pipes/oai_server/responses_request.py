from language_pipes.oai_server.chat import ChatMessage, ChatRole

# Missing from official schema to implement:
# https://developers.openai.com/api/reference/resources/responses/methods/create

class ChatResponsesRequest:
    conversation_id: str | None
    input: list[ChatMessage]

    def __init__(
            self,
            conversation_id: str | None,

        ):
        self.conversation_id = conversation_id


    @staticmethod
    def from_dict(data: dict) -> 'ChatResponsesRequest':
        conversation_id = data.get('conversation', None)
        data_input = data.get('input', None)
        if conversation_id is not None and type(conversation_id) == type({ }):
            conversation_id = data.get('id', None)

        if conversation_id is not None and type(conversation_id) != type(''):
            raise ValueError('invalid conversation id')

        if data_input is None:
            raise ValueError('input is a required property')

        if type(data_input) != type('') and type(data_input) != type([]):
            raise ValueError('invalid input property')

        messages = []
        if type(data_input) == type(''):
            messages.append(ChatMessage(ChatRole.USER, str(data_input)))
        elif type(data_input) == type([]):
            for item in data_input:
                if type(item) != type({ }):
                    raise ValueError('invalid input property')
                messages.append(parse_input_item(item)) # pyright: ignore[reportArgumentType]


        return ChatResponsesRequest(
            conversation_id=conversation_id,
        )
