import json
import re
from typing import Any, List, Optional
from uuid import uuid4

# Custom function tools for the Responses API. Hosted tools (web_search,
# file_search, computer_use, code_interpreter, ...) are intentionally not
# supported yet; only `type == "function"` tools are accepted.

class ResponsesTool:
    type: str
    name: str
    description: Optional[str]
    parameters: dict
    strict: Optional[bool]

    def __init__(self, name: str, description: Optional[str], parameters: dict, strict: Optional[bool] = None):
        self.type = "function"
        self.name = name
        self.description = description
        self.parameters = parameters
        self.strict = strict

    def to_schema(self) -> dict:
        schema: dict = {"name": self.name}
        if self.description is not None:
            schema["description"] = self.description
        schema["parameters"] = self.parameters
        return schema

class ParsedToolCall:
    call_id: str
    name: str
    arguments: str

    def __init__(self, call_id: str, name: str, arguments: str):
        self.call_id = call_id
        self.name = name
        self.arguments = arguments

def parse_tool_definitions(tools: Any) -> List[ResponsesTool]:
    """Parse and validate the request `tools` list.

    Raises ValueError for anything we cannot serve so the caller can return a
    400 with a clear message.
    """
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")

    parsed: List[ResponsesTool] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("each tool must be an object")

        tool_type = tool.get("type")
        if tool_type != "function":
            raise ValueError(
                f"unsupported tool type: {tool_type!r}; only 'function' tools are supported"
            )

        name = tool.get("name")
        if not isinstance(name, str) or name == "":
            raise ValueError("function tool requires a non-empty string 'name'")

        parameters = tool.get("parameters")
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ValueError("function tool 'parameters' must be an object")

        description = tool.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError("function tool 'description' must be a string")

        strict = tool.get("strict")
        parsed.append(ResponsesTool(name, description, parameters, strict))

    return parsed

def validate_tool_choice(tool_choice: Any, tools: List[ResponsesTool]) -> None:
    """Validate `tool_choice` against the parsed tools. Raises ValueError."""
    if tool_choice is None:
        return

    if isinstance(tool_choice, str):
        if tool_choice not in ("auto", "none", "required"):
            raise ValueError(
                f"invalid tool_choice: {tool_choice!r}; expected 'auto', 'none', or 'required'"
            )
        return

    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError("tool_choice object must have type 'function'")
        name = tool_choice.get("name")
        names = {t.name for t in tools}
        if name not in names:
            raise ValueError(f"tool_choice references unknown tool: {name!r}")
        return

    raise ValueError("tool_choice must be a string or an object")

def build_tool_instructions(tools: List[ResponsesTool], tool_choice: Any = None) -> str:
    """Build a model-agnostic instruction block describing the available tools.

    The model is asked to emit a single JSON object when it wants to call a
    tool, which `parse_tool_call` then recognizes.
    """
    schemas = [t.to_schema() for t in tools]

    if tool_choice == "none":
        lines = [
            "The following tools are available for reference, but you must NOT call "
            "any tool. Respond with text only.",
        ]
    else:
        lines = [
            "You may call one of the following tools. If you need a tool, respond "
            "ONLY with a JSON object in this exact format and nothing else:",
            '{"tool_call": {"name": "tool_name", "arguments": {}}}',
            "",
            "Do not wrap the JSON in markdown code fences. If you do not need a "
            "tool, respond normally with text.",
        ]
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            lines.append(f"You MUST call the tool named {tool_choice.get('name')!r}.")
        elif tool_choice == "required":
            lines.append("You MUST call one of the tools.")

    lines.append("")
    lines.append("Tools:")
    lines.append(json.dumps(schemas, indent=2))
    return "\n".join(lines)

def format_assistant_tool_call(name: Any, arguments: Any, call_id: Optional[str] = None) -> str:
    """Render a prior assistant `function_call` input item back into the same
    JSON shape `build_tool_instructions` asks the model to emit, so replayed
    continuations match the format the model was trained on in this request.

    `arguments` follows the Responses shape (a JSON string); it is parsed back
    into an object when possible so the replayed call reads naturally.
    """
    if isinstance(arguments, str):
        try:
            args_obj: Any = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            args_obj = arguments
    else:
        args_obj = arguments if arguments is not None else {}

    call: dict = {"name": name, "arguments": args_obj}
    if call_id is not None:
        call["call_id"] = call_id
    return json.dumps({"tool_call": call})

def format_tool_result(call_id: Optional[str], output: str) -> str:
    """Render a `function_call_output` item into a plain-text tool result that
    preserves the `call_id` so the model can match it to the prior call."""
    if call_id:
        return f"Tool result for call_id {call_id}:\n{output}"
    return f"Tool result:\n{output}"

_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```$", re.DOTALL)

def _strip_fences(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped

def parse_tool_call(text: Optional[str], tools: List[ResponsesTool]) -> Optional[ParsedToolCall]:
    """Classify model output as a tool call or plain text.

    Returns a ParsedToolCall when the output is recognizable JSON naming one of
    the provided tools, otherwise None. Conservative by design to avoid
    misclassifying ordinary assistant text.
    """
    if not text or not tools:
        return None

    candidate = _strip_fences(text)
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    call: Optional[dict] = None
    if isinstance(data.get("tool_call"), dict):
        call = data["tool_call"]
    elif "name" in data and ("arguments" in data or "parameters" in data):
        call = data

    if call is None:
        return None

    name = call.get("name")
    names = {t.name for t in tools}
    if not isinstance(name, str) or name not in names:
        return None

    args = call.get("arguments")
    if args is None:
        args = call.get("parameters")
    if isinstance(args, str):
        arguments = args
    else:
        arguments = json.dumps(args if args is not None else {})

    return ParsedToolCall(f"call_{uuid4().hex}", name, arguments)
