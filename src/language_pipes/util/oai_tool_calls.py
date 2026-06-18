import json
import re
from typing import Any, List, Optional, Tuple
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
# Reasoning models (Qwen, DeepSeek, ...) wrap chain-of-thought in <think> tags
# before the actual answer; strip those blocks so they do not break parsing.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

def _strip_reasoning(text: str) -> str:
    return _THINK_RE.sub("", text)

def split_reasoning(text: Optional[str]) -> Tuple[Optional[str], str]:
    """Split a completed model output into (reasoning, content).

    Reasoning is the concatenated text of any `<think>...</think>` blocks; the
    content is the remaining text with those blocks removed. Returns
    (None, content) when there is no reasoning block.
    """
    if not text:
        return None, text or ""
    matches = _THINK_RE.findall(text)
    reasoning = "\n".join(m.strip() for m in matches if m.strip()).strip()
    content = _strip_reasoning(text).strip()
    return (reasoning or None), content

class ReasoningStreamSplitter:
    """Incrementally split a streamed token sequence into reasoning and content.

    A leading `<think>...</think>` block is routed to the reasoning channel and
    everything after it to the content channel. Only a bounded lookahead (at
    most ``len(<think>/</think>) - 1`` characters) is held back so tags that
    span two token deltas are still detected; the answer text after `</think>`
    is released as soon as it is unambiguous.
    """

    def __init__(self):
        self._buf = ""
        # phase: "start" (deciding if output opens with <think>), "reasoning"
        # (inside the block), or "content" (after it / no block at all).
        self._phase = "start"

    @staticmethod
    def _safe_emit_len(buf: str, tag: str) -> int:
        """Length of `buf` safe to release without splitting a partial `tag`."""
        hold = min(len(buf), len(tag) - 1)
        while hold > 0:
            if tag.startswith(buf[-hold:]):
                return len(buf) - hold
            hold -= 1
        return len(buf)

    def feed(self, delta: Optional[str]) -> Tuple[str, str]:
        """Consume a token delta; return (reasoning_delta, content_delta)."""
        if delta:
            self._buf += delta
        reasoning_parts: List[str] = []
        content_parts: List[str] = []
        while True:
            if self._phase == "start":
                if len(self._buf) < len(_THINK_OPEN) and _THINK_OPEN.startswith(self._buf):
                    break  # could still become "<think>"; wait for more
                if self._buf.startswith(_THINK_OPEN):
                    self._buf = self._buf[len(_THINK_OPEN):]
                    self._phase = "reasoning"
                    continue
                self._phase = "content"
                continue
            if self._phase == "reasoning":
                idx = self._buf.find(_THINK_CLOSE)
                if idx != -1:
                    if idx > 0:
                        reasoning_parts.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(_THINK_CLOSE):]
                    self._phase = "content"
                    continue
                emit = self._safe_emit_len(self._buf, _THINK_CLOSE)
                if emit > 0:
                    reasoning_parts.append(self._buf[:emit])
                    self._buf = self._buf[emit:]
                break
            # content phase: release everything immediately
            if self._buf:
                content_parts.append(self._buf)
                self._buf = ""
            break
        return "".join(reasoning_parts), "".join(content_parts)

    def finalize(self) -> Tuple[str, str]:
        """Flush any held-back buffer at end of stream."""
        buf = self._buf
        self._buf = ""
        if self._phase == "reasoning":
            return buf, ""
        self._phase = "content"
        return "", buf

def _strip_fences(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped

def _try_load(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced top-level ``{...}`` substring, ignoring braces
    inside strings. Lets us recover a tool call when the model surrounds the
    JSON with prose (e.g. "Sure! {...}")."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def parse_tool_call(text: Optional[str], tools: List[ResponsesTool]) -> Optional[ParsedToolCall]:
    """Classify model output as a tool call or plain text.

    Returns a ParsedToolCall when the output is recognizable JSON naming one of
    the provided tools, otherwise None. Conservative by design to avoid
    misclassifying ordinary assistant text.
    """
    if not text or not tools:
        return None

    candidate = _strip_fences(_strip_reasoning(text))
    data = _try_load(candidate)
    if data is None:
        # The model may surround the JSON with prose or leftover reasoning;
        # recover the first balanced object and try again.
        obj = _extract_json_object(candidate)
        if obj is not None:
            data = _try_load(obj)

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
