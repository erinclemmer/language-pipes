import json
import time
import threading
from typing import Any, Callable

from promise import Promise
from http.server import BaseHTTPRequestHandler

from language_pipes.jobs.job import Job, chat_usage, responses_usage

from language_pipes.oai_server.chat import ChatMessage, ChatRole
from language_pipes.oai_server.completion_request import ChatCompletionRequest
from language_pipes.oai_server.oai_cache import CacheOptions, parse_cache_options

from language_pipes.util.http import _connection_alive, _respond_json, _send_code, _send_sse_headers
from language_pipes.util.oai_chunks import send_complete, send_error, send_initial_chunk, send_keepalive, send_update_chunk

from language_pipes.oai_server.oai_tool_calls import (
    ReasoningStreamSplitter,
    format_assistant_tool_call,
    format_tool_result,
    parse_tool_call,
    split_reasoning,
)

SSE_KEEPALIVE_INTERVAL = 10.0
DISCONNECT_CHECK_INTERVAL = 1.0

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)

def _response_input_to_messages(response_input: Any) -> tuple[list[ChatMessage], list[int]]:
    """The chat messages an `input` maps to, and where each one came from.

    The second list holds, per message, the index of the `input` item that
    produced it. A breakpoint is marked on an item but has to become a *message*
    index for the chat template to be re-rendered up to it, and the two do not
    line up on their own: an item the mapping skips produces no message at all.
    """
    if isinstance(response_input, str):
        return [ChatMessage(ChatRole.USER, response_input)], [0]

    items = response_input if isinstance(response_input, list) else [response_input]
    messages: list[ChatMessage] = []
    sources: list[int] = []
    for index, item in enumerate(items):
        produced = _input_item_to_messages(item)
        messages.extend(produced)
        sources.extend([index] * len(produced))

    return messages, sources


def _input_item_to_messages(item: Any) -> list[ChatMessage]:
    """The chat messages one `input` item maps to; none for an item we skip."""
    if isinstance(item, str):
        return [ChatMessage(ChatRole.USER, item)]
    if not isinstance(item, dict):
        return []

    item_type = item.get("type")
    role = item.get("role")

    if item_type == "function_call":
        # A prior assistant tool call. Replay it as an assistant message so
        # the model has the context of what it asked for before it sees the
        # corresponding tool result.
        return [ChatMessage(
            ChatRole.ASSISTANT,
            format_assistant_tool_call(
                item.get("name"),
                item.get("arguments", ""),
                item.get("call_id")
            )
        )]
    if item_type == "function_call_output":
        output = _content_to_text(item.get("output", ""))
        return [ChatMessage(
            ChatRole.USER,
            format_tool_result(item.get("call_id"), output)
        )]
    if item_type == "message" or role is not None:
        content = _content_to_text(item.get("content", ""))
        return [ChatMessage.from_dict({
            "role": role or "user",
            "content": content
        })]
    return []


def _breakpoints_to_message_indices(
    marked_items: list[int],
    sources: list[int],
    offset: int
) -> list[int]:
    """Turn `input`-item breakpoints into indices of the final message list.

    A breakpoint names the end of an item, so the prefix ends on the last
    message that item produced; an item that produced none is dropped. `offset`
    covers the messages inserted ahead of the input - `instructions` and the
    tool schemas - which are part of every prefix and shift every index by the
    same amount.
    """
    indices = []
    for item in marked_items:
        positions = [i for i, source in enumerate(sources) if source == item]
        if len(positions) > 0:
            indices.append(positions[-1] + offset)
    return sorted(set(indices))

def _reasoning_item(job: Any, reasoning_text: str) -> dict:
    return {
        "id": f"rs-{job.job_id}",
        "type": "reasoning",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": reasoning_text}]
    }

def _response_json(job: Any, req: ResponsesRequest, created_at: float):
    reasoning_text, content = split_reasoning(job.result)
    tool_call = parse_tool_call(job.result, req.tools)

    response_output = []
    if reasoning_text:
        response_output.append(_reasoning_item(job, reasoning_text))

    if tool_call is not None:
        response_output.append({
            "id": f"fc-{job.job_id}",
            "type": "function_call",
            "status": "completed",
            "call_id": tool_call.call_id,
            "name": tool_call.name,
            "arguments": tool_call.arguments
        })
        response_output_text = ""
    else:
        response_output.append({
            "id": f"msg-{job.job_id}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": content,
                "annotations": []
            }]
        })
        response_output_text = content

    return {
        "id": f"resp-{job.job_id}",
        "object": "response",
        "created_at": int(created_at),
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": req.instructions,
        "max_output_tokens": req.max_output_tokens,
        "model": job.model_id,
        "output": response_output,
        "output_text": response_output_text,
        "usage": responses_usage(job)
    }

def _write_response_event(handler: BaseHTTPRequestHandler, event_type: str, data: dict):
    msg = {"type": event_type}
    msg.update(data)
    try:
        handler.wfile.write(b"data: " + json.dumps(msg).encode("utf-8") + b"\n\n")
        handler.wfile.flush()
    except Exception:
        return False
    return True

def _start_disconnect_watchdog(
    handler: BaseHTTPRequestHandler,
    job: Job,
    stream: bool,
    write_lock: threading.Lock,
    last_write: list[float],
) -> threading.Event:
    """Poll the client connection for the life of the job so a drop is caught
    even mid prompt-processing, where no per-token/per-chunk update() call
    happens to observe it. Doubles as the SSE keepalive heartbeat while
    streaming, so slow prefill / inter-token gaps don't trip client or proxy
    idle timeouts. Marks the job stale on detecting a drop; job_processor
    halts at the next checkpoint once Job.send_update() sees that.
    """
    stop_event = threading.Event()

    def _watch():
        while not stop_event.wait(DISCONNECT_CHECK_INTERVAL):
            if not _connection_alive(handler):
                job.stale = True
                return
            if stream and time.time() - last_write[0] >= SSE_KEEPALIVE_INTERVAL:
                with write_lock:
                    if not send_keepalive(handler):
                        job.stale = True
                        return
                    last_write[0] = time.time()

    threading.Thread(target=_watch, daemon=True).start()
    return stop_event

def oai_chat_complete(handler: BaseHTTPRequestHandler, complete_cb: Callable, data: dict, api_key: str):
    req = ChatCompletionRequest.from_dict(data)
    created_at = time.time()

    # Serialize every write to the SSE socket: the watchdog thread and the
    # token-update callbacks both write to handler.wfile concurrently.
    write_lock = threading.Lock()
    last_write = [time.time()]
    stop_watchdog = threading.Event()

    def start(job: Job):
        nonlocal stop_watchdog
        stop_watchdog = _start_disconnect_watchdog(handler, job, req.stream, write_lock, last_write)
        if not req.stream:
            return
        _send_sse_headers(handler)
        with write_lock:
            send_initial_chunk(job, created_at, handler)
            last_write[0] = time.time()

    def update(job: Job):
        if not req.stream:
            return _connection_alive(handler)
        with write_lock:
            ok = send_update_chunk(job, {
                "content": job.delta
            }, created_at, None, handler)
            last_write[0] = time.time()
        return ok

    def complete(job: Job):
        stop_watchdog.set()
        if type(job) is type('') and job == 'NO_PIPE':
            _respond_json(handler, { "error": "no pipe available" })
        elif type(job) is type('') and job == 'NO_ENDS':
            _respond_json(handler, { "error": "no model ends available" })
        elif type(job) is type('') and job == 'MAX_JOBS':
            _respond_json(handler, { "error": "maximum number of jobs reached" })
        elif job.cancel_reason is not None:
            # The pipe stopped being able to run this job (e.g. a model was
            # unloaded). Report it now rather than letting the caller wait.
            if req.stream:
                with write_lock:
                    send_error(job, job.cancel_reason, created_at, handler)
            else:
                _respond_json(handler, { "error": job.cancel_reason })
        else:
            if req.stream:
                with write_lock:
                    send_complete(
                        job, created_at, handler,
                        chat_usage(job) if req.include_usage else None
                    )
            else:
                _respond_json(handler, {
                    "id": f"chatcmpl-{job.job_id}",
                    "object": "chat.completion",
                    "created": int(created_at),
                    "model": job.model_id,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": job.result
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": chat_usage(job)
                })

    def promise_fn(resolve: Callable, _: Callable):
        complete_cb(api_key, req.model, req.messages, req.cache_options, req.max_completion_tokens, req.temperature, req.top_k, req.top_p, req.min_p, req.presence_penalty, start, update, resolve)
    job = Promise(promise_fn).get()
    complete(job)

def oai_responses_create(handler: BaseHTTPRequestHandler, complete_cb: Callable, data: dict, api_key: str):
    try:
        req = ResponsesRequest.from_dict(data)
    except ValueError as e:
        _send_code(400, handler, str(e))
        return

    created_at = time.time()
    # When tools are present the output cannot be classified as text vs. a tool
    # call until generation finishes, so we buffer (suppress live deltas) and
    # emit every item at completion. Plain text requests stream token-by-token;
    # a leading <think> reasoning block streams live to a reasoning item, which
    # is closed when the answer begins.
    buffering = len(req.tools) > 0
    splitter = ReasoningStreamSplitter()
    # Serialize every write to the SSE socket: the watchdog thread and the
    # token-update callbacks both write to handler.wfile concurrently.
    write_lock = threading.Lock()
    last_write = [time.time()]
    stop_watchdog = threading.Event()
    # Live (non-buffered) streaming state, mutated across start/update/complete.
    sstate = {
        "reasoning_id": None,
        "message_id": None,
        "reasoning_index": None,
        "message_index": None,
        "next_index": 0,
        "reasoning_acc": "",
        "reasoning_open": False,
        "reasoning_closed": False,
        "message_open": False,
        "content_acc": "",
    }

    def _open_reasoning():
        sstate["reasoning_index"] = sstate["next_index"]
        sstate["next_index"] += 1
        sstate["reasoning_open"] = True
        _write_response_event(handler, "response.output_item.added", {
            "output_index": sstate["reasoning_index"],
            "item": {"id": sstate["reasoning_id"], "type": "reasoning", "status": "in_progress", "summary": []}
        })

    def _reasoning_delta(delta: str):
        # Stream reasoning tokens live to the reasoning item.
        if not sstate["reasoning_open"]:
            _open_reasoning()
        sstate["reasoning_acc"] += delta
        return _write_response_event(handler, "response.reasoning_summary_text.delta", {
            "item_id": sstate["reasoning_id"],
            "output_index": sstate["reasoning_index"],
            "summary_index": 0,
            "delta": delta
        })

    def _close_reasoning():
        if not sstate["reasoning_open"] or sstate["reasoning_closed"]:
            return
        sstate["reasoning_closed"] = True
        text = sstate["reasoning_acc"]
        _write_response_event(handler, "response.reasoning_summary_text.done", {
            "item_id": sstate["reasoning_id"], "output_index": sstate["reasoning_index"],
            "summary_index": 0, "text": text
        })
        _write_response_event(handler, "response.output_item.done", {
            "output_index": sstate["reasoning_index"],
            "item": {
                "id": sstate["reasoning_id"], "type": "reasoning", "status": "completed",
                "summary": [{"type": "summary_text", "text": text}] if text else []
            }
        })

    def _open_message():
        sstate["message_index"] = sstate["next_index"]
        sstate["next_index"] += 1
        sstate["message_open"] = True
        _write_response_event(handler, "response.output_item.added", {
            "output_index": sstate["message_index"],
            "item": {
                "id": sstate["message_id"], "type": "message", "status": "in_progress",
                "role": "assistant", "content": []
            }
        })

    def _content_delta(delta: str):
        # The reasoning block (if any) is fully streamed before the answer; close
        # it before the first content token.
        if sstate["reasoning_open"] and not sstate["reasoning_closed"]:
            _close_reasoning()
        if not sstate["message_open"]:
            _open_message()
        sstate["content_acc"] += delta
        return _write_response_event(handler, "response.output_text.delta", {
            "item_id": sstate["message_id"],
            "output_index": sstate["message_index"],
            "content_index": 0,
            "delta": delta
        })

    def start(job: Job):
        nonlocal stop_watchdog
        stop_watchdog = _start_disconnect_watchdog(handler, job, req.stream, write_lock, last_write)
        sstate["reasoning_id"] = f"rs-{job.job_id}"
        sstate["message_id"] = f"msg-{job.job_id}"
        if not req.stream:
            return
        with write_lock:
            _send_sse_headers(handler)
            response = {
                "id": f"resp-{job.job_id}",
                "object": "response",
                "created_at": int(created_at),
                "status": "in_progress",
                "error": None,
                "incomplete_details": None,
                "instructions": req.instructions,
                "max_output_tokens": req.max_output_tokens,
                "model": job.model_id,
                "output": [],
                "output_text": ""
            }
            # output_item.added is deferred until the output type/ordering is known
            # (a reasoning item may precede the message/function_call item).
            result = _write_response_event(handler, "response.created", {"response": response})
            last_write[0] = time.time()
            return result

    def update(job: Job):
        if not req.stream or buffering:
            return _connection_alive(handler)
        with write_lock:
            reasoning_delta, content_delta = splitter.feed(job.delta)
            ok = True
            if reasoning_delta:
                ok = _reasoning_delta(reasoning_delta) and ok
            if content_delta:
                ok = _content_delta(content_delta) and ok
            last_write[0] = time.time()
        return ok

    def _complete_buffered(response: dict):
        # Nothing has streamed yet; emit each output item in full.
        for index, item in enumerate(response["output"]):
            item_type = item["type"]
            if item_type == "reasoning":
                text = item["summary"][0]["text"] if item["summary"] else ""
                _write_response_event(handler, "response.output_item.added", {
                    "output_index": index,
                    "item": {"id": item["id"], "type": "reasoning", "status": "in_progress", "summary": []}
                })
                _write_response_event(handler, "response.reasoning_summary_text.delta", {
                    "item_id": item["id"], "output_index": index, "summary_index": 0, "delta": text
                })
                _write_response_event(handler, "response.reasoning_summary_text.done", {
                    "item_id": item["id"], "output_index": index, "summary_index": 0, "text": text
                })
            elif item_type == "function_call":
                _write_response_event(handler, "response.output_item.added", {
                    "output_index": index,
                    "item": {
                        "id": item["id"], "type": "function_call", "status": "in_progress",
                        "call_id": item["call_id"], "name": item["name"], "arguments": ""
                    }
                })
                _write_response_event(handler, "response.function_call_arguments.delta", {
                    "item_id": item["id"], "output_index": index, "delta": item["arguments"]
                })
                _write_response_event(handler, "response.function_call_arguments.done", {
                    "item_id": item["id"], "output_index": index, "arguments": item["arguments"]
                })
            else:  # message
                text = item["content"][0]["text"]
                _write_response_event(handler, "response.output_item.added", {
                    "output_index": index,
                    "item": {
                        "id": item["id"], "type": "message", "status": "in_progress",
                        "role": "assistant", "content": []
                    }
                })
                _write_response_event(handler, "response.output_text.delta", {
                    "item_id": item["id"], "output_index": index, "content_index": 0, "delta": text
                })
                _write_response_event(handler, "response.output_text.done", {
                    "item_id": item["id"], "output_index": index, "content_index": 0, "text": text
                })
            _write_response_event(handler, "response.output_item.done", {
                "output_index": index, "item": item
            })

    def _complete_live(job: Job, response: dict):
        # Flush any held-back lookahead, then reconcile against the authoritative
        # split of the full result in case deltas did not sum to job.result.
        flush_reasoning, flush_content = splitter.finalize()
        if flush_reasoning:
            _reasoning_delta(flush_reasoning)
        if flush_content:
            _content_delta(flush_content)

        final_reasoning, final_content = split_reasoning(job.result)
        # Reconcile reasoning, then close it before any content (in case deltas
        # did not stream it, e.g. when the pipeline delivered no live updates).
        if final_reasoning:
            r_remainder = final_reasoning[len(sstate["reasoning_acc"]):] \
                if final_reasoning.startswith(sstate["reasoning_acc"]) else final_reasoning
            if r_remainder:
                _reasoning_delta(r_remainder)
        if sstate["reasoning_open"] and not sstate["reasoning_closed"]:
            _close_reasoning()

        c_remainder = final_content[len(sstate["content_acc"]):] \
            if final_content.startswith(sstate["content_acc"]) else final_content
        if c_remainder:
            _content_delta(c_remainder)
        if not sstate["message_open"]:
            _open_message()

        message_item = next(i for i in response["output"] if i["type"] == "message")
        _write_response_event(handler, "response.output_text.done", {
            "item_id": sstate["message_id"],
            "output_index": sstate["message_index"],
            "content_index": 0,
            "text": final_content
        })
        _write_response_event(handler, "response.output_item.done", {
            "output_index": sstate["message_index"], "item": message_item
        })

    def complete_stream_failed(job: Job):
        with write_lock:
            _write_response_event(handler, "response.failed", {
                "response": {
                    "id": f"resp-{job.job_id}",
                    "object": "response",
                    "created_at": int(created_at),
                    "status": "failed",
                    "model": job.model_id,
                    "error": {"code": "job_canceled", "message": job.cancel_reason},
                    "output": []
                }
            })
            try:
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
            except Exception:
                pass
            last_write[0] = time.time()

    def complete_stream(job: Job, response: dict):
        with write_lock:
            if buffering:
                _complete_buffered(response)
            else:
                _complete_live(job, response)
            _write_response_event(handler, "response.completed", {"response": response})
            try:
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
            except Exception:
                pass
            last_write[0] = time.time()

    def complete(job: Job):
        stop_watchdog.set()
        if type(job) is type('') and job == 'NO_PIPE':
            _respond_json(handler, { "error": "no pipe available" })
        elif type(job) is type('') and job == 'NO_ENDS':
            _respond_json(handler, { "error": "no model ends available" })
        elif type(job) is type('') and job == 'MAX_JOBS':
            _respond_json(handler, { "error": "maximum number of jobs reached" })
        elif job.cancel_reason is not None:
            # The pipe stopped being able to run this job (e.g. a model was
            # unloaded). Report it now rather than letting the caller wait.
            if req.stream:
                complete_stream_failed(job)
            else:
                _respond_json(handler, { "error": job.cancel_reason })
        else:
            response = _response_json(job, req, created_at)
            if req.stream:
                complete_stream(job, response)
            else:
                _respond_json(handler, response)

    def promise_fn(resolve: Callable, _: Callable):
        complete_cb(api_key, req.model, req.messages, req.cache_options, req.max_output_tokens, req.temperature, req.top_k, req.top_p, req.min_p, req.presence_penalty, start, update, resolve)
    job = Promise(promise_fn).get()
    complete(job)

def get_models(handler: BaseHTTPRequestHandler, get_models: Callable):
    models = get_models()
    try:  # noqa: SIM105
        _respond_json(handler, {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": ""
                } for m in models
            ]
        })
    except:  # noqa: E722
        pass 
