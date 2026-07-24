import json
import os
import socket
import sys
import threading
import time
import unittest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.oai_server import OAIHttpServer
from language_pipes.util.chat import ChatRole
from language_pipes.util.oai import ResponsesRequest, _response_json
from language_pipes.util.oai_tool_calls import (
    ReasoningStreamSplitter,
    parse_tool_call,
    parse_tool_definitions,
    split_reasoning,
)


WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


class DummyJob:
    job_id = "job-1"
    model_id = "model-1"
    result = "Hello from Language Pipes"
    prompt_tokens = 4
    current_token = 3


class ToolJob(DummyJob):
    result = '{"tool_call": {"name": "get_weather", "arguments": {"city": "Chicago"}}}'


class ResponsesRequestTests(unittest.TestCase):
    def test_string_input_maps_to_user_message(self):
        req = ResponsesRequest.from_dict({
            "model": "model-1",
            "input": "Hello",
            "max_output_tokens": 25,
        })

        self.assertEqual(req.model, "model-1")
        self.assertEqual(req.max_output_tokens, 25)
        self.assertEqual(len(req.messages), 1)
        self.assertEqual(req.messages[0].role, ChatRole.USER)
        self.assertEqual(req.messages[0].content, "Hello")

    def test_instructions_and_message_items_map_to_chat_messages(self):
        req = ResponsesRequest.from_dict({
            "model": "model-1",
            "instructions": "Be concise",
            "input": [
                {"role": "user", "content": "Question"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Prior answer"}],
                },
            ],
        })

        self.assertEqual(len(req.messages), 3)
        self.assertEqual(req.messages[0].role, ChatRole.SYSTEM)
        self.assertEqual(req.messages[0].content, "Be concise")
        self.assertEqual(req.messages[1].role, ChatRole.USER)
        self.assertEqual(req.messages[1].content, "Question")
        self.assertEqual(req.messages[2].role, ChatRole.ASSISTANT)
        self.assertEqual(req.messages[2].content, "Prior answer")

    def test_response_json_contains_responses_api_output_shape(self):
        req = ResponsesRequest.from_dict({
            "model": "model-1",
            "instructions": "Be concise",
            "input": "Hello",
            "max_output_tokens": 25,
        })

        response = _response_json(DummyJob(), req, 1234.5)

        self.assertEqual(response["id"], "resp-job-1")
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["created_at"], 1234)
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["instructions"], "Be concise")
        self.assertEqual(response["max_output_tokens"], 25)
        self.assertEqual(response["output_text"], "Hello from Language Pipes")
        self.assertEqual(response["output"][0]["type"], "message")
        self.assertEqual(response["output"][0]["content"][0]["type"], "output_text")
        self.assertEqual(response["usage"]["input_tokens"], 4)
        self.assertEqual(response["usage"]["output_tokens"], 3)
        self.assertEqual(response["usage"]["total_tokens"], 7)

    def test_http_handler_routes_responses_endpoint(self):
        captured = {}

        def complete(api_key, model, messages, max_completion_tokens, temperature, top_k, top_p, min_p, presence_penalty, start, update, resolve):
            captured["model"] = model
            captured["messages"] = messages
            captured["max_completion_tokens"] = max_completion_tokens
            job = DummyJob()
            start(job)
            resolve(job)

        server = OAIHttpServer(
            port=5000, 
            api_keys=[], 
            complete=complete, 
            get_models=lambda: ["model-1"]
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            port = server.server_address[1]
            res = requests.post(f"http://127.0.0.1:{port}/v1/responses", json={
                "model": "model-1",
                "instructions": "Be concise",
                "input": "Hello",
                "max_output_tokens": 25,
            })

            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["object"], "response")
            self.assertEqual(data["output_text"], "Hello from Language Pipes")
            self.assertEqual(captured["model"], "model-1")
            self.assertEqual(captured["max_completion_tokens"], 25)
            self.assertEqual(captured["messages"][0].role, ChatRole.SYSTEM)
            self.assertEqual(captured["messages"][1].role, ChatRole.USER)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


class ToolDefinitionParsingTests(unittest.TestCase):
    def test_parses_valid_function_tool(self):
        tools = parse_tool_definitions([WEATHER_TOOL])
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].type, "function")
        self.assertEqual(tools[0].name, "get_weather")
        self.assertEqual(tools[0].description, "Get weather for a city")
        self.assertIn("city", tools[0].parameters["properties"])

    def test_rejects_hosted_tool(self):
        with self.assertRaises(ValueError):
            parse_tool_definitions([{"type": "web_search"}])

    def test_rejects_tools_not_a_list(self):
        with self.assertRaises(ValueError):
            parse_tool_definitions({"type": "function", "name": "x"})

    def test_rejects_function_tool_without_name(self):
        with self.assertRaises(ValueError):
            parse_tool_definitions([{"type": "function"}])

    def test_request_injects_tool_instructions_and_preserves_choice(self):
        req = ResponsesRequest.from_dict({
            "model": "model-1",
            "instructions": "Be concise",
            "input": "What is the weather in Chicago?",
            "tools": [WEATHER_TOOL],
            "tool_choice": {"type": "function", "name": "get_weather"},
        })

        self.assertEqual(len(req.tools), 1)
        self.assertEqual(req.tool_choice, {"type": "function", "name": "get_weather"})
        # instructions stay first, tool instructions injected right after
        self.assertEqual(req.messages[0].role, ChatRole.SYSTEM)
        self.assertEqual(req.messages[0].content, "Be concise")
        self.assertEqual(req.messages[1].role, ChatRole.SYSTEM)
        self.assertIn("get_weather", req.messages[1].content)
        self.assertEqual(req.messages[2].role, ChatRole.USER)

    def test_request_rejects_unknown_tool_choice(self):
        with self.assertRaises(ValueError):
            ResponsesRequest.from_dict({
                "model": "model-1",
                "input": "hi",
                "tools": [WEATHER_TOOL],
                "tool_choice": {"type": "function", "name": "does_not_exist"},
            })


class ToolCallParserTests(unittest.TestCase):
    def setUp(self):
        self.tools = parse_tool_definitions([WEATHER_TOOL])

    def test_parses_canonical_tool_call(self):
        call = parse_tool_call(
            '{"tool_call": {"name": "get_weather", "arguments": {"city": "Chicago"}}}',
            self.tools,
        )
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(json.loads(call.arguments), {"city": "Chicago"})
        self.assertTrue(call.call_id.startswith("call_"))

    def test_parses_fenced_json(self):
        call = parse_tool_call(
            '```json\n{"name": "get_weather", "arguments": {"city": "Chicago"}}\n```',
            self.tools,
        )
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "get_weather")

    def test_rejects_unknown_tool_name(self):
        call = parse_tool_call(
            '{"tool_call": {"name": "other_tool", "arguments": {}}}',
            self.tools,
        )
        self.assertIsNone(call)

    def test_plain_text_is_not_a_tool_call(self):
        self.assertIsNone(parse_tool_call("The weather is sunny.", self.tools))

    def test_malformed_json_returns_none(self):
        self.assertIsNone(parse_tool_call('{"tool_call": {', self.tools))

    def test_no_tools_returns_none(self):
        self.assertIsNone(parse_tool_call('{"tool_call": {"name": "x"}}', []))

    def test_parses_tool_call_after_think_block(self):
        text = (
            "<think>\nThe user wants the weather. I should call get_weather.\n</think>\n\n"
            '{"tool_call": {"name": "get_weather", "arguments": {"city": "Chicago"}}}'
        )
        call = parse_tool_call(text, self.tools)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(json.loads(call.arguments), {"city": "Chicago"})

    def test_parses_tool_call_with_surrounding_prose(self):
        text = (
            'Sure, let me check that for you.\n'
            '{"tool_call": {"name": "get_weather", "arguments": {"city": "Chicago"}}}'
        )
        call = parse_tool_call(text, self.tools)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.name, "get_weather")

    def test_think_block_with_braces_does_not_break_extraction(self):
        text = (
            "<think>maybe {city: chicago}?</think>\n"
            '{"tool_call": {"name": "get_weather", "arguments": {"city": "Chicago"}}}'
        )
        call = parse_tool_call(text, self.tools)
        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(json.loads(call.arguments), {"city": "Chicago"})

    def test_pure_reasoning_text_is_not_a_tool_call(self):
        text = "<think>I don't need a tool here.</think>\nThe weather is usually nice."
        self.assertIsNone(parse_tool_call(text, self.tools))


class ToolCallResponseShapeTests(unittest.TestCase):
    def _tool_request(self):
        return ResponsesRequest.from_dict({
            "model": "model-1",
            "input": "What is the weather in Chicago?",
            "tools": [WEATHER_TOOL],
        })

    def test_tool_call_output_returns_function_call_item(self):
        response = _response_json(ToolJob(), self._tool_request(), 1234.5)

        item = response["output"][0]
        self.assertEqual(item["type"], "function_call")
        self.assertEqual(item["status"], "completed")
        self.assertEqual(item["name"], "get_weather")
        self.assertEqual(json.loads(item["arguments"]), {"city": "Chicago"})
        self.assertTrue(item["call_id"].startswith("call_"))
        self.assertEqual(response["output_text"], "")

    def test_text_output_still_returns_message_item(self):
        response = _response_json(DummyJob(), self._tool_request(), 1234.5)
        self.assertEqual(response["output"][0]["type"], "message")
        self.assertEqual(response["output_text"], "Hello from Language Pipes")

    def test_usage_present_for_tool_call(self):
        response = _response_json(ToolJob(), self._tool_request(), 1234.5)
        self.assertEqual(response["usage"]["total_tokens"], 7)


class ToolResultContinuationTests(unittest.TestCase):
    def test_function_call_output_maps_to_tool_result_message(self):
        req = ResponsesRequest.from_dict({
            "model": "model-1",
            "input": [
                {"role": "user", "content": "What is the weather in Chicago?"},
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "Chicago"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": '{"temperature": 72}',
                },
            ],
            "tools": [WEATHER_TOOL],
        })

        # tool instruction block is injected first, then the transcript
        roles = [m.role for m in req.messages]
        self.assertEqual(roles[0], ChatRole.SYSTEM)
        self.assertEqual(req.messages[1].role, ChatRole.USER)

        assistant = next(m for m in req.messages if m.role == ChatRole.ASSISTANT)
        replay = json.loads(assistant.content)
        self.assertEqual(replay["tool_call"]["name"], "get_weather")
        self.assertEqual(replay["tool_call"]["arguments"], {"city": "Chicago"})
        self.assertEqual(replay["tool_call"]["call_id"], "call_abc")

        result_msg = req.messages[-1]
        self.assertEqual(result_msg.role, ChatRole.USER)
        self.assertIn("call_abc", result_msg.content)
        self.assertIn('{"temperature": 72}', result_msg.content)

    def test_function_call_output_without_call_id(self):
        req = ResponsesRequest.from_dict({
            "model": "model-1",
            "input": [
                {"type": "function_call_output", "output": "done"},
            ],
        })
        self.assertEqual(req.messages[-1].role, ChatRole.USER)
        self.assertIn("done", req.messages[-1].content)


def _parse_sse_events(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or not block.startswith("data: "):
            continue
        payload = block[len("data: "):]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


class StreamingTests(unittest.TestCase):
    def _serve(self, job):
        def complete(api_key, model, messages, max_completion_tokens, temperature, top_k, top_p, min_p, presence_penalty, start, update, resolve):
            start(job)
            resolve(job)

        server = OAIHttpServer(5000, [], complete, lambda: ["model-1"])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_streaming_text_emits_message_events(self):
        server, thread = self._serve(DummyJob())
        try:
            port = server.server_address[1]
            res = requests.post(f"http://127.0.0.1:{port}/v1/responses", json={
                "model": "model-1",
                "input": "Hello",
                "stream": True,
            })
            self.assertEqual(res.status_code, 200)
            events = _parse_sse_events(res.text)
            types = [e["type"] for e in events]
            self.assertEqual(types[0], "response.created")
            self.assertIn("response.output_item.done", types)
            completed = next(e for e in events if e["type"] == "response.completed")
            self.assertEqual(completed["response"]["output"][0]["type"], "message")
            self.assertEqual(completed["response"]["output_text"], "Hello from Language Pipes")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_streaming_tool_call_emits_function_call_events(self):
        server, thread = self._serve(ToolJob())
        try:
            port = server.server_address[1]
            res = requests.post(f"http://127.0.0.1:{port}/v1/responses", json={
                "model": "model-1",
                "input": "What is the weather in Chicago?",
                "tools": [WEATHER_TOOL],
                "stream": True,
            })
            self.assertEqual(res.status_code, 200)
            events = _parse_sse_events(res.text)
            types = [e["type"] for e in events]
            self.assertIn("response.function_call_arguments.delta", types)
            self.assertIn("response.function_call_arguments.done", types)

            added = next(e for e in events if e["type"] == "response.output_item.added")
            self.assertEqual(added["item"]["type"], "function_call")

            done = next(e for e in events if e["type"] == "response.function_call_arguments.done")
            self.assertEqual(json.loads(done["arguments"]), {"city": "Chicago"})

            completed = next(e for e in events if e["type"] == "response.completed")
            self.assertEqual(completed["response"]["output"][0]["type"], "function_call")
            self.assertEqual(completed["response"]["output_text"], "")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


class SplitReasoningTests(unittest.TestCase):
    def test_no_think_block(self):
        reasoning, content = split_reasoning("Just an answer.")
        self.assertIsNone(reasoning)
        self.assertEqual(content, "Just an answer.")

    def test_extracts_reasoning_and_content(self):
        reasoning, content = split_reasoning("<think>I should answer 42.</think>\nThe answer is 42.")
        self.assertEqual(reasoning, "I should answer 42.")
        self.assertEqual(content, "The answer is 42.")

    def test_reasoning_only(self):
        reasoning, content = split_reasoning("<think>hmm</think>")
        self.assertEqual(reasoning, "hmm")
        self.assertEqual(content, "")

    def test_empty(self):
        self.assertEqual(split_reasoning(""), (None, ""))


class ReasoningStreamSplitterTests(unittest.TestCase):
    def _feed_all(self, chunks):
        splitter = ReasoningStreamSplitter()
        reasoning, content = "", ""
        for chunk in chunks:
            r, c = splitter.feed(chunk)
            reasoning += r
            content += c
        r, c = splitter.finalize()
        return reasoning + r, content + c

    def test_plain_text_is_all_content(self):
        reasoning, content = self._feed_all(["Hello ", "world"])
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "Hello world")

    def test_reasoning_then_content_in_one_chunk(self):
        reasoning, content = self._feed_all(["<think>thinking</think>answer"])
        self.assertEqual(reasoning, "thinking")
        self.assertEqual(content, "answer")

    def test_tags_split_across_chunks(self):
        reasoning, content = self._feed_all(
            ["<th", "ink>think", "ing hard</th", "ink>The ans", "wer is 42."]
        )
        self.assertEqual(reasoning, "thinking hard")
        self.assertEqual(content, "The answer is 42.")

    def test_close_tag_split_at_every_boundary(self):
        text = "<think>abc</think>xyz"
        for i in range(1, len(text)):
            reasoning, content = self._feed_all([text[:i], text[i:]])
            self.assertEqual(reasoning, "abc", f"split at {i}")
            self.assertEqual(content, "xyz", f"split at {i}")

    def test_unterminated_think_flushes_as_reasoning(self):
        reasoning, content = self._feed_all(["<think>never closes"])
        self.assertEqual(reasoning, "never closes")
        self.assertEqual(content, "")

    def test_content_with_leading_angle_bracket_is_not_reasoning(self):
        reasoning, content = self._feed_all(["<answer>hi"])
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "<answer>hi")


class ReasoningJob(DummyJob):
    result = "<think>The user said hello.</think>Hello there!"
    delta = ""


class ReasoningResponseShapeTests(unittest.TestCase):
    def test_reasoning_item_precedes_message(self):
        req = ResponsesRequest.from_dict({"model": "model-1", "input": "Hi"})
        response = _response_json(ReasoningJob(), req, 1234.5)

        self.assertEqual(response["output"][0]["type"], "reasoning")
        self.assertEqual(
            response["output"][0]["summary"][0]["text"], "The user said hello."
        )
        self.assertEqual(response["output"][1]["type"], "message")
        # reasoning is stripped from the visible answer
        self.assertEqual(response["output_text"], "Hello there!")

    def test_reasoning_with_tool_call(self):
        req = ResponsesRequest.from_dict({
            "model": "model-1",
            "input": "What is the weather in Chicago?",
            "tools": [WEATHER_TOOL],
        })

        class ReasoningToolJob(DummyJob):
            result = (
                "<think>I need the weather tool.</think>"
                '{"tool_call": {"name": "get_weather", "arguments": {"city": "Chicago"}}}'
            )

        response = _response_json(ReasoningToolJob(), req, 1234.5)
        self.assertEqual(response["output"][0]["type"], "reasoning")
        self.assertEqual(response["output"][1]["type"], "function_call")
        self.assertEqual(response["output_text"], "")


class ReasoningStreamingTests(unittest.TestCase):
    def _serve_streaming(self, job, chunks):
        def complete(api_key, model, messages, max_completion_tokens, temperature, top_k, top_p, min_p, presence_penalty, start, update, resolve):
            start(job)
            for chunk in chunks:
                job.delta = chunk
                update(job)
            resolve(job)

        server = OAIHttpServer(5000, [], complete, lambda: ["model-1"])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_reasoning_streamed_live_then_content(self):
        job = ReasoningJob()
        # chunks deliberately split the tags across deltas
        chunks = ["<thi", "nk>The user ", "said hello.</thi", "nk>Hello ", "there!"]
        server, thread = self._serve_streaming(job, chunks)
        try:
            port = server.server_address[1]
            res = requests.post(f"http://127.0.0.1:{port}/v1/responses", json={
                "model": "model-1",
                "input": "Hi",
                "stream": True,
            })
            self.assertEqual(res.status_code, 200)
            events = _parse_sse_events(res.text)
            types = [e["type"] for e in events]

            # reasoning streamed live across multiple deltas, no <think> leak
            reasoning_deltas = [
                e["delta"] for e in events
                if e["type"] == "response.reasoning_summary_text.delta"
            ]
            self.assertGreater(len(reasoning_deltas), 1)
            reasoning = "".join(reasoning_deltas)
            self.assertEqual(reasoning, "The user said hello.")
            self.assertNotIn("<think>", reasoning)
            self.assertNotIn("</think>", reasoning)

            reasoning_done = next(
                e for e in events if e["type"] == "response.reasoning_summary_text.done"
            )
            self.assertEqual(reasoning_done["text"], "The user said hello.")

            # reasoning is closed before the first content delta
            reasoning_done_idx = types.index("response.reasoning_summary_text.done")
            content_idx = types.index("response.output_text.delta")
            self.assertLess(reasoning_done_idx, content_idx)
            # ...and the last reasoning delta precedes the close
            last_reasoning_delta_idx = max(
                i for i, t in enumerate(types)
                if t == "response.reasoning_summary_text.delta"
            )
            self.assertLess(last_reasoning_delta_idx, reasoning_done_idx)

            content = "".join(
                e["delta"] for e in events if e["type"] == "response.output_text.delta"
            )
            self.assertEqual(content, "Hello there!")
            self.assertNotIn("<think>", content)

            completed = next(e for e in events if e["type"] == "response.completed")
            self.assertEqual(completed["response"]["output"][0]["type"], "reasoning")
            self.assertEqual(completed["response"]["output"][1]["type"], "message")
            self.assertEqual(completed["response"]["output_text"], "Hello there!")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


class ToolCallHttpTests(unittest.TestCase):
    def _serve(self, job):
        captured = {}

        def complete(api_key, model, messages, max_completion_tokens, temperature, top_k, top_p, min_p, presence_penalty, start, update, resolve):
            captured["messages"] = messages
            start(job)
            resolve(job)

        server = OAIHttpServer(5000, [], complete, lambda: ["model-1"])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, captured

    def test_tool_call_json_produces_function_call_output(self):
        server, thread, captured = self._serve(ToolJob())
        try:
            port = server.server_address[1]
            res = requests.post(f"http://127.0.0.1:{port}/v1/responses", json={
                "model": "model-1",
                "input": "What is the weather in Chicago?",
                "tools": [WEATHER_TOOL],
            })

            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["output"][0]["type"], "function_call")
            self.assertEqual(data["output"][0]["name"], "get_weather")
            self.assertEqual(data["output_text"], "")
            # tool instructions reached the completion callback
            self.assertTrue(any("get_weather" in m.content for m in captured["messages"]))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_invalid_tools_return_400(self):
        server, thread, _ = self._serve(DummyJob())
        try:
            port = server.server_address[1]
            res = requests.post(f"http://127.0.0.1:{port}/v1/responses", json={
                "model": "model-1",
                "input": "hi",
                "tools": [{"type": "web_search"}],
            })
            self.assertEqual(res.status_code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


class DisconnectDetectionTests(unittest.TestCase):
    """update() must be able to observe a dropped client connection even when
    it has nothing to write yet (non-streaming, or buffering a tool call) —
    otherwise the job never learns the client is gone and runs to completion."""

    def _post_and_drop(self, port: int, body: dict):
        payload = json.dumps(body).encode("utf-8")
        request = (
            b"POST /v1/responses HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
            b"Connection: close\r\n\r\n"
        ) + payload
        sock = socket.create_connection(("127.0.0.1", port))
        sock.sendall(request)
        sock.close()

    def test_non_streaming_update_returns_false_after_client_disconnects(self):
        client_gone = threading.Event()
        result = {}

        def complete(api_key, model, messages, max_completion_tokens, temperature, top_k, top_p, min_p, presence_penalty, start, update, resolve):
            job = DummyJob()
            start(job)
            client_gone.wait(timeout=5)
            deadline = time.time() + 5
            while time.time() < deadline:
                result["alive"] = update(job)
                if result["alive"] is False:
                    break
                time.sleep(0.02)
            resolve(job)

        server = OAIHttpServer(5000, [], complete, lambda: ["model-1"])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            self._post_and_drop(port, {"model": "model-1", "input": "Hello"})
            client_gone.set()
            deadline = time.time() + 5
            while "alive" not in result and time.time() < deadline:
                time.sleep(0.02)
            self.assertIn("alive", result)
            self.assertFalse(result["alive"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_buffered_tool_call_update_returns_false_after_client_disconnects(self):
        client_gone = threading.Event()
        result = {}

        def complete(api_key, model, messages, max_completion_tokens, temperature, top_k, top_p, min_p, presence_penalty, start, update, resolve):
            job = ToolJob()
            start(job)
            client_gone.wait(timeout=5)
            deadline = time.time() + 5
            while time.time() < deadline:
                result["alive"] = update(job)
                if result["alive"] is False:
                    break
                time.sleep(0.02)
            resolve(job)

        server = OAIHttpServer(5000, [], complete, lambda: ["model-1"])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            self._post_and_drop(port, {
                "model": "model-1",
                "input": "What is the weather in Chicago?",
                "tools": [WEATHER_TOOL],
                "stream": True,
            })
            client_gone.set()
            deadline = time.time() + 5
            while "alive" not in result and time.time() < deadline:
                time.sleep(0.02)
            self.assertIn("alive", result)
            self.assertFalse(result["alive"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


class DisconnectWatchdogTests(unittest.TestCase):
    """A dropped connection must be caught even mid prompt-processing, i.e.
    while nothing has called update() yet — the watchdog started in start()
    is the only thing that can observe that."""

    def _post_and_drop(self, port: int, body: dict):
        payload = json.dumps(body).encode("utf-8")
        request = (
            b"POST /v1/responses HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n"
            b"Connection: close\r\n\r\n"
        ) + payload
        sock = socket.create_connection(("127.0.0.1", port))
        sock.sendall(request)
        sock.close()

    def test_watchdog_marks_job_stale_without_any_update_call(self):
        client_gone = threading.Event()
        result = {}

        def complete(api_key, model, messages, max_completion_tokens, temperature, top_k, top_p, min_p, presence_penalty, start, update, resolve):
            job = DummyJob()
            start(job)  # only this launches the watchdog; update() is never called below
            client_gone.wait(timeout=5)
            deadline = time.time() + 5
            while time.time() < deadline and not getattr(job, "stale", False):
                time.sleep(0.05)
            result["stale"] = getattr(job, "stale", False)
            resolve(job)

        server = OAIHttpServer(5000, [], complete, lambda: ["model-1"])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            self._post_and_drop(port, {"model": "model-1", "input": "Hello"})
            client_gone.set()
            deadline = time.time() + 5
            while "stale" not in result and time.time() < deadline:
                time.sleep(0.05)
            self.assertIn("stale", result)
            self.assertTrue(result["stale"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()