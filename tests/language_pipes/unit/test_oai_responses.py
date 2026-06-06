import json
import os
import sys
import threading
import unittest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.oai_server import OAIHttpServer
from language_pipes.util.chat import ChatRole
from language_pipes.util.oai import ResponsesRequest, _response_json
from language_pipes.util.oai_tool_calls import (
    parse_tool_call,
    parse_tool_definitions,
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

        def complete(model, messages, max_completion_tokens, temperature, top_k, top_p, min_p, presence_penalty, start, update, resolve):
            captured["model"] = model
            captured["messages"] = messages
            captured["max_completion_tokens"] = max_completion_tokens
            job = DummyJob()
            start(job)
            resolve(job)

        server = OAIHttpServer(0, [], complete, lambda: ["model-1"])
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
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(json.loads(call.arguments), {"city": "Chicago"})
        self.assertTrue(call.call_id.startswith("call_"))

    def test_parses_fenced_json(self):
        call = parse_tool_call(
            '```json\n{"name": "get_weather", "arguments": {"city": "Chicago"}}\n```',
            self.tools,
        )
        self.assertIsNotNone(call)
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


class ToolCallHttpTests(unittest.TestCase):
    def _serve(self, job):
        captured = {}

        def complete(model, messages, max_completion_tokens, temperature, top_k, top_p, min_p, presence_penalty, start, update, resolve):
            captured["messages"] = messages
            start(job)
            resolve(job)

        server = OAIHttpServer(0, [], complete, lambda: ["model-1"])
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


if __name__ == "__main__":
    unittest.main()