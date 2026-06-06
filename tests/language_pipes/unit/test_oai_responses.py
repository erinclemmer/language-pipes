import os
import sys
import threading
import unittest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.oai_server import OAIHttpServer
from language_pipes.util.chat import ChatRole
from language_pipes.util.oai import ResponsesRequest, _response_json


class DummyJob:
    job_id = "job-1"
    model_id = "model-1"
    result = "Hello from Language Pipes"
    prompt_tokens = 4
    current_token = 3


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


if __name__ == "__main__":
    unittest.main()