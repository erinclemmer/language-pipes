"""A canceled job (e.g. its model was unloaded mid-request) has to come back to
the API caller as an error instead of leaving the request open."""
import json
import os
import sys
import threading
import unittest

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.oai_server import OAIHttpServer

CANCEL_REASON = "layers for model-1 unloaded"


class CanceledJob:
    job_id = "job-1"
    model_id = "model-1"
    result = None
    prompt_tokens = 4
    current_token = 0
    cancel_reason = CANCEL_REASON


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


class CanceledJobResponseTests(unittest.TestCase):
    def _serve(self):
        def complete(api_key, model, messages, max_completion_tokens, temperature, top_k, top_p, min_p, presence_penalty, start, update, resolve):
            job = CanceledJob()
            start(job)
            resolve(job)

        server = OAIHttpServer(0, [], complete, lambda: ["model-1"])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _post(self, path, payload):
        server, thread = self._serve()
        try:
            port = server.server_address[1]
            return requests.post(f"http://127.0.0.1:{port}{path}", json=payload, timeout=10)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_chat_completion_returns_the_cancel_reason(self):
        res = self._post("/v1/chat/completions", {
            "model": "model-1",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"error": CANCEL_REASON})

    def test_chat_completion_stream_reports_the_error_in_band(self):
        res = self._post("/v1/chat/completions", {
            "model": "model-1",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })

        self.assertEqual(res.status_code, 200)
        events = _parse_sse_events(res.text)
        self.assertEqual(events[-1]["error"]["message"], CANCEL_REASON)
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "error")
        self.assertIn("data: [DONE]", res.text)

    def test_responses_returns_the_cancel_reason(self):
        res = self._post("/v1/responses", {
            "model": "model-1",
            "input": "Hello",
        })

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"error": CANCEL_REASON})

    def test_responses_stream_emits_response_failed(self):
        res = self._post("/v1/responses", {
            "model": "model-1",
            "input": "Hello",
            "stream": True,
        })

        self.assertEqual(res.status_code, 200)
        events = _parse_sse_events(res.text)
        types = [e["type"] for e in events]
        self.assertIn("response.failed", types)
        failed = next(e for e in events if e["type"] == "response.failed")
        self.assertEqual(failed["response"]["status"], "failed")
        self.assertEqual(failed["response"]["error"]["message"], CANCEL_REASON)
        self.assertIn("data: [DONE]", res.text)


if __name__ == "__main__":
    unittest.main()
