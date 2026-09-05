"""The rest of the OpenAI caching surface: `prompt_cache_options`,
`prompt_cache_retention` and explicit breakpoints.

Two rules run through all of it. Unknown values on `/v1/responses` are a `400`,
because a client that asked for a 24h retention and silently got the node's
default has no way to find out. And none of it exists on `/v1/chat/completions`,
where OpenAI exposes only `prompt_cache_key`, so sending it there is ignored
rather than reported.
"""

import os
import sys
import threading
import unittest

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.jobs.job_cache import JobCache
from language_pipes.oai_server import OAIHttpServer
from language_pipes.util.oai import ChatCompletionRequest, ResponsesRequest
from language_pipes.util.oai_cache import CacheOptionError, parse_cache_options


def responses(data, authenticated: bool = False):
    return parse_cache_options(data, authenticated, responses=True)


def marked(text: str = "stable"):
    """An `input_text` content block carrying a breakpoint."""
    return {
        "type": "input_text",
        "text": text,
        "prompt_cache_breakpoint": {"mode": "explicit"}
    }


class ModeTests(unittest.TestCase):
    def test_implicit_is_the_default(self):
        self.assertEqual(responses({"input": "hi"}).mode, "implicit")

    def test_explicit_is_accepted(self):
        options = responses({
            "input": "hi", "prompt_cache_options": {"mode": "explicit"}
        })

        self.assertEqual(options.mode, "explicit")

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(CacheOptionError) as raised:
            responses({"input": "hi", "prompt_cache_options": {"mode": "auto"}})

        self.assertIn("auto", str(raised.exception))

    def test_prompt_cache_options_must_be_an_object(self):
        with self.assertRaises(CacheOptionError):
            responses({"input": "hi", "prompt_cache_options": "explicit"})


class TtlTests(unittest.TestCase):
    def test_no_ttl_leaves_the_node_default(self):
        self.assertIsNone(responses({"input": "hi"}).ttl_seconds)

    def test_a_ttl_is_read_in_seconds(self):
        options = responses({
            "input": "hi", "prompt_cache_options": {"ttl": "30m"}
        })

        self.assertEqual(options.ttl_seconds, 1800)

    def test_an_unknown_ttl_is_rejected(self):
        with self.assertRaises(CacheOptionError) as raised:
            responses({"input": "hi", "prompt_cache_options": {"ttl": "5h"}})

        self.assertIn("5h", str(raised.exception))

    def test_retention_is_accepted_as_an_alias(self):
        options = responses({"input": "hi", "prompt_cache_retention": "24h"})

        self.assertEqual(options.ttl_seconds, 86400)

    def test_in_memory_retention_means_the_node_default(self):
        """The older spelling of "however long you normally keep it"."""
        for spelling in ("in_memory", "in-memory"):
            options = responses({"input": "hi", "prompt_cache_retention": spelling})

            self.assertIsNone(options.ttl_seconds)

    def test_an_unknown_retention_is_rejected(self):
        with self.assertRaises(CacheOptionError):
            responses({"input": "hi", "prompt_cache_retention": "forever"})

    def test_the_current_spelling_wins_over_the_alias(self):
        options = responses({
            "input": "hi",
            "prompt_cache_options": {"ttl": "30m"},
            "prompt_cache_retention": "24h"
        })

        self.assertEqual(options.ttl_seconds, 1800)


class BreakpointTests(unittest.TestCase):
    def test_a_marked_block_marks_its_message(self):
        options = responses({"input": [
            {"role": "system", "content": [marked()]},
            {"role": "user", "content": "question"},
        ]})

        self.assertEqual(options.breakpoints, [0])

    def test_an_unmarked_request_has_none(self):
        options = responses({"input": [{"role": "user", "content": "hi"}]})

        self.assertEqual(options.breakpoints, [])

    def test_at_most_four_are_kept(self):
        """§2.1: extras are ignored, not an error."""
        options = responses({"input": [
            {"role": "user", "content": [marked(str(i))]} for i in range(6)
        ]})

        self.assertEqual(options.breakpoints, [0, 1, 2, 3])

    def test_a_breakpoint_on_instructions_is_rejected(self):
        """`instructions` is a string, so there is nowhere for a prefix to end."""
        with self.assertRaises(CacheOptionError) as raised:
            responses({"input": "hi", "instructions": [marked("be concise")]})

        self.assertIn("instructions", str(raised.exception))

    def test_a_plain_string_instructions_is_untouched(self):
        options = responses({"input": "hi", "instructions": "Be concise"})

        self.assertEqual(options.breakpoints, [])

    def test_a_non_object_breakpoint_is_rejected(self):
        with self.assertRaises(CacheOptionError):
            responses({"input": [
                {"role": "user", "content": [
                    {"type": "input_text", "text": "x", "prompt_cache_breakpoint": True}
                ]}
            ]})

    def test_an_unknown_breakpoint_mode_is_rejected(self):
        with self.assertRaises(CacheOptionError):
            responses({"input": [
                {"role": "user", "content": [
                    {"type": "input_text", "text": "x",
                     "prompt_cache_breakpoint": {"mode": "auto"}}
                ]}
            ]})


class BreakpointIndexTests(unittest.TestCase):
    """A breakpoint names an `input` item; the write path needs the index of the
    message it became, after everything inserted ahead of it."""

    def test_instructions_shift_every_index(self):
        req = ResponsesRequest.from_dict({
            "model": "m",
            "instructions": "Be concise",
            "input": [
                {"role": "system", "content": [marked()]},
                {"role": "user", "content": "question"},
            ],
            "prompt_cache_options": {"mode": "explicit"},
        })

        # messages: [instructions, marked system, user]
        self.assertEqual(req.cache_options.breakpoints, [1])
        self.assertEqual(req.messages[1].content, "stable")

    def test_tool_instructions_shift_them_too(self):
        req = ResponsesRequest.from_dict({
            "model": "m",
            "input": [
                {"role": "system", "content": [marked()]},
                {"role": "user", "content": "question"},
            ],
            "tools": [{
                "type": "function", "name": "f", "description": "d",
                "parameters": {"type": "object", "properties": {}}
            }],
            "prompt_cache_options": {"mode": "explicit"},
        })

        self.assertEqual(req.cache_options.breakpoints, [1])
        self.assertEqual(req.messages[1].content, "stable")

    def test_a_tool_result_can_carry_the_marker(self):
        """Its text lives under `output`, not `content`, and a long tool result
        is exactly the kind of stable block a client marks."""
        req = ResponsesRequest.from_dict({
            "model": "m",
            "input": [
                {"role": "user", "content": "question"},
                {"type": "function_call", "name": "f", "arguments": "{}",
                 "call_id": "c1"},
                {"type": "function_call_output", "call_id": "c1",
                 "output": [marked("result")]},
            ],
            "prompt_cache_options": {"mode": "explicit"},
        })

        self.assertEqual(req.cache_options.breakpoints, [2])

    def test_an_item_that_produced_no_message_is_dropped(self):
        req = ResponsesRequest.from_dict({
            "model": "m",
            "input": [
                {"type": "not_a_message", "content": [marked()]},
                {"role": "user", "content": "question"},
            ],
            "prompt_cache_options": {"mode": "explicit"},
        })

        self.assertEqual(req.cache_options.breakpoints, [])


class ChatEndpointTests(unittest.TestCase):
    """§2.2: only `prompt_cache_key` exists here, so the rest is neither read
    nor rejected."""

    def test_the_full_surface_is_ignored_silently(self):
        req = ChatCompletionRequest.from_dict({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_key": "k",
            "prompt_cache_options": {"mode": "nonsense", "ttl": "5h"},
            "prompt_cache_retention": "forever",
        })

        self.assertEqual(req.cache_options.mode, "implicit")
        self.assertIsNone(req.cache_options.ttl_seconds)
        self.assertEqual(req.cache_options.breakpoints, [])
        self.assertTrue(req.cache_options.enabled)


class DummyJob:
    job_id = "job-1"
    model_id = "model-1"
    result = "hello"
    prompt_tokens = 4
    current_token = 3
    cancel_reason = None
    caching = JobCache()


class ValidationOverHttpTests(unittest.TestCase):
    """The parse errors have to reach the client as a 400, not a 500."""

    def setUp(self):
        def complete(api_key, model, messages, max_completion_tokens, temperature,
                     top_k, top_p, min_p, presence_penalty, start, update, resolve,
                     cache_options=None):
            job = DummyJob()
            start(job)
            resolve(job)

        self.server = OAIHttpServer(0, [], complete, lambda: ["model-1"])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    def post(self, path, body):
        return requests.post(f"http://127.0.0.1:{self.port}{path}", json=body)

    def test_an_unknown_mode_is_a_400(self):
        res = self.post("/v1/responses", {
            "model": "model-1", "input": "hi",
            "prompt_cache_options": {"mode": "auto"},
        })

        self.assertEqual(res.status_code, 400)

    def test_an_unknown_retention_is_a_400(self):
        res = self.post("/v1/responses", {
            "model": "model-1", "input": "hi", "prompt_cache_retention": "forever",
        })

        self.assertEqual(res.status_code, 400)

    def test_a_breakpoint_on_instructions_is_a_400(self):
        res = self.post("/v1/responses", {
            "model": "model-1", "input": "hi", "instructions": [marked()],
        })

        self.assertEqual(res.status_code, 400)

    def test_a_valid_request_still_succeeds(self):
        res = self.post("/v1/responses", {
            "model": "model-1",
            "input": [{"role": "user", "content": [marked("hi")]}],
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "prompt_cache_key": "k",
        })

        self.assertEqual(res.status_code, 200)

    def test_the_chat_endpoint_never_400s_on_them(self):
        res = self.post("/v1/chat/completions", {
            "model": "model-1",
            "messages": [{"role": "user", "content": "hi"}],
            "prompt_cache_options": {"mode": "auto"},
        })

        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
