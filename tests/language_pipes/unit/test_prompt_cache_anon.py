"""§2.3: on a server with no `api_keys`, every caller is `"anon"` and shares one
scope. A request without a `prompt_cache_key` there would land in a scope shared
by everyone who can reach the port, which is a cross-tenant `cached_tokens`
oracle - so it runs uncached instead.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.jobs.prompt_cache import new_secret, scope
from language_pipes.util.oai import ChatCompletionRequest, ResponsesRequest
from language_pipes.util.oai_cache import parse_cache_options


class UnauthenticatedServerTests(unittest.TestCase):
    def test_a_request_with_no_cache_key_is_not_cached(self):
        options = parse_cache_options({}, authenticated=False)

        self.assertFalse(options.enabled)
        self.assertEqual(options.prompt_cache_key, "")

    def test_an_empty_cache_key_counts_as_absent(self):
        options = parse_cache_options({"prompt_cache_key": ""}, authenticated=False)

        self.assertFalse(options.enabled)

    def test_a_request_that_supplies_a_key_caches_normally(self):
        options = parse_cache_options(
            {"prompt_cache_key": "tenant-7"}, authenticated=False
        )

        self.assertTrue(options.enabled)
        self.assertEqual(options.prompt_cache_key, "tenant-7")

    def test_two_keys_do_not_share_a_scope(self):
        secret = new_secret()
        self.assertNotEqual(
            scope(secret, "node-a", "anon", "tenant-7"),
            scope(secret, "node-a", "anon", "tenant-8")
        )


class AuthenticatedServerTests(unittest.TestCase):
    def test_an_absent_cache_key_still_caches(self):
        """With `api_keys` set the API key is doing the isolating, so the cache
        key is back to being an optional partitioning label."""
        options = parse_cache_options({}, authenticated=True)

        self.assertTrue(options.enabled)
        self.assertEqual(options.prompt_cache_key, "")

    def test_the_api_key_separates_scopes_on_its_own(self):
        secret = new_secret()
        self.assertNotEqual(
            scope(secret, "node-a", "key-1", ""),
            scope(secret, "node-a", "key-2", "")
        )


class ParsingTests(unittest.TestCase):
    def test_a_non_string_cache_key_is_ignored_rather_than_rejected(self):
        """Phase 1 must not start 400ing requests the server accepts today."""
        options = parse_cache_options({"prompt_cache_key": 7}, authenticated=False)

        self.assertEqual(options.prompt_cache_key, "")
        self.assertFalse(options.enabled)

    def test_implicit_is_the_only_mode_in_this_release(self):
        options = parse_cache_options(
            {"prompt_cache_key": "k", "prompt_cache_options": {"mode": "explicit"}},
            authenticated=False
        )

        self.assertEqual(options.mode, "implicit")
        self.assertEqual(options.breakpoints, [])
        self.assertIsNone(options.ttl_seconds)

    def test_both_endpoints_read_the_cache_key(self):
        chat = ChatCompletionRequest.from_dict(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}],
             "prompt_cache_key": "k"},
            authenticated=False
        )
        responses = ResponsesRequest.from_dict(
            {"model": "m", "input": "hi", "prompt_cache_key": "k"},
            authenticated=False
        )

        self.assertEqual(chat.cache_options.prompt_cache_key, "k")
        self.assertTrue(chat.cache_options.enabled)
        self.assertEqual(responses.cache_options.prompt_cache_key, "k")
        self.assertTrue(responses.cache_options.enabled)

    def test_a_plain_request_on_an_open_server_carries_caching_disabled(self):
        chat = ChatCompletionRequest.from_dict(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            authenticated=False
        )
        responses = ResponsesRequest.from_dict(
            {"model": "m", "input": "hi"}, authenticated=False
        )

        self.assertFalse(chat.cache_options.enabled)
        self.assertFalse(responses.cache_options.enabled)


if __name__ == "__main__":
    unittest.main()
