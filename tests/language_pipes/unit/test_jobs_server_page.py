"""The Jobs / Server page after the two prompt-cache rows were inserted.

Every focus index shifted, so the navigation and the enter-key routing are what
these tests are actually guarding.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from ansinout import PressedKey

from language_pipes.jobs.prompt_cache import CacheStats
from language_pipes.tui.components.jobs_server.top_state import TopPageState
from language_pipes.tui.frame.tips import TIPS

PORT = 0
NODE_JOBS = 1
API_JOBS = 2
CACHE_TIME = 3
CACHE_TOKENS = 4
API_KEYS = 5
START = 6


class FakeJobProvider:
    def __init__(self):
        self.job_port = 8000
        self.max_node_jobs = 10
        self.max_api_jobs = 5
        self.max_cache_time = 300
        self.max_cache_tokens = 16384
        self.api_keys = []
        self.cache_stats = None
        self.running = False

    def get_job_port(self):
        return self.job_port

    def set_job_port(self, value):
        self.job_port = value

    def get_max_node_jobs(self):
        return self.max_node_jobs

    def set_max_node_jobs(self, value):
        self.max_node_jobs = value

    def get_max_api_jobs(self):
        return self.max_api_jobs

    def set_max_api_jobs(self, value):
        self.max_api_jobs = value

    def get_max_cache_time(self):
        return self.max_cache_time

    def set_max_cache_time(self, value):
        self.max_cache_time = value

    def get_max_cache_tokens(self):
        return self.max_cache_tokens

    def set_max_cache_tokens(self, value):
        self.max_cache_tokens = value

    def get_api_keys(self):
        return self.api_keys

    def set_api_keys(self, keys):
        self.api_keys = keys

    def get_cache_stats(self):
        return self.cache_stats

    def oai_server_running(self):
        return self.running

    def start_oai_server(self):
        self.running = True


class FakeNetworkStatus:
    running = True


class FakeNetworkProvider:
    def get_network_status(self):
        return FakeNetworkStatus()


class FakeProvider:
    def __init__(self):
        self.job_provider = FakeJobProvider()
        self.network_provider = FakeNetworkProvider()


def make_state(can_start: bool = True) -> TopPageState:
    state = TopPageState()
    state.provider = FakeProvider()  # type: ignore[assignment]
    state.changed_to = []
    state.change_state = lambda name, args: state.changed_to.append(name)
    state.exit_page = lambda: None
    # The port-availability probe touches a real socket; the tests care about
    # focus routing, not about whether 8000 happens to be free here.
    state.can_start_server = lambda: can_start  # type: ignore[method-assign]
    return state


class NavigationTests(unittest.TestCase):
    def test_arrow_down_walks_every_row_and_wraps(self):
        state = make_state()

        seen = [state.focus_idx]
        for _ in range(START + 1):
            state.on_key(PressedKey.ArrowDown, "")
            seen.append(state.focus_idx)

        self.assertEqual(seen, [0, 1, 2, 3, 4, 5, 6, 0])

    def test_arrow_up_wraps_to_the_last_row(self):
        state = make_state()

        state.on_key(PressedKey.ArrowUp, "")

        self.assertEqual(state.focus_idx, START)

    def test_the_start_row_is_skipped_when_the_server_cannot_start(self):
        state = make_state(can_start=False)

        state.on_key(PressedKey.ArrowUp, "")

        self.assertEqual(state.focus_idx, API_KEYS)


class EditingTests(unittest.TestCase):
    def test_typing_on_the_cache_time_row_persists_it(self):
        state = make_state()
        state.focus_idx = CACHE_TIME

        # Clear the default 300 a character at a time, then type a new value.
        for _ in range(3):
            state.on_key(PressedKey.Backspace, "")
        for ch in "60":
            state.on_key(PressedKey.Alpha, ch)

        self.assertEqual(state.provider.job_provider.max_cache_time, 60)

    def test_typing_on_the_cache_tokens_row_persists_it(self):
        state = make_state()
        state.focus_idx = CACHE_TOKENS
        state.edit_max_cache_tokens = ""

        for ch in "4096":
            state.on_key(PressedKey.Alpha, ch)

        self.assertEqual(state.provider.job_provider.max_cache_tokens, 4096)

    def test_zero_is_accepted_because_it_disables_the_cache(self):
        state = make_state()
        state.focus_idx = CACHE_TIME
        state.edit_max_cache_time = ""

        state.on_key(PressedKey.Alpha, "0")

        self.assertTrue(state._valid_max_cache_time())
        self.assertEqual(state.provider.job_provider.max_cache_time, 0)

    def test_a_non_numeric_value_is_reported_and_not_saved(self):
        state = make_state()
        state.focus_idx = CACHE_TOKENS
        state.edit_max_cache_tokens = "12x"

        self.assertFalse(state._valid_max_cache_tokens())
        self.assertIn("   Error: Invalid max cache tokens value", state.get_view())
        self.assertEqual(state.provider.job_provider.max_cache_tokens, 16384)

    def test_editing_the_cache_rows_leaves_the_job_limits_alone(self):
        state = make_state()
        state.focus_idx = CACHE_TIME
        state.on_key(PressedKey.Alpha, "1")

        self.assertEqual(state.provider.job_provider.max_node_jobs, 10)
        self.assertEqual(state.provider.job_provider.max_api_jobs, 5)


class EnterTests(unittest.TestCase):
    def test_enter_advances_through_the_cache_rows(self):
        state = make_state()
        state.focus_idx = API_JOBS

        state.on_key(PressedKey.Enter, "")
        self.assertEqual(state.focus_idx, CACHE_TIME)

        state.on_key(PressedKey.Enter, "")
        self.assertEqual(state.focus_idx, CACHE_TOKENS)

        state.on_key(PressedKey.Enter, "")
        self.assertEqual(state.focus_idx, API_KEYS)

    def test_enter_on_the_api_keys_row_still_opens_the_keys_page(self):
        state = make_state()
        state.focus_idx = API_KEYS

        state.on_key(PressedKey.Enter, "")

        self.assertEqual(state.changed_to, ["keys"])

    def test_enter_on_the_start_row_saves_both_cache_limits(self):
        state = make_state()
        state.focus_idx = START
        state.edit_max_cache_time = "45"
        state.edit_max_cache_tokens = "2048"

        state.on_key(PressedKey.Enter, "")

        self.assertTrue(state.provider.job_provider.running)
        self.assertEqual(state.provider.job_provider.max_cache_time, 45)
        self.assertEqual(state.provider.job_provider.max_cache_tokens, 2048)


class ViewTests(unittest.TestCase):
    def test_both_rows_are_rendered_with_their_values(self):
        lines = make_state().get_view()

        self.assertIn("   Max Cache Time: 300", lines)
        self.assertIn("   Max Cache Tokens: 16384", lines)

    def test_the_cursor_sits_on_the_focused_cache_row(self):
        state = make_state()
        state.focus_idx = CACHE_TOKENS

        lines = state.get_view()

        self.assertIn("   Max Cache Tokens: 16384|", lines)
        self.assertIn("   Max Cache Time: 300", lines)

    def test_each_cache_row_shows_its_own_tip(self):
        state = make_state()

        state.focus_idx = CACHE_TIME
        self.assertIn(TIPS["jobs_server"]["max_cache_time"], state.get_view())

        state.focus_idx = CACHE_TOKENS
        self.assertIn(TIPS["jobs_server"]["max_cache_tokens"], state.get_view())

    def test_the_stats_line_reports_usage_against_the_budget(self):
        state = make_state()
        state.provider.job_provider.cache_stats = CacheStats(
            entries=6, tokens=11200, reserved=3100, budget=16384,
            size_gb=1.2, hits=3, misses=1
        )

        line = next(line for line in state.get_view() if line.startswith("   Cache:"))

        self.assertIn("6 entries", line)
        self.assertIn("11200/16384 tokens", line)
        self.assertIn("(3100 reserved)", line)
        self.assertIn("1.2 GB", line)
        self.assertIn("75% hit rate", line)

    def test_a_zero_budget_reads_as_disabled(self):
        state = make_state()
        state.provider.job_provider.cache_stats = CacheStats(budget=0)

        self.assertIn("   Cache: disabled", state.get_view())

    def test_no_stats_line_before_the_network_is_up(self):
        state = make_state()

        self.assertFalse(any(line.startswith("   Cache:") for line in state.get_view()))


if __name__ == "__main__":
    unittest.main()
