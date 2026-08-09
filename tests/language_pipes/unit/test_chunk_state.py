import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.util.chunk_state import ChunkState

@patch("language_pipes.util.chunk_state.CHUNK_SIZE", 32)
class TestChunkState(unittest.TestCase):
    """Chunk ranges must reflect the real prompt length, not a whole-chunk estimate."""

    def make(self, prompt_length: int) -> ChunkState:
        state = ChunkState("job-1")
        state.init(prompt_length)
        return state

    def test_short_prompt_is_not_chunked(self):
        state = self.make(10)

        self.assertFalse(state.is_active())
        self.assertEqual(state.get_range(), (0, 10))
        self.assertEqual(state.get_chunk_length(), 10)
        self.assertEqual(state.get_tokens_processed(), 0)

    def test_ranges_cover_the_prompt_exactly(self):
        state = self.make(70)
        self.assertEqual(state.total_chunks, 3)

        ranges = [state.get_range()]
        while state.has_more():
            state.advance()
            ranges.append(state.get_range())

        self.assertEqual(ranges, [(0, 32), (32, 64), (64, 70)])
        self.assertEqual(sum(end - start for start, end in ranges), 70)

    def test_progress_never_exceeds_prompt_length(self):
        state = self.make(70)

        while state.has_more():
            state.advance()

        # Final chunk is short, so processed tokens stop below the prompt length
        self.assertEqual(state.get_tokens_processed(), 64)
        self.assertEqual(state.get_chunk_length(), 6)
        self.assertLess(state.get_tokens_processed(), state.prompt_length)

    def test_disable_clears_chunking(self):
        state = self.make(70)
        state.advance()
        state.disable()

        self.assertFalse(state.is_active())
        self.assertTrue(state.is_final())
        self.assertEqual(state.get_range(), (0, 70))

if __name__ == "__main__":
    unittest.main()
