from typing import Optional, Tuple

from transformers.cache_utils import Cache


class PartialCacheMaskView:
    """Sizes attention masks against a known token count instead of the local cache.

    A node that owns only a slice of the decoder stack only ever populates the
    cache layers it computes. `Cache.get_seq_length` and `Cache.get_mask_sizes`
    both resolve through the first *attention* layer, so a node whose slice
    contains no attention layer reads a length of 0 forever. Hybrid
    linear-attention stacks hit this immediately: Qwen3.5 opens with three
    `linear_attention` layers, so the first attention layer is index 3 and an
    embedding node holding one local layer never advances it.

    Masks are built once on the embedding node and shipped to every layer node,
    so they have to describe the whole sequence rather than the part of the
    cache that happens to be local. This view keeps the real cache for
    layer-type and sliding-window metadata but answers every size question from
    `past_seen_tokens`. When the cache holds the whole stack the answers are
    identical to the cache's own, since all layers of a mask type have then seen
    the same tokens.
    """

    def __init__(self, cache: Cache, past_seen_tokens: int):
        self._cache = cache
        self._past_seen_tokens = past_seen_tokens

    def __getattr__(self, name: str):
        return getattr(self._cache, name)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._past_seen_tokens

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self._past_seen_tokens

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        sliding_window = self._sliding_window(layer_idx)
        if sliding_window is None:
            return self._past_seen_tokens + query_length, 0

        # Mirrors DynamicSlidingWindowLayer.get_mask_sizes
        kv_offset = max(self._past_seen_tokens - sliding_window + 1, 0)
        if self._past_seen_tokens >= sliding_window:
            return sliding_window - 1 + query_length, kv_offset

        return self._past_seen_tokens + query_length, kv_offset

    def _sliding_window(self, layer_idx: int) -> Optional[int]:
        layers = self._cache.layers
        if layer_idx >= len(layers):
            return None

        layer = layers[layer_idx]
        if not getattr(layer, 'is_sliding', False):
            return None

        return getattr(layer, 'sliding_window', None)
