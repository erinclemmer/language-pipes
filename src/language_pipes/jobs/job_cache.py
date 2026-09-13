from transformers import DynamicCache, PretrainedConfig

from language_pipes.util.oai_cache import CacheOptions


class JobCache:
    options: CacheOptions

    def __init__(self, options: CacheOptions, config: PretrainedConfig):
        self.options = options
        self.data = DynamicCache(config=config)