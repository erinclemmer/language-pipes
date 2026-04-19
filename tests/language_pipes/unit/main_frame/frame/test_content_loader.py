import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from language_pipes.content_loader import ContentLoader

class TestContentLoader(unittest.TestCase):
    """Unit tests for ContentLoader caching and provider dispatch."""

    def test_load_without_provider_returns_placeholder(self):
        loader = ContentLoader(providers=None)
        result = loader.load("Network", "Status", update_status=False, force=False)
        self.assertIn("state", result)
        # Without a provider, should return placeholder
        self.assertEqual(result["state"], "placeholder")

    def test_load_caches_result(self):
        loader = ContentLoader(providers=None)
        r1 = loader.load("Network", "Status", update_status=False, force=False)
        r2 = loader.load("Network", "Status", update_status=False, force=False)
        self.assertIs(r1, r2)

    def test_load_force_bypasses_cache(self):
        loader = ContentLoader(providers=None)
        r1 = loader.load("Network", "Status", update_status=False, force=False)
        r2 = loader.load("Network", "Status", update_status=False, force=True)
        # Both are placeholders but force=True creates a new dict
        self.assertIsNot(r1, r2)

    def test_invalidate_clears_specific_key(self):
        loader = ContentLoader(providers=None)
        loader.load("Network", "Status", update_status=False, force=False)
        self.assertIn(("Network", "Status"), loader._cache)
        loader.invalidate("Network", "Status")
        self.assertNotIn(("Network", "Status"), loader._cache)

    def test_invalidate_all_clears_cache(self):
        loader = ContentLoader(providers=None)
        loader.load("Network", "Status", update_status=False, force=False)
        loader.load("Network", "Peers", update_status=False, force=False)
        loader.invalidate_all()
        self.assertEqual(len(loader._cache), 0)

    def test_provider_available_false_when_none(self):
        from language_pipes.content_provider.provider_calls import ProviderCall
        loader = ContentLoader(providers=None)
        self.assertFalse(loader.provider_available(ProviderCall.get_network_config))

    def test_provider_available_true_with_dict(self):
        from language_pipes.content_provider.provider_calls import ProviderCall
        providers = {ProviderCall.get_network_config: lambda: None}
        loader = ContentLoader(providers=providers)
        self.assertTrue(loader.provider_available(ProviderCall.get_network_config))

    def test_load_with_provider_calls_formatter(self):
        """When a provider is available, load should call it and format the result."""
        from language_pipes.distributed_state_network.objects.config import DSNodeConfig
        from language_pipes.content_provider.provider_calls import ProviderCall

        fake_config = DSNodeConfig(
            node_id="test",
            credential_dir="/tmp",
            port=5000,
            network_ip="",
            aes_key="",
            whitelist_ips=[],
            whitelist_node_ids=[],
            bootstrap_nodes=[],
        )

        providers = {ProviderCall.get_network_config: lambda: fake_config}
        loader = ContentLoader(providers=providers)
        result = loader.load("Network", "Configure", update_status=True, force=True)
        self.assertEqual(result["state"], "ok")

