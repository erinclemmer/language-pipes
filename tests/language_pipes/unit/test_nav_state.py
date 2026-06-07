import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from language_pipes.tui.frame.nav_state import NavState
from language_pipes.content_provider.content_provider import ProviderState


def _state(sub_menu):
    return ProviderState(
        visible_headers=list(sub_menu.keys()),
        visible_sub_menu=sub_menu,
    )


class NavStateTests(unittest.TestCase):
    def test_preserves_side_option_when_submenu_grows(self):
        nav = NavState()
        # Start with the Network tab showing only "Configure" (no node_id set).
        nav.sync_provider_state(_state({"Network": ["Configure"]}))
        nav.set_side_nav("Configure")
        self.assertEqual(nav.active_side_option(), "Configure")

        # Setting a node_id prepends "Status"; sub_idx must follow "Configure"
        # by name rather than staying at position 0 (which is now "Status").
        nav.sync_provider_state(_state({"Network": ["Status", "Configure"]}))

        self.assertEqual(nav.active_side_option(), "Configure")
        self.assertEqual(nav.sub_idx, 1)

    def test_clamps_when_selected_option_removed(self):
        nav = NavState()
        nav.sync_provider_state(_state({"Network": ["Status", "Peers", "Configure"]}))
        nav.set_side_nav("Peers")
        self.assertEqual(nav.sub_idx, 1)

        # The selected "Peers" option disappears; sub_idx must clamp into range
        # rather than indexing out of bounds.
        nav.sync_provider_state(_state({"Network": ["Status", "Configure"]}))
        self.assertLess(nav.sub_idx, len(nav.active_sub_options()))
        self.assertEqual(nav.sub_idx, 1)


if __name__ == "__main__":
    unittest.main()
