import sys
from typing import Tuple, List
from language_pipes.tui.tui import TuiWindow, TermText

class SideNav:
    window: TuiWindow
    focused_idx: int
    l_cursor_id: int
    r_cursor_id: int

    def __init__(
            self, 
            size: Tuple[int, int], 
            pos: Tuple[int, int],
            options: List[str]
        ):
        self.window = TuiWindow(size, pos)
        self.window.add_text(TermText("|>"), (0, 0))
        for i, opt in enumerate(options):
            self.window.add_text(TermText(opt), (3, i * 2))
        self.window.paint()
        self.focused_idx = 0

class TopNav:
    def __init__(self, size: Tuple[int, int], pos: Tuple[int, int], headers: List[str]):
        self.window = TuiWindow(size, pos)
        self.window.add_text(TermText("||"), (3, 0))
        self.window.add_text(TermText("||"), (12, 0))

        self.headers = headers
        for i, h in enumerate(self.headers):
            self.window.add_text(TermText(h), (5 + i * 15, 0))

        self.focused_idx = 0

        self.window.paint()

class MainFrame:
    top_nav: TopNav
    network_side_nav: SideNav
    window: TuiWindow
    focus_depth: int

    def __init__(self, size: Tuple[int, int], pos: Tuple[int, int]):
        self.window = TuiWindow(size, pos)
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, 2))
        self.window.add_text(TermText("|\n" * (size[1] - 2)), (15, 3))
        
        self.window.paint()

        self.top_nav = TopNav((80, 1), (pos[0], pos[1] + 1), ["Network", "Models", "Pipes", "Jobs", "Activity"])
        self.network_side_nav = SideNav((13, size[1]), (pos[0] + 1, pos[1] + 4), ["Status", "Peers", "Configure"])
        self.focus_depth = 2

        sys.stdin.read(1)
