
from typing import Tuple, Optional
from language_pipes.tui.util.prompt import prompt
from language_pipes.tui.tui import TuiWindow, TermText

class TextField:
    window_id: int
    window: TuiWindow
    field_name: str
    position: Tuple[int, int]
    value: str

    def __init__(self, window: TuiWindow, field_name: str, pos: Tuple[int, int], initial: Optional[str] = None):
        self.window = window
        self.field_name = field_name
        self.window_id = window.add_text(TermText(field_name + "|>"), pos)
        self.position = pos
        self.value = initial if initial is not None else ""

    def edit(self) -> Optional[str]:
        self.window.hide_txt(self.window_id)
        self.window.paint()
        res = prompt(TermText(self.field_name), self.window, self.position, self.value)
        self.window.show_txt(self.window_id)
        if res is None:
            return None
        self.window.update_text(self.window_id, TermText(self.field_name + "|> " + res))
        self.window.paint()
        self.value = res
        return res