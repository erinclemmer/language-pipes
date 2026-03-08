from typing import List, Tuple, Optional
from language_pipes.tui.util.screen_utils import print_pos

class TermText:
    value: str
    fg: Optional[int]
    bg: Optional[int]
    bold: bool

    def __init__(self, v: str, fg: Optional[int] = None, bg: Optional[int] = None, bold: bool = False):
        self.value = v
        self.fg = fg
        self.bg = bg
        self.bold = bold

class TuiCell:
    text: TermText
    
    position: Tuple[int, int]
    committed: bool

    def __init__(self, v: TermText, pos: Tuple[int, int]):
        self.text = v
        self.position = pos
        self.committed = True

    def set_value(self, tt: TermText):
        if not isinstance(tt, TermText):
            raise Exception(f"Tried to set cell value for non string {tt}")
        if len(tt.value) > 1:
            raise Exception("Tried to set cell with more than one character")
        if tt == self.text:
            return
        self.text = tt
        self.committed = False

    def paint(self, abs_position: Tuple[int, int]):
        if self.committed:
            return
        print_pos(abs_position[1], abs_position[0], self.text.value, self.text.fg, self.text.bg, self.text.bold)
        self.committed = True

class TuiGrid:
    size: Tuple[int, int]
    position: Tuple[int, int]
    grid: List[List[TuiCell]]

    def __init__(self, size: Tuple[int, int], pos: Tuple[int, int]):
        self.size = size
        self.position = pos
        self.grid = []
        for x in range(0, self.size[0]):
            rows = []
            for y in range(0, self.size[1]):
                rows.append(TuiCell(TermText(""), (x, y)))
            self.grid.append(rows)

    def set_cell(self, pos: Tuple[int, int], tt: TermText):
        if pos[0] > self.size[0] or pos[1] > self.size[1]:
            return
        if pos[0] < 0 or pos[1] < 0:
            return
        try:
            self.grid[pos[0]][pos[1]].set_value(tt)
        except IndexError:
            pass

    def paint(self):
        for row in self.grid:
            for cell in row:
                cell.paint((
                    self.position[0] + cell.position[0],
                    self.position[1] + cell.position[1]
                ))
        print_pos(0, 0, '')

class TuiText:
    id: int
    hidden: bool
    text: TermText
    size: Tuple[int, int]
    position: Tuple[int, int]

    def __init__(self, id: int, v: TermText, pos: Tuple[int, int]):
        self.id = id
        if not isinstance(v, TermText):
            raise Exception("TuiText must use TermText")
        self.text = v
        self.position = pos
        self.hidden = False
        self._update_size()
        
    def _update_size(self):
        max_len = 0
        lines = self.text.value.split("\n")
        for i in range(0, len(lines)):
            if len(lines[i]) > max_len:
                max_len = len(lines[i])
        
        self.size = (max_len, len(lines))

    def update_value(self, v: TermText):
        if not isinstance(v, TermText):
            raise Exception("TuiText must use TermText")
        self.text = v
        self._update_size()

    def get_cells(self) -> List[Tuple[TermText, Tuple[int, int]]]:
        cells = []
        lines = self.text.value.split("\n")
        for x in range(0, self.size[0]):
            for y in range(0, self.size[1]):
                if x > len(lines[y]) - 1:
                    continue
                tt = TermText(lines[y][x], self.text.fg, self.text.bg, self.text.bold)
                cells.append((tt, (
                    self.position[0] + x, 
                    self.position[1] + y
                )))
        return cells

class TuiWindow(TuiGrid):
    _current_id: int
    text_objects: List[TuiText]

    def __init__(self, size: Tuple[int, int], pos: Tuple[int, int]):
        super().__init__(size, pos)
        self._current_id = 0
        self.text_objects = []
    
    def get_text(self, id: int) -> TuiText:
        objs = [o for o in self.text_objects if o.id == id]
        if len(objs) == 0:
            raise Exception("Could not find text id for TuiWindow")
        return objs[0]

    def add_text(self, v: TermText, pos: Tuple[int, int]) -> int:
        self.text_objects.append(TuiText(self._current_id, v, pos))
        self._current_id += 1
        self._update_grid()
        return self._current_id - 1
    
    def clear_text(self, id: int):
        txt = self.get_text(id)
        whitespace_v = TermText(''.join((c if c.isspace() else ' ') for c in txt.text.value))
        whitespace_obj = TuiText(-1, whitespace_v, txt.position)
        for clear_v, rel_pos in whitespace_obj.get_cells():
            self.set_cell((
                rel_pos[0],
                rel_pos[1]
            ), clear_v)

    def hide_txt(self, id: int):
        txt = self.get_text(id)
        txt.hidden = True
        self.clear_text(id)

    def show_txt(self, id: int):
        txt = self.get_text(id)
        txt.hidden = False
        self._update_grid()

    def remove_txt(self, id: int):
        self.clear_text(id)
        self.text_objects = [t for t in self.text_objects if t.id != id]

    def remove_all(self):
        for txt in list(self.text_objects):
            self.remove_txt(txt.id)

    def update_text(self, id: int, v: Optional[TermText], pos: Optional[Tuple[int, int]] = None):
        if v is not None and not isinstance(v, TermText):
            raise Exception("TuiWindow must use TermText")
        txtObj = self.get_text(id)

        self.clear_text(id)

        if v is not None:
            txtObj.update_value(v)
        
        if pos is not None:
            txtObj.position = pos

        self._update_grid()
    
    def _update_grid(self):
        for obj in self.text_objects:
            if obj.hidden:
                continue
            for v, rel_pos in obj.get_cells():
                    self.set_cell((
                        rel_pos[0],
                        rel_pos[1]
                    ), v)
