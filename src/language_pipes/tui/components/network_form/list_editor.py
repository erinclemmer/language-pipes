from abc import ABC, abstractmethod
from typing import Callable, Generic, List, TypeVar

from language_pipes.tui.util.kb_utils import PressedKey
from language_pipes.tui.components.confirm import Confirm
from language_pipes.tui.content_loader import ContentLoader

T = TypeVar("T")


class ListEditor(ABC, Generic[T]):
    """Base class for TUI editors that present a selectable list of items
    with an option to add a new item via an inline input form.

    Subclasses must implement:
      - load_items()          – return the current list of items
      - reset_input_fields()  – clear any new-item input state
      - is_input_valid()      – whether the current input can be saved
      - input_field_count()   – number of focusable fields in the input form
      - on_save_new()         – persist the new item and call restart()
      - on_select_existing(item) – handle selecting an existing item
      - on_delete_existing(item) – handle deleting an existing item
      - format_item(item)     – render an item as a display string
      - get_input_lines()     – lines to show when in input mode
      - input_footer()        – footer text when in input mode
      - list_footer()         – footer text when in list mode
      - list_header()         – header label shown above the item list
      - add_new_label()       – label for the "add new" option
      - on_char_to_field(ch)  – append a character to the focused input field
      - on_backspace_field()  – delete a character from the focused input field
    """

    confirm: Confirm
    loader: ContentLoader
    exit_editor: Callable

    select_idx: int
    focus_idx: int
    items: List[T]
    adding: bool

    def __init__(self, loader: ContentLoader, confirm: Confirm, exit_editor: Callable):
        self.exit_editor = exit_editor
        self.confirm = confirm
        self.loader = loader
        self.restart()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_items(self) -> List[T]:
        """Return the current list of items from the data source."""
        ...

    @abstractmethod
    def reset_input_fields(self) -> None:
        """Reset all new-item input fields to their defaults."""
        ...

    @abstractmethod
    def is_input_valid(self) -> bool:
        """Return True when the current input state is valid for saving."""
        ...

    @abstractmethod
    def input_field_count(self) -> int:
        """Number of focusable input fields in the add-new form."""
        ...

    @abstractmethod
    def on_save_new(self, discard: Callable) -> None:
        """Persist the new item.  Call ``self.restart()`` when done.
        Open a Confirm dialog as needed, passing *discard* as the
        on_discard callback."""
        ...

    @abstractmethod
    def on_select_existing(self, item: T, discard: Callable) -> None:
        """Handle the user pressing Enter on an existing item."""
        ...

    @abstractmethod
    def on_delete_existing(self, item: T) -> None:
        """Handle the user pressing Delete on an existing item."""
        ...

    @abstractmethod
    def format_item(self, item: T) -> str:
        """Return a display string for *item*."""
        ...

    @abstractmethod
    def get_input_lines(self) -> List[str]:
        """Return the lines to render when the user is adding a new item."""
        ...

    @abstractmethod
    def input_footer(self) -> str:
        """Footer hint shown while the add-new form is active."""
        ...

    @abstractmethod
    def list_footer(self) -> str:
        """Footer hint shown while the item list is active."""
        ...

    @abstractmethod
    def list_header(self) -> str:
        """Header label displayed above the item list."""
        ...

    @abstractmethod
    def add_new_label(self) -> str:
        """Label for the trailing 'add new' option in the list."""
        ...

    @abstractmethod
    def on_char_to_field(self, ch: str) -> None:
        """Append *ch* to the currently focused input field."""
        ...

    @abstractmethod
    def on_backspace_field(self) -> None:
        """Delete the last character from the currently focused input field."""
        ...

    # ------------------------------------------------------------------
    # Concrete shared logic
    # ------------------------------------------------------------------

    def restart(self, reset_select: bool = True) -> None:
        if reset_select:
            self.select_idx = 0
        self.items = self.load_items()
        self.adding = True if len(self.items) == 0 else False
        self.focus_idx = 0
        self.reset_input_fields()

    def on_key(self, key: PressedKey, ch: str) -> None:
        if key == PressedKey.ArrowUp:
            self.on_prev()
        elif key == PressedKey.ArrowDown:
            self.on_next()
        elif key == PressedKey.Enter:
            self.on_enter()
        elif key == PressedKey.Backspace:
            self.on_backspace()
        elif key == PressedKey.Alpha:
            self.on_char(ch)
        elif key == PressedKey.Delete:
            self.on_delete()

    def on_next(self) -> None:
        if self.adding:
            self.focus_idx += 1
            if self.focus_idx >= self.input_field_count():
                self.focus_idx = 0
        else:
            self.select_idx += 1
            if self.select_idx > len(self.items):
                self.select_idx = 0

    def on_prev(self) -> None:
        if self.adding:
            self.focus_idx -= 1
            if self.focus_idx < 0:
                self.focus_idx = self.input_field_count() - 1
        else:
            self.select_idx -= 1
            if self.select_idx < 0:
                self.select_idx = len(self.items)

    def back(self) -> bool:
        was_adding = self.adding
        if was_adding:
            self.restart(False)
        return not was_adding or len(self.items) == 0

    def on_enter(self) -> None:
        def discard_choice():
            self.restart()
            self.confirm.close()

        if self.adding:
            if not self.is_input_valid():
                return
            self.on_save_new(discard_choice)
            return

        if self.select_idx == len(self.items):
            self.adding = True
        else:
            self.on_select_existing(self.items[self.select_idx], discard_choice)

    def on_char(self, ch: str) -> None:
        if not self.adding:
            return
        self.on_char_to_field(ch)

    def on_backspace(self) -> None:
        if not self.adding:
            return
        self.on_backspace_field()

    def on_delete(self) -> None:
        if self.select_idx >= len(self.items):
            return
        self.on_delete_existing(self.items[self.select_idx])

    def get_footer(self) -> str:
        if self.adding:
            return self.input_footer()
        return self.list_footer()

    def get_lines(self) -> List[str]:
        lines: List[str] = []

        if self.adding:
            lines.extend(self.get_input_lines())
        else:
            if len(self.items) > 0:
                lines.append(self.list_header())
                for i, item in enumerate(self.items):
                    l_cursor = "|>" if i == self.select_idx else "  "
                    r_cursor = "<|" if i == self.select_idx else "  "
                    lines.append(f" {l_cursor} {self.format_item(item)} {r_cursor}")
                lines.append("")
            l_cursor = "|>" if self.select_idx == len(self.items) else "  "
            r_cursor = "<|" if self.select_idx == len(self.items) else "  "
            lines.append(f" {l_cursor} {self.add_new_label()} {r_cursor}")

        return lines
