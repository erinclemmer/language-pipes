# Skill: custom_tui — Language Pipes terminal UI library

## Overview

Language Pipes includes a small custom terminal UI stack under `src/language_pipes/tui/`.
It is a low-level, paint-by-cells library that supports:

- alternate-screen terminal mode
- raw-ish key reading (arrows, enter, escape, delete, alphanumerics)
- text objects with optional ANSI foreground/background/bold styling
- composable windows/grids for multi-panel layouts

This skill explains how to use those primitives safely and consistently.

---

## Core modules and responsibilities

- `screen_utils.py`
  - terminal mode enter/exit (`enable_vt_mode`, `exit_vt_mode`, `restore_mode`)
  - cursor movement and ANSI color output
- `kb_utils.py`
  - key parsing via `read_key()` and `PressedKey`
- `tui.py`
  - render primitives: `TermText`, `TuiText`, `TuiWindow`
- `prompt.py`
  - reusable text prompt and option picker helpers
- `text_field.py`
  - small editable field wrapper built on `prompt`
- `__init__.py`
  - startup lifecycle (`initialize_tui`)

---

## Startup and teardown (required)

Always run interactive TUI code inside the terminal mode lifecycle:

1. `enable_vt_mode()` before reading keys or painting
2. execute your UI loop
3. in `finally`: `exit_vt_mode()` then `restore_mode(fd, attrs)`

Pattern used by project:

```python
fd_in, old_in_attrs = enable_vt_mode()
try:
    main_menu((cols, rows))
finally:
    exit_vt_mode()
    restore_mode(fd_in, old_in_attrs)
```

Why: this prevents terminal corruption (stuck alt screen, broken echo/cbreak state).

---

## Rendering model (how drawing works)

### `TermText`

`TermText(value, fg=None, bg=None, bold=False)` stores text plus style.

### `TuiWindow`

`TuiWindow(size=(width, height), pos=(x, y))` is the main drawing surface.

Common methods:

- `add_text(TermText, (x, y)) -> text_id`
- `update_text(text_id, TermText | None, pos=None)`
- `remove_txt(text_id)`
- `hide_txt(text_id)` / `show_txt(text_id)`
- `remove_all()`
- `paint()`

Important: mutate text first, then call `paint()` to flush changes.

### Positioning rules

- window positions are relative to terminal origin
- text positions are relative to window origin
- multiline strings (`"\n"`) are supported

---

## Input model

Use:

```python
key, raw = read_key()
```

`PressedKey` supports:

- `Alpha`
- `ArrowUp`, `ArrowDown`, `ArrowLeft`, `ArrowRight`
- `Enter`, `Escape`, `Backspace`, `Delete`
- `Nop`

For alpha input, inspect `raw` (e.g. `if key == PressedKey.Alpha and raw.lower() == "q": ...`).

Escape handling is robust to CSI sequences: bare `Esc` and arrow/delete sequences are distinguished internally.

---

## Recommended controller pattern

For screens like `MainFrame`, use a controller class with:

- state fields (`focus_depth`, selected indices, running flag)
- `_init_layout()` to allocate text objects once
- `_handle_key()` for state transitions
- `_render_all()` to sync visual state
- `run()` for loop + dispatch

Skeleton:

```python
class Screen:
    def __init__(self):
        self.running = False
        self.window = TuiWindow((80, 24), (0, 0))
        self._init_layout()
        self._render_all()

    def run(self):
        self.running = True
        while self.running:
            key, ch = read_key()
            self._handle_key(key, ch)
            self._render_all()
```

---

## Built-in UX helpers

### `prompt(...)`

Single-line editable input with cursor control.

```python
value = prompt(TermText("Node ID"), window, (2, 4), initial="my-node")
```

- Enter accepts
- Escape cancels (`None`)

### `select_option(...)`

Arrow-key menu picker.

```python
res = select_option((10, 5), ["Create", "Load", "Exit"])
```

Returns `(selected_text, command)` where command is:

- `0` for enter/accept
- `1` for delete (only when `allow_delete=True`)

### `TextField`

Convenient wrapper for labeled form fields in an existing window.

---

## Practical styling guidance

- Use `bold=True` for active selection labels
- Use color sparingly (`fg` 256-color index) for focus/severity
- Keep a persistent footer with key hints by focus depth
- Prefer updating existing text IDs over removing/re-adding everything

---

## Common pitfalls and how to avoid them

1. **Forgetting `paint()`**
   - Symptom: state changes but screen does not update.

2. **Terminal not restored on exception**
   - Always keep mode restoration in `finally`.

3. **Using absolute coordinates in child windows**
   - Text coords in `add_text` are window-relative.

4. **Blocking constructor logic**
   - Keep constructors non-blocking; put loops in `run()`.

5. **Recreating text objects each frame**
   - Store IDs and update with `update_text` for less flicker and simpler state management.

---

## Suggested workflow for new TUI features

1. Add state fields and constants
2. Add/adjust layout text IDs in `_init_layout()`
3. Implement key transitions in `_handle_key()`
4. Update `_render_all()` and sub-renderers
5. Verify enter/escape behavior and cleanup
6. Manually validate in terminal (arrows, enter, esc, q, redraw behavior)

---

## Reference locations

- Core primitives: `src/language_pipes/tui/tui.py`
- Terminal lifecycle: `src/language_pipes/tui/screen_utils.py`, `src/language_pipes/tui/__init__.py`
- Keyboard parser: `src/language_pipes/tui/kb_utils.py`
- Prompt helpers: `src/language_pipes/tui/prompt.py`, `src/language_pipes/tui/text_field.py`
- Example controller usage: `src/language_pipes/tui/main_frame.py`, `src/language_pipes/tui/main_menu.py`
