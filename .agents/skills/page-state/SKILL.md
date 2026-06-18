---
name: page-state
description: How to build multi-view TUI pages with the Page / PageState state-machine base classes (src/language_pipes/tui/components/page.py), using the models_layers package as the reference implementation.
---

# Skill: page-state — multi-view TUI pages

## Overview

A "page" in the Language Pipes TUI is one selectable section of the frame (e.g.
`Models > Layer Models`). Pages that need more than one screen — a list, an
editor, a confirmation menu, a picker — are built as a small **state machine**
on top of two base classes in `src/language_pipes/tui/components/page.py`:

- `PageState` — one view (one screen). Owns its own fields, rendering, and key
  handling.
- `Page` — a container that holds a list of `PageState`s, injects shared
  dependencies, tracks which state is active, and routes input/rendering to it.

Only one state is active at a time. States hand control to each other by name
via `change_state(...)`. This replaces the older pattern of a single monolithic
class with an `Enum` state field and giant `if self.state == ...` branches (see
the history of `models_layers.py`, which was refactored into the
`models_layers/` package described below).

Use this skill when adding a new multi-screen page, or adding a screen to an
existing one.

each page state should have its own file in a folder for the page

---

## The base classes (`page.py`)

### `PageState`

```python
class PageState:
    name: str                                   # unique key within the page
    provider: ContentProvider                   # injected by Page
    confirm: Confirm                            # injected by Page
    exit_page: Callable                         # injected by Page
    change_state: Callable[[str, Dict], None]   # injected by Page

    def __init__(self, name: str): ...
    def on_change(self, args: Dict): ...        # called every time this state becomes active
    def on_key(self, key: PressedKey, ch: str): ...
    def get_view(self) -> List[str]: ...        # body lines
    def get_footer(self) -> str: ...            # key-hint footer (default "")
```

Subclass it, call `super().__init__('<name>')`, and override the hooks you need.
`provider`, `confirm`, `exit_page`, and `change_state` are **set by `Page`
after construction** — never passed to a `PageState.__init__`. Don't reference
them in `__init__`; they aren't bound yet.

### `Page`

```python
class Page:
    def __init__(self, states, provider, confirm, exit_page):
        # wires provider/confirm/exit_page/change_state into every state
        # active state = states[0]
```

A concrete page subclasses `Page` and just registers its states. The first
state in the list is the entry/landing view.

```python
class ModelsLayers(Page):
    def __init__(self, provider, confirm, exit_page):
        super().__init__(
            [
                ListPageState(),       # states[0] -> landing view
                EditPageState(),
                OptionsPageState(),
                ChooseModelPageState(),
                ChooseDevicePageState(),
            ],
            provider, confirm, exit_page,
        )
```

`Page` exposes `on_key`, `get_view`, `get_footer` — all delegate to the active
state — so the page object plugs straight into the page router (see Wiring).

---

## State lifecycle and transitions

- **Transition:** `self.change_state('edit', { ... })`. `Page` finds the state
  whose `name == 'edit'`, makes it active, and calls its `on_change(args)`.
- **`on_change(args)` runs on *every* entry** into a state — both first-time and
  returns from a sub-screen. The `args` dict is how the previous state passes
  data in. If a state needs no input, pass `{ }`.
- **`exit_page()`** leaves the page entirely (back up to the frame nav). Call it
  from the landing state's Escape handler.

### Standard key wiring

Every state follows the same `on_key` shape, dispatching to private helpers:

```python
def on_key(self, key: PressedKey, ch: str):
    if key == PressedKey.ArrowUp:   self._on_prev()
    if key == PressedKey.ArrowDown: self._on_next()
    if key == PressedKey.Enter:     self._on_enter()
    if key == PressedKey.Escape:    self._on_escape()
    # text-entry states also handle PressedKey.Alpha / Backspace
    # list states also handle PressedKey.Delete
```

Selection indices wrap with modulo against the live option count, e.g.
`self.select_idx = (self.select_idx + 1) % (n + 1)` (the `+ 1` accounts for a
trailing "Add ..." row).

---

## The args convention (most important part)

`args` is the *only* channel between states — they don't share fields. Two
distinct uses:

### 1. Opening a state fresh

The opener passes the full context the target needs:

```python
# list -> edit (or list -> options)
self.change_state('edit', { "model": current_model, "model_idx": self.model_idx })
```

### 2. Round-tripping through a sub-selector

When a state opens a picker and expects a value back, the picker returns to the
opener with a **partial** args dict naming just the field it resolved. The
opener's `on_change` must distinguish this from a fresh open so it does **not**
wipe the in-progress form:

```python
# edit.on_change
def on_change(self, args: Dict):
    if "model_id" in args:        # returning from choose_model
        self.model_id = args["model_id"]; return
    if "device" in args:          # returning from choose_device
        self.device_name = args["device"]; return
    if "model" not in args:       # cancelled out of a picker (empty args) -> keep form
        return
    # fresh open: reset the whole form from args["model"] / args["model_idx"]
    ...
```

Pickers send results back, and cancellation sends nothing:

```python
# choose_model
def _on_enter(self):  self.change_state('edit', { "model_id": self.installed_models[self.select_idx] })
def _on_escape(self): self.change_state('edit', { })   # empty args => opener keeps its state
```

**Rule of thumb:** a fresh-open args dict carries a sentinel key the round-trip
never uses (here `"model"`), so the target can branch on key presence. Reset
form state (and `select_idx`, caches) **only** on the fresh-open branch.

---

## Reference implementation

`src/language_pipes/tui/components/models_layers/` is the canonical example:

| File | State | Role |
|------|-------|------|
| `list_state.py` | `list` | landing: lists configured layer models + "Add" row; Enter → `options` (or `edit` when adding / network down), Delete → confirm-remove, Escape → `exit_page()` |
| `edit_state.py` | `edit` | create/edit form; Enter on a field opens `choose_model` / `choose_device`; Save persists and optionally prompts to load |
| `options_state.py` | `options` | per-model menu (Edit / Load·Unload / Back) |
| `choose_model_state.py` | `choose_model` | installed-model picker, returns `{ "model_id": ... }` |
| `choose_device_state.py` | `choose_device` | device picker, opened with `{ "device": ... }`, returns `{ "device": ... }` |

`__init__.py` defines `ModelsLayers(Page)` and registers all five.

---

## Wiring a page into the frame

In `src/language_pipes/tui/frame/page_router.py`:

1. Import the page class: `from language_pipes.tui.components.models_layers import ModelsLayers`
2. Construct it in `__init__` with `(provider, confirm, self.exit_page)`.
3. Return it from `get_page()` for the matching `(tab, section)`.

The router calls `on_key` / `get_view` / `get_footer` on whatever `get_page()`
returns — `Page` satisfies that interface, so no per-page glue is needed.

---

## Rendering helpers (from `tui/util/text.py`)

- `make_selectable_text(text, selected)` — wraps a row with `|> ... <|` cursors
  when selected. Use for every navigable row.
- `make_window_text(entries, selected_idx, max_height)` — scroll-windows a list
  of multi-line entries around the selection, adding `^`/`V` clip indicators.
  Each entry is a `List[str]`; append `""` for inter-row spacing.
- `make_footer_text([...])` — evenly spreads key hints across the footer width.
  Build footers from the active `select_idx` so hints match the focused field.

---

## Gotchas (learned the hard way)

1. **`change_state` must `return` after dispatching.** `Page._change_state`
   loops the states; the success branch needs an explicit `return`, otherwise it
   falls through and raises "State change failed" even on success.
2. **`on_change` fires on re-entry, not just first entry.** Guard form resets
   behind a fresh-open check (see the args convention) or returning from a
   picker will blank the user's input.
3. **Initialize `Optional` fields in `__init__`.** `get_view` may read
   `self.editing_model` etc.; set them to `None` in `__init__` so the state is
   valid even before its first `on_change`.
4. **Derive "adding vs editing" from the payload, not the cursor index.** When
   the landing list's "Add" row is selected it passes `model=None`; map that to
   an index of `None` so a save appends instead of indexing out of range.
5. **Validate the field, not the cursor.** A `_valid_model_id` must test
   `self.model_id`, not `self.select_idx` — easy copy-paste slip when porting
   from index-based code.
6. **Pickers must no-op on empty option lists.** Guard modulo navigation and
   Enter with `if len(options) > 0` to avoid `% 0` / index errors.

---

## Workflow for adding a screen to a page

1. Create `<name>_state.py` with a `PageState` subclass; `super().__init__('<name>')`.
2. Implement `on_change` (branch fresh-open vs round-trip), `on_key`,
   `get_view`, and `get_footer`.
3. Register the instance in the page's `Page` subclass state list.
4. From the opening state, `change_state('<name>', {...})`; have the new state
   `change_state(...)` back with its result (or `{ }` on cancel).
5. Sanity-check: every `change_state('x', ...)` target has a registered state
   with `name == 'x'` (grep both and diff), then import the page router to
   confirm the module graph loads.

---

## Reference locations

- Base classes: `src/language_pipes/tui/components/page.py`
- Reference page: `src/language_pipes/tui/components/models_layers/`
- Router: `src/language_pipes/tui/frame/page_router.py`
- Render helpers: `src/language_pipes/tui/util/text.py`
- Lower-level TUI primitives: see the `tui` skill.
