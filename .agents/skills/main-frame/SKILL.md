---
name: main-frame
description: Reference for MainFrame TUI architecture, key handling, form editing, NodeIdEditor flows, and related tests.
---

## Skill: MainFrame TUI

### Architecture overview
`MainFrame` (`src/language_pipes/tui/frame/main_frame.py`) is the top-level TUI controller. Key dispatch flows through this chain:

```
FrameKeyHandler.handle_key()
  → if editor.edit_mode: _handle_edit_mode_key()
    → Editor methods (prev_field, next_field, on_enter, on_backspace, on_alpha, change_field_editor)
      → Editor delegates to the active form (e.g. NetworkForm)
        → NetworkForm delegates field-specific editing to sub-editors (e.g. NodeIdEditor)
```

### Key components
- **Editor** (`src/language_pipes/tui/frame/editor.py`): State machine for form editing. Tracks `edit_mode`, `edit_field_idx`, `field_editor_visible`, and the active `form`.
- **NetworkForm** (`src/language_pipes/tui/components/network_form.py`): Network config form. Contains a `NodeIdEditor` sub-editor for the node_id field.
- **NodeIdEditor**: Manages node ID selection/registration. Has `registering_node_id`, `new_node_id`, `node_ids`, `select_idx` state.
- **FrameKeyHandler** (`src/language_pipes/tui/frame/frame_key_handler.py`): Routes keypresses based on current UI state (exit_confirm open → confirm open → edit_mode → normal navigation).

### Field editor lifecycle
1. In edit mode, **Enter** on a field calls `Editor.change_field_editor(True)` → sets `field_editor_visible = True`
2. Subsequent keypresses are delegated to the form's sub-editor (e.g. `NodeIdEditor.on_key()`)
3. **Escape** while `field_editor_visible` calls `Editor.change_field_editor(False)` → `form.exit_field_editor()` → sub-editor `restart()` → `form.start()` (re-enters edit mode with field_editor_visible=False)

### Test utilities (`tests/language_pipes/unit/main_frame/util.py`)
- `_make_main_frame(providers=dict)`: Creates a MainFrame with all terminal I/O mocked. The `providers` dict maps `ProviderCall` enum values to callables. The frame auto-navigates to Network/Configure and calls `activate_selection()`.
- `_simulate_keys(frame, [(PressedKey, ch), ...])`: Feeds keypresses through `frame.key_handler.handle_key()` with rendering suppressed.

### Required providers by test scenario
- **Basic edit mode**: `{ProviderCall.get_network_config: ..., ProviderCall.save_network_config: ...}`
- **Node ID editor flows**: Also needs `ProviderCall.get_registered_node_ids: lambda: []` (or a list of existing IDs)
- **Node ID save flows**: Additionally needs `ProviderCall.save_new_node_id: ...`

### NodeIdEditor key sequences
- **Enter "Register new" mode** (no existing IDs): `Enter` (open field editor) → `Enter` (select "Register new node id" at idx 0)
- **Type a node ID**: `(PressedKey.Alpha, "c")` per character
- **Back out to form**: `Escape` (triggers restart → exit_field_editor → back to form field list)

### Test file location
Tests for NetworkForm/NodeIdEditor: `tests/language_pipes/unit/main_frame/components/test_network_form.py`
