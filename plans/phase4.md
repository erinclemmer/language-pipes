# MainFrame Phase 4 Plan (Edit Workflows and Persistence)

This plan defines the implementation phase after Phase 3 provider data wiring. It is aligned with:

- `plans/mainframe_tui_design_and_implementation_plan.md`
- `plans/phase3.md`

Phase 3 delivered live provider-backed summaries, resilient refresh/error handling, and the `NavState` / `ConfirmDialog` / `ContentLoader` / `view_state` architecture. Phase 4 now focuses on **edit workflows and persistence** so operators can configure and manage the node from within the TUI.

---

## Phase 4 Objective

Replace read-only placeholder sections with interactive edit forms that read from and write to the node's configuration and runtime state.

Primary outcomes:

1. `Network / Configure` edit form (node identity, bootstrap, key)
2. `Models / Assignments` edit form (layer and end-model assignments)
3. `Models / Validation` toggle
4. Save/apply confirmation pattern with inline validation feedback
5. Lightweight tests for form dispatch and validation paths

---

## Architecture Context

Phase 4 builds on the Phase 3 module split. The relevant extension points are:

| Module | Role in Phase 4 |
|---|---|
| `NavState` | Unchanged; focus depth 2 now enters an edit sub-mode |
| `ConfirmDialog` | Reused for save/apply and destructive-action confirmations |
| `ContentLoader` | Extended with write-back methods alongside existing read methods |
| `view_state.py` | New `form_view_state` helper for rendering field-level validation errors |
| `MainFrame` | Gains `_enter_edit_mode()` / `_exit_edit_mode()` and routes `Enter` at depth 2 to the active form |

---

## In Scope (Phase 4)

### 1) `Network / Configure` edit form

Fields (using existing `TextField` / `prompt` primitives):

- `node_id` (string)
- `network_key` / `aes_key` (string, masked display)
- `bootstrap_address` (string)
- `bootstrap_port` (integer, validated 1–65535)

Behavior:

- Pre-populate fields from the current config file (via a `get_network_config() -> dict` provider)
- Validate each field on `Enter`; show inline error in the content pane if invalid
- On final field confirmation, show a `ConfirmDialog`-style prompt: `"Apply changes? Network reconnect may take a few seconds."`
- On confirm: call a `save_network_config(data: dict)` provider; set success/error status
- On cancel: discard edits and return to read-only view

**Why:** Operators need to configure bootstrap and identity without leaving the TUI.

### 2) `Models / Assignments` edit form

Fields:

- Layer model assignments (list of `{layer_idx, model_id}` pairs)
- End model assignment (`model_id`)

Behavior:

- Show current assignments from `list_models()` provider
- Allow adding/editing/removing assignments via `select_option` + `prompt` primitives
- Validate that referenced model IDs exist in the installed model list
- Confirm before applying; call a `save_model_assignments(data: dict)` provider

**Why:** Model assignment is a core operational action that currently requires CLI.

### 3) `Models / Validation` toggle

Behavior:

- Show current validation mode (on/off) from provider
- `Enter` at depth 2 toggles the setting
- Confirm before applying if toggling off (destructive direction)
- Call a `set_validation_mode(enabled: bool)` provider

**Why:** Operators need a quick way to enable/disable hash validation without editing config files.

### 4) Save/apply confirmation pattern

Introduce a reusable `EditConfirmDialog` (or extend `ConfirmDialog`) with options:

- `"Apply"` / `"Discard"` / `"Cancel"`

This replaces ad-hoc inline prompts and keeps the confirmation UX consistent with the existing exit-confirm pattern.

**Why:** Consistent confirmation reduces operator error and aligns with the design plan's "safe" principle.

### 5) Inline validation feedback

Extend `view_state.py` with:

```python
def form_view_state(
    fields: List[Dict[str, Any]],   # [{name, value, error}]
    hint: str,
    level: str,
) -> Dict[str, Any]: ...
```

`MainFrame._render_content()` renders field rows with inline error annotations when `state == "form"`.

**Why:** Operators need to see exactly which field failed and why without leaving the form.

### 6) Lightweight tests for edit dispatch

Add unit tests in `tests/language_pipes/unit/` covering:

- `Enter` at depth 2 on an editable section opens the form
- Field validation errors produce `form` view state with correct error annotations
- Save provider is called with correct payload on confirm
- Discard returns to the previous read-only view state
- Provider write failure sets error status without crashing

---

## Out of Scope (Defer to Later Phases)

- Background auto-refresh threading/timers
- Rich table widgets, pagination, or search UX
- Deep diagnostics drilldowns / modal framework
- `Pipes / Routes` and `Pipes / Health` edit flows (Phase 5)
- `Activity` write actions (Phase 5)
- Broad integration-test expansion beyond focused unit coverage

---

## New Provider Contracts

Phase 4 requires write-side providers in addition to the existing read-side ones:

| Provider name              | Signature                                    | Used by                        |
|----------------------------|----------------------------------------------|--------------------------------|
| `get_network_config`       | `() -> dict`                                 | Network / Configure (pre-fill) |
| `save_network_config`      | `(data: dict) -> None`                       | Network / Configure (save)     |
| `save_model_assignments`   | `(data: dict) -> None`                       | Models / Assignments (save)    |
| `set_validation_mode`      | `(enabled: bool) -> None`                    | Models / Validation (toggle)   |

All write providers follow the same injection pattern as Phase 3 read providers: passed via `providers` dict/object at `MainFrame` construction time. Missing write providers disable the edit action and show an informational status.

---

## Proposed Implementation Steps

### Step 1: Add `form_view_state` to `view_state.py`

Extend the stateless formatting module with a `form` state type and field-level error rendering.

**Done when:** `_render_content()` can display a field list with inline error annotations.

### Step 2: Add `EditConfirmDialog` (or extend `ConfirmDialog`)

Reusable `Apply / Discard / Cancel` overlay, consistent with the existing exit-confirm pattern.

**Done when:** any edit form can invoke a standard save-confirmation flow.

### Step 3: Implement `Network / Configure` form

Wire `get_network_config` pre-fill, `TextField`-based editing, field validation, and `save_network_config` write-back.

**Done when:** operator can edit and save network config from within the TUI.

### Step 4: Implement `Models / Assignments` form

Wire `list_models` for available model IDs, assignment list editing, and `save_model_assignments` write-back.

**Done when:** operator can add/edit/remove model assignments from within the TUI.

### Step 5: Implement `Models / Validation` toggle

Wire current mode display and `set_validation_mode` toggle with confirmation for the off direction.

**Done when:** operator can toggle validation mode with a single `Enter` + confirm.

### Step 6: Add targeted unit tests

Cover form dispatch, validation error paths, save provider calls, and discard behavior.

**Done when:** edit form state machine is regression-protected.

---

## Files Expected to Change

Primary:
- `src/language_pipes/tui/main_frame.py` (edit mode routing, form rendering)
- `src/language_pipes/tui/view_state.py` (add `form_view_state`)
- `src/language_pipes/tui/content_loader.py` (add write-provider dispatch)

New:
- `src/language_pipes/tui/edit_confirm_dialog.py` (or extend `confirm_dialog.py`)

Possible supporting updates:
- `src/language_pipes/tui/placeholders.py` (update Configure/Assignments/Validation placeholders)
- `tests/language_pipes/unit/test_main_frame.py` (or new `test_edit_forms.py`)

---

## Acceptance Criteria (Phase 4)

All should be true:

1. `Network / Configure` shows current config values and allows editing and saving.
2. `Models / Assignments` shows current assignments and allows add/edit/remove.
3. `Models / Validation` shows current mode and allows toggling with confirmation.
4. Invalid field input shows an inline error message without crashing or leaving the form.
5. Save confirmation uses a consistent `Apply / Discard / Cancel` pattern.
6. Missing write providers disable the edit action gracefully with an informational status.
7. Provider write failures set error status without crashing the frame.
8. Unit tests cover form dispatch, validation, save, and discard paths.

---

## Manual Validation Checklist

1. Navigate to `Network / Configure`; verify fields pre-populate from config.
2. Edit a field with invalid input; verify inline error appears.
3. Complete all fields and confirm; verify `save_network_config` is called and status reflects success.
4. Press `Discard` on the confirmation; verify edits are dropped and read-only view is restored.
5. Navigate to `Models / Assignments`; add and remove an assignment; confirm save.
6. Navigate to `Models / Validation`; toggle off; verify confirmation prompt appears.
7. Simulate a write provider exception; verify frame remains responsive and error status is shown.
8. Exit via root confirmation and verify lifecycle handoff is unchanged.

---

## Phase 4 Time Budget (Suggested 150–240 min)

- 20 min: `form_view_state` + `EditConfirmDialog`
- 40 min: `Network / Configure` form
- 40 min: `Models / Assignments` form
- 20 min: `Models / Validation` toggle
- 30–80 min: unit tests + manual validation

---

## Next Phase Preview (Phase 5)

After Phase 4 edit workflows are stable:

- `Pipes / Routes` and `Pipes / Health` edit/action flows
- `Activity` write actions (clear logs, acknowledge events)
- Comprehensive test suite expansion including navigation regression suite
- Optional: background auto-refresh with non-blocking timer thread
