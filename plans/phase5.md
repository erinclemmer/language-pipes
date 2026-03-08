# MainFrame Phase 5 Plan (Operational Actions and Workflow Hardening)

This document provides an overview of **current MainFrame TUI functionality** after Phase 4 work, then defines the Phase 5 direction.

It is aligned with:

- `plans/mainframe_tui_design_and_implementation_plan.md`
- `plans/phase3.md`
- `plans/phase4.md`

---

## Current Functionality Overview (Post-Phase 4)

The TUI now supports both read-side provider summaries and targeted edit workflows.

### 1) Navigation and frame lifecycle

- Three-depth navigation model remains in place (`focus_depth`):
  - `0`: top tabs
  - `1`: side sections
  - `2`: content
- Exit flow uses `ConfirmDialog` (`Return to menu` / `Exit TUI` / `Cancel`)
- Footer/status behavior remains consistent and non-crashing

### 2) Provider-backed read views (Phase 3 baseline retained)

`ContentLoader` still handles mapped provider dispatch with cache + status updates:

- Network: `Status`, `Peers`
- Models: `Installed`, `Assignments`, `Validation`
- Pipes: `Overview`, `Routes`, `Configure`
- Jobs: `Queue`, `History`, `Stats`
- Activity: `Logs`, `Events`, `Metrics`

Fallback behavior for missing providers is still placeholder-first and safe.

### 3) Edit architecture now available

The frame now includes explicit edit-mode state and confirmation flow:

- `edit_mode` field editing state in `MainFrame`
- `EditConfirmDialog` with `Apply / Discard / Cancel`
- Inline field-level validation surfaced through `form_view_state(...)`

This gives a consistent save/apply UX without disrupting existing exit confirmation UX.

### 4) Implemented edit workflows

#### Network / Configure

- Pre-fill via `get_network_config`
- Editable fields:
  - `node_id`
  - `network_key` (masked in form display)
  - `bootstrap_address`
  - `bootstrap_port`
- Validation:
  - required text fields
  - `bootstrap_port` must be integer in `1..65535`
- Save flow:
  - confirm with `EditConfirmDialog`
  - apply calls `save_network_config(data=...)`
  - discard exits edit mode without saving

#### Models / Assignments

- Source data from `list_models`
- Editable fields:
  - `layer_assignments` string (`layer:model` comma format)
  - `end_model_id`
- Validation:
  - parse/format checks
  - duplicate/negative/non-int layer checks
  - model ID existence check against installed IDs
- Save flow:
  - confirm with `EditConfirmDialog`
  - apply calls `save_model_assignments(data=...)`

#### Models / Validation

- Current mode read from `get_validation_mode` (or fallback config read path)
- `Enter` on section toggles mode
- Disabling requires confirmation dialog (safety direction)
- Apply calls `set_validation_mode(enabled=...)`

### 5) Error handling and safety guarantees

- Missing write providers disable edit actions gracefully with info status
- Provider exceptions during save/apply set error status and keep frame responsive
- No crash path was introduced in edit state machine under tested scenarios

### 6) Unit test coverage now in place

`tests/language_pipes/unit/test_main_frame.py` includes coverage for:

- Edit dispatch at depth 2 for editable sections
- Inline validation error behavior for forms
- Apply path calling save providers with expected payload
- Discard path restoring read-only state
- Provider write-failure behavior (error status, no crash)

Current targeted test count: **18 passing tests**.

---

## Phase 5 Objective

Build on the now-stable edit foundation to support operational actions and broaden coverage without regressing the safe interaction model.

Primary outcomes:

1. Add action/edit flows for `Pipes / Routes` and `Pipes / Health`-adjacent operational controls
2. Add `Activity` write actions (e.g., clear/acknowledge) with confirmation safety
3. Normalize edit flow internals (shared field descriptors + reusable validators)
4. Expand unit tests around multi-step edit/confirm state transitions
5. Add focused integration-level smoke checks for provider wiring and action sequencing

---

## Proposed Phase 5 Work Items

### Step 1: Consolidate form/action state helpers

- Extract common edit field descriptor patterns and validation helpers from `MainFrame`
- Keep `MainFrame` as coordinator, reduce per-section branching complexity

### Step 2: Add `Pipes` operational actions

- Implement safe action flows for route-level operations
- Reuse `EditConfirmDialog` for destructive actions

### Step 3: Add `Activity` write actions

- Introduce explicit action handlers for event/log maintenance operations
- Ensure clear status messaging + failure isolation

### Step 4: Harden test coverage

- Add regression tests for:
  - confirm-cancel loops
  - repeated apply/discard cycles
  - provider unavailability for action endpoints
  - state reset correctness when leaving edit/action mode

### Step 5: Documentation updates

- Update TUI behavior docs once Phase 5 actions are merged
- Keep provider contract docs aligned with newly required write-side endpoints

---

## Files Most Likely to Change in Phase 5

Primary:

- `src/language_pipes/tui/main_frame.py`
- `src/language_pipes/tui/content_loader.py`
- `src/language_pipes/tui/view_state.py`
- `src/language_pipes/tui/edit_confirm_dialog.py` (if additional options/metadata needed)

Tests:

- `tests/language_pipes/unit/test_main_frame.py`
- Potential new focused test modules under `tests/language_pipes/unit/`

---

## Acceptance Criteria for Phase 5 Planning Baseline

This plan is ready when all are true:

1. Current post-Phase-4 functionality is captured accurately in one place
2. Phase 5 scope is concrete and limited to operational actions + hardening
3. Safety model (`Apply/Discard/Cancel` + explicit destructive confirms) remains central
4. Testing and documentation expansion are part of the phase definition
