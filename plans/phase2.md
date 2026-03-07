# MainFrame Phase 2 Plan (Post Session 1)

This plan defines the **next implementation phase** after the Session 1 navigation skeleton, aligned with the original design in:

- `plans/mainframe_tui_design_and_implementation_plan.md`
- `plans/mainframe_first_session_steps.md`

Session 1 established the event loop, focus-depth navigation, placeholder content, and main-menu lifecycle handoff. Phase 2 should now focus on **UX polish and interaction safety** before full data wiring.

---

## Phase 2 Objective

Upgrade `MainFrame` from a working skeleton to a more operator-ready interface by adding:

1. Footer hint and status messaging system
2. Safe quit/back confirmation behavior at root
3. Better focus/activation behavior in content pane
4. Refresh action (`r`) and predictable no-op handling
5. Clear empty/error placeholders with actionable next steps

---

## In Scope (Phase 2)

### 1) Footer and status line system
- Add persistent footer rendering that combines:
  - context key hints (by focus depth)
  - transient status message (`status_message`, `status_level`)
- Add helper API:
  - `_set_status(message: str, level: str = "info")`
  - `_clear_status()`

**Why:** Matches the original UX requirement for transparent operator feedback.

### 2) Root-level quit/back confirmation
- On `Esc` at focus depth 0, show a minimal confirm interaction:
  - `Return to menu` / `Exit TUI` / `Cancel`
- Keep behavior keyboard-driven and recoverable (`Esc` should never hard-kill unexpectedly).

**Why:** Aligns with safe/recoverable UX principles.

### 3) Content-pane interaction contract
- Define content focus (`focus_depth == 2`) behavior:
  - `Up/Down` supports scrolling/select cursor where applicable (or explicit no-op status for now)
  - `Enter` can trigger placeholder action per section
- Add `_activate_selection()` stub method to centralize future actions.

**Why:** Keeps architecture ready for Phase 3/4 without reworking key dispatch.

### 4) Refresh behavior
- Add `r` key handling to refresh current view (`_refresh_current_view()`).
- If no provider yet, set info status: "Refreshed (placeholder view)".

**Why:** Required by original interaction model and useful for operational rhythm.

### 5) Better empty/error placeholders
- For each tab/section placeholder, include:
  - current state summary
  - explicit next action (e.g., "Open Network -> Configure")
- Distinguish informational vs warning placeholder text.

**Why:** Original design requires actionable empty/error states.

---

## Out of Scope (Defer to Later Phases)

- Real backend provider integration (Phase 3)
- Full edit workflows and persistence forms (Phase 4)
- Modal/dialog framework beyond simple confirm flow
- Comprehensive test suite expansion beyond lightweight unit checks

---

## Proposed Implementation Steps

### Step 1: Extend `MainFrame` state model
Add fields in `src/language_pipes/tui/main_frame.py`:

- `status_message: str`
- `status_level: str` (`info|success|warning|error`)
- optional `content_cursor_idx: int`

**Done when:** footer can render both hints and transient status.

### Step 2: Refactor footer rendering
- Split/clean footer rendering into a dedicated `_render_footer()` method.
- Keep key hints deterministic by focus depth.

**Done when:** every render pass shows stable hints and most-recent status.

### Step 3: Add root confirmation flow
- Introduce a small confirm prompt on `Esc` at root.
- Route user choice to:
  - return to main menu
  - exit process
  - cancel and continue frame

**Done when:** operator cannot accidentally leave frame with a single root `Esc`.

### Step 4: Add dispatch hooks for activate/refresh
- Add `_activate_selection()` and `_refresh_current_view()` methods.
- Wire key handling:
  - `Enter` at content depth calls `_activate_selection()`
  - `r` calls `_refresh_current_view()`

**Done when:** key dispatch remains simple while behavior is extensible.

### Step 5: Improve placeholders per section
- Expand `_render_content()` to use per-section templates/messages.
- Include concrete next-step guidance in each section.

**Done when:** every tab/section view communicates what to do next.

### Step 6: Lightweight tests for state dispatch
Add unit tests for key transitions (target: `tests/language_pipes/unit/`), covering:

- focus-depth movement (`Enter`/`Esc`)
- root confirmation path selection
- refresh status updates
- per-tab side selection retention

**Done when:** key state machine behavior is regression-protected.

---

## Files Expected to Change

Primary:
- `src/language_pipes/tui/main_frame.py`

Possible supporting updates:
- `src/language_pipes/tui/main_menu.py` (if return-to-menu behavior needs explicit handoff handling)
- `src/language_pipes/tui/kb_utils.py` (only if additional key parsing is required)
- `tests/language_pipes/unit/...` (new/updated unit tests)

---

## Acceptance Criteria (Phase 2)

All should be true:

1. Footer always shows valid key hints for current focus depth.
2. Status messages can be set/cleared and are visible without breaking layout.
3. Root `Esc` uses a safe confirmation path (no accidental immediate termination).
4. `r` refresh key is handled and provides user feedback.
5. Content pane gives section-specific, actionable placeholder guidance.
6. At least lightweight unit coverage exists for key state transitions.

---

## Manual Validation Checklist

1. Open TUI, reach `MainFrame`, verify footer hints at each focus depth.
2. Press `r` in top/side/content focus and verify refresh feedback.
3. Press `Esc` from content -> side -> top and confirm stepwise back behavior.
4. Press `Esc` at root and verify confirm flow options work correctly.
5. Navigate all tabs/sections and verify placeholder text remains actionable.
6. Exit and confirm terminal state remains clean.

---

## Phase 2 Time Budget (Suggested 90–150 min)

- 25 min: state + footer/status plumbing
- 25 min: root confirmation flow
- 20 min: refresh/activate dispatch hooks
- 20 min: section placeholder quality pass
- 20–60 min: unit tests + manual run-through

---

## Next Phase Preview (Phase 3)

After Phase 2, proceed with provider-based data integration from the original design:

- `get_network_status`, `list_peers`, `list_models`, `get_pipe_health`, `list_jobs`, `list_activity`
- Replace placeholders with live summaries while preserving the stabilized interaction model.
