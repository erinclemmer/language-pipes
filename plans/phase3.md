# MainFrame Phase 3 Plan (Provider Data Wiring)

This plan defines the implementation phase after Phase 2 UX stabilization. It is aligned with:

- `plans/mainframe_tui_design_and_implementation_plan.md`
- `plans/phase2.md`

Phase 2 delivered safer navigation, status/footer behavior, refresh hooks, and actionable placeholders. Phase 3 now focuses on replacing placeholder-only content with **live provider-backed summaries** while preserving the interaction model.

---

## Phase 3 Objective

Integrate provider callables into `MainFrame` so each top-level domain can render current operational state without coupling the TUI directly to backend internals.

Primary outcomes:

1. Provider injection contract in `MainFrame`
2. Provider-backed content rendering per tab/section
3. Resilient refresh/error handling and status reporting
4. Predictable fallback behavior when providers are missing/unavailable
5. Lightweight tests for provider dispatch and failure paths

---

## In Scope (Phase 3)

### 1) Provider interface + injection
Introduce optional provider callables in `MainFrame` (constructor arg or structured dict/object), using the design-plan targets:

- `get_network_status() -> dict`
- `list_peers() -> list[dict]`
- `list_models() -> list[dict]`
- `get_pipe_health() -> dict`
- `list_jobs(state: str | None = None) -> list[dict]`
- `list_activity(level: str | None = None) -> list[dict]`

If a provider is not supplied, render the existing placeholder guidance and set an informational status when refreshed.

**Why:** Keeps MainFrame modular and ready for incremental backend integration.

### 2) Section-to-provider rendering map
Replace static placeholder-only content with provider-backed summaries for sections where data exists.

Suggested mapping:

- **Network / Status** → `get_network_status`
- **Network / Peers** → `list_peers`
- **Models** sections → `list_models` (filtered presentation per section)
- **Pipes** sections → `get_pipe_health`
- **Jobs** sections → `list_jobs(state=...)`
- **Activity** sections → `list_activity(level=...)`

Keep output concise and line-oriented (state summary first, then key metrics/list preview).

**Why:** Converts the frame into a real monitoring console without redesigning navigation.

### 3) Refresh and load behavior
Enhance `_refresh_current_view()` to:

- invoke the relevant provider for current tab/section
- update cached view data
- set success/info status on completion
- catch exceptions and set error/warning status with next-step guidance

Refresh should remain explicit (`r`) and deterministic.

**Why:** Operators need a reliable action rhythm and clear feedback when data calls fail.

### 4) Error/empty-state normalization
For provider-backed sections, normalize rendering states:

- **ok:** show summary + key values
- **empty:** explicit "no data yet" + action hint
- **error:** clear failure message + likely recovery step (refresh/reconfigure/check connectivity)

Continue using Phase 2’s actionable tone.

**Why:** Ensures operational clarity under both normal and degraded conditions.

### 5) Lightweight tests for data dispatch
Add unit tests to verify:

- correct provider function is used by active tab/section
- refresh updates status on success
- provider exceptions map to non-crashing error statuses
- missing providers preserve placeholder fallback behavior

**Why:** Protects against regressions as live wiring expands.

---

## Out of Scope (Defer to Later Phases)

- Full edit/configuration forms and persistence workflows (Phase 4)
- Background auto-refresh threading/timers
- Rich table widgets, pagination, or search UX
- Deep diagnostics drilldowns/modal framework
- Broad integration-test expansion beyond focused unit coverage

---

## Proposed Implementation Steps

### Step 1: Extend `MainFrame` constructor for providers
Add optional provider registry parameter and internal defaults.

**Done when:** `MainFrame` can run with or without external providers.

### Step 2: Add provider dispatch helpers
Create helper(s) to resolve current section -> provider call + view formatter.

**Done when:** provider lookup is centralized and not duplicated in key handling.

### Step 3: Implement section formatters
Add compact formatter methods for each top domain (network/models/pipes/jobs/activity).

**Done when:** live data renders cleanly in existing content pane boundaries.

### Step 4: Wire refresh + initial load behavior
Update `_refresh_current_view()` and selected render flow to load/refresh provider data safely.

**Done when:** `r` performs provider call paths and status reflects success/failure.

### Step 5: Harden fallback/error handling
Ensure missing provider, empty payload, malformed payload, and raised exceptions all render actionable outcomes.

**Done when:** no provider path can crash the frame loop.

### Step 6: Add targeted unit tests
Add/extend tests in `tests/language_pipes/unit/` for provider dispatch and refresh status behavior.

**Done when:** section-provider routing and error handling are regression-protected.

---

## Files Expected to Change

Primary:
- `src/language_pipes/tui/main_frame.py`

Possible supporting updates:
- `src/language_pipes/tui/main_menu.py` (if provider objects are passed during frame construction)
- `tests/language_pipes/unit/test_main_frame.py` (or related new unit files)

---

## Acceptance Criteria (Phase 3)

All should be true:

1. `MainFrame` accepts optional provider callables without breaking current flow.
2. Each top domain has at least one section showing live provider-backed summary output.
3. `r` triggers provider refresh for the active view and sets meaningful status.
4. Missing providers gracefully fall back to actionable placeholder messaging.
5. Provider errors do not crash the frame and produce clear error status + next step.
6. Lightweight unit coverage exists for provider dispatch, refresh success, and refresh failure.

---

## Manual Validation Checklist

1. Launch TUI and enter MainFrame with no providers; confirm fallback placeholders still work.
2. Inject mock providers and verify each tab/section shows live summary content.
3. Press `r` across top/side/content focus depths and verify refresh feedback.
4. Simulate provider exception; confirm frame remains responsive and status indicates recovery action.
5. Navigate tabs/sections after repeated refreshes and confirm selection/focus behavior remains stable.
6. Exit via root confirmation and confirm lifecycle handoff behavior is unchanged.

---

## Phase 3 Time Budget (Suggested 120–210 min)

- 30 min: provider contract + constructor/state integration
- 35 min: section dispatch + formatter pass
- 25 min: refresh/error/fallback hardening
- 30–90 min: unit tests + manual validation

---

## Next Phase Preview (Phase 4)

After Phase 3 data visibility is stable, implement edit workflows and persistence UX:

- Network configure editing flow
- Model assignment/edit actions
- save/apply confirmation patterns
- validation messaging and guardrails for disruptive changes
