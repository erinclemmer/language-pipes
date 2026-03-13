# MainFrame Phase 3 Plan (Provider Data Wiring)

This plan defines the implementation phase after Phase 2 UX stabilization. It is aligned with:

- `plans/mainframe_tui_design_and_implementation_plan.md`
- `plans/phase2.md`

Phase 2 delivered safer navigation, status/footer behavior, refresh hooks, and actionable placeholders. Phase 3 focused on replacing placeholder-only content with **live provider-backed summaries** while preserving the interaction model.

---

## Architecture (as of Phase 3 completion)

The original monolithic `MainFrame` class has been refactored into four focused modules. `MainFrame` itself is now a thin coordinator that owns layout, rendering, and the run loop; all domain concerns live in the classes below.

### `src/language_pipes/tui/nav_state.py` — `NavState`

Owns all navigation cursor state:

- `focus_depth` (0 = top-nav, 1 = side-nav, 2 = content pane)
- `active_top_idx` / `side_idx_by_tab` / `content_cursor_idx`
- Derived helpers: `active_tab()`, `active_side_option()`, `active_view_key()`
- Mutations: `tab_next/prev`, `side_next/prev` (takes `SideNav` to sync widget), `focus_deeper/shallower`, `content_cursor_down/up`

### `src/language_pipes/tui/confirm_dialog.py` — `ExitConfirm`

Owns the exit-confirmation overlay:

- `is_open`, `choice_idx`
- `open()` / `close()`
- `render() -> str` – produces the prompt text
- `handle_key(key) -> str` – returns an action token (`"prev"`, `"next"`, `"confirm"`, `"cancel"`, `"nop"`)

### `src/language_pipes/tui/content_loader.py` — `ContentLoader`

Owns provider resolution, data fetching, and the per-section view-state cache:

- Constructor accepts an optional `providers` dict or object
- `load(tab, section, *, update_status, force) -> dict` – returns a view-state dict, using the cache unless `force=True`
- `invalidate(tab, section)` / `invalidate_all()`
- After each load, `last_status_message` / `last_status_level` reflect the outcome for the caller to surface
- Falls back to `PLACEHOLDERS` guidance when no provider is available

### `src/language_pipes/tui/view_state.py` — pure formatting functions

Stateless module; no class required:

- `build_view_state(state, summary, details, hint, level) -> dict`
- `error_view_state(summary, hint)` / `empty_view_state(summary, hint)`
- Per-domain formatters: `format_network`, `format_models`, `format_pipes`, `format_jobs`, `format_activity`, `format_unknown`
- `section_provider_spec(tab, section) -> (provider_name, kwargs, formatter)` – single source of truth for the section → provider mapping

### `src/language_pipes/tui/main_frame.py` — `MainFrame` (coordinator)

Responsibilities retained in `MainFrame`:

- Window/layout construction (`_init_layout`)
- Rendering (`_render_all`, `_render_content`, `_render_footer`, `_sync_navigation`)
- Input dispatch (`_handle_key`, `_handle_confirm_key`)
- Status string management (`_set_status`, `_clear_status`)
- Run loop (`run`)
- Backward-compatible `@property` shims so existing tests continue to access `focus_depth`, `active_top_idx`, `side_idx_by_tab`, `confirm_escape_open`, `confirm_choice_idx`, `content_cursor_idx`, `view_state_by_section` directly on the frame

---

## Provider Interface

Providers are injected at `MainFrame(size, pos, providers=...)` construction time. `providers` may be a `dict` of callables or any object with matching method names.

| Provider name        | Signature                                  | Used by                          |
|----------------------|--------------------------------------------|----------------------------------|
| `get_network_status` | `() -> dict`                               | Network / Status                 |
| `list_peers`         | `() -> list[dict]`                         | Network / Peers                  |
| `list_models`        | `() -> list[dict]`                         | Models / Installed, Download, Cache |
| `get_pipe_health`    | `() -> dict`                               | Pipes / Overview, Routes, Configure |
| `list_jobs`          | `(state: str | None = None) -> list[dict]` | Jobs / Queue, History, Stats     |
| `list_activity`      | `(level: str | None = None) -> list[dict]` | Activity / Logs, Events, Metrics |

If a provider is absent or raises an exception, `ContentLoader` falls back to placeholder guidance and sets an appropriate status level without crashing the frame.

---

## Rendering States

Each view-state dict produced by `ContentLoader` / `view_state.py` carries a `state` field:

| State         | Meaning                                      |
|---------------|----------------------------------------------|
| `ok`          | Provider returned usable data                |
| `empty`       | Provider returned an empty collection        |
| `error`       | Provider raised an exception or returned malformed data |
| `placeholder` | No provider mapped or provider unavailable   |

---

## Refresh Behavior

- `r` key → `MainFrame._refresh_current_view()` → `ContentLoader.load(..., force=True, update_status=True)`
- On success: status shows `"Refreshed {tab} -> {section}"`
- On empty: status shows `"No data for {tab} -> {section} yet"`
- On error: status shows `"Refresh failed for {tab} -> {section}; check provider"` at `error` level
- On missing provider: status shows `"Provider '{name}' unavailable for {tab} -> {section}; showing guidance"`

---

## Test Coverage

`tests/language_pipes/unit/test_main_frame.py` covers:

- Focus depth transitions (Enter / Escape)
- Root-escape confirmation dialog (open, navigate, confirm, cancel)
- `q` key opens confirmation instead of immediate exit
- Per-tab side-selection retention across tab switches
- Refresh dispatches to the correct provider with correct kwargs
- Provider exception → non-crashing error status
- Missing provider → placeholder fallback

All 13 tests pass.

---

## Files Changed in Phase 3

New:
- `src/language_pipes/tui/nav_state.py`
- `src/language_pipes/tui/confirm_dialog.py`
- `src/language_pipes/tui/content_loader.py`
- `src/language_pipes/tui/view_state.py`

Modified:
- `src/language_pipes/tui/main_frame.py` (refactored to use the four new modules)

Unchanged:
- `src/language_pipes/tui/top_nav.py`
- `src/language_pipes/tui/side_nav.py`
- `src/language_pipes/tui/tui.py`
- `src/language_pipes/tui/placeholders.py`
- `src/language_pipes/tui/kb_utils.py`
- `src/language_pipes/tui/main_menu.py`
- `tests/language_pipes/unit/test_main_frame.py`

---

## Out of Scope (Deferred to Later Phases)

- Full edit/configuration forms and persistence workflows (Phase 4)
- Background auto-refresh threading/timers
- Rich table widgets, pagination, or search UX
- Deep diagnostics drilldowns/modal framework
- Broad integration-test expansion beyond focused unit coverage

---

## Next Phase Preview (Phase 4)

After Phase 3 data visibility is stable, implement edit workflows and persistence UX:

- Network configure editing flow
- Model assignment/edit actions
- Save/apply confirmation patterns
- Validation messaging and guardrails for disruptive changes
