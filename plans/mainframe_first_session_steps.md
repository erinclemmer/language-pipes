# MainFrame: First Session Implementation Steps

This document defines a **realistic first coding session** to make meaningful progress on `MainFrame` without trying to finish the full TUI architecture at once.

The goal is to deliver a usable vertical slice that is achievable in one focused context window.

---

## Session Objective

Implement a **functional navigation skeleton** for `MainFrame`:

1. MainFrame enters a key-driven loop (`run()`)
2. User can switch top tabs and side-nav options
3. Content pane updates with placeholder text for selected view
4. User can exit cleanly with `Esc`/`q`
5. `main_menu.py` calls `frame.run()` instead of relying on constructor blocking

This gives a complete interaction backbone for future data wiring.

---

## Why this is achievable in one session

- No backend data integration required yet
- No new dependencies
- Reuses existing primitives (`TuiWindow`, `read_key`, `TermText`)
- Changes concentrated in 2 files:
  - `src/language_pipes/tui/main_frame.py`
  - `src/language_pipes/tui/main_menu.py`

---

## Scope for Session 1 (Do / Don’t)

## Do
- Add `MainFrame.run()` event loop
- Add tab/section state tracking
- Render selected section title and placeholder content
- Implement keyboard controls:
  - arrows for nav
  - `Enter` to move focus deeper
  - `Esc` to move focus up / exit
  - `q` for quick exit
- Add basic footer key hints

## Don’t (defer)
- Live network/model/jobs data providers
- Edit forms and persistence from content pane
- Advanced modal/dialog framework
- Automated tests for this first pass (optional next session)

---

## Concrete Step-by-Step Plan

### Step 1: Refactor `main_frame.py` into controller structure
- Move constructor-only behavior into:
  - `_init_layout()`
  - `_render_all()`
  - `run()`
- Remove `sys.stdin.read(1)` blocking pattern.

**Done when:** constructing `MainFrame(...)` does not block; `run()` drives UI.

### Step 2: Complete TopNav/SideNav interaction methods
- Add stored option text IDs and cursor marker IDs
- Add methods:
  - `move_next()` / `move_prev()`
  - `set_focus(is_focused: bool)`
  - `set_options(options: list[str])` for side-nav updates by tab

**Done when:** arrows visibly move selection and focused control is obvious.

### Step 3: Implement MainFrame state and dispatch
- Add state:
  - `active_top_idx`
  - `focus_depth`
  - `side_idx_by_tab`
  - `running`
- Add tab→section mapping constant.
- Add `_handle_key()` with core behavior.

**Done when:** user can navigate top/side levels and exit safely.

### Step 4: Render placeholder content pane
- Add `_render_content()` that displays:
  - active top tab
  - active side section
  - placeholder lines indicating future data panel

**Done when:** selection changes are immediately visible in content area.

### Step 5: Wire `main_menu.py` to lifecycle
- Replace direct call:
  - from: `MainFrame(...)`
  - to:
    - `frame = MainFrame(...)`
    - `frame.run()`

**Done when:** user reaches interactive frame after config load.

### Step 6: Manual validation pass
Run TUI and verify:
1. open TUI and load/create config
2. navigate tabs with arrows
3. move through side nav
4. `Esc` backs out and can leave frame
5. no crashes on empty/default config values

---

## Acceptance Criteria for Session 1

All must be true:

1. `MainFrame` has a non-blocking constructor and a `run()` loop
2. Top and side navigation are keyboard-operable
3. Focus depth behavior is predictable (top ↔ side ↔ content)
4. Content pane updates for every tab/section combination
5. `main_menu.py` starts and runs the frame loop
6. User can exit frame without terminal corruption

---

## Suggested Session Time Budget (90–120 min)

- 20 min: nav class updates (TopNav/SideNav)
- 35 min: MainFrame state + event loop + dispatch
- 20 min: content/footer rendering
- 10 min: main_menu integration
- 15–30 min: manual run-through and bug fixes

---

## Next Session (after this one)

1. Add provider-based data hooks for Network/Peers/Jobs/Activity
2. Implement `Network -> Configure` editing flow
3. Add lightweight unit tests for key dispatch/state transitions
