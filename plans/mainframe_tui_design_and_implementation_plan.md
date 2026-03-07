# MainFrame TUI Design and Implementation Plan

## Scope
This document outlines a comprehensive **user experience design** and **implementation plan** to complete `src/language_pipes/tui/main_frame.py` as a functional operations console for Language Pipes.

It is based on existing project interfaces and behavior in:
- `src/language_pipes/tui/main_menu.py`
- `src/language_pipes/tui/prompt.py`
- `src/language_pipes/tui/text_field.py`
- `src/language_pipes/tui/kb_utils.py`
- CLI interaction style in `src/language_pipes/commands/start.py` and `src/language_pipes/util/user_prompts.py`

---

## 1) UX Goals and Principles

The TUI should feel like an operator dashboard that is:

1. **Fast to learn**
   - Consistent keyboard controls
   - Clear context and visible active selection
2. **Safe**
   - Confirm destructive actions
   - Avoid silent data loss
3. **Recoverable**
   - `Esc` always steps back
   - User can return to main menu without killing the process
4. **Action-oriented**
   - Every empty/error state has a next step
   - Common operations reachable in 1–2 navigational steps
5. **Transparent**
   - Show status, progress, and errors in clear language
   - Surface network/model/job state summaries first

---

## 2) End-to-End User Journey

### Entry
1. User launches: `language-pipes tui`
2. Banner and load animation shown (existing behavior)
3. User creates/loads config
4. `MainFrame` opens in `Network -> Status`

### First-run readiness
1. User sees node/network summary
2. If setup is incomplete, UI suggests `Network -> Configure`
3. User edits values and saves

### Capacity and topology validation
1. User checks `Models` for local model readiness
2. User checks `Pipes` for route/capacity overview
3. User checks `Network -> Peers` for connectivity

### Runtime monitoring
1. User inspects `Jobs` (queue/running/history)
2. User monitors `Activity` (events/warnings/errors)

### Exit
- `Esc` backs out by focus depth and modal stack
- At root: offer `Return to main menu`, `Exit TUI`, `Cancel`

---

## 3) Information Architecture

## Top-level tabs
- Network
- Models
- Pipes
- Jobs
- Activity

## Side navigation per tab

### Network
- Status
- Peers
- Configure

### Models
- Installed
- Assignments
- Validation

### Pipes
- Local Capacity
- Routes
- Health

### Jobs
- Queue
- Running
- History

### Activity
- Events
- Warnings/Errors
- Diagnostics

This preserves the architecture and operational concepts already represented in docs and CLI.

---

## 4) Interaction Model

## Focus depths
- `0`: Top nav
- `1`: Side nav
- `2`: Content pane

## Key bindings
- `Left/Right`: switch top tabs (when top focused)
- `Up/Down`: move selection in side/content lists
- `Enter`: activate selected action/item
- `Esc`: back one level / close modal
- `r`: refresh current view
- `q`: quit prompt

Optional enhancement:
- Add `Tab` and `Shift+Tab` support in `kb_utils` for depth switching.

## Navigation behavior rules
1. Wrap-around selection in menu lists
2. Keep per-tab side-nav selection state when switching tabs
3. Never steal focus during periodic refresh
4. Use footer hints based on current focus depth

---

## 5) Screen Layout Design

## Persistent frame regions
1. **Header**: app name/version/config context
2. **Top nav**: high-level domain switching
3. **Side nav**: local section navigation
4. **Content pane**: details and actions
5. **Footer**: key hints + transient status messages

## Visual indicators
- Focused target uses explicit delimiters (`|> ... <|`) or color emphasis
- Section titles bolded where supported
- Severity colors for status/events:
  - info (neutral),
  - warning (yellow),
  - error (red),
  - success (green)

---

## 6) Screen-by-Screen UX Spec

## Network / Status
Displays:
- Node ID
- Listening port / address
- Bootstrap status
- Connected peer count
- Connectivity state badge (Connected / Degraded / Offline)

Actions:
- Refresh
- Open Configure

Empty/error guidance:
- "Configuration incomplete. Open Configure to continue."

## Network / Peers
Displays list/table:
- Peer ID, address, latency, last seen, trust info

Actions:
- Refresh list
- Inspect peer details (content pane subview)

## Network / Configure
Editable fields using `TextField`/`prompt` style:
- `node_id`
- `network_key`
- `bootstrap_address`
- `bootstrap_port`
- whitelist values

Behavior:
- Validate as user confirms each field
- Show inline field error in content area
- Confirm before applying disruptive changes

## Models / Installed
Displays:
- discovered local model IDs
- readiness/availability summary

If none:
- "No local models found" + actionable guidance

## Models / Assignments
Displays:
- current layer model assignments
- end model assignments

Actions:
- add/edit/remove assignment with validation

## Models / Validation
Displays:
- hash validation mode and latest status

Actions:
- toggle validation setting

## Pipes / Local Capacity
Displays:
- max pipes
- current utilization
- estimated available capacity

## Pipes / Routes
Displays:
- model-to-route summary
- local vs remote route distribution

## Pipes / Health
Displays:
- route health summary and degraded segments

## Jobs / Queue, Running, History
Displays:
- queued jobs
- active jobs + key runtime metrics
- historical jobs with status filters

## Activity / Events, Warnings/Errors, Diagnostics
Displays:
- chronological event stream
- severity-focused views
- diagnostics details for troubleshooting

---

## 7) Copywriting and Feedback Standards

Message format:
- **What happened**
- **Why it matters** (if needed)
- **What to do next**

Examples:
- "Could not parse bootstrap port. Enter a number between 1 and 65535."
- "Configuration saved. Network reconnect may take a few seconds."
- "No peers connected. Check bootstrap settings in Configure."

Destructive confirmation style:
- "Delete `<name>`? This action cannot be undone."

---

## 8) Implementation Architecture

## MainFrame responsibilities
`MainFrame` should become a controller with:
- state model
- input dispatch
- region rendering
- action execution hooks

### Proposed state fields
- `active_top_idx: int`
- `focus_depth: int`
- `side_idx_by_tab: dict[int, int]`
- `content_cursor_idx: int`
- `running: bool`
- `status_message: str`
- `status_level: str`
- `view_cache` for currently displayed content

### Proposed methods
- `run()`
- `_handle_key(key, raw)`
- `_set_focus(depth)`
- `_switch_top_tab(delta)`
- `_switch_side_item(delta)`
- `_activate_selection()`
- `_render_all()`
- `_render_shell()`
- `_render_top_nav()`
- `_render_side_nav()`
- `_render_content()`
- `_render_footer()`

### TopNav / SideNav improvements
- store option IDs for cursor movement
- maintain left/right marker IDs
- explicit `set_focus`, `move_next`, `move_prev`
- avoid full redraw where possible (update text IDs)

---

## 9) Data Integration Plan (Decoupled)

Avoid hard-coding backend objects in TUI layer. Use provider callables:

- `get_network_status() -> dict`
- `list_peers() -> list[dict]`
- `list_models() -> list[dict]`
- `get_pipe_health() -> dict`
- `list_jobs(state: str | None) -> list[dict]`
- `list_activity(level: str | None) -> list[dict]`

MainFrame should accept optional providers and fallback to placeholder/mock summaries if absent.

---

## 10) Concrete File-Level Plan

1. **`src/language_pipes/tui/main_frame.py`**
   - Complete interactive nav classes and MainFrame controller loop
   - Add render methods and content placeholders
2. **`src/language_pipes/tui/main_menu.py`**
   - Replace direct constructor-only usage with:
     - `frame = MainFrame(...)`
     - `frame.run()`
3. **`src/language_pipes/tui/kb_utils.py`** (optional)
   - Add Tab/BackTab key parsing for focus cycling
4. **Optional new file**: `src/language_pipes/tui/view_models.py`
   - constants/enums for tabs, sections, status levels

---

## 11) Delivery Phases

## Phase 1: Functional skeleton
- Full keyboard loop
- Top/side navigation
- Placeholder content pages for all sections

## Phase 2: UX polish
- Footer hint system
- Status notifications
- Quit/back confirmation flows

## Phase 3: Data wiring
- Inject providers
- Populate pages with live summaries

## Phase 4: Edit workflows
- Configure + model assignment forms
- Validation and save/apply messaging

## Phase 5: Tests
- Unit-test state transitions and dispatch logic
- Basic regression around navigation behavior

---

## 12) Definition of Done for MainFrame Completion

`MainFrame` is considered complete when:

1. User can navigate all top/side sections via keyboard
2. Focus handling is deterministic and visible
3. Esc/back behavior is consistent and safe
4. Content pane updates according to selected section
5. Error/empty states are explicit and actionable
6. Main menu integration calls `frame.run()`
7. Core navigation logic has unit tests

---

## 13) Risks and Mitigations

1. **Terminal rendering artifacts**
   - Mitigate with incremental updates and strict region boundaries
2. **Input ambiguity for Escape sequences**
   - Keep current robust `read_key` fallback behavior
3. **Over-coupling UI to runtime internals**
   - Use provider interfaces and dependency injection
4. **Growing complexity in single file**
   - Split enums/view models if class grows too large

---

## 14) Recommended First Implementation Cut

If implementing immediately, start with:
1. MainFrame event loop and focus handling
2. Interactive TopNav/SideNav with proper cursor movement
3. Per-tab side mappings + placeholder content rendering
4. Footer help/status messaging

This creates a complete, testable UX skeleton before backend wiring.
