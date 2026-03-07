from typing import Any, Callable, Dict, List, Optional, Tuple

from language_pipes.tui.top_nav import TopNav
from language_pipes.tui.side_nav import SideNav
from language_pipes.tui.tui import TuiWindow, TermText
from language_pipes.tui.kb_utils import PressedKey, read_key

class MainFrame:
    TOP_HEADERS = ["Network", "Models", "Pipes", "Jobs", "Activity"]
    SIDE_OPTIONS_BY_TAB: Dict[str, List[str]] = {
        "Network": ["Status", "Peers", "Configure"],
        "Models": ["Installed", "Download", "Cache"],
        "Pipes": ["Overview", "Routes", "Configure"],
        "Jobs": ["Queue", "History", "Stats"],
        "Activity": ["Logs", "Events", "Metrics"],
    }

    top_nav: TopNav
    side_nav: SideNav
    window: TuiWindow
    focus_depth: int
    active_top_idx: int
    side_idx_by_tab: Dict[str, int]
    running: bool
    exit_tui: bool
    content_id: int
    footer_id: int
    status_message: str
    status_level: str
    content_cursor_idx: int
    confirm_escape_open: bool
    confirm_choice_idx: int
    content_bg_id: int
    content_area_size: Tuple[int, int]
    providers: Optional[object]
    view_state_by_section: Dict[Tuple[str, str], Dict[str, Any]]

    def __init__(self, size: Tuple[int, int], pos: Tuple[int, int], providers: Optional[object] = None):
        self.window = TuiWindow(size, pos)
        self.active_top_idx = 0
        self.focus_depth = 0
        self.side_idx_by_tab = {
            tab: 0 for tab in self.TOP_HEADERS
        }
        self.running = False
        self.exit_tui = False
        self.status_message = ""
        self.status_level = "info"
        self.content_cursor_idx = 0
        self.confirm_escape_open = False
        self.confirm_choice_idx = 2
        self.providers = providers
        self.view_state_by_section = {}

        self._init_layout(size, pos)
        self._render_all()

    def _init_layout(self, size: Tuple[int, int], pos: Tuple[int, int]):
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, 2))
        self.window.add_text(TermText("|\n" * (size[1] - 5)), (15, 3))
        self.window.add_text(TermText("_" * (size[0] - 2)), (1, size[1] - 3))

        self.content_area_size = (max(1, size[0] - 19), max(1, size[1] - 7))
        self.content_bg_id = self.window.add_text(
            TermText(self._content_blank_block()),
            (17, 4),
        )
        self.content_id = self.window.add_text(TermText(""), (17, 4))
        self.footer_id = self.window.add_text(TermText(""), (2, size[1] - 2))

        self.top_nav = TopNav((80, 1), (pos[0], pos[1] + 1), self.TOP_HEADERS)
        self.side_nav = SideNav(
            (13, size[1] - 5),
            (pos[0] + 1, pos[1] + 4),
            self._active_side_options(),
        )

    def _active_tab(self) -> str:
        return self.TOP_HEADERS[self.active_top_idx]

    def _active_side_options(self) -> List[str]:
        return self.SIDE_OPTIONS_BY_TAB.get(self._active_tab(), ["Overview"])

    def _active_side_idx(self) -> int:
        return self.side_idx_by_tab.get(self._active_tab(), 0)

    def _active_side_option(self) -> str:
        options = self._active_side_options()
        if len(options) == 0:
            return ""
        idx = min(self._active_side_idx(), len(options) - 1)
        return options[idx]

    def _footer_text(self) -> str:
        if self.confirm_escape_open:
            return "Confirm Exit: Arrows U/D or L/R to choose   Enter: Confirm   Esc: Cancel"
        if self.focus_depth == 0:
            return "Arrows L/R: Switch Tab   Enter: Side Nav   Esc: Back/Quit Options   q: Exit"
        if self.focus_depth == 1:
            return "Arrows U/D: Switch Section   Enter: Content   Esc: Top Tabs   q: Exit"
        return "Arrows U/D: Navigate Placeholder   Enter: Activate   r: Refresh   Esc: Back   q: Exit"

    def _status_text(self) -> str:
        if self.status_message == "":
            return ""
        return f"[{self.status_level.upper()}] {self.status_message}"

    def _set_status(self, message: str, level: str = "info"):
        self.status_message = message
        self.status_level = level

    def _clear_status(self):
        self.status_message = ""
        self.status_level = "info"

    def _confirm_options(self) -> List[str]:
        return ["Return to menu", "Exit TUI", "Cancel"]

    def _render_confirm_prompt(self) -> str:
        options = self._confirm_options()
        lines = [
            "Safe exit confirmation",
            "",
            "Use arrows to choose an option, then press Enter:",
        ]
        for i, opt in enumerate(options):
            cursor = ">" if i == self.confirm_choice_idx else " "
            lines.append(f"{cursor} {opt}")
        lines.extend([
            "",
            "Esc: cancel and continue working in MainFrame",
        ])
        return "\n".join(lines)

    def _section_placeholder(self) -> Tuple[str, str, str]:
        placeholders: Dict[str, Dict[str, Tuple[str, str, str]]] = {
            "Network": {
                "Status": (
                    "No live network provider connected yet.",
                    "Next: Open Network -> Configure to review bootstrap and identity settings.",
                    "info",
                ),
                "Peers": (
                    "Peer list unavailable in placeholder mode.",
                    "Next: Open Network -> Configure, then refresh after provider wiring in Phase 3.",
                    "warning",
                ),
                "Configure": (
                    "Network configuration editor is not wired in this phase.",
                    "Next: Keep credentials/config ready; edit workflow arrives in Phase 4.",
                    "info",
                ),
            },
            "Models": {
                "Installed": (
                    "Installed model inventory is not loaded yet.",
                    "Next: Use CLI model commands now; this panel will show provider data in Phase 3.",
                    "info",
                ),
                "Download": (
                    "Download workflow placeholder only.",
                    "Next: Pre-stage model assets, then return here for guided actions in Phase 4.",
                    "warning",
                ),
                "Cache": (
                    "Cache usage and eviction stats unavailable.",
                    "Next: Run refresh after provider wiring to inspect cache health.",
                    "info",
                ),
            },
            "Pipes": {
                "Overview": (
                    "Pipe topology summary not yet connected.",
                    "Next: Validate your routes in config; live topology appears in Phase 3.",
                    "info",
                ),
                "Routes": (
                    "Route table is currently placeholder-only.",
                    "Next: Open Pipes -> Configure for route editing in a later phase.",
                    "warning",
                ),
                "Configure": (
                    "Pipe configuration editor pending.",
                    "Next: Use current config files as source of truth until edit flows land.",
                    "info",
                ),
            },
            "Jobs": {
                "Queue": (
                    "Active job queue telemetry not connected.",
                    "Next: Trigger a workload, then refresh once job provider integration is enabled.",
                    "info",
                ),
                "History": (
                    "Historical jobs are not surfaced in this phase.",
                    "Next: Capture logs now; timeline UI arrives with backend integration.",
                    "warning",
                ),
                "Stats": (
                    "Job throughput stats unavailable.",
                    "Next: Use Activity -> Metrics placeholder for expected future summary shape.",
                    "info",
                ),
            },
            "Activity": {
                "Logs": (
                    "Live logs are not connected to the content pane yet.",
                    "Next: Tail CLI logs externally; in-TUI stream arrives in a later phase.",
                    "info",
                ),
                "Events": (
                    "Event timeline provider is not active.",
                    "Next: Refresh after Phase 3 data wiring to inspect operational events.",
                    "warning",
                ),
                "Metrics": (
                    "Metrics feed is currently placeholder-only.",
                    "Next: Use this section to validate navigation while wiring metrics provider next.",
                    "info",
                ),
            },
        }

        tab = self._active_tab()
        section = self._active_side_option()
        return placeholders.get(tab, {}).get(
            section,
            (
                "No placeholder registered for this section.",
                "Next: Return to top tabs and choose a known section.",
                "warning",
            ),
        )

    def _active_view_key(self) -> Tuple[str, str]:
        return self._active_tab(), self._active_side_option()

    def _view_state(self, state: str, summary: str, details: List[str], hint: str, level: str) -> Dict[str, Any]:
        normalized_details = [str(line) for line in details if str(line).strip() != ""]
        return {
            "state": state,
            "summary": summary,
            "details": normalized_details,
            "hint": hint,
            "level": level,
        }

    def _placeholder_view_state(self) -> Dict[str, Any]:
        summary, hint, level = self._section_placeholder()
        return self._view_state(
            "placeholder",
            summary,
            [],
            hint,
            level,
        )

    def _error_view_state(self, summary: str, hint: str) -> Dict[str, Any]:
        return self._view_state("error", summary, [], hint, "error")

    def _empty_view_state(self, summary: str, hint: str) -> Dict[str, Any]:
        return self._view_state("empty", summary, [], hint, "info")

    def _get_provider(self, name: str) -> Optional[Callable[..., Any]]:
        if self.providers is None:
            return None
        if isinstance(self.providers, dict):
            provider = self.providers.get(name)
            return provider if callable(provider) else None

        provider = getattr(self.providers, name, None)
        return provider if callable(provider) else None

    def _section_provider_spec(self) -> Tuple[Optional[str], Dict[str, Any], Callable[[str, str, Any], Dict[str, Any]]]:
        tab = self._active_tab()
        section = self._active_side_option()

        if tab == "Network" and section == "Status":
            return "get_network_status", {}, self._format_network
        if tab == "Network" and section == "Peers":
            return "list_peers", {}, self._format_network

        if tab == "Models":
            return "list_models", {}, self._format_models

        if tab == "Pipes":
            return "get_pipe_health", {}, self._format_pipes

        if tab == "Jobs":
            state_map = {
                "Queue": "queued",
                "History": "completed",
                "Stats": None,
            }
            return "list_jobs", {"state": state_map.get(section)}, self._format_jobs

        if tab == "Activity":
            level_map = {
                "Logs": "info",
                "Events": "event",
                "Metrics": "metrics",
            }
            return "list_activity", {"level": level_map.get(section)}, self._format_activity

        return None, {}, self._format_unknown

    def _dict_preview(self, payload: Dict[str, Any], limit: int = 5) -> List[str]:
        preview: List[str] = []
        for idx, key in enumerate(payload.keys()):
            if idx >= limit:
                break
            preview.append(f"- {key}: {payload[key]}")
        if len(payload) > limit:
            preview.append(f"- ... ({len(payload) - limit} more)")
        return preview

    def _item_name(self, item: Dict[str, Any], fallback_idx: int) -> str:
        for key in ("name", "id", "model", "peer", "route", "event"):
            value = item.get(key)
            if value:
                return str(value)
        return f"item-{fallback_idx + 1}"

    def _format_network(self, tab: str, section: str, payload: Any) -> Dict[str, Any]:
        if section == "Status":
            if not isinstance(payload, dict):
                return self._error_view_state(
                    "Malformed network status payload.",
                    "Next: Confirm get_network_status returns a dict, then press r.",
                )
            if len(payload) == 0:
                return self._empty_view_state(
                    "No network status data yet.",
                    "Next: Start networking components and press r to refresh.",
                )

            health = payload.get("status") or payload.get("state") or payload.get("health") or "unknown"
            details = self._dict_preview(payload)
            return self._view_state(
                "ok",
                f"Network health: {health}",
                details,
                "Next: Use Network -> Peers for connected node details.",
                "info",
            )

        if section == "Peers":
            if not isinstance(payload, list):
                return self._error_view_state(
                    "Malformed peers payload.",
                    "Next: Confirm list_peers returns a list of dict items, then press r.",
                )
            if len(payload) == 0:
                return self._empty_view_state(
                    "No peers connected yet.",
                    "Next: Check bootstrap connectivity and press r.",
                )

            lines: List[str] = []
            for i, peer in enumerate(payload[:5]):
                if isinstance(peer, dict):
                    name = self._item_name(peer, i)
                    addr = peer.get("address") or peer.get("host") or "unknown"
                    state = peer.get("state") or peer.get("status") or "unknown"
                    lines.append(f"- {name} @ {addr} ({state})")
                else:
                    lines.append(f"- {peer}")

            if len(payload) > 5:
                lines.append(f"- ... ({len(payload) - 5} more peers)")

            return self._view_state(
                "ok",
                f"Connected peers: {len(payload)}",
                lines,
                "Next: Press r to re-check discovery/connectivity.",
                "info",
            )

        return self._format_unknown(tab, section, payload)

    def _format_models(self, _: str, section: str, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, list):
            return self._error_view_state(
                "Malformed models payload.",
                "Next: Confirm list_models returns a list of dict items, then press r.",
            )
        if len(payload) == 0:
            return self._empty_view_state(
                "No model data available.",
                "Next: Ensure model manager is initialized and press r.",
            )

        lines: List[str] = []
        for i, model in enumerate(payload[:6]):
            if isinstance(model, dict):
                name = self._item_name(model, i)
                state = model.get("status")
                if section == "Cache":
                    cache_size = model.get("cache_size") or model.get("cache")
                    if cache_size is not None:
                        lines.append(f"- {name}: cache={cache_size}")
                    else:
                        lines.append(f"- {name}: cache info unavailable")
                elif state is not None:
                    lines.append(f"- {name}: {state}")
                else:
                    lines.append(f"- {name}")
            else:
                lines.append(f"- {model}")

        if len(payload) > 6:
            lines.append(f"- ... ({len(payload) - 6} more models)")

        section_label = {
            "Installed": "Installed models",
            "Download": "Model download candidates",
            "Cache": "Model cache summary",
        }.get(section, "Model summary")

        return self._view_state(
            "ok",
            f"{section_label}: {len(payload)}",
            lines,
            "Next: Press r to refresh model inventory.",
            "info",
        )

    def _format_pipes(self, _: str, section: str, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return self._error_view_state(
                "Malformed pipe health payload.",
                "Next: Confirm get_pipe_health returns a dict, then press r.",
            )
        if len(payload) == 0:
            return self._empty_view_state(
                "No pipe health data yet.",
                "Next: Start routing/services and press r.",
            )

        summary_health = payload.get("health") or payload.get("status") or payload.get("state") or "unknown"
        details = self._dict_preview(payload)
        return self._view_state(
            "ok",
            f"{section} health: {summary_health}",
            details,
            "Next: Verify route configuration if health is degraded.",
            "info",
        )

    def _format_jobs(self, _: str, section: str, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, list):
            return self._error_view_state(
                "Malformed jobs payload.",
                "Next: Confirm list_jobs returns a list of dict items, then press r.",
            )
        if len(payload) == 0:
            return self._empty_view_state(
                "No job data in this view.",
                "Next: Run workloads and press r.",
            )

        if section == "Stats":
            by_state: Dict[str, int] = {}
            for item in payload:
                if isinstance(item, dict):
                    state = str(item.get("state") or "unknown")
                else:
                    state = "unknown"
                by_state[state] = by_state.get(state, 0) + 1
            lines = [f"- {state}: {count}" for state, count in sorted(by_state.items())]
            return self._view_state(
                "ok",
                f"Tracked jobs: {len(payload)}",
                lines,
                "Next: Use Queue/History sections for item-level detail.",
                "info",
            )

        lines: List[str] = []
        for i, job in enumerate(payload[:6]):
            if isinstance(job, dict):
                job_id = self._item_name(job, i)
                state = job.get("state") or "unknown"
                progress = job.get("progress")
                if progress is None:
                    lines.append(f"- {job_id} ({state})")
                else:
                    lines.append(f"- {job_id} ({state}, {progress})")
            else:
                lines.append(f"- {job}")
        if len(payload) > 6:
            lines.append(f"- ... ({len(payload) - 6} more jobs)")

        return self._view_state(
            "ok",
            f"{section} jobs: {len(payload)}",
            lines,
            "Next: Press r to sync the latest job state.",
            "info",
        )

    def _format_activity(self, _: str, section: str, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, list):
            return self._error_view_state(
                "Malformed activity payload.",
                "Next: Confirm list_activity returns a list, then press r.",
            )
        if len(payload) == 0:
            return self._empty_view_state(
                "No activity captured yet.",
                "Next: Trigger workload/network activity and press r.",
            )

        lines: List[str] = []
        for i, item in enumerate(payload[:6]):
            if isinstance(item, dict):
                name = self._item_name(item, i)
                level = item.get("level") or item.get("type") or "info"
                message = item.get("message") or item.get("summary")
                if message:
                    lines.append(f"- [{level}] {name}: {message}")
                else:
                    lines.append(f"- [{level}] {name}")
            else:
                lines.append(f"- {item}")

        if len(payload) > 6:
            lines.append(f"- ... ({len(payload) - 6} more events)")

        return self._view_state(
            "ok",
            f"{section} entries: {len(payload)}",
            lines,
            "Next: Press r to pull the latest activity snapshot.",
            "info",
        )

    def _format_unknown(self, tab: str, section: str, _: Any) -> Dict[str, Any]:
        return self._view_state(
            "placeholder",
            f"No provider mapping for {tab} -> {section}.",
            [],
            "Next: Select another section or wire a provider for this view.",
            "warning",
        )

    def _load_active_view_data(self, update_status: bool, force: bool) -> Dict[str, Any]:
        tab, section = self._active_view_key()
        view_key = (tab, section)

        if not force and view_key in self.view_state_by_section:
            return self.view_state_by_section[view_key]

        provider_name, kwargs, formatter = self._section_provider_spec()
        if provider_name is None:
            view_state = self._placeholder_view_state()
            self.view_state_by_section[view_key] = view_state
            if update_status:
                self._set_status(
                    f"No provider mapping for {tab} -> {section}; showing guidance",
                    "info",
                )
            return view_state

        provider = self._get_provider(provider_name)
        if provider is None:
            view_state = self._placeholder_view_state()
            self.view_state_by_section[view_key] = view_state
            if update_status:
                self._set_status(
                    f"Provider '{provider_name}' unavailable for {tab} -> {section}; showing guidance",
                    "info",
                )
            return view_state

        try:
            provider_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            payload = provider(**provider_kwargs)
            view_state = formatter(tab, section, payload)
        except Exception as ex:
            view_state = self._error_view_state(
                f"Provider call failed for {tab} -> {section}: {ex}",
                "Next: Verify provider connectivity/configuration, then press r.",
            )

        self.view_state_by_section[view_key] = view_state

        if update_status:
            if view_state["state"] == "ok":
                self._set_status(f"Refreshed {tab} -> {section}", "info")
            elif view_state["state"] == "empty":
                self._set_status(f"No data for {tab} -> {section} yet", "info")
            elif view_state["state"] == "error":
                self._set_status(f"Refresh failed for {tab} -> {section}; check provider", "error")
            else:
                self._set_status(f"Showing guidance for {tab} -> {section}", "info")

        return view_state

    def _render_content(self):
        self.window.update_text(self.content_bg_id, TermText(self._content_blank_block()))

        if self.confirm_escape_open:
            self.window.update_text(self.content_id, TermText(self._render_confirm_prompt()))
            return

        tab = self._active_tab()
        section = self._active_side_option()
        view_state = self._load_active_view_data(update_status=False, force=False)
        state_summary = str(view_state.get("summary", "No summary available."))
        next_action = str(view_state.get("hint", "Next: Press r to refresh."))
        level = str(view_state.get("level", "info"))
        details = view_state.get("details", [])

        detail_lines: List[str] = []
        if isinstance(details, list) and len(details) > 0:
            detail_lines.extend([str(line) for line in details])

        selection_hint = f"Selection index: {self.content_cursor_idx + 1}"
        state_label = str(view_state.get("state", "placeholder")).upper()

        content_parts = [
            f"View: {tab}",
            f"Section: {section}",
            "",
            f"State ({state_label}/{level.upper()}): {state_summary}",
        ]

        if len(detail_lines) > 0:
            content_parts.extend([
                "",
                "Details:",
                *detail_lines,
            ])

        content_parts.extend([
            "",
            f"Next Action: {next_action}",
            "",
            selection_hint,
            f"Focus depth: {self.focus_depth} (0=top, 1=side, 2=content)",
        ])

        content = "\n".join([
            *content_parts,
        ])
        self.window.update_text(self.content_id, TermText(content))

    def _content_blank_block(self) -> str:
        width, height = self.content_area_size
        return "\n".join([" " * width for _ in range(height)])

    def _sync_navigation(self):
        active_options = self._active_side_options()
        self.side_nav.focused_idx = min(self._active_side_idx(), len(active_options) - 1) if len(active_options) > 0 else 0
        self.side_nav.set_options(active_options)

        self.top_nav.focused_idx = self.active_top_idx
        self.top_nav.set_focus(self.focus_depth == 0 and not self.confirm_escape_open)
        self.side_nav.set_focus(self.focus_depth == 1 and not self.confirm_escape_open)

    def _render_footer(self):
        footer_base = self._footer_text()
        status = self._status_text()
        footer_text = footer_base if status == "" else f"{footer_base}   |   {status}"
        self.window.update_text(self.footer_id, TermText(footer_text))

    def _activate_selection(self):
        self._set_status(
            f"Activated {self._active_tab()} -> {self._active_side_option()} (placeholder action)",
            "info",
        )

    def _refresh_current_view(self):
        self._load_active_view_data(update_status=True, force=True)

    def _confirm_prev(self):
        options = self._confirm_options()
        self.confirm_choice_idx = (self.confirm_choice_idx - 1) % len(options)

    def _confirm_next(self):
        options = self._confirm_options()
        self.confirm_choice_idx = (self.confirm_choice_idx + 1) % len(options)

    def _resolve_confirm_choice(self):
        choice = self._confirm_options()[self.confirm_choice_idx]
        self.confirm_escape_open = False

        if choice == "Return to menu":
            self.exit_tui = False
            self.running = False
            return
        if choice == "Exit TUI":
            self.exit_tui = True
            self.running = False
            return

        self._set_status("Exit canceled", "info")

    def _handle_confirm_key(self, key: PressedKey):
        if key in (PressedKey.ArrowUp, PressedKey.ArrowLeft):
            self._confirm_prev()
            return
        if key in (PressedKey.ArrowDown, PressedKey.ArrowRight):
            self._confirm_next()
            return
        if key == PressedKey.Enter:
            self._resolve_confirm_choice()
            return
        if key == PressedKey.Escape:
            self.confirm_escape_open = False
            self._set_status("Exit canceled", "info")

    def _open_exit_confirm(self):
        self.confirm_escape_open = True
        self.confirm_choice_idx = 2
        self._set_status("Choose: Return to menu, Exit TUI, or Cancel", "warning")

    def _render_all(self):
        self._sync_navigation()
        self._render_content()
        self._render_footer()

        self.window.paint()
        self.top_nav.window.paint()
        self.side_nav.window.paint()

    def _teardown_windows(self):
        self.window.remove_all()
        self.top_nav.window.remove_all()
        self.side_nav.window.remove_all()

        self.window.paint()
        self.top_nav.window.paint()
        self.side_nav.window.paint()

    def _handle_key(self, key: PressedKey, ch: str):
        if self.confirm_escape_open:
            self._handle_confirm_key(key)
            return

        if key == PressedKey.Alpha and ch.lower() == "q":
            self._open_exit_confirm()
            return

        if key == PressedKey.Alpha and ch.lower() == "r":
            self._refresh_current_view()
            return

        if key == PressedKey.Escape:
            if self.focus_depth > 0:
                self.focus_depth -= 1
                if self.focus_depth < 2:
                    self.content_cursor_idx = 0
            else:
                self._open_exit_confirm()
            return

        if key == PressedKey.Enter:
            if self.focus_depth < 2:
                self.focus_depth += 1
            else:
                self._activate_selection()
            return

        if self.focus_depth == 0:
            if key == PressedKey.ArrowRight:
                self.active_top_idx = (self.active_top_idx + 1) % len(self.TOP_HEADERS)
                self.content_cursor_idx = 0
                self._clear_status()
            elif key == PressedKey.ArrowLeft:
                self.active_top_idx = (self.active_top_idx - 1) % len(self.TOP_HEADERS)
                self.content_cursor_idx = 0
                self._clear_status()
            return

        if self.focus_depth == 1:
            if key == PressedKey.ArrowDown:
                self.side_nav.move_next()
                self.side_idx_by_tab[self._active_tab()] = self.side_nav.focused_idx
                self.content_cursor_idx = 0
                self._clear_status()
            elif key == PressedKey.ArrowUp:
                self.side_nav.move_prev()
                self.side_idx_by_tab[self._active_tab()] = self.side_nav.focused_idx
                self.content_cursor_idx = 0
                self._clear_status()
            return

        if self.focus_depth == 2:
            if key == PressedKey.ArrowDown:
                self.content_cursor_idx += 1
                self._set_status(
                    "Moved selection cursor (placeholder content)",
                    "info",
                )
            elif key == PressedKey.ArrowUp:
                self.content_cursor_idx = max(0, self.content_cursor_idx - 1)
                self._set_status(
                    "Moved selection cursor (placeholder content)",
                    "info",
                )
            elif key in (PressedKey.ArrowLeft, PressedKey.ArrowRight):
                self._set_status("No horizontal action in placeholder content", "info")

    def run(self) -> str:
        self.running = True
        self.exit_tui = False
        self._render_all()
        while self.running:
            key, ch = read_key()
            self._handle_key(key, ch)
            self._render_all()

        self._teardown_windows()

        return "exit" if self.exit_tui else "menu"
