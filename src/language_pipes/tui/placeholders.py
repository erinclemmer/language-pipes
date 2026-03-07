PLACEHOLDERS = {
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