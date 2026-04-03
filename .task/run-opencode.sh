#!/usr/bin/env bash
set -euo pipefail
cd "/home/erin/hermes_data/subagents/active/20260403-132423-show-model-loading-state-unloaded-loading-or-loaded-on-models-hosted-page/worktree"
OPENCODE_BIN="${OPENCODE_BIN:-opencode}"
PROMPT_FILE=.task/opencode-prompt.txt
exec "$OPENCODE_BIN" run "$(cat "$PROMPT_FILE")" -f .task/plan.md -f .task/notes.md -f .task/close-template.md
