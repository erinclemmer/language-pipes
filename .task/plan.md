# Task Plan

Repo: /home/erin/prog/language-pipes
Task folder: /home/erin/hermes_data/subagents/active/20260403-180842-models-instance-list
Agent guidelines: /home/erin/hermes_data/AGENT.md
Workflow references:
- /home/erin/hermes_data/README.md
- /home/erin/hermes_data/knowledgebase/README.md

Goal:
- Fix Models Hosted page to show all instances spawned from host_model calls instead of just configured models

Constraints:
- Follow /home/erin/hermes_data/AGENT.md.
- Read /home/erin/hermes_data/README.md and /home/erin/hermes_data/knowledgebase/README.md before making changes.
- Do not lint files.
- Do not add tests unless the user explicitly asked for them.
- Only change the minimum required lines.
- Work in an isolated git worktree and task branch.
- Write close.md in this task folder. If that is blocked, write opencode-close.md in the worktree and copy it back later.
- Ensure the work is committed on task/models-instance-list before cleanup.

Starting point:
- Base branch in main checkout: tui
- Task branch: task/models-instance-list

Relevant files to inspect first:
- None identified yet.

Pre-check:
-  M src/language_pipes/tui/components/dashboard.py
-  M tests/language_pipes/unit/main_frame/components/test_dashboard.py
- ?? test_dashboard_logic.py
- ?? test_dashboard_logic2.py

