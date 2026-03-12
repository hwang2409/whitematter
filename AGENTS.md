# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

**Branch per agent:** Each agent MUST work on its own branch. Create the branch at the **start** of the task and push to that branch when done. Do not work on `main` (or the default branch) for agent-delivered changes.

**dongha.md:** When you make substantive changes to the project, update `dongha.md` (local dev log, gitignored) with a new numbered section describing what was implemented. Add entries above the final “*When you implement another plan...*” line.
## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Starting Work (Create Your Branch First)

**Before making any changes**, create and switch to a new branch for this agent’s work:

```bash
git fetch origin
git checkout -b agent/<branch-name>
```

- **Branch name:** Use a short, descriptive name, e.g. `agent/fix-predict-tab`, `agent/add-models-filter`, or `agent/<issue-id>-<slug>` if working on a bd issue.
- All commits for this task MUST be on this branch. Do not commit to `main` or another shared branch.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY. Push **this agent’s branch** (not necessarily main):
   ```bash
   git pull --rebase origin main   # or default branch, to stay current
   bd sync
   git push -u origin HEAD        # push this branch and set upstream
   git status                     # MUST show "up to date with origin/<branch>"
   ```
5. **Clean up** - Clear stashes, prune remote branches (do not delete the branch you just pushed)
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Create a **new branch** at the start of the task; do all work on that branch
- Work is NOT complete until `git push` succeeds (pushing this agent’s branch)
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

