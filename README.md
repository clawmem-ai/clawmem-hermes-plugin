# ClawMem Hermes Plugin

Persistent cross-session memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent), powered by [ClawMem](https://clawmem.ai).

Conversations are automatically mirrored as issues, memories are extracted at session end, and recalled semantically each turn — all backed by a GitHub-compatible Git server. No API key needed upfront; the installer auto-registers an agent identity.

## Prerequisites

- Hermes Agent installed and working (`hermes version` should print version info)
- An internet connection (the plugin talks to the ClawMem Git Server)

## Quick Install

Review the script first if you prefer, then run:

```bash
curl -fsSL https://raw.githubusercontent.com/clawmem-ai/clawmem-hermes-plugin/main/install.sh | bash
```

This single command:

1. Installs the plugin via `hermes plugins install`
2. Creates a symlink at `plugins/memory/clawmem` inside the Hermes repo
3. Auto-registers an agent identity (no API key needed)
4. Saves token to `HERMES_HOME/.env` and config to `clawmem.json`
5. Activates `memory.provider=clawmem`

If activation succeeds, start a new Hermes session — ClawMem is active. If the script reports a connectivity failure, run `hermes memory setup` to reconfigure.

## Manual Install

```bash
# 1. Install the plugin files
hermes plugins install clawmem-ai/clawmem-hermes-plugin

# 2. Link into the Hermes memory provider directory
bash "${HERMES_HOME:-$HOME/.hermes}/plugins/clawmem/scripts/link-memory-provider.sh"

# 3. Run the interactive setup (auto-bootstrap + activation)
hermes memory setup        # select "clawmem" in the picker
```

During `hermes memory setup`, the plugin auto-registers an agent identity with the ClawMem server. No API key or account is required.

## Upgrade

```bash
hermes plugins update clawmem
```

This runs `git pull` inside the installed plugin directory. Start a new Hermes session to pick up the changes. Your `.env` and `clawmem.json` are not touched.

If the symlink is missing after an upgrade (e.g. Hermes was reinstalled), re-run:

```bash
bash "${HERMES_HOME:-$HOME/.hermes}/plugins/clawmem/scripts/link-memory-provider.sh"
```

## Verify

```bash
hermes memory status
```

You should see `clawmem` listed as the active memory provider.

## Features

- **Auto-recall** — relevant memories are fetched every turn (prefetch)
- **52 agent tools** (all prefixed `clawmem_`):
  - **Memory CRUD (7)** — `recall`, `store`, `list`, `get`, `update`, `forget`, `console`
  - **Memory schema / routing (5)** — `labels`, `repos`, `repo_create`, `repo_set_default`, `review`
  - **Generic issues (6)** — `issue_create`, `issue_list`, `issue_get`, `issue_update`, `issue_comment_add`, `issue_comments_list`
  - **Collaboration (34)** — orgs, teams, repo collaborators, repo transfer, org invitations, self-side invitations, outside collaborators, and a repo-access inspector
- **Confirmed-write gate** — every destructive collaboration / repo-set-default tool requires `confirmed=true` after explicit user approval
- **Conversation mirroring** — each session is mirrored as a `type:conversation` issue with per-turn comments
- **Session-end extraction** — facts are extracted from the conversation via LLM at session end
- **Memory write mirroring** — Hermes `memory add` commands are mirrored to ClawMem
- **SHA-256 deduplication** — duplicate memories are detected and merged
- **Web console** — browse your memories at `console.clawmem.ai`

> Three OpenClaw features are not portable to Hermes's `manifest_version: 1` contract: per-agent credential maps, a dedicated turn-start review-nudge injection, and multi-file skill packs. See [`docs/plugins/hermes-features-diff.md`](../../docs/plugins/hermes-features-diff.md) for the full parity table and rationale.

## Configuration

All files live under `${HERMES_HOME:-$HOME/.hermes}/`.

**`.env`** — token and optional overrides (env vars take precedence over `clawmem.json`):

| Variable | Description |
|----------|-------------|
| `CLAWMEM_TOKEN` | API token (auto-provisioned by installer) |
| `CLAWMEM_GIT_BASE_URL` | Git server URL override (default: `https://git.clawmem.ai`) |
| `CLAWMEM_CONSOLE_BASE_URL` | Console URL override (default: `https://console.clawmem.ai`) |
| `CLAWMEM_LOGIN` | Login override |
| `CLAWMEM_DEFAULT_REPO` | Default repo override |

**`clawmem.json`** — provider config:

| Key | Description |
|-----|-------------|
| `git_base_url` | Git server URL (default: `https://git.clawmem.ai`) |
| `console_base_url` | Console URL (default: `https://console.clawmem.ai`) |
| `login` | Registered agent login |
| `default_repo` | Default memory repository (e.g. `hermes-abc123/hermes-memory`) |

## Uninstall

```bash
# Remove the symlink and plugin files
rm -f "$(hermes version 2>/dev/null | awk '/Project:/{print $2}')/plugins/memory/clawmem"
hermes plugins remove clawmem

# Switch to another provider or disable memory
hermes config set memory.provider ""
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `hermes memory status` doesn't show clawmem | Run `link-memory-provider.sh` — the symlink may be missing |
| "ClawMem not configured" during chat | Check that `CLAWMEM_TOKEN` is set in `HERMES_HOME/.env` and `clawmem.json` exists |
| Memories not appearing | Check `hermes memory status` and `agent.log` for errors; recall runs every turn |
| Connection errors | Verify `CLAWMEM_GIT_BASE_URL` and check `HERMES_HOME/logs/agent.log` with `logging.level: DEBUG` in `config.yaml` |
| "requires API key" in picker | Should not happen — if it does, ensure `get_config_schema()` returns an empty list |
