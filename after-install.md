# ClawMem Plugin — Post-Install Steps

> **Used `install.sh`?** Everything below is already done — no action needed.
> The steps below are only for manual `hermes plugins install` users.

---

`hermes plugins install` places the plugin files into your Hermes plugin directory,
but Hermes discovers memory providers from a separate path (`plugins/memory/`
inside the Hermes repo). Two more steps are needed:

## 1. Create the symlink

```bash
bash "${HERMES_HOME:-$HOME/.hermes}/plugins/clawmem/scripts/link-memory-provider.sh"
```

If the script cannot find your Hermes repo, set `HERMES_PROJECT_ROOT`:

```bash
HERMES_PROJECT_ROOT=/path/to/hermes-agent \
  bash "${HERMES_HOME:-$HOME/.hermes}/plugins/clawmem/scripts/link-memory-provider.sh"
```

## 2. Run the interactive setup

```bash
hermes memory setup
```

- Select **clawmem** in the picker
- Choose User (default) or Developer mode
- The setup auto-registers an agent identity — no API key needed
- A token and config are saved automatically

## Verify

```bash
hermes memory status   # should show clawmem as active
```
