#!/usr/bin/env bash

set -euo pipefail

PLUGIN_NAME="clawmem"
PLUGIN_REPO="clawmem-ai/clawmem-hermes-plugin"
DEFAULT_GIT_BASE_URL="https://git.clawmem.ai"
DEFAULT_CONSOLE_BASE_URL="https://console.clawmem.ai"
DEFAULT_REPO_NAME="hermes-memory"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGIN_DIR="$HERMES_HOME/plugins/$PLUGIN_NAME"
HERMES_ENV_FILE="$HERMES_HOME/.env"
CLAWMEM_CONFIG_FILE="$HERMES_HOME/clawmem.json"
CLAWMEM_GIT_BASE_URL="${CLAWMEM_GIT_BASE_URL:-}"
CLAWMEM_CONSOLE_BASE_URL="${CLAWMEM_CONSOLE_BASE_URL:-}"
HERMES_PROJECT_HINT="${HERMES_PROJECT_ROOT:-}"
FORCE_INSTALL=0

info() {
  printf -- '-> %s\n' "$*"
}

success() {
  printf 'OK %s\n' "$*"
}

warn() {
  printf 'WARN %s\n' "$*" >&2
}

fail() {
  printf 'ERROR %s\n' "$*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

require_cmd() {
  have_cmd "$1" || fail "missing required command: $1"
}

is_hermes_project_root() {
  local candidate="$1"
  [ -n "$candidate" ] || return 1
  [ -f "$candidate/hermes_cli/main.py" ] || return 1
  [ -d "$candidate/plugins/memory" ] || return 1
}

extract_project_path() {
  awk '
    BEGIN { IGNORECASE = 1 }
    /^[[:space:]]*Project:[[:space:]]*/ {
      sub(/^[[:space:]]*Project:[[:space:]]*/, "", $0)
      print
      exit
    }
  '
}

detect_hermes_project_root() {
  local candidate=""
  local output=""

  if is_hermes_project_root "$HERMES_PROJECT_HINT"; then
    printf '%s\n' "$HERMES_PROJECT_HINT"
    return 0
  fi

  if have_cmd hermes; then
    output="$(HERMES_HOME="$HERMES_HOME" hermes version 2>/dev/null || true)"
    candidate="$(printf '%s\n' "$output" | extract_project_path)"
    if is_hermes_project_root "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi

    output="$(HERMES_HOME="$HERMES_HOME" hermes status 2>/dev/null || true)"
    candidate="$(printf '%s\n' "$output" | extract_project_path)"
    if is_hermes_project_root "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  for candidate in "$HERMES_HOME/hermes-agent" "$HOME/.hermes/hermes-agent"; do
    if is_hermes_project_root "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

read_env_value() {
  local key="$1"
  local file="$2"

  [ -f "$file" ] || return 0
  awk -F= -v key="$key" '
    $1 == key {
      value = substr($0, index($0, "=") + 1)
    }
    END {
      if (value != "") {
        print value
      }
    }
  ' "$file"
}

read_json_value() {
  local key="$1"
  local file="$2"
  local python_bin="$3"

  [ -f "$file" ] || return 0
  "$python_bin" - "$file" "$key" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
key = sys.argv[2]

try:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

value = payload.get(key)
if value is None:
    raise SystemExit(0)
if isinstance(value, str):
    value = value.strip()
if value == "":
    raise SystemExit(0)

print(value)
PY
}

upsert_env_value() {
  local key="$1"
  local value="$2"
  local file="$3"
  local tmp_file

  mkdir -p "$(dirname "$file")"
  tmp_file="${file}.tmp.$$"

  if [ -f "$file" ]; then
    awk -v key="$key" -v value="$value" '
      BEGIN { updated = 0 }
      index($0, key "=") == 1 {
        if (!updated) {
          print key "=" value
          updated = 1
        }
        next
      }
      { print }
      END {
        if (!updated) {
          print key "=" value
        }
      }
    ' "$file" >"$tmp_file"
  else
    printf '%s=%s\n' "$key" "$value" >"$tmp_file"
  fi

  mv "$tmp_file" "$file"
}

derive_prefix_login() {
  # Match _get_profile_name() logic in __init__.py
  local home="$HERMES_HOME"
  local parent
  parent="$(dirname "$home")"
  if [ "$(basename "$parent")" = "profiles" ]; then
    basename "$home"
  else
    printf 'hermes\n'
  fi
}

bootstrap_agent() {
  local git_url="$1"
  local prefix_login="$2"
  local repo_name="$3"

  require_cmd curl

  local api_url="${git_url%/}/api/v3/agents"
  local response

  info "registering agent identity at $api_url"
  response="$(
    curl -fsSL \
      --connect-timeout 5 \
      --max-time 20 \
      --retry 2 \
      --retry-delay 1 \
      --retry-connrefused \
      -X POST \
      -H "Accept: application/vnd.github+json" \
      -H "Content-Type: application/json" \
      -d "{\"prefix_login\":\"$prefix_login\",\"default_repo_name\":\"$repo_name\"}" \
      "$api_url"
  )"

  printf '%s' "$response"
}

parse_json_field() {
  local json_str="$1"
  local field="$2"
  local python_bin="$3"

  "$python_bin" -c "
import json, sys
data = json.loads(sys.argv[1])
val = data.get(sys.argv[2], '')
if val:
    print(val)
" "$json_str" "$field"
}

verify_token() {
  local token="$1"
  local git_url="$2"
  local login="$3"
  local repo="$4"

  require_cmd curl

  local api_url="${git_url%/}/api/v3"

  info "verifying token connectivity"
  if curl -fsSL \
       --connect-timeout 5 \
       --max-time 10 \
       -H "Authorization: token $token" \
       -H "Accept: application/vnd.github+json" \
       "${api_url}/repos/${repo}" >/dev/null 2>&1; then
    success "token is valid"
    return 0
  else
    return 1
  fi
}

get_hermes_python() {
  local hermes_project_root="$1"
  local hermes_python="$hermes_project_root/.venv/bin/python"

  if [ ! -x "$hermes_python" ]; then
    hermes_python="$(command -v python3 || true)"
  fi

  [ -n "$hermes_python" ] || fail "unable to find a Python interpreter"
  printf '%s\n' "$hermes_python"
}

write_provider_config() {
  local python_bin="$1"
  local git_url="$2"
  local console_url="$3"
  local login="$4"
  local default_repo="$5"

  "$python_bin" - "$CLAWMEM_CONFIG_FILE" "$git_url" "$console_url" "$login" "$default_repo" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
git_url = sys.argv[2]
console_url = sys.argv[3]
login = sys.argv[4]
default_repo = sys.argv[5]

existing = {}
if config_path.exists():
    try:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        existing = {}

existing["git_base_url"] = git_url
existing["console_base_url"] = console_url
existing["login"] = login
existing["default_repo"] = default_repo

config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
PY
}

ensure_memory_symlink() {
  local hermes_project_root="$1"
  local memory_dir="$hermes_project_root/plugins/memory"
  local memory_link="$memory_dir/$PLUGIN_NAME"

  mkdir -p "$memory_dir"

  if [ -L "$memory_link" ]; then
    if [ "$(readlink "$memory_link")" = "$PLUGIN_DIR" ]; then
      success "memory symlink already points at $PLUGIN_DIR"
      return 0
    fi

    rm "$memory_link"
    ln -s "$PLUGIN_DIR" "$memory_link"
    success "updated memory symlink at $memory_link"
    return 0
  fi

  if [ -e "$memory_link" ]; then
    fail "found an existing path at $memory_link; move it away before enabling clawmem"
  fi

  ln -s "$PLUGIN_DIR" "$memory_link"
  success "linked clawmem into $memory_link"
}

cleanup_memory_symlink() {
  local hermes_project_root="$1"
  local memory_link="$hermes_project_root/plugins/memory/$PLUGIN_NAME"

  if [ -L "$memory_link" ] && [ "$(readlink "$memory_link")" = "$PLUGIN_DIR" ]; then
    rm "$memory_link"
    success "removed clawmem symlink from $memory_link"
  fi
}

activate_provider_with_python() {
  local hermes_project_root="$1"
  local python_bin="$2"

  HERMES_HOME="$HERMES_HOME" "$python_bin" - "$hermes_project_root" <<'PY'
import sys

project_root = sys.argv[1]
sys.path.insert(0, project_root)

from hermes_cli.config import load_config, save_config

config = load_config()
memory_config = config.get("memory")
if not isinstance(memory_config, dict):
    memory_config = {}
    config["memory"] = memory_config

memory_config["provider"] = "clawmem"
save_config(config)
PY
}

activate_provider() {
  local hermes_project_root="$1"
  local python_bin="$2"

  if have_cmd hermes; then
    if HERMES_HOME="$HERMES_HOME" hermes config set memory.provider "$PLUGIN_NAME" >/dev/null; then
      success "activated memory.provider=$PLUGIN_NAME via hermes CLI"
      return 0
    fi
    warn "hermes config set failed; trying Python fallback"
  fi

  if activate_provider_with_python "$hermes_project_root" "$python_bin"; then
    success "activated memory.provider=$PLUGIN_NAME via Python fallback"
    return 0
  fi

  return 1
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [ "$#" -gt 0 ]; do
  case "$1" in
    --force)
      FORCE_INSTALL=1
      shift
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

have_cmd hermes || fail "hermes command is required; install Hermes first"

install_args=()
if [ "$FORCE_INSTALL" = "1" ]; then
  install_args+=("--force")
fi

info "installing clawmem via hermes plugins install"
HERMES_HOME="$HERMES_HOME" hermes plugins install "$PLUGIN_REPO" "${install_args[@]}"

[ -f "$PLUGIN_DIR/__init__.py" ] || fail "installed plugin is missing __init__.py"
[ -f "$PLUGIN_DIR/plugin.yaml" ] || fail "installed plugin is missing plugin.yaml"

hermes_project_root="$(detect_hermes_project_root)" || fail "unable to locate the Hermes repo; check HERMES_HOME or Hermes installation"
hermes_python="$(get_hermes_python "$hermes_project_root")"
info "using HERMES_HOME=$HERMES_HOME"
info "using Hermes project root $hermes_project_root"

# Resolve URLs
if [ -z "$CLAWMEM_GIT_BASE_URL" ]; then
  existing_git_url="$(read_json_value "git_base_url" "$CLAWMEM_CONFIG_FILE" "$hermes_python")"
  CLAWMEM_GIT_BASE_URL="${existing_git_url:-$DEFAULT_GIT_BASE_URL}"
fi
if [ -z "$CLAWMEM_CONSOLE_BASE_URL" ]; then
  existing_console_url="$(read_json_value "console_base_url" "$CLAWMEM_CONFIG_FILE" "$hermes_python")"
  CLAWMEM_CONSOLE_BASE_URL="${existing_console_url:-$DEFAULT_CONSOLE_BASE_URL}"
fi

# Check for existing identity
existing_token="$(read_env_value "CLAWMEM_TOKEN" "$HERMES_ENV_FILE")"
existing_login="$(read_json_value "login" "$CLAWMEM_CONFIG_FILE" "$hermes_python")"
existing_repo="$(read_json_value "default_repo" "$CLAWMEM_CONFIG_FILE" "$hermes_python")"

token_verified=1
token_is_new=0

if [ -n "$existing_token" ] && [ -n "$existing_login" ] && [ -n "$existing_repo" ]; then
  success "found existing ClawMem identity: $existing_login / $existing_repo"
  if ! verify_token "$existing_token" "$CLAWMEM_GIT_BASE_URL" "$existing_login" "$existing_repo"; then
    warn "existing token failed connectivity check (network issue or wrong CLAWMEM_GIT_BASE_URL?)"
    warn "keeping existing token — skipping activation"
    token_verified=0
  fi
  login="$existing_login"
  default_repo="$existing_repo"
else
  # Auto-bootstrap: register a new agent identity (no API key needed!)
  prefix_login="$(derive_prefix_login)"
  bootstrap_response="$(bootstrap_agent "$CLAWMEM_GIT_BASE_URL" "$prefix_login" "$DEFAULT_REPO_NAME")"

  login="$(parse_json_field "$bootstrap_response" "login" "$hermes_python")"
  token="$(parse_json_field "$bootstrap_response" "token" "$hermes_python")"
  default_repo="$(parse_json_field "$bootstrap_response" "repo_full_name" "$hermes_python")"

  [ -n "$login" ] || fail "server did not return a login"
  [ -n "$token" ] || fail "server did not return a token"
  [ -n "$default_repo" ] || fail "server did not return a repo_full_name"

  existing_token="$token"
  token_is_new=1
  success "registered new agent identity: $login / $default_repo"
fi

# Save token to .env
upsert_env_value "CLAWMEM_TOKEN" "$existing_token" "$HERMES_ENV_FILE"
if [ "$token_is_new" = "1" ]; then
  success "saved CLAWMEM_TOKEN to $HERMES_ENV_FILE"
else
  success "CLAWMEM_TOKEN unchanged in $HERMES_ENV_FILE"
fi

# Save config
write_provider_config "$hermes_python" "$CLAWMEM_GIT_BASE_URL" "$CLAWMEM_CONSOLE_BASE_URL" "$login" "$default_repo"
if [ -f "$CLAWMEM_CONFIG_FILE" ] && [ "$token_is_new" = "0" ]; then
  success "clawmem config up to date in $CLAWMEM_CONFIG_FILE"
else
  success "saved clawmem config to $CLAWMEM_CONFIG_FILE"
fi

# Symlink
ensure_memory_symlink "$hermes_project_root"

# Activate
if [ "$token_verified" = "1" ]; then
  if ! activate_provider "$hermes_project_root" "$hermes_python"; then
    cleanup_memory_symlink "$hermes_project_root"
    fail "unable to activate memory.provider=$PLUGIN_NAME"
  fi
  printf '\n'
  success "clawmem is ready"
  printf 'Next: start a new Hermes session and run `hermes memory status` to verify.\n'
else
  printf '\n'
  warn "clawmem is installed but NOT activated (token could not be verified)"
  printf 'Next: fix the connectivity issue, then run `hermes memory setup` to complete activation.\n'
fi
