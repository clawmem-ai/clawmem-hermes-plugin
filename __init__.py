"""ClawMem memory plugin — GitHub-compatible issue-backed long-term memory.

Provides 7 agent-facing tools, auto-recall (prefetch), conversation mirroring
to ``type:conversation`` issues, session-end memory extraction via auxiliary
LLM, and built-in memory write mirroring.

Config chain:
  1. Environment variables (CLAWMEM_*)
  2. $HERMES_HOME/clawmem.json
  3. Hardcoded defaults

See docs/specs/memory-plugin-clawmem.md for the full design spec.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_GIT_BASE_URL = "https://git.clawmem.ai"
_DEFAULT_CONSOLE_BASE_URL = "https://console.clawmem.ai"


def _load_config() -> dict:
    """Load ClawMem config with env var overrides.

    Priority: env var > clawmem.json > default.
    """
    from hermes_constants import get_hermes_home

    defaults: dict[str, str] = {
        "git_base_url": _DEFAULT_GIT_BASE_URL,
        "console_base_url": _DEFAULT_CONSOLE_BASE_URL,
        "login": "",
        "default_repo": "",
        "token": "",
    }

    config_path = get_hermes_home() / "clawmem.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            defaults.update({k: v for k, v in file_cfg.items()
                             if v is not None and v != ""})
        except Exception:
            pass

    env_map = {
        "CLAWMEM_GIT_BASE_URL": "git_base_url",
        "CLAWMEM_CONSOLE_BASE_URL": "console_base_url",
        "CLAWMEM_TOKEN": "token",
        "CLAWMEM_LOGIN": "login",
        "CLAWMEM_DEFAULT_REPO": "default_repo",
    }
    for env_var, key in env_map.items():
        val = os.environ.get(env_var, "").strip()
        if val:
            defaults[key] = val

    # Token from .env (loaded by hermes_cli.env_loader at startup)
    if not defaults.get("token"):
        defaults["token"] = os.environ.get("CLAWMEM_TOKEN", "")

    return defaults


def _get_profile_name() -> str:
    """Derive current profile name from HERMES_HOME."""
    hermes_home = os.environ.get("HERMES_HOME", "")
    if not hermes_home:
        return "hermes"
    p = Path(hermes_home)
    if p.parent.name == "profiles":
        return p.name
    return "hermes"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "clawmem_recall",
    "description": (
        "Search ClawMem active memories for relevant prior facts, decisions, "
        "preferences, and lessons. Use before answering questions about prior "
        "conversations or user preferences."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to recall from memory.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 5, max: 20).",
            },
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "clawmem_store",
    "description": (
        "Store one atomic durable memory. Keep each write to a single fact, "
        "preference, decision, or lesson."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Optional human-readable title.",
            },
            "detail": {
                "type": "string",
                "description": "The durable fact to remember.",
            },
            "kind": {
                "type": "string",
                "description": "Optional kind label (e.g. core-fact, preference, lesson).",
            },
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional topic labels for retrieval (max 10).",
            },
        },
        "required": ["detail"],
    },
}

LIST_SCHEMA = {
    "name": "clawmem_list",
    "description": "List ClawMem memories by status, kind, or topic.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "stale", "all"],
                "description": "Which memories to list (default: active).",
            },
            "kind": {
                "type": "string",
                "description": "Optional kind filter.",
            },
            "topic": {
                "type": "string",
                "description": "Optional topic filter.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 20, max: 200).",
            },
        },
        "required": [],
    },
}

GET_SCHEMA = {
    "name": "clawmem_get",
    "description": "Fetch one ClawMem memory by ID (issue number).",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID or issue number.",
            },
        },
        "required": ["memory_id"],
    },
}

UPDATE_SCHEMA = {
    "name": "clawmem_update",
    "description": (
        "Update an existing ClawMem memory in place when a canonical fact "
        "has evolved."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID or issue number to update.",
            },
            "title": {
                "type": "string",
                "description": "Optional replacement title.",
            },
            "detail": {
                "type": "string",
                "description": "Optional replacement detail.",
            },
            "kind": {
                "type": "string",
                "description": "Optional replacement kind.",
            },
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional replacement topics.",
            },
        },
        "required": ["memory_id"],
    },
}

FORGET_SCHEMA = {
    "name": "clawmem_forget",
    "description": (
        "Mark an active ClawMem memory as stale when it is no longer true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID or issue number to forget.",
            },
        },
        "required": ["memory_id"],
    },
}

CONSOLE_SCHEMA = {
    "name": "clawmem_console",
    "description": (
        "Return a URL to the ClawMem Console where the user can browse, "
        "search, and manage their memories in a web interface. "
        "Use when the user asks to view or manage memories in a browser."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

ALL_TOOL_SCHEMAS = [
    RECALL_SCHEMA, STORE_SCHEMA, LIST_SCHEMA, GET_SCHEMA,
    UPDATE_SCHEMA, FORGET_SCHEMA, CONSOLE_SCHEMA,
]


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction assistant. Your task is to identify durable facts, \
preferences, decisions, and lessons from the conversation transcript below.

Rules:
- Extract ONLY facts that would be useful across future sessions
- Each memory should be ONE atomic fact (not a session summary)
- Skip ephemeral task details, debugging steps, or transient state
- Skip facts that are obvious from code/git (e.g. "the project uses Python")
- Prefer the user's own words for preferences and corrections

Output a JSON array. Each element:
{
  "title": "short human-readable title (required)",
  "detail": "the durable fact, preference, or decision (required)",
  "kind": "one of: core-fact, preference, decision, lesson, convention, task (optional)",
  "topics": ["relevant", "topic", "labels"] (optional, max 5)
}

If no durable facts are found, return an empty array: []

Output ONLY the JSON array, no markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# Normalize label helpers (ported from memory.ts normalizeLabelValue)
# ---------------------------------------------------------------------------

def _normalize_label_value(value: str | None, prefix: str) -> str | None:
    """Normalize a label value: lowercase, dashes for separators."""
    if not value:
        return None
    raw = value.strip()
    if raw.lower().startswith(prefix):
        raw = raw[len(prefix):]
    import unicodedata
    normalized = unicodedata.normalize("NFKC", raw).lower()
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"[^\w-]", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or None


def _mem_labels(kind: str | None, topics: list[str] | None) -> list[str]:
    """Build the labels list for a type:memory issue."""
    labels = ["type:memory"]
    if kind:
        labels.append(f"kind:{kind}")
    for topic in (topics or []):
        if topic:
            labels.append(f"topic:{topic}")
    return labels


# ---------------------------------------------------------------------------
# ClawMemProvider
# ---------------------------------------------------------------------------

class ClawMemProvider(MemoryProvider):
    """ClawMem issue-backed long-term memory with hybrid recall."""

    def __init__(self):
        self._client = None  # ClawMemClient
        self._login = ""
        self._default_repo = ""
        self._token = ""
        self._console_base_url = _DEFAULT_CONSOLE_BASE_URL

        # Prefetch
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._auto_recall_limit = 5

        # Sync / conversation mirroring
        self._sync_thread: Optional[threading.Thread] = None
        self._conversation_issue_number: Optional[int] = None
        self._mirrored_turn_count = 0

        # Session
        self._session_id = ""

    @property
    def name(self) -> str:
        return "clawmem"

    # -- Config / availability ------------------------------------------------

    def is_available(self) -> bool:
        cfg = _load_config()
        return bool(cfg.get("token") and cfg.get("default_repo"))

    def get_config_schema(self) -> list[dict[str, Any]]:
        # Return empty — post_setup handles all config including auto-bootstrap.
        # This makes the picker show "no setup needed" instead of "requires API key",
        # which is accurate: users don't need to prepare any key beforehand.
        return []

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "clawmem.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # -- Setup wizard ---------------------------------------------------------

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Run the full ClawMem setup wizard with auto-bootstrap."""
        from hermes_cli.config import save_config
        from plugins.memory.clawmem.client import ClawMemClient, run_sync

        home = Path(hermes_home)

        # Step 1: Developer or User?
        print("\n  ClawMem Setup\n")
        mode = input("  Are you a ClawMem developer or a user? [1] User  [2] Developer (default: 1): ").strip()
        if mode == "2":
            git_url = input(f"  ClawMem Git Server URL [{_DEFAULT_GIT_BASE_URL}]: ").strip()
            git_base_url = git_url or _DEFAULT_GIT_BASE_URL
            console_url = input(f"  ClawMem Console URL [{_DEFAULT_CONSOLE_BASE_URL}]: ").strip()
            console_base_url = console_url or _DEFAULT_CONSOLE_BASE_URL
        else:
            git_base_url = _DEFAULT_GIT_BASE_URL
            console_base_url = _DEFAULT_CONSOLE_BASE_URL

        # Step 2: Agent identity
        default_prefix = _get_profile_name()
        prefix_login = input(f"  Agent prefix login [{default_prefix}]: ").strip() or default_prefix
        default_repo_name = input("  Default repo name [hermes-memory]: ").strip() or "hermes-memory"

        # Step 3: Bootstrap — check for existing identity
        config_path = home / "clawmem.json"
        existing_cfg: dict = {}
        if config_path.exists():
            try:
                existing_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        existing_token = os.environ.get("CLAWMEM_TOKEN", "").strip()
        existing_login = existing_cfg.get("login", "")
        existing_repo = existing_cfg.get("default_repo", "")

        if existing_login and existing_repo and existing_token:
            print(f"\n  Existing ClawMem identity found: {existing_login} / {existing_repo}")
            re_register = input("  Re-register? [y/N]: ").strip().lower()
            if re_register != "y":
                # Skip bootstrap, keep existing
                file_cfg = {
                    "git_base_url": git_base_url,
                    "console_base_url": console_base_url,
                    "login": existing_login,
                    "default_repo": existing_repo,
                }
                config_path.write_text(json.dumps(file_cfg, indent=2), encoding="utf-8")
                # Activate
                if not isinstance(config.get("memory"), dict):
                    config["memory"] = {}
                config["memory"]["provider"] = "clawmem"
                save_config(config)
                print(f"\n  \u2713 Config saved to {config_path}")
                print(f"  \u2713 memory.provider: clawmem saved to config.yaml")
                print("\n  Start a new session to activate.\n")
                return

        # Bootstrap: POST /agents
        print("\n  Registering agent identity...")
        try:
            result = run_sync(ClawMemClient.register_agent(
                git_base_url, prefix_login, default_repo_name,
            ))
        except Exception as e:
            print(f"\n  \u2717 Registration failed: {e}")
            print("  Setup aborted. Check your network and try again.\n")
            return

        login = result.get("login", "")
        token = result.get("token", "")
        repo_full_name = result.get("repo_full_name", "")

        if not login or not token or not repo_full_name:
            print(f"\n  \u2717 Unexpected response from server: {result}")
            print("  Setup aborted.\n")
            return

        print(f"  \u2713 Agent registered")
        print(f"    Login: {login}")
        print(f"    Repo:  {repo_full_name}")

        # Step 4: Save
        file_cfg = {
            "git_base_url": git_base_url,
            "console_base_url": console_base_url,
            "login": login,
            "default_repo": repo_full_name,
        }
        config_path.write_text(json.dumps(file_cfg, indent=2), encoding="utf-8")

        # Write token to .env
        env_path = home / ".env"
        _write_env_var(env_path, "CLAWMEM_TOKEN", token)

        # Activate in config.yaml
        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = "clawmem"
        save_config(config)

        print(f"\n  \u2713 Config saved to {config_path}")
        print(f"  \u2713 Token saved to {env_path}")
        print(f"  \u2713 memory.provider: clawmem saved to config.yaml")
        print("\n  Start a new session to activate.\n")

    # -- Lifecycle ------------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """Load config, create client, start conversation mirroring."""
        try:
            from plugins.memory.clawmem.client import ClawMemClient, run_sync

            cfg = _load_config()
            token = cfg.get("token", "")
            default_repo = cfg.get("default_repo", "")
            if not token or not default_repo:
                logger.debug("ClawMem not configured — plugin inactive")
                return

            self._token = token
            self._default_repo = default_repo
            self._login = cfg.get("login", "")
            self._console_base_url = cfg.get("console_base_url", _DEFAULT_CONSOLE_BASE_URL)
            self._session_id = session_id

            self._client = ClawMemClient(
                base_url=cfg.get("git_base_url", _DEFAULT_GIT_BASE_URL),
                token=token,
                default_repo=default_repo,
            )

            # Create conversation issue in background
            platform = kwargs.get("platform", "cli")

            def _create_conversation():
                try:
                    title = f"Session: {session_id[:12]}"
                    labels = [
                        "type:conversation", "status:active",
                    ]
                    if self._login:
                        labels.append(f"agent:{self._login}")
                    body = (
                        f"platform: {platform}\n"
                        f"started: {datetime.utcnow().isoformat()}Z\n"
                        f"session_id: {session_id}\n"
                    )
                    run_sync(self._client.ensure_labels(labels))
                    issue = run_sync(self._client.create_issue(title, body, labels))
                    self._conversation_issue_number = issue.get("number")
                    logger.debug("ClawMem conversation issue #%s created", self._conversation_issue_number)
                except Exception as e:
                    logger.debug("ClawMem conversation issue creation failed: %s", e)

            t = threading.Thread(target=_create_conversation, daemon=True,
                                 name="clawmem-conv-init")
            t.start()
            # Store so shutdown can join
            self._sync_thread = t

        except Exception as e:
            logger.warning("ClawMem init failed: %s", e)
            self._client = None

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        return (
            "# ClawMem Memory\n"
            "Active (hybrid mode). Relevant memories are auto-injected before each turn. "
            "Memory tools are also available:\n"
            "- clawmem_recall: search memories by relevance\n"
            "- clawmem_store: save a new durable fact\n"
            "- clawmem_list: browse memories by kind/topic\n"
            "- clawmem_get: fetch one memory by ID\n"
            "- clawmem_update: update an existing memory in place\n"
            "- clawmem_forget: mark a memory as stale\n"
            "- clawmem_console: get a URL to browse memories in the web console"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## ClawMem Memories\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._client or not query:
            return

        from plugins.memory.clawmem.client import run_sync, parse_memory_issue, format_memory_line

        client = self._client
        default_repo = self._default_repo
        limit = self._auto_recall_limit

        def _run():
            try:
                results = run_sync(client.search_issues(
                    f"{query} repo:{default_repo} label:type:memory state:open",
                    per_page=min(limit * 3, 60),
                ))
                if results:
                    lines = []
                    for issue in results:
                        parsed = parse_memory_issue(issue)
                        if parsed and parsed["status"] == "active":
                            lines.append(f"- {format_memory_line(parsed)}")
                            if len(lines) >= limit:
                                break
                    if lines:
                        with self._prefetch_lock:
                            self._prefetch_result = "\n".join(lines)
            except Exception as e:
                logger.debug("ClawMem prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="clawmem-prefetch",
        )
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._client or not self._conversation_issue_number:
            return

        from plugins.memory.clawmem.client import run_sync

        client = self._client
        issue_number = self._conversation_issue_number
        turn = self._mirrored_turn_count
        self._mirrored_turn_count += 1

        def _sync():
            try:
                parts = []
                if user_content:
                    parts.append(f"**User:**\n{user_content}")
                if assistant_content:
                    parts.append(f"**Assistant:**\n{assistant_content}")
                if not parts:
                    return
                comment_body = "\n\n".join(parts)
                run_sync(client.create_comment(issue_number, comment_body))
            except Exception as e:
                logger.debug("ClawMem sync_turn comment failed: %s", e)

        # Wait for previous sync to finish
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="clawmem-sync",
        )
        self._sync_thread.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._client:
            return

        # Wait for pending sync_turn
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

        try:
            self._extract_memories(messages)
        except Exception as e:
            logger.warning("ClawMem memory extraction failed: %s", e)

        # Close conversation issue
        if self._conversation_issue_number:
            try:
                from plugins.memory.clawmem.client import run_sync
                run_sync(self._client.update_issue(
                    self._conversation_issue_number, state="closed",
                ))
                # Update labels: replace status:active with status:closed
                run_sync(self._client.sync_managed_labels(
                    self._conversation_issue_number,
                    ["type:conversation", "status:closed"]
                    + ([f"agent:{self._login}"] if self._login else []),
                ))
            except Exception as e:
                logger.debug("ClawMem conversation close failed: %s", e)

    def _extract_memories(self, messages: List[Dict[str, Any]]) -> None:
        """Extract durable memories from conversation via auxiliary LLM."""
        # Filter to user/assistant messages only
        filtered = []
        for msg in messages:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multi-part message: extract text parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)
            if not content or not isinstance(content, str):
                continue
            filtered.append({"role": role, "content": content})

        if not filtered:
            return

        # Truncate to last 50 messages
        if len(filtered) > 50:
            filtered = filtered[-50:]

        # Build transcript
        transcript_lines = []
        for i, msg in enumerate(filtered, 1):
            transcript_lines.append(f"{i}. {msg['role']}: {msg['content']}")
        transcript_text = "\n\n".join(transcript_lines)

        # LLM call
        try:
            from agent.auxiliary_client import call_llm

            response = call_llm(
                task="memory_extraction",
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": transcript_text},
                ],
                temperature=0.0,
                max_tokens=2000,
                timeout=30.0,
            )

            raw = response.choices[0].message.content
            if not raw:
                return
        except Exception as e:
            logger.warning("ClawMem extraction LLM call failed: %s", e)
            return

        # Parse response
        candidates = _parse_extraction_response(raw)
        if not candidates:
            return

        # Store each candidate (with dedup)
        from plugins.memory.clawmem.client import (
            run_sync, sha256_hex, render_memory_body, render_memory_title,
            parse_memory_issue,
        )

        for candidate in candidates:
            try:
                detail = candidate.get("detail", "").strip()
                if not detail:
                    continue

                hash_val = sha256_hex(detail)
                title = candidate.get("title")
                kind = _normalize_label_value(candidate.get("kind"), "kind:")
                topics_raw = candidate.get("topics", [])
                topics = [
                    _normalize_label_value(t, "topic:")
                    for t in (topics_raw or [])
                ]
                topics = [t for t in topics if t][:10]

                # Dedup check
                existing = run_sync(self._client.search_issues(
                    f"{hash_val} repo:{self._default_repo} label:type:memory state:open",
                    per_page=5,
                ))
                is_dup = False
                for issue in existing:
                    parsed = parse_memory_issue(issue)
                    if parsed and parsed.get("memory_hash") == hash_val:
                        is_dup = True
                        break

                if is_dup:
                    continue

                # Store
                labels = _mem_labels(kind, topics)
                run_sync(self._client.ensure_labels(labels))
                issue_title = render_memory_title(detail, title)
                body = render_memory_body(detail, hash_val)
                run_sync(self._client.create_issue(issue_title, body, labels))

            except Exception as e:
                logger.debug("ClawMem extraction store failed for one candidate: %s", e)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if action != "add" or not content:
            return
        if not self._client:
            return

        from plugins.memory.clawmem.client import (
            run_sync, sha256_hex, render_memory_body, render_memory_title,
            parse_memory_issue,
        )

        client = self._client
        default_repo = self._default_repo

        def _write():
            try:
                detail = content.strip()
                if not detail:
                    return
                hash_val = sha256_hex(detail)

                # Dedup check
                existing = run_sync(client.search_issues(
                    f"{hash_val} repo:{default_repo} label:type:memory state:open",
                    per_page=5,
                ))
                for issue in existing:
                    parsed = parse_memory_issue(issue)
                    if parsed and parsed.get("memory_hash") == hash_val:
                        return  # already exists

                # Store
                title = render_memory_title(detail)
                body = render_memory_body(detail, hash_val)
                labels = ["type:memory"]
                if target == "user":
                    labels.append("kind:user-profile")
                run_sync(client.ensure_labels(labels))
                run_sync(client.create_issue(title, body, labels))
            except Exception as e:
                logger.debug("ClawMem memory mirror failed: %s", e)

        t = threading.Thread(target=_write, daemon=True, name="clawmem-memwrite")
        t.start()

    # -- Tools ----------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not self._client:
            return []
        return list(ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._client:
            return tool_error("ClawMem is not active for this session.")

        try:
            if tool_name == "clawmem_recall":
                return self._handle_recall(args)
            elif tool_name == "clawmem_store":
                return self._handle_store(args)
            elif tool_name == "clawmem_list":
                return self._handle_list(args)
            elif tool_name == "clawmem_get":
                return self._handle_get(args)
            elif tool_name == "clawmem_update":
                return self._handle_update(args)
            elif tool_name == "clawmem_forget":
                return self._handle_forget(args)
            elif tool_name == "clawmem_console":
                return self._handle_console()
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error("ClawMem tool %s failed: %s", tool_name, e)
            return tool_error(f"ClawMem {tool_name} failed: {e}")

    def _handle_recall(self, args: dict) -> str:
        from plugins.memory.clawmem.client import (
            run_sync, parse_memory_issue, format_memory_block,
        )

        query = args.get("query", "")
        if not query:
            return tool_error("Missing required parameter: query")
        limit = min(int(args.get("limit", 5)), 20)

        results = run_sync(self._client.search_issues(
            f"{query} repo:{self._default_repo} label:type:memory state:open",
            per_page=min(limit * 3, 60),
        ))

        memories = []
        for issue in results:
            parsed = parse_memory_issue(issue)
            if parsed and parsed["status"] == "active":
                memories.append(parsed)
                if len(memories) >= limit:
                    break

        if not memories:
            return json.dumps({"result": "No relevant memories found.", "count": 0})

        blocks = [format_memory_block(m) for m in memories]
        return json.dumps({
            "result": "\n\n---\n\n".join(blocks),
            "count": len(memories),
        })

    def _handle_store(self, args: dict) -> str:
        from plugins.memory.clawmem.client import (
            run_sync, sha256_hex, render_memory_body, render_memory_title,
            parse_memory_issue, format_memory_block,
        )

        detail = (args.get("detail") or "").strip()
        if not detail:
            return tool_error("Missing required parameter: detail")

        title = args.get("title")
        kind = _normalize_label_value(args.get("kind"), "kind:")
        topics_raw = args.get("topics", [])
        topics = [
            _normalize_label_value(t, "topic:")
            for t in (topics_raw or [])
        ]
        topics = [t for t in topics if t][:10]

        hash_val = sha256_hex(detail)

        # Dedup check
        existing = run_sync(self._client.search_issues(
            f"{hash_val} repo:{self._default_repo} label:type:memory state:open",
            per_page=5,
        ))
        for issue in existing:
            parsed = parse_memory_issue(issue)
            if parsed and parsed.get("memory_hash") == hash_val:
                # Merge schema labels if needed
                new_labels = _mem_labels(kind, topics)
                current_labels = _mem_labels(parsed.get("kind"), parsed.get("topics", []))
                if set(new_labels) != set(current_labels):
                    merged = _mem_labels(
                        kind or parsed.get("kind"),
                        list(dict.fromkeys((parsed.get("topics") or []) + topics)),
                    )
                    run_sync(self._client.ensure_labels(merged))
                    run_sync(self._client.sync_managed_labels(
                        parsed["issue_number"], merged,
                    ))
                return json.dumps({
                    "result": "Memory already exists (dedup).",
                    "memory": format_memory_block(parsed),
                    "created": False,
                })

        # Create new
        labels = _mem_labels(kind, topics)
        run_sync(self._client.ensure_labels(labels))
        issue_title = render_memory_title(detail, title)
        body = render_memory_body(detail, hash_val)
        issue = run_sync(self._client.create_issue(issue_title, body, labels))

        return json.dumps({
            "result": f"Memory stored (#{issue.get('number', '?')}).",
            "memory_id": str(issue.get("number", "")),
            "created": True,
        })

    def _handle_list(self, args: dict) -> str:
        from plugins.memory.clawmem.client import (
            run_sync, parse_memory_issue, format_memory_line,
        )

        status = args.get("status", "active")
        kind = _normalize_label_value(args.get("kind"), "kind:")
        topic = _normalize_label_value(args.get("topic"), "topic:")
        limit = min(int(args.get("limit", 20)), 200)

        labels = ["type:memory"]
        if kind:
            labels.append(f"kind:{kind}")
        if topic:
            labels.append(f"topic:{topic}")

        # Map status to API state
        state = "open" if status == "active" else ("closed" if status == "stale" else "all")

        memories = []
        page = 1
        while len(memories) < limit and page <= 20:
            batch = run_sync(self._client.list_issues(
                labels=labels,
                state=state,
                page=page,
                per_page=min(100, limit),
            ))
            for issue in batch:
                parsed = parse_memory_issue(issue)
                if not parsed:
                    continue
                if status != "all" and parsed["status"] != status:
                    continue
                memories.append(parsed)
                if len(memories) >= limit:
                    break
            if len(batch) < min(100, limit):
                break
            page += 1

        if not memories:
            return json.dumps({"result": "No memories found.", "count": 0})

        lines = [format_memory_line(m) for m in memories]
        return json.dumps({
            "result": "\n".join(lines),
            "count": len(memories),
        })

    def _handle_get(self, args: dict) -> str:
        from plugins.memory.clawmem.client import (
            run_sync, parse_memory_issue, format_memory_block,
        )

        memory_id = (args.get("memory_id") or "").strip()
        if not memory_id:
            return tool_error("Missing required parameter: memory_id")

        if not memory_id.isdigit():
            return tool_error("memory_id must be a numeric issue number.")

        issue = run_sync(self._client.get_issue(int(memory_id)))
        if not issue:
            return tool_error(f"Memory #{memory_id} not found.")

        parsed = parse_memory_issue(issue)
        if not parsed:
            return tool_error(f"Issue #{memory_id} is not a type:memory issue.")

        return json.dumps({
            "result": format_memory_block(parsed),
            "memory": parsed,
        })

    def _handle_update(self, args: dict) -> str:
        from plugins.memory.clawmem.client import (
            run_sync, sha256_hex, render_memory_body, render_memory_title,
            parse_memory_issue, format_memory_block,
        )

        memory_id = (args.get("memory_id") or "").strip()
        if not memory_id or not memory_id.isdigit():
            return tool_error("Missing or invalid memory_id.")

        issue = run_sync(self._client.get_issue(int(memory_id)))
        if not issue:
            return tool_error(f"Memory #{memory_id} not found.")

        current = parse_memory_issue(issue)
        if not current:
            return tool_error(f"Issue #{memory_id} is not a type:memory issue.")

        # Merge fields
        new_detail = (args.get("detail") or "").strip() or current["detail"]
        new_title_raw = args.get("title")
        new_kind = _normalize_label_value(args.get("kind"), "kind:") if "kind" in args else current.get("kind")
        new_topics = (
            [_normalize_label_value(t, "topic:") for t in args["topics"]]
            if "topics" in args
            else current.get("topics", [])
        )
        new_topics = [t for t in (new_topics or []) if t][:10]

        # Recompute hash if detail changed
        new_hash = sha256_hex(new_detail)
        if new_hash != current.get("memory_hash") and new_hash != sha256_hex(current["detail"]):
            # Check for collision with another active memory
            existing = run_sync(self._client.search_issues(
                f"{new_hash} repo:{self._default_repo} label:type:memory state:open",
                per_page=5,
            ))
            for iss in existing:
                parsed = parse_memory_issue(iss)
                if parsed and parsed.get("memory_hash") == new_hash and parsed["issue_number"] != current["issue_number"]:
                    return tool_error(
                        f"Another active memory (#{parsed['memory_id']}) already stores this detail."
                    )

        # Update
        new_title = render_memory_title(new_detail, new_title_raw)
        new_body = render_memory_body(new_detail, new_hash)
        run_sync(self._client.update_issue(
            current["issue_number"],
            title=new_title,
            body=new_body,
        ))

        # Sync labels
        new_labels = _mem_labels(new_kind, new_topics)
        run_sync(self._client.ensure_labels(new_labels))
        run_sync(self._client.sync_managed_labels(current["issue_number"], new_labels))

        updated = {
            **current,
            "title": new_title,
            "detail": new_detail,
            "memory_hash": new_hash,
            "kind": new_kind,
            "topics": new_topics,
        }
        return json.dumps({
            "result": f"Memory #{memory_id} updated.",
            "memory": format_memory_block(updated),
        })

    def _handle_forget(self, args: dict) -> str:
        from plugins.memory.clawmem.client import (
            run_sync, parse_memory_issue, format_memory_line,
        )

        memory_id = (args.get("memory_id") or "").strip()
        if not memory_id or not memory_id.isdigit():
            return tool_error("Missing or invalid memory_id.")

        issue = run_sync(self._client.get_issue(int(memory_id)))
        if not issue:
            return tool_error(f"Memory #{memory_id} not found.")

        parsed = parse_memory_issue(issue)
        if not parsed:
            return tool_error(f"Issue #{memory_id} is not a type:memory issue.")

        if parsed["status"] != "active":
            return tool_error(f"Memory #{memory_id} is already stale.")

        run_sync(self._client.update_issue(int(memory_id), state="closed"))

        return json.dumps({
            "result": f"Memory #{memory_id} marked as stale.",
            "memory": format_memory_line({**parsed, "status": "stale"}),
        })

    def _handle_console(self) -> str:
        url = f"{self._console_base_url}/{self._default_repo}?token={self._token}"
        return json.dumps({
            "url": url,
            "message": "Open this URL to browse your memories.",
        })

    # -- Shutdown -------------------------------------------------------------

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_env_var(env_path: Path, key: str, value: str) -> None:
    """Write or update a single env var in a .env file."""
    env_path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines = []
    for line in existing_lines:
        line_key = line.split("=", 1)[0].strip() if "=" in line else ""
        if line_key == key:
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _parse_extraction_response(raw: str) -> list[dict]:
    """Parse the LLM extraction response into a list of memory candidates."""
    text = raw.strip()

    # Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [c for c in result if isinstance(c, dict) and c.get("detail")]
        if isinstance(result, dict) and "candidates" in result:
            return [c for c in result["candidates"]
                    if isinstance(c, dict) and c.get("detail")]
        return []
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown fences
    import re
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        try:
            result = json.loads(fenced.group(1).strip())
            if isinstance(result, list):
                return [c for c in result if isinstance(c, dict) and c.get("detail")]
            return []
        except json.JSONDecodeError:
            pass

    # Try finding JSON array in text
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return [c for c in result if isinstance(c, dict) and c.get("detail")]
        except json.JSONDecodeError:
            pass

    logger.warning("ClawMem extraction: could not parse LLM response as JSON")
    return []


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register ClawMem as a memory provider plugin."""
    ctx.register_memory_provider(ClawMemProvider())
