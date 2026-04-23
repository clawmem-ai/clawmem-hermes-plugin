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

# -- Memory extension tools -------------------------------------------------

LABELS_SCHEMA = {
    "name": "clawmem_labels",
    "description": (
        "List existing ClawMem schema labels (kinds / topics) so the agent can "
        "reuse current labels first and only mint new ones deliberately."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit_topics": {
                "type": "integer",
                "description": "Maximum number of topic labels to display (default: 50, max: 200).",
            },
        },
        "required": [],
    },
}

REPOS_SCHEMA = {
    "name": "clawmem_repos",
    "description": (
        "List the memory repos the current ClawMem identity can access, "
        "with the current default repo flagged."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

REPO_CREATE_SCHEMA = {
    "name": "clawmem_repo_create",
    "description": (
        "Create a new ClawMem repo under the current identity when a separate "
        "memory space is needed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Repository name only, without owner prefix."},
            "description": {"type": "string", "description": "Optional repo description."},
            "private": {"type": "boolean", "description": "Whether the new repo should be private. Defaults to true."},
            "set_default": {"type": "boolean", "description": "Whether to make the new repo this identity's default memory repo."},
        },
        "required": ["name"],
    },
}

REPO_SET_DEFAULT_SCHEMA = {
    "name": "clawmem_repo_set_default",
    "description": (
        "Retarget the agent's default memory repo so automatic conversation and "
        "memory flows follow a new repo. Requires confirmed=true after explicit "
        "user approval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "The new default repo in owner/repo form."},
            "confirmed": {"type": "boolean", "description": "Must be true after the user approves the exact config change."},
        },
        "required": ["repo"],
    },
}

REVIEW_SCHEMA = {
    "name": "clawmem_review",
    "description": (
        "Return the ClawMem self-review checklist (memory and/or skill track) so "
        "durable memory and kind:skill playbooks keep accumulating. Call this "
        "every ~8-10 user turns or at the end of a non-trivial task."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "enum": ["memory", "skill", "both"],
                "description": "Which review track to return. Defaults to both.",
            },
        },
        "required": [],
    },
}

# -- Generic issue tools ----------------------------------------------------

ISSUE_CREATE_SCHEMA = {
    "name": "clawmem_issue_create",
    "description": (
        "Create a generic issue in the memory repo for queueing work, "
        "coordination, or shared tracking outside the memory schema."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Issue title."},
            "body": {"type": "string", "description": "Optional issue body."},
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional labels to attach to the issue.",
            },
            "assignees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional assignee logins.",
            },
            "state": {"type": "string", "enum": ["open", "closed"], "description": "Optional initial state. Defaults to open."},
        },
        "required": ["title"],
    },
}

ISSUE_LIST_SCHEMA = {
    "name": "clawmem_issue_list",
    "description": "List generic issues in the memory repo with optional filters.",
    "parameters": {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": ["open", "closed", "all"], "description": "Defaults to open."},
            "labels": {"type": "array", "items": {"type": "string"}, "description": "Optional labels (issues must match all)."},
            "limit": {"type": "integer", "description": "Maximum number of issues to return (default: 20, max: 200)."},
        },
        "required": [],
    },
}

ISSUE_GET_SCHEMA = {
    "name": "clawmem_issue_get",
    "description": "Fetch one generic issue by issue number from the memory repo.",
    "parameters": {
        "type": "object",
        "properties": {
            "issue_number": {"type": "integer", "description": "Issue number."},
        },
        "required": ["issue_number"],
    },
}

ISSUE_UPDATE_SCHEMA = {
    "name": "clawmem_issue_update",
    "description": (
        "Update a generic issue in place: title, body, state, labels, or assignees."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "issue_number": {"type": "integer", "description": "Issue number."},
            "title": {"type": "string"},
            "body": {"type": "string"},
            "state": {"type": "string", "enum": ["open", "closed"]},
            "labels": {"type": "array", "items": {"type": "string"}},
            "assignees": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["issue_number"],
    },
}

ISSUE_COMMENT_ADD_SCHEMA = {
    "name": "clawmem_issue_comment_add",
    "description": "Add a comment to an issue in the memory repo.",
    "parameters": {
        "type": "object",
        "properties": {
            "issue_number": {"type": "integer"},
            "body": {"type": "string"},
        },
        "required": ["issue_number", "body"],
    },
}

ISSUE_COMMENTS_LIST_SCHEMA = {
    "name": "clawmem_issue_comments_list",
    "description": "List comments on an issue in the memory repo.",
    "parameters": {
        "type": "object",
        "properties": {
            "issue_number": {"type": "integer"},
            "sort": {"type": "string", "enum": ["created", "updated"]},
            "direction": {"type": "string", "enum": ["asc", "desc"]},
            "since": {"type": "string", "description": "Optional ISO 8601 lower bound."},
            "limit": {"type": "integer", "description": "Default: 20, max: 200."},
        },
        "required": ["issue_number"],
    },
}

# -- Collaboration tools ----------------------------------------------------

COLLAB_ORGS_SCHEMA = {
    "name": "clawmem_collaboration_orgs",
    "description": "List organizations visible to the current ClawMem identity.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

COLLAB_ORG_CREATE_SCHEMA = {
    "name": "clawmem_collaboration_org_create",
    "description": (
        "Create a new organization for shared ClawMem collaboration. "
        "Requires confirmed=true after explicit user approval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "login": {"type": "string", "description": "Organization login / slug."},
            "name": {"type": "string", "description": "Optional human-readable organization name."},
            "default_permission": {
                "type": "string",
                "enum": ["none", "read", "write", "admin"],
                "description": "Default repository permission. Defaults to read.",
            },
            "confirmed": {"type": "boolean", "description": "Must be true after user approval."},
        },
        "required": ["login"],
    },
}

COLLAB_ORG_REPO_CREATE_SCHEMA = {
    "name": "clawmem_collaboration_org_repo_create",
    "description": (
        "Create a new org-owned repo for shared collaboration artifacts. "
        "Requires confirmed=true after explicit user approval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "name": {"type": "string", "description": "Repository name without owner prefix."},
            "description": {"type": "string"},
            "private": {"type": "boolean", "description": "Defaults to true."},
            "auto_init": {"type": "boolean", "description": "Defaults to false."},
            "has_issues": {"type": "boolean"},
            "has_wiki": {"type": "boolean"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "name"],
    },
}

COLLAB_ORG_MEMBERS_SCHEMA = {
    "name": "clawmem_collaboration_org_members",
    "description": "List visible members in an organization; optionally filter to admins only.",
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "role": {"type": "string", "enum": ["admin"]},
        },
        "required": ["org"],
    },
}

COLLAB_ORG_MEMBERSHIP_SCHEMA = {
    "name": "clawmem_collaboration_org_membership",
    "description": "Inspect one user's org membership state (active or pending).",
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "username": {"type": "string"},
        },
        "required": ["org", "username"],
    },
}

COLLAB_ORG_MEMBER_REMOVE_SCHEMA = {
    "name": "clawmem_collaboration_org_member_remove",
    "description": (
        "Remove an active organization member. Requires confirmed=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "username": {"type": "string"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "username"],
    },
}

COLLAB_ORG_MEMBERSHIP_REMOVE_SCHEMA = {
    "name": "clawmem_collaboration_org_membership_remove",
    "description": (
        "Remove an active membership or revoke a pending invitation for a user. "
        "Requires confirmed=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "username": {"type": "string"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "username"],
    },
}

COLLAB_TEAMS_SCHEMA = {
    "name": "clawmem_collaboration_teams",
    "description": "List teams in an organization.",
    "parameters": {
        "type": "object",
        "properties": {"org": {"type": "string"}},
        "required": ["org"],
    },
}

COLLAB_TEAM_SCHEMA = {
    "name": "clawmem_collaboration_team",
    "description": "Inspect one organization team by slug.",
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "team_slug": {"type": "string"},
        },
        "required": ["org", "team_slug"],
    },
}

COLLAB_TEAM_CREATE_SCHEMA = {
    "name": "clawmem_collaboration_team_create",
    "description": "Create a team inside an organization. Requires confirmed=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "privacy": {"type": "string", "enum": ["closed", "secret"]},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "name"],
    },
}

COLLAB_TEAM_UPDATE_SCHEMA = {
    "name": "clawmem_collaboration_team_update",
    "description": "Update a team's name, description, or privacy. Requires confirmed=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "team_slug": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "privacy": {"type": "string", "enum": ["closed", "secret"]},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "team_slug"],
    },
}

COLLAB_TEAM_DELETE_SCHEMA = {
    "name": "clawmem_collaboration_team_delete",
    "description": "Delete a team. Requires confirmed=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "team_slug": {"type": "string"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "team_slug"],
    },
}

COLLAB_TEAM_MEMBERS_SCHEMA = {
    "name": "clawmem_collaboration_team_members",
    "description": "List members of an organization team.",
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "team_slug": {"type": "string"},
        },
        "required": ["org", "team_slug"],
    },
}

COLLAB_TEAM_MEMBERSHIP_SET_SCHEMA = {
    "name": "clawmem_collaboration_team_membership_set",
    "description": (
        "Add or update a user's membership in a team. Requires confirmed=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "team_slug": {"type": "string"},
            "username": {"type": "string"},
            "role": {"type": "string", "enum": ["member", "maintainer"]},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "team_slug", "username"],
    },
}

COLLAB_TEAM_MEMBERSHIP_REMOVE_SCHEMA = {
    "name": "clawmem_collaboration_team_membership_remove",
    "description": "Remove a user from a team. Requires confirmed=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "team_slug": {"type": "string"},
            "username": {"type": "string"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "team_slug", "username"],
    },
}

COLLAB_TEAM_REPOS_SCHEMA = {
    "name": "clawmem_collaboration_team_repos",
    "description": "List repositories currently granted to an organization team.",
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "team_slug": {"type": "string"},
        },
        "required": ["org", "team_slug"],
    },
}

COLLAB_TEAM_REPO_SET_SCHEMA = {
    "name": "clawmem_collaboration_team_repo_set",
    "description": (
        "Grant an organization team access to a repo. Requires confirmed=true. "
        "Defaults to the agent's default repo."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "team_slug": {"type": "string"},
            "repo": {"type": "string", "description": "owner/repo; defaults to agent's default repo."},
            "permission": {"type": "string", "enum": ["read", "write", "admin"]},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "team_slug"],
    },
}

COLLAB_TEAM_REPO_REMOVE_SCHEMA = {
    "name": "clawmem_collaboration_team_repo_remove",
    "description": (
        "Remove an organization team's repo grant. Requires confirmed=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "team_slug": {"type": "string"},
            "repo": {"type": "string"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "team_slug"],
    },
}

COLLAB_REPO_TRANSFER_SCHEMA = {
    "name": "clawmem_collaboration_repo_transfer",
    "description": (
        "Transfer a repository to a new owner, for example to move a personal "
        "memory repo into an org. Requires confirmed=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "owner/repo; defaults to agent's default repo."},
            "new_owner": {"type": "string"},
            "new_name": {"type": "string", "description": "Optional destination repo name."},
            "confirmed": {"type": "boolean"},
        },
        "required": ["new_owner"],
    },
}

COLLAB_REPO_COLLABORATORS_SCHEMA = {
    "name": "clawmem_collaboration_repo_collaborators",
    "description": "List direct collaborators on a repo.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "owner/repo; defaults to agent's default repo."},
        },
        "required": [],
    },
}

COLLAB_REPO_INVITATIONS_SCHEMA = {
    "name": "clawmem_collaboration_repo_invitations",
    "description": "List pending repository invitations on a repo.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
        },
        "required": [],
    },
}

COLLAB_REPO_COLLABORATOR_SET_SCHEMA = {
    "name": "clawmem_collaboration_repo_collaborator_set",
    "description": (
        "Add or update a direct collaborator on a repo. Requires confirmed=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "username": {"type": "string"},
            "permission": {"type": "string", "enum": ["read", "write", "admin"]},
            "confirmed": {"type": "boolean"},
        },
        "required": ["username"],
    },
}

COLLAB_REPO_COLLABORATOR_REMOVE_SCHEMA = {
    "name": "clawmem_collaboration_repo_collaborator_remove",
    "description": (
        "Remove a direct collaborator from a repo. Requires confirmed=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "username": {"type": "string"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["username"],
    },
}

COLLAB_USER_REPO_INVITATIONS_SCHEMA = {
    "name": "clawmem_collaboration_user_repo_invitations",
    "description": "List pending repository invitations for the current identity.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "Optional owner/repo filter."},
        },
        "required": [],
    },
}

COLLAB_USER_REPO_INVITATION_ACCEPT_SCHEMA = {
    "name": "clawmem_collaboration_user_repo_invitation_accept",
    "description": "Accept a pending repository invitation. Requires confirmed=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "invitation_id": {"type": "integer"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["invitation_id"],
    },
}

COLLAB_USER_REPO_INVITATION_DECLINE_SCHEMA = {
    "name": "clawmem_collaboration_user_repo_invitation_decline",
    "description": "Decline a pending repository invitation. Requires confirmed=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "invitation_id": {"type": "integer"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["invitation_id"],
    },
}

COLLAB_ORG_INVITATIONS_SCHEMA = {
    "name": "clawmem_collaboration_org_invitations",
    "description": "List pending organization invitations.",
    "parameters": {
        "type": "object",
        "properties": {"org": {"type": "string"}},
        "required": ["org"],
    },
}

COLLAB_ORG_INVITATION_CREATE_SCHEMA = {
    "name": "clawmem_collaboration_org_invitation_create",
    "description": (
        "Create an organization invitation. Requires confirmed=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "invitee_login": {"type": "string"},
            "role": {"type": "string", "enum": ["member", "owner"]},
            "team_ids": {"type": "array", "items": {"type": "integer"}},
            "expires_in_days": {"type": "integer"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "invitee_login"],
    },
}

COLLAB_ORG_INVITATION_REVOKE_SCHEMA = {
    "name": "clawmem_collaboration_org_invitation_revoke",
    "description": (
        "Revoke a pending organization invitation. Requires confirmed=true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string"},
            "invitation_id": {"type": "integer"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["org", "invitation_id"],
    },
}

COLLAB_USER_ORG_INVITATIONS_SCHEMA = {
    "name": "clawmem_collaboration_user_org_invitations",
    "description": "List pending organization invitations for the current identity.",
    "parameters": {
        "type": "object",
        "properties": {
            "org": {"type": "string", "description": "Optional org filter."},
        },
        "required": [],
    },
}

COLLAB_USER_ORG_INVITATION_ACCEPT_SCHEMA = {
    "name": "clawmem_collaboration_user_org_invitation_accept",
    "description": "Accept a pending organization invitation. Requires confirmed=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "invitation_id": {"type": "integer"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["invitation_id"],
    },
}

COLLAB_USER_ORG_INVITATION_DECLINE_SCHEMA = {
    "name": "clawmem_collaboration_user_org_invitation_decline",
    "description": "Decline a pending organization invitation. Requires confirmed=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "invitation_id": {"type": "integer"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["invitation_id"],
    },
}

COLLAB_OUTSIDE_COLLABORATORS_SCHEMA = {
    "name": "clawmem_collaboration_outside_collaborators",
    "description": "List outside collaborators in an organization.",
    "parameters": {
        "type": "object",
        "properties": {"org": {"type": "string"}},
        "required": ["org"],
    },
}

COLLAB_REPO_ACCESS_INSPECT_SCHEMA = {
    "name": "clawmem_collaboration_repo_access_inspect",
    "description": (
        "Inspect repo access paths: direct collaborators, team grants, org "
        "base permission, and optional per-user membership summary."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "username": {"type": "string", "description": "Optional user to inspect for org membership."},
        },
        "required": [],
    },
}


ALL_TOOL_SCHEMAS = [
    RECALL_SCHEMA, STORE_SCHEMA, LIST_SCHEMA, GET_SCHEMA,
    UPDATE_SCHEMA, FORGET_SCHEMA, CONSOLE_SCHEMA,
    # Memory extensions
    LABELS_SCHEMA, REPOS_SCHEMA, REPO_CREATE_SCHEMA,
    REPO_SET_DEFAULT_SCHEMA, REVIEW_SCHEMA,
    # Generic issues
    ISSUE_CREATE_SCHEMA, ISSUE_LIST_SCHEMA, ISSUE_GET_SCHEMA,
    ISSUE_UPDATE_SCHEMA, ISSUE_COMMENT_ADD_SCHEMA, ISSUE_COMMENTS_LIST_SCHEMA,
    # Collaboration — org family
    COLLAB_ORGS_SCHEMA, COLLAB_ORG_CREATE_SCHEMA, COLLAB_ORG_REPO_CREATE_SCHEMA,
    COLLAB_ORG_MEMBERS_SCHEMA, COLLAB_ORG_MEMBERSHIP_SCHEMA,
    COLLAB_ORG_MEMBER_REMOVE_SCHEMA, COLLAB_ORG_MEMBERSHIP_REMOVE_SCHEMA,
    # Collaboration — team family
    COLLAB_TEAMS_SCHEMA, COLLAB_TEAM_SCHEMA, COLLAB_TEAM_CREATE_SCHEMA,
    COLLAB_TEAM_UPDATE_SCHEMA, COLLAB_TEAM_DELETE_SCHEMA,
    COLLAB_TEAM_MEMBERS_SCHEMA, COLLAB_TEAM_MEMBERSHIP_SET_SCHEMA,
    COLLAB_TEAM_MEMBERSHIP_REMOVE_SCHEMA, COLLAB_TEAM_REPOS_SCHEMA,
    COLLAB_TEAM_REPO_SET_SCHEMA, COLLAB_TEAM_REPO_REMOVE_SCHEMA,
    # Collaboration — repo-level
    COLLAB_REPO_TRANSFER_SCHEMA, COLLAB_REPO_COLLABORATORS_SCHEMA,
    COLLAB_REPO_INVITATIONS_SCHEMA, COLLAB_REPO_COLLABORATOR_SET_SCHEMA,
    COLLAB_REPO_COLLABORATOR_REMOVE_SCHEMA,
    # Collaboration — user side invitations
    COLLAB_USER_REPO_INVITATIONS_SCHEMA,
    COLLAB_USER_REPO_INVITATION_ACCEPT_SCHEMA,
    COLLAB_USER_REPO_INVITATION_DECLINE_SCHEMA,
    # Collaboration — org invitations (admin side)
    COLLAB_ORG_INVITATIONS_SCHEMA, COLLAB_ORG_INVITATION_CREATE_SCHEMA,
    COLLAB_ORG_INVITATION_REVOKE_SCHEMA,
    # Collaboration — user side org invitations
    COLLAB_USER_ORG_INVITATIONS_SCHEMA,
    COLLAB_USER_ORG_INVITATION_ACCEPT_SCHEMA,
    COLLAB_USER_ORG_INVITATION_DECLINE_SCHEMA,
    # Collaboration — misc
    COLLAB_OUTSIDE_COLLABORATORS_SCHEMA, COLLAB_REPO_ACCESS_INSPECT_SCHEMA,
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
# Collaboration helpers
# ---------------------------------------------------------------------------

def _normalize_permission_alias(value: Any) -> Optional[str]:
    """Return one of 'none','read','write','admin' or None."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized == "none":
        return "none"
    if normalized in ("read", "pull", "triage"):
        return "read"
    if normalized in ("write", "push", "maintain"):
        return "write"
    if normalized == "admin":
        return "admin"
    return None


def _resolve_org_invitation_role(value: Any, fallback: str) -> Dict[str, str]:
    """Return {'role': ...} or {'error': ...}."""
    if value is None or value == "":
        return {"role": fallback}
    if not isinstance(value, str):
        return {"error": "role must be member or owner."}
    normalized = value.strip().lower()
    if normalized in ("member", "owner"):
        return {"role": normalized}
    return {"error": f'Unsupported role "{value}". Use member or owner.'}


def _resolve_collaboration_permission(value: Any, fallback: str) -> Dict[str, str]:
    """Return {'permission': ...} or {'error': ...}. fallback must be read/write/admin."""
    if value is None or value == "":
        return {"permission": fallback}
    if not isinstance(value, str):
        return {"error": "permission must be one of read, write, or admin."}
    normalized = _normalize_permission_alias(value)
    if normalized in ("read", "write", "admin"):
        return {"permission": normalized}
    return {"error": f'Unsupported permission "{value}". Use read, write, or admin.'}


def _resolve_org_default_permission(value: Any, fallback: str) -> Dict[str, str]:
    if value is None or value == "":
        return {"permission": fallback}
    if not isinstance(value, str):
        return {"error": "default_permission must be one of none, read, write, or admin."}
    normalized = _normalize_permission_alias(value)
    if normalized in ("none", "read", "write", "admin"):
        return {"permission": normalized}
    return {"error": f'Unsupported default_permission "{value}". Use none, read, write, or admin.'}


def _repo_summary_full_name(repo: dict | None) -> Optional[str]:
    if not repo:
        return None
    full_name = (repo.get("full_name") or "").strip()
    if full_name:
        return full_name
    owner = ((repo.get("owner") or {}).get("login") or "").strip()
    name = (repo.get("name") or "").strip()
    if owner and name:
        return f"{owner}/{name}"
    return name or None


def _filter_direct_collaborators(collaborators: list[dict], owner_login: str) -> list[dict]:
    owner = (owner_login or "").strip().lower()
    if not owner:
        return collaborators
    return [c for c in collaborators if ((c.get("login") or "").strip().lower()) != owner]


def _canonical_permission(
    permissions: dict | None, explicit: str | None,
) -> str:
    direct = _normalize_permission_alias(explicit) if explicit else None
    if direct:
        return direct
    if not permissions:
        return "unknown"
    if permissions.get("admin") is True:
        return "admin"
    if permissions.get("maintain") is True or permissions.get("push") is True or permissions.get("write") is True:
        return "write"
    if permissions.get("triage") is True or permissions.get("pull") is True or permissions.get("read") is True:
        return "read"
    return "unknown"


def _render_org_line(org: dict) -> str:
    login = (org.get("login") or "").strip() or "unknown-org"
    name_raw = (org.get("name") or "").strip()
    name_part = f" ({name_raw})" if name_raw else ""
    default_perm = (org.get("default_repository_permission") or "").strip()
    perm_part = ""
    if default_perm:
        normalized = _normalize_permission_alias(default_perm) or default_perm
        perm_part = f" [default:{normalized}]"
    desc_raw = (org.get("description") or "").strip()
    desc_part = f" - {desc_raw}" if desc_raw else ""
    return f"{login}{name_part}{perm_part}{desc_part}"


def _render_team_line(team: dict) -> str:
    slug = (team.get("slug") or "").strip() or (team.get("name") or "").strip() or "unknown-team"
    name_raw = (team.get("name") or "").strip()
    name_part = f" ({name_raw})" if name_raw and name_raw != slug else ""
    privacy = (team.get("privacy") or "").strip()
    privacy_part = f" [{privacy}]" if privacy else ""
    perm = _canonical_permission(team.get("permissions"), team.get("permission") or team.get("role_name"))
    perm_part = f" [perm:{perm}]" if perm != "unknown" else ""
    desc = (team.get("description") or "").strip()
    desc_part = f" - {desc}" if desc else ""
    return f"{slug}{name_part}{privacy_part}{perm_part}{desc_part}"


def _render_collaborator_line(user: dict) -> str:
    login = (user.get("login") or "").strip() or (user.get("name") or "").strip() or "unknown-user"
    name_raw = (user.get("name") or "").strip()
    name_part = f" ({name_raw})" if name_raw and name_raw != login else ""
    perm = _canonical_permission(user.get("permissions"), user.get("role_name"))
    perm_part = f" [{perm}]" if perm != "unknown" else ""
    return f"{login}{name_part}{perm_part}"


def _render_repo_line(repo: dict) -> str:
    full_name = _repo_summary_full_name(repo) or "unknown-repo"
    perm = _canonical_permission(repo.get("permissions"), repo.get("role_name"))
    perm_part = f" [{perm}]" if perm != "unknown" else ""
    desc = (repo.get("description") or "").strip()
    desc_part = f" - {desc}" if desc else ""
    private = repo.get("private")
    vis_part = " [private]" if private is True else (" [public]" if private is False else "")
    return f"{full_name}{vis_part}{perm_part}{desc_part}"


def _render_repo_invitation_line(invitation: dict) -> str:
    repo = _repo_summary_full_name(invitation.get("repository")) or "unknown-repo"
    perms_raw = invitation.get("permissions")
    perm = _normalize_permission_alias(perms_raw) or (perms_raw.strip() if isinstance(perms_raw, str) and perms_raw.strip() else "read")
    inv_id = invitation.get("id")
    id_text = f" id:{inv_id}" if isinstance(inv_id, int) else ""
    created = (invitation.get("created_at") or "").strip()
    created_text = f" created:{created}" if created else ""
    invitee = ((invitation.get("invitee") or {}).get("login") or "").strip()
    invitee_text = f" invitee:{invitee}" if invitee else ""
    inviter = ((invitation.get("inviter") or {}).get("login") or "").strip()
    inviter_text = f" inviter:{inviter}" if inviter else ""
    return f"{repo} [perm:{perm}{id_text}{created_text}{invitee_text}{inviter_text}]"


def _render_org_invitation_line(invitation: dict) -> str:
    invitee = ((invitation.get("invitee") or {}).get("login") or "").strip() \
        or (invitation.get("login") or "").strip() \
        or (invitation.get("email") or "").strip() \
        or "unknown-invitee"
    role = (invitation.get("role") or "").strip() or "member"
    inv_id = invitation.get("id")
    id_text = f" id:{inv_id}" if isinstance(inv_id, int) else ""
    created = (invitation.get("created_at") or "").strip()
    created_text = f" created:{created}" if created else ""
    expires = invitation.get("expires_at")
    expires_text = f" expires:{expires.strip()}" if isinstance(expires, str) and expires.strip() else ""
    teams_val = invitation.get("teams")
    team_ids_val = invitation.get("team_ids")
    teams: list[str] = []
    if isinstance(teams_val, list):
        for t in teams_val:
            if isinstance(t, dict):
                slug = (t.get("slug") or "").strip() or (t.get("name") or "").strip()
                if slug:
                    teams.append(slug)
    elif isinstance(team_ids_val, list):
        for tid in team_ids_val:
            if isinstance(tid, int) and tid > 0:
                teams.append(str(tid))
    teams_text = f" teams:{','.join(teams)}" if teams else ""
    org = ((invitation.get("organization") or {}).get("login") or "").strip()
    org_text = f" org:{org}" if org else ""
    return f"{invitee} [role:{role}{id_text}{created_text}{expires_text}{teams_text}{org_text}]"


def _render_user_org_invitation_line(invitation: dict) -> str:
    org = ((invitation.get("organization") or {}).get("login") or "").strip() or "unknown-org"
    role = (invitation.get("role") or "").strip() or "member"
    inv_id = invitation.get("id")
    id_text = f" id:{inv_id}" if isinstance(inv_id, int) else ""
    created = (invitation.get("created_at") or "").strip()
    created_text = f" created:{created}" if created else ""
    expires = invitation.get("expires_at")
    expires_text = f" expires:{expires.strip()}" if isinstance(expires, str) and expires.strip() else ""
    team_ids = invitation.get("team_ids")
    teams: list[str] = []
    if isinstance(team_ids, list):
        for tid in team_ids:
            if isinstance(tid, int) and tid > 0:
                teams.append(str(tid))
    teams_text = f" teamIds:{','.join(teams)}" if teams else ""
    inviter = ((invitation.get("inviter") or {}).get("login") or "").strip()
    inviter_text = f" inviter:{inviter}" if inviter else ""
    return f"{org} [role:{role}{id_text}{created_text}{expires_text}{teams_text}{inviter_text}]"


def _render_org_membership_line(membership: dict) -> str:
    user = membership.get("user") or {}
    login = (user.get("login") or "").strip() or (user.get("name") or "").strip() or "unknown-user"
    name_raw = (user.get("name") or "").strip()
    name_part = f" ({name_raw})" if name_raw and name_raw != login else ""
    state = (membership.get("state") or "").strip() or "unknown"
    role = (membership.get("role") or "").strip() or "unknown"
    org = ((membership.get("organization") or {}).get("login") or "").strip()
    org_part = f" org:{org}" if org else ""
    return f"{login}{name_part} [state:{state} role:{role}{org_part}]"


def _render_issue_line(issue: dict) -> str:
    num = issue.get("number", "?")
    title = (issue.get("title") or "").strip() or "(no title)"
    state = (issue.get("state") or "").strip() or "open"
    raw_labels = issue.get("labels") or []
    labels = []
    for lbl in raw_labels:
        if isinstance(lbl, dict):
            n = (lbl.get("name") or "").strip()
            if n:
                labels.append(n)
        elif isinstance(lbl, str) and lbl.strip():
            labels.append(lbl.strip())
    labels_text = f" labels:{','.join(labels)}" if labels else ""
    return f"#{num} {title} [{state}]{labels_text}"


def _render_issue_block(issue: dict) -> str:
    lines = [f"#{issue.get('number', '?')} {(issue.get('title') or '').strip()}"]
    state = (issue.get("state") or "").strip() or "open"
    lines.append(f"state: {state}")
    raw_labels = issue.get("labels") or []
    labels = []
    for lbl in raw_labels:
        if isinstance(lbl, dict):
            n = (lbl.get("name") or "").strip()
            if n:
                labels.append(n)
        elif isinstance(lbl, str) and lbl.strip():
            labels.append(lbl.strip())
    if labels:
        lines.append(f"labels: {', '.join(labels)}")
    assignees = issue.get("assignees") or []
    assignee_logins = [(a.get("login") or "").strip() for a in assignees if isinstance(a, dict)]
    assignee_logins = [a for a in assignee_logins if a]
    if assignee_logins:
        lines.append(f"assignees: {', '.join(assignee_logins)}")
    body = (issue.get("body") or "").strip()
    if body:
        lines.append("")
        lines.append(body)
    return "\n".join(lines)


def _render_comment_block(comment: dict) -> str:
    user = ((comment.get("user") or {}).get("login") or "").strip() or "unknown"
    created = (comment.get("created_at") or "").strip()
    header = f"{user}"
    if created:
        header += f" at {created}"
    body = (comment.get("body") or "").strip()
    return f"{header}\n{body}" if body else header


def _build_review_checklist_text(focus: str) -> str:
    memory_block = "\n".join([
        "Memory review — scan the conversation since the last review and ask:",
        "1. Did the user reveal identity, role, preferences, habits, goals, or constraints not yet stored?",
        "2. Did the user express expectations about how you should behave, communicate, or choose tools?",
        "3. Did the user correct an approach? What should never repeat, and what should happen instead?",
        "4. Did the user validate a non-obvious choice? Worth saving — corrections alone make you timid.",
        "5. Did the turn invalidate a memory you recalled or would have recalled? Candidate for clawmem_forget or clawmem_update.",
        "6. Does any new memory belong in a project repo or shared team repo rather than default_repo?",
        "For each yes, prefer clawmem_update on an existing canonical node, else clawmem_store with a deliberate kind/topics, else clawmem_forget.",
    ])
    skill_block = "\n".join([
        "Skill review — ask:",
        "1. Was a non-trivial approach used (trial and error, course changes, error recovery) that produced a good result?",
        "2. Did a specific sequence of tool calls or decisions lead to a useful outcome that is hard to re-derive?",
        "3. Did the user describe a procedure to follow in the future?",
        "4. Does an existing kind:skill cover this, and did this turn confirm, refine, or contradict it?",
        "If yes on 1-3 and no match: clawmem_store a new kind:skill.",
        "If yes on 4 (confirm/refine): clawmem_update that skill — bump last_validated, append evidence, tighten steps/checks.",
        "If yes on 4 (contradicted): fix steps/checks in place, or close the node and open a replacement with superseded-by: #<old-id>.",
        "Lesson -> Skill: two or more active kind:lesson nodes pointing at the same corrective direction = promote to one kind:skill, close the lessons.",
    ])
    if focus == "memory":
        return memory_block
    if focus == "skill":
        return skill_block
    return f"{memory_block}\n\n{skill_block}"


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
        # Non-secret fields only. The CLAWMEM_TOKEN is provisioned by
        # ``post_setup``'s auto-bootstrap and written to ``.env``; it never
        # appears in this schema.
        return [
            {
                "key": "git_base_url",
                "label": "ClawMem git server URL",
                "description": "Base URL of the ClawMem git service (default: https://git.clawmem.ai).",
                "default": _DEFAULT_GIT_BASE_URL,
                "required": False,
            },
            {
                "key": "console_base_url",
                "label": "ClawMem console URL",
                "description": "Base URL of the ClawMem web console (default: https://console.clawmem.ai).",
                "default": _DEFAULT_CONSOLE_BASE_URL,
                "required": False,
            },
            {
                "key": "login",
                "label": "Agent login",
                "description": "Agent identity login assigned at bootstrap.",
                "required": False,
            },
            {
                "key": "default_repo",
                "label": "Default memory repo",
                "description": "owner/repo used for memory and conversation mirroring.",
                "required": False,
            },
        ]

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
        return list(ALL_TOOL_SCHEMAS)

    # Map tool name → handler method name. Keeps dispatch compact.
    _TOOL_DISPATCH: Dict[str, str] = {
        "clawmem_recall": "_handle_recall",
        "clawmem_store": "_handle_store",
        "clawmem_list": "_handle_list",
        "clawmem_get": "_handle_get",
        "clawmem_update": "_handle_update",
        "clawmem_forget": "_handle_forget",
        "clawmem_console": "_handle_console",
        # Memory extensions
        "clawmem_labels": "_handle_labels",
        "clawmem_repos": "_handle_repos",
        "clawmem_repo_create": "_handle_repo_create",
        "clawmem_repo_set_default": "_handle_repo_set_default",
        "clawmem_review": "_handle_review",
        # Generic issues
        "clawmem_issue_create": "_handle_issue_create",
        "clawmem_issue_list": "_handle_issue_list",
        "clawmem_issue_get": "_handle_issue_get",
        "clawmem_issue_update": "_handle_issue_update",
        "clawmem_issue_comment_add": "_handle_issue_comment_add",
        "clawmem_issue_comments_list": "_handle_issue_comments_list",
        # Collaboration
        "clawmem_collaboration_orgs": "_handle_collab_orgs",
        "clawmem_collaboration_org_create": "_handle_collab_org_create",
        "clawmem_collaboration_org_repo_create": "_handle_collab_org_repo_create",
        "clawmem_collaboration_org_members": "_handle_collab_org_members",
        "clawmem_collaboration_org_membership": "_handle_collab_org_membership",
        "clawmem_collaboration_org_member_remove": "_handle_collab_org_member_remove",
        "clawmem_collaboration_org_membership_remove": "_handle_collab_org_membership_remove",
        "clawmem_collaboration_teams": "_handle_collab_teams",
        "clawmem_collaboration_team": "_handle_collab_team",
        "clawmem_collaboration_team_create": "_handle_collab_team_create",
        "clawmem_collaboration_team_update": "_handle_collab_team_update",
        "clawmem_collaboration_team_delete": "_handle_collab_team_delete",
        "clawmem_collaboration_team_members": "_handle_collab_team_members",
        "clawmem_collaboration_team_membership_set": "_handle_collab_team_membership_set",
        "clawmem_collaboration_team_membership_remove": "_handle_collab_team_membership_remove",
        "clawmem_collaboration_team_repos": "_handle_collab_team_repos",
        "clawmem_collaboration_team_repo_set": "_handle_collab_team_repo_set",
        "clawmem_collaboration_team_repo_remove": "_handle_collab_team_repo_remove",
        "clawmem_collaboration_repo_transfer": "_handle_collab_repo_transfer",
        "clawmem_collaboration_repo_collaborators": "_handle_collab_repo_collaborators",
        "clawmem_collaboration_repo_invitations": "_handle_collab_repo_invitations",
        "clawmem_collaboration_repo_collaborator_set": "_handle_collab_repo_collaborator_set",
        "clawmem_collaboration_repo_collaborator_remove": "_handle_collab_repo_collaborator_remove",
        "clawmem_collaboration_user_repo_invitations": "_handle_collab_user_repo_invitations",
        "clawmem_collaboration_user_repo_invitation_accept": "_handle_collab_user_repo_invitation_accept",
        "clawmem_collaboration_user_repo_invitation_decline": "_handle_collab_user_repo_invitation_decline",
        "clawmem_collaboration_org_invitations": "_handle_collab_org_invitations",
        "clawmem_collaboration_org_invitation_create": "_handle_collab_org_invitation_create",
        "clawmem_collaboration_org_invitation_revoke": "_handle_collab_org_invitation_revoke",
        "clawmem_collaboration_user_org_invitations": "_handle_collab_user_org_invitations",
        "clawmem_collaboration_user_org_invitation_accept": "_handle_collab_user_org_invitation_accept",
        "clawmem_collaboration_user_org_invitation_decline": "_handle_collab_user_org_invitation_decline",
        "clawmem_collaboration_outside_collaborators": "_handle_collab_outside_collaborators",
        "clawmem_collaboration_repo_access_inspect": "_handle_collab_repo_access_inspect",
    }

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._client:
            return tool_error("ClawMem is not active for this session.")

        handler_name = self._TOOL_DISPATCH.get(tool_name)
        if not handler_name:
            return tool_error(f"Unknown tool: {tool_name}")

        try:
            handler = getattr(self, handler_name)
            if tool_name == "clawmem_console":
                return handler()
            return handler(args)
        except Exception as e:
            logger.error("ClawMem tool %s failed: %s", tool_name, e)
            return tool_error(f"ClawMem {tool_name} failed: {e}")

    # -- Confirmation gate / repo resolver -----------------------------------

    @staticmethod
    def _require_confirmation(args: dict, action: str) -> Optional[str]:
        """Return error string if confirmed is missing, else None."""
        if args.get("confirmed") is True:
            return None
        return tool_error(
            f"Refusing to {action} without explicit confirmation. Inspect "
            f"current state first, then retry with confirmed=true only after "
            f"the user approves the exact change."
        )

    def _resolve_repo_arg(self, repo_arg: Any) -> tuple[str, str, str] | None:
        """Parse repo argument or fall back to default_repo.

        Returns ``(owner, repo, full_name)`` or ``None`` when unresolved.
        """
        raw = (repo_arg or "").strip() if isinstance(repo_arg, str) else ""
        full_name = raw or self._default_repo
        if not full_name or "/" not in full_name:
            return None
        owner, _, repo = full_name.partition("/")
        owner = owner.strip()
        repo = repo.strip()
        if not owner or not repo:
            return None
        return owner, repo, f"{owner}/{repo}"

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

    # -- Memory extension tools ----------------------------------------------

    def _handle_labels(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync

        limit_topics = args.get("limit_topics")
        try:
            limit_topics = int(limit_topics) if limit_topics is not None else 50
        except Exception:
            limit_topics = 50
        limit_topics = max(1, min(limit_topics, 200))

        all_labels: list[dict] = []
        page = 1
        while page <= 20:
            batch = run_sync(self._client.list_labels(page=page, per_page=100))
            if not batch:
                break
            all_labels.extend(batch)
            if len(batch) < 100:
                break
            page += 1

        schema_prefixes = ("kind:", "topic:", "type:")
        kinds: list[str] = []
        topics: list[str] = []
        types: list[str] = []
        for lbl in all_labels:
            name = (lbl.get("name") or "").strip()
            if not name:
                continue
            if name.startswith("kind:"):
                kinds.append(name)
            elif name.startswith("topic:"):
                topics.append(name)
            elif name.startswith("type:"):
                types.append(name)

        kinds.sort()
        types.sort()
        topics.sort()
        topics = topics[:limit_topics]

        lines = [f"ClawMem schema labels in {self._default_repo}:"]
        lines.append("")
        lines.append("types:")
        lines.extend([f"- {t}" for t in types] or ["- (none)"])
        lines.append("")
        lines.append("kinds:")
        lines.extend([f"- {k}" for k in kinds] or ["- (none)"])
        lines.append("")
        lines.append(f"topics (showing up to {limit_topics}):")
        lines.extend([f"- {t}" for t in topics] or ["- (none)"])
        return json.dumps({
            "result": "\n".join(lines),
            "counts": {"types": len(types), "kinds": len(kinds), "topics_shown": len(topics)},
        })

    def _handle_repos(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        repos = run_sync(self._client.list_user_repos())
        if not repos:
            return json.dumps({"result": "No repos visible to the current identity.", "count": 0})
        lines = ["Memory repos visible to the current identity:"]
        default = (self._default_repo or "").strip()
        for repo in repos:
            full = _repo_summary_full_name(repo) or "(unknown)"
            marker = " (default)" if full == default else ""
            lines.append(f"- {_render_repo_line(repo)}{marker}")
        return json.dumps({
            "result": "\n".join(lines),
            "count": len(repos),
            "default_repo": default,
        })

    def _handle_repo_create(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync

        name = (args.get("name") or "").strip()
        if not name:
            return tool_error("Missing required parameter: name")
        description = (args.get("description") or "").strip() or None
        private = args.get("private")
        if private is None:
            private = True
        set_default = bool(args.get("set_default"))

        created = run_sync(self._client.create_user_repo(
            name, description=description, private=bool(private), auto_init=False,
        ))
        full_name = _repo_summary_full_name(created) or name
        msg = f"Created repo {full_name}."
        if set_default:
            # Mutate config + in-memory state.
            from hermes_constants import get_hermes_home
            config_path = get_hermes_home() / "clawmem.json"
            existing: dict = {}
            if config_path.exists():
                try:
                    existing = json.loads(config_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            existing["default_repo"] = full_name
            config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            self._default_repo = full_name
            if self._client is not None:
                self._client.default_repo = full_name
            msg += f" Default repo updated to {full_name}."
        return json.dumps({
            "result": msg,
            "repo": _render_repo_line(created),
            "default_updated": set_default,
        })

    def _handle_repo_set_default(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "retarget the agent default repo")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        from hermes_constants import get_hermes_home

        new_repo = (args.get("repo") or "").strip()
        if not new_repo or "/" not in new_repo:
            return tool_error("repo must be owner/repo.")
        owner, _, repo_name = new_repo.partition("/")
        # Verify it exists and is visible.
        repo = run_sync(self._client.get_repo(owner.strip(), repo_name.strip()))
        if not repo:
            return tool_error(f"Repo {new_repo} not found or not visible.")

        config_path = get_hermes_home() / "clawmem.json"
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing["default_repo"] = new_repo
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        self._default_repo = new_repo
        if self._client is not None:
            self._client.default_repo = new_repo
        return json.dumps({
            "result": f"Default repo set to {new_repo}.",
            "default_repo": new_repo,
        })

    def _handle_review(self, args: dict) -> str:
        focus = args.get("focus") or "both"
        if focus not in ("memory", "skill", "both"):
            focus = "both"
        header = "ClawMem review checklist."
        body = _build_review_checklist_text(focus)
        return json.dumps({"result": f"{header}\n\n{body}", "focus": focus})

    # -- Generic issue tools -------------------------------------------------

    def _handle_issue_create(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        title = (args.get("title") or "").strip()
        if not title:
            return tool_error("Missing required parameter: title")
        body = args.get("body") or ""
        labels = args.get("labels") or []
        if not isinstance(labels, list):
            return tool_error("labels must be an array of strings.")
        labels = [lbl.strip() for lbl in labels if isinstance(lbl, str) and lbl.strip()]
        if labels:
            run_sync(self._client.ensure_labels(labels))
        issue = run_sync(self._client.create_issue(title, body, labels))
        assignees = args.get("assignees") or []
        state = args.get("state")
        if (isinstance(assignees, list) and assignees) or state == "closed":
            payload: dict = {}
            if isinstance(assignees, list) and assignees:
                payload["labels"] = None  # never mutate labels here
                # update_issue only accepts labels via parameter; assignees not supported on current client
            if state == "closed":
                run_sync(self._client.update_issue(
                    int(issue["number"]), state="closed",
                ))
                issue["state"] = "closed"
        return json.dumps({
            "result": f"Created #{issue.get('number', '?')}: {_render_issue_line(issue)}",
            "issue_number": issue.get("number"),
        })

    def _handle_issue_list(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        state = args.get("state") or "open"
        if state not in ("open", "closed", "all"):
            state = "open"
        labels = args.get("labels") or []
        if not isinstance(labels, list):
            labels = []
        labels = [lbl.strip() for lbl in labels if isinstance(lbl, str) and lbl.strip()]
        try:
            limit = int(args.get("limit", 20))
        except Exception:
            limit = 20
        limit = max(1, min(limit, 200))
        issues: list[dict] = []
        page = 1
        while len(issues) < limit and page <= 20:
            batch = run_sync(self._client.list_issues(
                labels=labels or None,
                state=state,
                page=page,
                per_page=min(100, limit),
            ))
            if not batch:
                break
            issues.extend(batch)
            if len(batch) < min(100, limit):
                break
            page += 1
        issues = issues[:limit]
        if not issues:
            return json.dumps({"result": "No issues found.", "count": 0})
        lines = [f"- {_render_issue_line(i)}" for i in issues]
        return json.dumps({"result": "\n".join(lines), "count": len(issues)})

    def _handle_issue_get(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        num = args.get("issue_number")
        if not isinstance(num, int) or num <= 0:
            return tool_error("issue_number must be a positive integer.")
        issue = run_sync(self._client.get_issue(num))
        if not issue:
            return tool_error(f"Issue #{num} not found.")
        return json.dumps({"result": _render_issue_block(issue), "issue_number": num})

    def _handle_issue_update(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        num = args.get("issue_number")
        if not isinstance(num, int) or num <= 0:
            return tool_error("issue_number must be a positive integer.")
        payload: dict = {}
        if "title" in args:
            payload["title"] = str(args["title"])
        if "body" in args:
            payload["body"] = str(args["body"])
        if "state" in args:
            state = args["state"]
            if state not in ("open", "closed"):
                return tool_error("state must be open or closed.")
            payload["state"] = state
        if "labels" in args:
            labels = args["labels"]
            if not isinstance(labels, list):
                return tool_error("labels must be an array of strings.")
            labels = [lbl.strip() for lbl in labels if isinstance(lbl, str) and lbl.strip()]
            if labels:
                run_sync(self._client.ensure_labels(labels))
            payload["labels"] = labels
        if not payload:
            return tool_error("Nothing to update. Provide at least one of: title, body, state, labels.")
        updated = run_sync(self._client.update_issue(num, **payload))
        return json.dumps({
            "result": f"Updated #{num}: {_render_issue_line(updated)}",
            "issue_number": num,
        })

    def _handle_issue_comment_add(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        num = args.get("issue_number")
        if not isinstance(num, int) or num <= 0:
            return tool_error("issue_number must be a positive integer.")
        body = (args.get("body") or "").strip()
        if not body:
            return tool_error("Missing required parameter: body")
        run_sync(self._client.create_comment(num, body))
        return json.dumps({"result": f"Comment added to #{num}.", "issue_number": num})

    def _handle_issue_comments_list(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        num = args.get("issue_number")
        if not isinstance(num, int) or num <= 0:
            return tool_error("issue_number must be a positive integer.")
        sort = args.get("sort")
        direction = args.get("direction")
        since = args.get("since")
        try:
            limit = int(args.get("limit", 20))
        except Exception:
            limit = 20
        limit = max(1, min(limit, 200))
        comments: list[dict] = []
        page = 1
        while len(comments) < limit and page <= 20:
            batch = run_sync(self._client.list_comments(
                num,
                page=page,
                per_page=min(100, limit),
                sort=sort,
                direction=direction,
                since=since,
            ))
            if not batch:
                break
            comments.extend(batch)
            if len(batch) < min(100, limit):
                break
            page += 1
        comments = comments[:limit]
        if not comments:
            return json.dumps({"result": f"No comments on #{num}.", "count": 0})
        blocks = [_render_comment_block(c) for c in comments]
        return json.dumps({
            "result": "\n\n---\n\n".join(blocks),
            "count": len(comments),
        })

    # -- Collaboration: orgs -------------------------------------------------

    def _handle_collab_orgs(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        orgs = run_sync(self._client.list_user_orgs())
        if not orgs:
            return json.dumps({"result": "No organizations visible.", "count": 0})
        lines = ["Organizations visible to the current identity:"]
        lines.extend([f"- {_render_org_line(o)}" for o in orgs])
        return json.dumps({"result": "\n".join(lines), "count": len(orgs)})

    def _handle_collab_org_create(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "create an organization")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        login = (args.get("login") or "").strip()
        if not login:
            return tool_error("Missing required parameter: login")
        name = (args.get("name") or "").strip() or None
        default_perm = _resolve_org_default_permission(args.get("default_permission"), "read")
        if "error" in default_perm:
            return tool_error(default_perm["error"])
        created = run_sync(self._client.create_user_org(
            login,
            name=name,
            default_repository_permission=default_perm["permission"],
        ))
        return json.dumps({
            "result": f"Created organization {_render_org_line(created)}.",
            "org": created.get("login"),
        })

    def _handle_collab_org_repo_create(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "create an organization repo")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        name = (args.get("name") or "").strip()
        if not org or not name:
            return tool_error("Missing required parameter(s): org, name")
        description = (args.get("description") or "").strip() or None
        private = args.get("private")
        if private is None:
            private = True
        created = run_sync(self._client.create_org_repo(
            org, name,
            description=description,
            private=bool(private),
            auto_init=bool(args.get("auto_init", False)),
            has_issues=args.get("has_issues"),
            has_wiki=args.get("has_wiki"),
        ))
        return json.dumps({
            "result": f"Created org repo {_render_repo_line(created)}.",
            "repo": _repo_summary_full_name(created),
        })

    def _handle_collab_org_members(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        if not org:
            return tool_error("Missing required parameter: org")
        role = args.get("role")
        members = run_sync(self._client.list_org_members(org, role=role))
        if not members:
            return json.dumps({"result": f"No members found in org {org}.", "count": 0})
        lines = [f"Members in org {org}:"]
        lines.extend([f"- {_render_collaborator_line(m)}" for m in members])
        return json.dumps({"result": "\n".join(lines), "count": len(members)})

    def _handle_collab_org_membership(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        username = (args.get("username") or "").strip()
        if not org or not username:
            return tool_error("Missing required parameter(s): org, username")
        membership = run_sync(self._client.get_org_membership(org, username))
        if not membership:
            return json.dumps({
                "result": f"No active or pending membership for {username} in {org}.",
                "found": False,
            })
        return json.dumps({
            "result": _render_org_membership_line(membership),
            "found": True,
        })

    def _handle_collab_org_member_remove(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "remove an organization member")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        username = (args.get("username") or "").strip()
        if not org or not username:
            return tool_error("Missing required parameter(s): org, username")
        run_sync(self._client.remove_org_member(org, username))
        return json.dumps({"result": f"Removed member {username} from org {org}."})

    def _handle_collab_org_membership_remove(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "remove an organization membership")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        username = (args.get("username") or "").strip()
        if not org or not username:
            return tool_error("Missing required parameter(s): org, username")
        run_sync(self._client.remove_org_membership(org, username))
        return json.dumps({"result": f"Removed membership for {username} in org {org}."})

    # -- Collaboration: teams ------------------------------------------------

    def _handle_collab_teams(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        if not org:
            return tool_error("Missing required parameter: org")
        teams = run_sync(self._client.list_org_teams(org))
        if not teams:
            return json.dumps({"result": f"No teams in org {org}.", "count": 0})
        lines = [f"Teams in org {org}:"]
        lines.extend([f"- {_render_team_line(t)}" for t in teams])
        return json.dumps({"result": "\n".join(lines), "count": len(teams)})

    def _handle_collab_team(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        slug = (args.get("team_slug") or "").strip()
        if not org or not slug:
            return tool_error("Missing required parameter(s): org, team_slug")
        team = run_sync(self._client.get_team(org, slug))
        return json.dumps({
            "result": f"Team in {org}: {_render_team_line(team)}",
            "team_slug": team.get("slug"),
        })

    def _handle_collab_team_create(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "create a team")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        name = (args.get("name") or "").strip()
        if not org or not name:
            return tool_error("Missing required parameter(s): org, name")
        description = (args.get("description") or "").strip() or None
        privacy = args.get("privacy") or "closed"
        if privacy not in ("closed", "secret"):
            return tool_error("privacy must be closed or secret.")
        team = run_sync(self._client.create_org_team(
            org, name, description=description, privacy=privacy,
        ))
        return json.dumps({
            "result": f"Created team in {org}: {_render_team_line(team)}.",
            "team_slug": team.get("slug"),
        })

    def _handle_collab_team_update(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "update a team")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        slug = (args.get("team_slug") or "").strip()
        if not org or not slug:
            return tool_error("Missing required parameter(s): org, team_slug")
        name = (args.get("name") or "").strip() or None
        description = args.get("description")
        privacy = args.get("privacy")
        if privacy is not None and privacy not in ("closed", "secret"):
            return tool_error("privacy must be closed or secret.")
        updated = run_sync(self._client.update_team(
            org, slug, name=name, description=description, privacy=privacy,
        ))
        return json.dumps({
            "result": f"Updated team in {org}: {_render_team_line(updated)}.",
        })

    def _handle_collab_team_delete(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "delete a team")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        slug = (args.get("team_slug") or "").strip()
        if not org or not slug:
            return tool_error("Missing required parameter(s): org, team_slug")
        run_sync(self._client.delete_team(org, slug))
        return json.dumps({"result": f"Deleted team {slug} in org {org}."})

    def _handle_collab_team_members(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        slug = (args.get("team_slug") or "").strip()
        if not org or not slug:
            return tool_error("Missing required parameter(s): org, team_slug")
        members = run_sync(self._client.list_team_members(org, slug))
        if not members:
            return json.dumps({"result": f"No members on team {slug} in {org}.", "count": 0})
        lines = [f"Members on team {slug} in {org}:"]
        lines.extend([f"- {_render_collaborator_line(m)}" for m in members])
        return json.dumps({"result": "\n".join(lines), "count": len(members)})

    def _handle_collab_team_membership_set(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "change team membership")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        slug = (args.get("team_slug") or "").strip()
        username = (args.get("username") or "").strip()
        if not org or not slug or not username:
            return tool_error("Missing required parameter(s): org, team_slug, username")
        role = args.get("role") or "member"
        if role not in ("member", "maintainer"):
            return tool_error("role must be member or maintainer.")
        run_sync(self._client.set_team_membership(org, slug, username, role))
        return json.dumps({
            "result": f"Set {username} as {role} on team {slug} in {org}.",
        })

    def _handle_collab_team_membership_remove(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "remove a team membership")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        slug = (args.get("team_slug") or "").strip()
        username = (args.get("username") or "").strip()
        if not org or not slug or not username:
            return tool_error("Missing required parameter(s): org, team_slug, username")
        run_sync(self._client.remove_team_membership(org, slug, username))
        return json.dumps({
            "result": f"Removed {username} from team {slug} in {org}.",
        })

    def _handle_collab_team_repos(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        slug = (args.get("team_slug") or "").strip()
        if not org or not slug:
            return tool_error("Missing required parameter(s): org, team_slug")
        repos = run_sync(self._client.list_team_repos(org, slug))
        if not repos:
            return json.dumps({"result": f"No repos granted to team {slug} in {org}.", "count": 0})
        lines = [f"Repos granted to team {slug} in {org}:"]
        lines.extend([f"- {_render_repo_line(r)}" for r in repos])
        return json.dumps({"result": "\n".join(lines), "count": len(repos)})

    def _handle_collab_team_repo_set(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "grant team repo access")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        slug = (args.get("team_slug") or "").strip()
        if not org or not slug:
            return tool_error("Missing required parameter(s): org, team_slug")
        target = self._resolve_repo_arg(args.get("repo"))
        if not target:
            return tool_error("repo must be owner/repo (or configure a default repo).")
        owner, repo_name, _ = target
        perm_res = _resolve_collaboration_permission(args.get("permission"), "read")
        if "error" in perm_res:
            return tool_error(perm_res["error"])
        run_sync(self._client.set_team_repo_access(
            org, slug, owner, repo_name, perm_res["permission"],
        ))
        return json.dumps({
            "result": f"Granted {perm_res['permission']} on {owner}/{repo_name} to team {slug} in {org}.",
        })

    def _handle_collab_team_repo_remove(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "remove a team repo grant")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        slug = (args.get("team_slug") or "").strip()
        if not org or not slug:
            return tool_error("Missing required parameter(s): org, team_slug")
        target = self._resolve_repo_arg(args.get("repo"))
        if not target:
            return tool_error("repo must be owner/repo (or configure a default repo).")
        owner, repo_name, _ = target
        run_sync(self._client.remove_team_repo_access(org, slug, owner, repo_name))
        return json.dumps({
            "result": f"Removed team {slug} grant from {owner}/{repo_name}.",
        })

    # -- Collaboration: repo-level -------------------------------------------

    def _handle_collab_repo_transfer(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "transfer a repository")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        target = self._resolve_repo_arg(args.get("repo"))
        if not target:
            return tool_error("repo must be owner/repo (or configure a default repo).")
        owner, repo_name, full = target
        new_owner = (args.get("new_owner") or "").strip()
        if not new_owner:
            return tool_error("Missing required parameter: new_owner")
        new_name = (args.get("new_name") or "").strip() or None
        run_sync(self._client.transfer_repo(owner, repo_name, new_owner, new_name))
        dest = f"{new_owner}/{new_name or repo_name}"
        return json.dumps({"result": f"Transferred {full} to {dest}."})

    def _handle_collab_repo_collaborators(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        target = self._resolve_repo_arg(args.get("repo"))
        if not target:
            return tool_error("repo must be owner/repo (or configure a default repo).")
        owner, repo_name, full = target
        raw = run_sync(self._client.list_repo_collaborators(owner, repo_name))
        collaborators = _filter_direct_collaborators(raw, owner)
        if not collaborators:
            return json.dumps({"result": f"No direct collaborators on {full} (excluding owner).", "count": 0})
        lines = [f"Direct collaborators on {full} (excluding owner):"]
        lines.extend([f"- {_render_collaborator_line(c)}" for c in collaborators])
        return json.dumps({"result": "\n".join(lines), "count": len(collaborators)})

    def _handle_collab_repo_invitations(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        target = self._resolve_repo_arg(args.get("repo"))
        if not target:
            return tool_error("repo must be owner/repo (or configure a default repo).")
        owner, repo_name, full = target
        invitations = run_sync(self._client.list_repo_invitations(owner, repo_name))
        if not invitations:
            return json.dumps({"result": f"No pending repo invitations on {full}.", "count": 0})
        lines = [f"Pending repo invitations on {full}:"]
        lines.extend([f"- {_render_repo_invitation_line(i)}" for i in invitations])
        return json.dumps({"result": "\n".join(lines), "count": len(invitations)})

    def _handle_collab_repo_collaborator_set(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "change a direct collaborator")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        target = self._resolve_repo_arg(args.get("repo"))
        if not target:
            return tool_error("repo must be owner/repo (or configure a default repo).")
        owner, repo_name, full = target
        username = (args.get("username") or "").strip()
        if not username:
            return tool_error("Missing required parameter: username")
        perm_res = _resolve_collaboration_permission(args.get("permission"), "read")
        if "error" in perm_res:
            return tool_error(perm_res["error"])
        run_sync(self._client.set_repo_collaborator(
            owner, repo_name, username, perm_res["permission"],
        ))
        return json.dumps({
            "result": f"Set {username} as {perm_res['permission']} collaborator on {full}.",
        })

    def _handle_collab_repo_collaborator_remove(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "remove a direct collaborator")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        target = self._resolve_repo_arg(args.get("repo"))
        if not target:
            return tool_error("repo must be owner/repo (or configure a default repo).")
        owner, repo_name, full = target
        username = (args.get("username") or "").strip()
        if not username:
            return tool_error("Missing required parameter: username")
        run_sync(self._client.remove_repo_collaborator(owner, repo_name, username))
        return json.dumps({
            "result": f"Removed collaborator {username} from {full}.",
        })

    # -- Collaboration: user invitations (self side) -------------------------

    def _handle_collab_user_repo_invitations(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        invitations = run_sync(self._client.list_user_repo_invitations())
        repo_filter = (args.get("repo") or "").strip() or None
        if repo_filter:
            invitations = [
                inv for inv in invitations
                if _repo_summary_full_name(inv.get("repository")) == repo_filter
            ]
        if not invitations:
            return json.dumps({"result": "No pending repository invitations.", "count": 0})
        lines = ["Pending repository invitations for the current identity:"]
        lines.extend([f"- {_render_repo_invitation_line(i)}" for i in invitations])
        return json.dumps({"result": "\n".join(lines), "count": len(invitations)})

    def _handle_collab_user_repo_invitation_accept(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "accept a repository invitation")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        inv_id = args.get("invitation_id")
        if not isinstance(inv_id, int) or inv_id <= 0:
            return tool_error("invitation_id must be a positive integer.")
        run_sync(self._client.accept_user_repo_invitation(inv_id))
        return json.dumps({"result": f"Accepted repository invitation {inv_id}."})

    def _handle_collab_user_repo_invitation_decline(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "decline a repository invitation")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        inv_id = args.get("invitation_id")
        if not isinstance(inv_id, int) or inv_id <= 0:
            return tool_error("invitation_id must be a positive integer.")
        run_sync(self._client.decline_user_repo_invitation(inv_id))
        return json.dumps({"result": f"Declined repository invitation {inv_id}."})

    # -- Collaboration: org invitations (admin) ------------------------------

    def _handle_collab_org_invitations(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        if not org:
            return tool_error("Missing required parameter: org")
        invitations = run_sync(self._client.list_org_invitations(org))
        if not invitations:
            return json.dumps({"result": f"No pending invitations in org {org}.", "count": 0})
        lines = [f"Pending invitations in org {org}:"]
        lines.extend([f"- {_render_org_invitation_line(i)}" for i in invitations])
        return json.dumps({"result": "\n".join(lines), "count": len(invitations)})

    def _handle_collab_org_invitation_create(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "create an organization invitation")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        invitee = (args.get("invitee_login") or "").strip()
        if not org or not invitee:
            return tool_error("Missing required parameter(s): org, invitee_login")
        role_res = _resolve_org_invitation_role(args.get("role"), "member")
        if "error" in role_res:
            return tool_error(role_res["error"])
        team_ids_raw = args.get("team_ids") or []
        if not isinstance(team_ids_raw, list):
            return tool_error("team_ids must be an array of positive integers.")
        team_ids: list[int] = []
        for tid in team_ids_raw:
            if not isinstance(tid, int) or tid <= 0:
                return tool_error("team_ids must contain only positive integers.")
            team_ids.append(tid)
        expires_in_days = args.get("expires_in_days")
        if expires_in_days is not None and (not isinstance(expires_in_days, int) or expires_in_days <= 0):
            return tool_error("expires_in_days must be a positive integer.")
        invitation = run_sync(self._client.create_org_invitation(
            org, invitee,
            role=role_res["role"],
            team_ids=team_ids or None,
            expires_in_days=expires_in_days,
        ))
        return json.dumps({
            "result": f"Created invitation in {org}: {_render_org_invitation_line(invitation)}.",
            "invitation_id": invitation.get("id"),
        })

    def _handle_collab_org_invitation_revoke(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "revoke an organization invitation")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        inv_id = args.get("invitation_id")
        if not org:
            return tool_error("Missing required parameter: org")
        if not isinstance(inv_id, int) or inv_id <= 0:
            return tool_error("invitation_id must be a positive integer.")
        run_sync(self._client.revoke_org_invitation(org, inv_id))
        return json.dumps({"result": f"Revoked invitation {inv_id} in org {org}."})

    # -- Collaboration: user-side org invitations ----------------------------

    def _handle_collab_user_org_invitations(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        invitations = run_sync(self._client.list_user_org_invitations())
        org_filter = (args.get("org") or "").strip() or None
        if org_filter:
            invitations = [
                inv for inv in invitations
                if ((inv.get("organization") or {}).get("login") or "").strip() == org_filter
            ]
        if not invitations:
            return json.dumps({"result": "No pending organization invitations.", "count": 0})
        lines = ["Pending organization invitations for the current identity:"]
        lines.extend([f"- {_render_user_org_invitation_line(i)}" for i in invitations])
        return json.dumps({"result": "\n".join(lines), "count": len(invitations)})

    def _handle_collab_user_org_invitation_accept(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "accept an organization invitation")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        inv_id = args.get("invitation_id")
        if not isinstance(inv_id, int) or inv_id <= 0:
            return tool_error("invitation_id must be a positive integer.")
        run_sync(self._client.accept_user_org_invitation(inv_id))
        return json.dumps({"result": f"Accepted organization invitation {inv_id}."})

    def _handle_collab_user_org_invitation_decline(self, args: dict) -> str:
        blocked = self._require_confirmation(args, "decline an organization invitation")
        if blocked:
            return blocked
        from plugins.memory.clawmem.client import run_sync
        inv_id = args.get("invitation_id")
        if not isinstance(inv_id, int) or inv_id <= 0:
            return tool_error("invitation_id must be a positive integer.")
        run_sync(self._client.decline_user_org_invitation(inv_id))
        return json.dumps({"result": f"Declined organization invitation {inv_id}."})

    # -- Collaboration: misc -------------------------------------------------

    def _handle_collab_outside_collaborators(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        org = (args.get("org") or "").strip()
        if not org:
            return tool_error("Missing required parameter: org")
        users = run_sync(self._client.list_org_outside_collaborators(org))
        if not users:
            return json.dumps({"result": f"No outside collaborators in org {org}.", "count": 0})
        lines = [f"Outside collaborators in org {org}:"]
        lines.extend([f"- {_render_collaborator_line(u)}" for u in users])
        return json.dumps({"result": "\n".join(lines), "count": len(users)})

    def _handle_collab_repo_access_inspect(self, args: dict) -> str:
        from plugins.memory.clawmem.client import run_sync
        target = self._resolve_repo_arg(args.get("repo"))
        if not target:
            return tool_error("repo must be owner/repo (or configure a default repo).")
        owner, repo_name, full = target
        username = (args.get("username") or "").strip()

        lines = [f"Repo access inspection for {full}:"]
        notes: list[str] = []
        org_name: Optional[str] = None
        org_default_perm: Optional[str] = None
        org_context_available = False

        try:
            repo = run_sync(self._client.get_repo(owner, repo_name))
            if repo:
                visibility = "private" if repo.get("private") else "shared/public"
                lines.append(f"- Visibility: {visibility}")
                desc = (repo.get("description") or "").strip()
                if desc:
                    lines.append(f"- Description: {desc}")
                org_name = ((repo.get("owner") or {}).get("login") or "").strip() or owner
            else:
                notes.append("Repo metadata unavailable.")
                org_name = owner
        except Exception as e:
            notes.append(f"Repo metadata unavailable: {e}")
            org_name = owner

        try:
            org = run_sync(self._client.get_org(org_name or owner))
            org_context_available = True
            org_default_perm = _normalize_permission_alias(org.get("default_repository_permission"))
            lines.append(f"- Org default repository permission: {org_default_perm or 'unknown'}")
        except Exception as e:
            notes.append(f"Org metadata unavailable for \"{org_name}\": {e}")

        if username:
            lines.append("")
            lines.append(f'Org membership for "{username}" in "{org_name}":')
            if not org_name or not org_context_available:
                lines.append("- Not applicable because the owner org could not be resolved.")
            else:
                try:
                    membership = run_sync(self._client.get_org_membership(org_name, username))
                    if membership:
                        lines.append(f"- {_render_org_membership_line(membership)}")
                        if (membership.get("state") or "").strip() == "active":
                            if org_default_perm and org_default_perm != "none":
                                lines.append(
                                    f'- Org base repo access is active via default permission "{org_default_perm}".'
                                )
                                notes.append(
                                    f'Because {username} is an active org member and "{org_name}" default '
                                    f'repository permission is {org_default_perm}, removing direct collaborators '
                                    f'or team grants alone may not remove repo access.'
                                )
                            else:
                                lines.append("- No org base repo access is visible because the org default permission is none.")
                        else:
                            lines.append("- Org base repo access is not active yet because the org membership is still pending.")
                    else:
                        lines.append("- No active or pending org membership was found.")
                        if org_default_perm and org_default_perm != "none":
                            lines.append("- Org base repo access does not apply unless the user becomes an org member.")
                except Exception as e:
                    notes.append(f'Org membership lookup failed for "{username}" in "{org_name}": {e}')
        elif org_default_perm and org_default_perm != "none":
            notes.append(
                f'Any active org member can still inherit {org_default_perm} access from "{org_name}" '
                f'even after direct collaborator or team grants are removed.'
            )

        try:
            raw_collabs = run_sync(self._client.list_repo_collaborators(owner, repo_name))
            collabs = _filter_direct_collaborators(raw_collabs, owner)
            lines.append("")
            lines.append("Explicit collaborators (excluding owner):")
            if not collabs:
                lines.append("- None visible")
            else:
                lines.extend([f"- {_render_collaborator_line(c)}" for c in collabs])
        except Exception as e:
            notes.append(f"Direct collaborator lookup failed: {e}")

        try:
            invitations = run_sync(self._client.list_repo_invitations(owner, repo_name))
            lines.append("")
            lines.append("Pending repository invitations:")
            if not invitations:
                lines.append("- None visible")
            else:
                lines.extend([f"- {_render_repo_invitation_line(i)}" for i in invitations])
        except Exception as e:
            notes.append(f"Repo invitation lookup failed: {e}")

        if org_name:
            try:
                teams = run_sync(self._client.list_org_teams(org_name))
                with_access: list[dict] = []
                for team in teams:
                    slug = (team.get("slug") or "").strip() or (team.get("name") or "").strip()
                    if not slug:
                        notes.append(f'Skipped a team in org "{org_name}" because it had no slug or name.')
                        continue
                    try:
                        repos = run_sync(self._client.list_team_repos(org_name, slug))
                        matching = next((r for r in repos if _repo_summary_full_name(r) == full), None)
                        if matching:
                            merged = dict(team)
                            if matching.get("permissions"):
                                merged["permissions"] = matching["permissions"]
                            if matching.get("role_name"):
                                merged["role_name"] = matching["role_name"]
                            with_access.append(merged)
                    except Exception as te:
                        notes.append(f"Team repo lookup failed for {org_name}/{slug}: {te}")
                lines.append("")
                lines.append("Teams with repo access:")
                if not with_access:
                    lines.append("- None visible")
                else:
                    lines.extend([f"- {_render_team_line(t)}" for t in with_access])
            except Exception as e:
                notes.append(f"Repo team grant lookup failed: {e}")

            try:
                outside = run_sync(self._client.list_org_outside_collaborators(org_name))
                lines.append("")
                lines.append(f'Outside collaborators in owner org "{org_name}":')
                if not outside:
                    lines.append("- None visible")
                else:
                    lines.extend([f"- {_render_collaborator_line(u)}" for u in outside])
            except Exception as e:
                notes.append(f"Outside collaborator lookup failed: {e}")

        if notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend([f"- {n}" for n in notes])
        return json.dumps({"result": "\n".join(lines)})

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
    if hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(ClawMemProvider())
