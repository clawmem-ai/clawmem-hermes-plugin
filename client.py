"""ClawMem REST client — httpx wrapper for the ClawMem Git Server API.

Stateless async client constructed with config.  All public methods are
async; callers use ``_run_sync()`` from the module-level persistent event
loop when invoked from synchronous MemoryProvider hooks.

API surface mirrors the subset of GitHub Issues API that ClawMem's Gitea
backend exposes:

  - Agent bootstrap (``POST /agents``)
  - Issue CRUD (create / update / get / list)
  - Comments (create)
  - Search (``GET /search/issues``)
  - Labels (idempotent ensure)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from datetime import date
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sync ↔ async bridging (persistent event loop)
# ---------------------------------------------------------------------------

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
        return _loop


def run_sync(coro):
    """Run an async coroutine from a synchronous context.

    Uses a module-level persistent event loop so cached httpx clients stay
    valid across calls (same pattern as ``model_tools._get_tool_loop``).
    """
    loop = _get_loop()
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Label helpers (ported from config.ts)
# ---------------------------------------------------------------------------


def label_color(label: str) -> str:
    """Return a hex color for a ClawMem-managed label."""
    if label.startswith("type:"):
        return "5319e7" if label == "type:memory" else "1d76db"
    if label.startswith("kind:"):
        return "5319e7"
    if label.startswith("topic:"):
        return "fbca04"
    if label.startswith("status:"):
        return "b60205"
    if label.startswith("agent:"):
        return "1d76db"
    if label.startswith("date:"):
        return "c5def5"
    return "0e8a16"


def label_description(label: str) -> str:
    """Return a description for a ClawMem-managed label."""
    prefixes = {
        "type:": "Issue type",
        "kind:": "Memory kind",
        "status:": "Conversation lifecycle status",
        "agent:": "Agent",
        "date:": "Date",
        "topic:": "Topic",
    }
    for pfx, desc in prefixes.items():
        if label.startswith(pfx):
            return f"{desc} label managed by clawmem."
    return "Label managed by clawmem."


# ---------------------------------------------------------------------------
# Issue body helpers
# ---------------------------------------------------------------------------


def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest of *text* (stripped, UTF-8)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def render_memory_body(detail: str, hash_val: str) -> str:
    """Render the flat-YAML body stored in a ``type:memory`` issue."""
    return render_flat_yaml([
        ("memory_hash", hash_val),
        ("date", date.today().isoformat()),
        ("detail", detail.strip()),
    ])


def render_memory_title(detail: str, title: str | None = None) -> str:
    """Render the title for a ``type:memory`` issue."""
    raw = title.strip() if title and title.strip() else detail.strip()
    prefix = "Memory: "
    if raw.startswith(prefix):
        return raw
    return f"{prefix}{raw[:120]}"


def render_flat_yaml(pairs: list[tuple[str, str]]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in pairs)


def parse_flat_yaml(text: str) -> dict[str, str]:
    """Parse simple ``key: value`` lines (one per line, no nesting)."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key and value:
            result[key] = value
    return result


def extract_label_names(labels: list) -> list[str]:
    """Extract label name strings from GitHub API label objects."""
    result: list[str] = []
    for item in labels:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = (item.get("name") or "").strip()
        else:
            continue
        if name:
            result.append(name)
    return result


def label_val(labels: list[str], prefix: str) -> str | None:
    """Extract value from the first label matching *prefix*."""
    for lbl in labels:
        if lbl.startswith(prefix):
            val = lbl[len(prefix):].strip()
            if val:
                return val
    return None


def parse_memory_issue(issue: dict) -> dict | None:
    """Parse a GitHub issue dict into a ClawMem memory record.

    Returns ``None`` if the issue is not a ``type:memory`` issue or has no
    detail text.
    """
    labels = extract_label_names(issue.get("labels", []))
    if "type:memory" not in labels:
        return None

    body = (issue.get("body") or "").strip()
    parsed = parse_flat_yaml(body)
    detail = (parsed.get("detail") or body).strip()
    if not detail:
        # List API may omit body — fall back to title (strip "Memory: " prefix)
        title_raw = (issue.get("title") or "").strip()
        detail = title_raw.removeprefix("Memory: ").strip() if title_raw else ""
    if not detail:
        return None

    kind = label_val(labels, "kind:")
    topics = [
        lbl[6:].strip()
        for lbl in labels
        if lbl.startswith("topic:") and lbl[6:].strip()
    ]
    status = "stale" if issue.get("state") == "closed" else "active"

    return {
        "issue_number": issue.get("number"),
        "title": (issue.get("title") or "").strip(),
        "memory_id": str(issue.get("number", "")),
        "memory_hash": parsed.get("memory_hash", ""),
        "date": parsed.get("date", ""),
        "detail": detail,
        "kind": kind,
        "topics": topics,
        "status": status,
    }


def format_memory_line(mem: dict) -> str:
    """One-line summary for tool results."""
    parts = [f"[{mem['memory_id']}]"]
    if mem.get("kind"):
        parts.append(f"({mem['kind']})")
    parts.append(mem["detail"][:120])
    if mem.get("topics"):
        parts.append(f"  topics: {', '.join(mem['topics'])}")
    return " ".join(parts)


def format_memory_block(mem: dict) -> str:
    """Multi-line detail for tool results."""
    lines = [
        f"ID: {mem['memory_id']}",
        f"Title: {mem['title']}",
        f"Status: {mem['status']}",
        f"Detail: {mem['detail']}",
    ]
    if mem.get("kind"):
        lines.append(f"Kind: {mem['kind']}")
    if mem.get("topics"):
        lines.append(f"Topics: {', '.join(mem['topics'])}")
    if mem.get("date"):
        lines.append(f"Date: {mem['date']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClawMemClient
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30.0


class ClawMemClient:
    """Async REST client for the ClawMem Git Server API."""

    def __init__(
        self,
        base_url: str,
        token: str,
        default_repo: str,
        auth_scheme: str = "token",
    ):
        # Normalize base_url: ensure it ends with /api/v3
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/api/v3"):
            base_url = f"{base_url}/api/v3"
        self.base_url = base_url
        self.token = token
        self.default_repo = default_repo
        self.auth_scheme = auth_scheme

    # -- Internal request helper --------------------------------------------

    def _headers(self, *, omit_auth: bool = False) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        }
        if not omit_auth and self.token:
            if self.auth_scheme == "bearer":
                headers["Authorization"] = f"Bearer {self.token}"
            else:
                headers["Authorization"] = f"token {self.token}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict | None = None,
        allow_not_found: bool = False,
        allow_validation_error: bool = False,
        omit_auth: bool = False,
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            resp = await client.request(
                method,
                url,
                json=body if body is not None else None,
                headers=self._headers(omit_auth=omit_auth),
            )

        if resp.status_code == 404 and allow_not_found:
            return None
        if resp.status_code == 422 and allow_validation_error:
            return None
        if resp.status_code == 204:
            return None

        if not resp.is_success:
            text = resp.text[:500] if resp.text else resp.reason_phrase
            raise RuntimeError(f"ClawMem API {method} {path}: HTTP {resp.status_code}: {text}")

        raw = resp.text.strip()
        if not raw:
            return None
        try:
            return resp.json()
        except Exception:
            logger.warning("ClawMem: failed to parse API response for %s %s", method, path)
            return None

    # -- Repo path helper ---------------------------------------------------

    def _repo_path(self, suffix: str) -> str:
        if not self.default_repo:
            raise RuntimeError("ClawMem repository is not configured")
        owner, _, repo = self.default_repo.partition("/")
        return f"repos/{quote(owner, safe='')}/{quote(repo, safe='')}/{suffix}"

    # -- Bootstrap (no auth) ------------------------------------------------

    @staticmethod
    async def register_agent(
        base_url: str,
        prefix_login: str,
        default_repo_name: str,
    ) -> dict:
        """``POST /agents`` — register an agent identity (no auth required).

        Returns ``{"login": "...", "token": "...", "repo_full_name": "..."}``.
        """
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/api/v3"):
            base_url = f"{base_url}/api/v3"
        url = f"{base_url}/agents"
        payload = {
            "prefix_login": prefix_login,
            "default_repo_name": default_repo_name,
        }
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Content-Type": "application/json",
                },
            )
        if not resp.is_success:
            text = resp.text[:500] if resp.text else resp.reason_phrase
            raise RuntimeError(
                f"ClawMem agent registration failed: HTTP {resp.status_code}: {text}"
            )
        return resp.json()

    # -- Issues -------------------------------------------------------------

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str],
    ) -> dict:
        """``POST /repos/{repo}/issues``"""
        return await self._request(
            "POST",
            self._repo_path("issues"),
            body={"title": title, "body": body, "labels": labels},
        )

    async def update_issue(
        self,
        issue_number: int,
        *,
        title: str | None = None,
        body: str | None = None,
        state: str | None = None,
        labels: list[str] | None = None,
    ) -> dict:
        """``PATCH /repos/{repo}/issues/{issue_number}``"""
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if state is not None:
            payload["state"] = state
        if labels is not None:
            payload["labels"] = labels
        return await self._request(
            "PATCH",
            self._repo_path(f"issues/{issue_number}"),
            body=payload,
        )

    async def get_issue(self, issue_number: int) -> dict:
        """``GET /repos/{repo}/issues/{issue_number}``"""
        return await self._request(
            "GET",
            self._repo_path(f"issues/{issue_number}"),
        )

    async def list_issues(
        self,
        *,
        labels: list[str] | None = None,
        state: str = "open",
        page: int = 1,
        per_page: int = 100,
    ) -> list[dict]:
        """``GET /repos/{repo}/issues``"""
        params = f"state={state}&page={page}&per_page={per_page}"
        if labels:
            params += f"&labels={','.join(labels)}"
        result = await self._request(
            "GET",
            self._repo_path(f"issues?{params}"),
        )
        return result if isinstance(result, list) else []

    # -- Comments -----------------------------------------------------------

    async def create_comment(self, issue_number: int, body: str) -> None:
        """``POST /repos/{repo}/issues/{issue_number}/comments``"""
        await self._request(
            "POST",
            self._repo_path(f"issues/{issue_number}/comments"),
            body={"body": body},
        )

    # -- Search -------------------------------------------------------------

    async def search_issues(
        self,
        query: str,
        *,
        page: int = 1,
        per_page: int = 100,
    ) -> list[dict]:
        """``GET /search/issues?q=...``"""
        from urllib.parse import urlencode
        params = urlencode({"q": query, "page": page, "per_page": per_page})
        result = await self._request("GET", f"search/issues?{params}")
        if isinstance(result, dict):
            return result.get("items", [])
        return []

    # -- Labels -------------------------------------------------------------

    async def ensure_labels(self, labels: list[str]) -> None:
        """Create labels if they don't exist (ignores 422 = already exists)."""
        for lbl in labels:
            lbl = lbl.strip()
            if not lbl:
                continue
            await self._request(
                "POST",
                self._repo_path("labels"),
                body={
                    "name": lbl,
                    "color": label_color(lbl),
                    "description": label_description(lbl),
                },
                allow_validation_error=True,
            )

    async def sync_managed_labels(
        self,
        issue_number: int,
        desired: list[str],
    ) -> None:
        """Replace managed labels on an issue while keeping user labels."""
        managed_prefixes = ("type:", "kind:", "session:", "date:", "topic:",
                            "agent:", "status:", "memory-status:")
        managed_exact = {"status:active", "status:closed",
                         "memory-status:active", "memory-status:stale"}

        issue = await self.get_issue(issue_number)
        current = extract_label_names(issue.get("labels", []))

        def is_managed(lbl: str) -> bool:
            if lbl in managed_exact:
                return True
            return any(lbl.startswith(p) for p in managed_prefixes)

        unmanaged = [lbl for lbl in current if not is_managed(lbl)]
        merged = list(dict.fromkeys(unmanaged + desired))  # dedupe, preserve order
        await self.update_issue(issue_number, labels=merged)

    async def list_labels(
        self,
        *,
        page: int = 1,
        per_page: int = 100,
    ) -> list[dict]:
        """``GET /repos/{repo}/labels``"""
        result = await self._request(
            "GET",
            self._repo_path(f"labels?page={page}&per_page={per_page}"),
        )
        return result if isinstance(result, list) else []

    # -- Comments list ------------------------------------------------------

    async def list_comments(
        self,
        issue_number: int,
        *,
        page: int = 1,
        per_page: int = 100,
        sort: str | None = None,
        direction: str | None = None,
        since: str | None = None,
        threaded: bool | None = None,
    ) -> list[dict]:
        """``GET /repos/{repo}/issues/{n}/comments``"""
        from urllib.parse import urlencode
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if sort:
            params["sort"] = sort
        if direction:
            params["direction"] = direction
        if since:
            params["since"] = since
        if threaded:
            params["threaded"] = "true"
        result = await self._request(
            "GET",
            self._repo_path(f"issues/{issue_number}/comments?{urlencode(params)}"),
        )
        return result if isinstance(result, list) else []

    # -- Repos --------------------------------------------------------------

    async def list_user_repos(self) -> list[dict]:
        """``GET /user/repos``"""
        result = await self._request("GET", "user/repos")
        return result if isinstance(result, list) else []

    async def create_user_repo(
        self,
        name: str,
        *,
        description: str | None = None,
        private: bool = True,
        auto_init: bool = False,
    ) -> dict:
        """``POST /user/repos``"""
        body: dict[str, Any] = {
            "name": name,
            "private": private,
            "auto_init": auto_init,
        }
        if description:
            body["description"] = description
        return await self._request("POST", "user/repos", body=body)

    async def create_org_repo(
        self,
        org: str,
        name: str,
        *,
        description: str | None = None,
        private: bool = True,
        auto_init: bool = False,
        has_issues: bool | None = None,
        has_wiki: bool | None = None,
    ) -> dict:
        """``POST /orgs/{org}/repos``"""
        body: dict[str, Any] = {
            "name": name,
            "private": private,
            "auto_init": auto_init,
        }
        if description:
            body["description"] = description
        if has_issues is not None:
            body["has_issues"] = has_issues
        if has_wiki is not None:
            body["has_wiki"] = has_wiki
        return await self._request(
            "POST",
            f"orgs/{quote(org, safe='')}/repos",
            body=body,
        )

    async def get_repo(self, owner: str, repo: str) -> dict | None:
        """``GET /repos/{owner}/{repo}``"""
        return await self._request(
            "GET",
            f"repos/{quote(owner, safe='')}/{quote(repo, safe='')}",
            allow_not_found=True,
        )

    async def transfer_repo(
        self,
        owner: str,
        repo: str,
        new_owner: str,
        new_repo_name: str | None = None,
    ) -> dict:
        """``POST /repos/{owner}/{repo}/transfer``"""
        body: dict[str, Any] = {"new_owner": new_owner}
        if new_repo_name:
            body["new_repo_name"] = new_repo_name
        return await self._request(
            "POST",
            f"repos/{quote(owner, safe='')}/{quote(repo, safe='')}/transfer",
            body=body,
        )

    async def rename_repo(self, owner: str, repo: str, new_name: str) -> dict:
        """``PATCH /repos/{owner}/{repo}``"""
        return await self._request(
            "PATCH",
            f"repos/{quote(owner, safe='')}/{quote(repo, safe='')}",
            body={"name": new_name},
        )

    # -- User / Orgs --------------------------------------------------------

    async def get_current_user(self) -> dict:
        """``GET /user``"""
        return await self._request("GET", "user")

    async def list_user_orgs(self) -> list[dict]:
        """``GET /user/orgs``"""
        result = await self._request("GET", "user/orgs")
        return result if isinstance(result, list) else []

    async def create_user_org(
        self,
        login: str,
        *,
        name: str | None = None,
        default_repository_permission: str | None = None,
    ) -> dict:
        """``POST /user/orgs``"""
        body: dict[str, Any] = {"login": login}
        if name:
            body["name"] = name
        if default_repository_permission:
            body["default_repository_permission"] = default_repository_permission
        return await self._request("POST", "user/orgs", body=body)

    async def get_org(self, org: str) -> dict:
        """``GET /orgs/{org}``"""
        return await self._request("GET", f"orgs/{quote(org, safe='')}")

    async def list_org_members(
        self, org: str, role: str | None = None,
    ) -> list[dict]:
        """``GET /orgs/{org}/members``"""
        path = f"orgs/{quote(org, safe='')}/members"
        if role:
            path += f"?role={role}"
        result = await self._request("GET", path)
        return result if isinstance(result, list) else []

    async def get_org_membership(self, org: str, username: str) -> dict | None:
        """``GET /orgs/{org}/memberships/{username}``"""
        return await self._request(
            "GET",
            f"orgs/{quote(org, safe='')}/memberships/{quote(username, safe='')}",
            allow_not_found=True,
        )

    async def remove_org_member(self, org: str, username: str) -> None:
        """``DELETE /orgs/{org}/members/{username}``"""
        await self._request(
            "DELETE",
            f"orgs/{quote(org, safe='')}/members/{quote(username, safe='')}",
        )

    async def remove_org_membership(self, org: str, username: str) -> None:
        """``DELETE /orgs/{org}/memberships/{username}``"""
        await self._request(
            "DELETE",
            f"orgs/{quote(org, safe='')}/memberships/{quote(username, safe='')}",
        )

    async def list_org_outside_collaborators(self, org: str) -> list[dict]:
        """``GET /orgs/{org}/outside_collaborators``"""
        result = await self._request(
            "GET",
            f"orgs/{quote(org, safe='')}/outside_collaborators",
        )
        return result if isinstance(result, list) else []

    # -- Teams --------------------------------------------------------------

    async def list_org_teams(self, org: str) -> list[dict]:
        """``GET /orgs/{org}/teams``"""
        result = await self._request("GET", f"orgs/{quote(org, safe='')}/teams")
        return result if isinstance(result, list) else []

    async def get_team(self, org: str, team_slug: str) -> dict:
        """``GET /orgs/{org}/teams/{team_slug}``"""
        return await self._request(
            "GET",
            f"orgs/{quote(org, safe='')}/teams/{quote(team_slug, safe='')}",
        )

    async def create_org_team(
        self,
        org: str,
        name: str,
        *,
        description: str | None = None,
        privacy: str = "closed",
    ) -> dict:
        """``POST /orgs/{org}/teams``"""
        body: dict[str, Any] = {"name": name, "privacy": privacy}
        if description:
            body["description"] = description
        return await self._request(
            "POST",
            f"orgs/{quote(org, safe='')}/teams",
            body=body,
        )

    async def update_team(
        self,
        org: str,
        team_slug: str,
        *,
        name: str | None = None,
        description: str | None = None,
        privacy: str | None = None,
    ) -> dict:
        """``PATCH /orgs/{org}/teams/{team_slug}``"""
        body: dict[str, Any] = {}
        if name:
            body["name"] = name
        if description:
            body["description"] = description
        if privacy:
            body["privacy"] = privacy
        return await self._request(
            "PATCH",
            f"orgs/{quote(org, safe='')}/teams/{quote(team_slug, safe='')}",
            body=body,
        )

    async def delete_team(self, org: str, team_slug: str) -> None:
        """``DELETE /orgs/{org}/teams/{team_slug}``"""
        await self._request(
            "DELETE",
            f"orgs/{quote(org, safe='')}/teams/{quote(team_slug, safe='')}",
        )

    async def list_team_members(self, org: str, team_slug: str) -> list[dict]:
        """``GET /orgs/{org}/teams/{team_slug}/members``"""
        result = await self._request(
            "GET",
            f"orgs/{quote(org, safe='')}/teams/{quote(team_slug, safe='')}/members",
        )
        return result if isinstance(result, list) else []

    async def set_team_membership(
        self,
        org: str,
        team_slug: str,
        username: str,
        role: str,
    ) -> dict:
        """``PUT /orgs/{org}/teams/{team_slug}/memberships/{username}``"""
        return await self._request(
            "PUT",
            f"orgs/{quote(org, safe='')}/teams/{quote(team_slug, safe='')}"
            f"/memberships/{quote(username, safe='')}",
            body={"role": role},
        )

    async def remove_team_membership(
        self, org: str, team_slug: str, username: str,
    ) -> None:
        """``DELETE /orgs/{org}/teams/{team_slug}/memberships/{username}``"""
        await self._request(
            "DELETE",
            f"orgs/{quote(org, safe='')}/teams/{quote(team_slug, safe='')}"
            f"/memberships/{quote(username, safe='')}",
        )

    async def list_team_repos(self, org: str, team_slug: str) -> list[dict]:
        """``GET /orgs/{org}/teams/{team_slug}/repos``"""
        result = await self._request(
            "GET",
            f"orgs/{quote(org, safe='')}/teams/{quote(team_slug, safe='')}/repos",
        )
        return result if isinstance(result, list) else []

    async def set_team_repo_access(
        self,
        org: str,
        team_slug: str,
        owner: str,
        repo: str,
        permission: str,
    ) -> None:
        """``PUT /orgs/{org}/teams/{team_slug}/repos/{owner}/{repo}``"""
        await self._request(
            "PUT",
            f"orgs/{quote(org, safe='')}/teams/{quote(team_slug, safe='')}"
            f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}",
            body={"permission": permission},
        )

    async def remove_team_repo_access(
        self, org: str, team_slug: str, owner: str, repo: str,
    ) -> None:
        """``DELETE /orgs/{org}/teams/{team_slug}/repos/{owner}/{repo}``"""
        await self._request(
            "DELETE",
            f"orgs/{quote(org, safe='')}/teams/{quote(team_slug, safe='')}"
            f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}",
        )

    # -- Repo collaborators -------------------------------------------------

    async def list_repo_collaborators(
        self, owner: str, repo: str,
    ) -> list[dict]:
        """``GET /repos/{owner}/{repo}/collaborators``"""
        result = await self._request(
            "GET",
            f"repos/{quote(owner, safe='')}/{quote(repo, safe='')}/collaborators",
        )
        return result if isinstance(result, list) else []

    async def list_repo_invitations(
        self, owner: str, repo: str,
    ) -> list[dict]:
        """``GET /repos/{owner}/{repo}/invitations``"""
        result = await self._request(
            "GET",
            f"repos/{quote(owner, safe='')}/{quote(repo, safe='')}/invitations",
        )
        return result if isinstance(result, list) else []

    async def set_repo_collaborator(
        self,
        owner: str,
        repo: str,
        username: str,
        permission: str,
    ) -> dict | None:
        """``PUT /repos/{owner}/{repo}/collaborators/{username}``"""
        return await self._request(
            "PUT",
            f"repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
            f"/collaborators/{quote(username, safe='')}",
            body={"permission": permission},
        )

    async def remove_repo_collaborator(
        self, owner: str, repo: str, username: str,
    ) -> None:
        """``DELETE /repos/{owner}/{repo}/collaborators/{username}``"""
        await self._request(
            "DELETE",
            f"repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
            f"/collaborators/{quote(username, safe='')}",
        )

    # -- User invitations (self-side) ---------------------------------------

    async def list_user_repo_invitations(self) -> list[dict]:
        """``GET /user/repository_invitations``"""
        result = await self._request("GET", "user/repository_invitations")
        return result if isinstance(result, list) else []

    async def accept_user_repo_invitation(self, invitation_id: int) -> None:
        """``PATCH /user/repository_invitations/{id}``"""
        await self._request(
            "PATCH", f"user/repository_invitations/{invitation_id}",
        )

    async def decline_user_repo_invitation(self, invitation_id: int) -> None:
        """``DELETE /user/repository_invitations/{id}``"""
        await self._request(
            "DELETE", f"user/repository_invitations/{invitation_id}",
        )

    async def list_user_org_invitations(self) -> list[dict]:
        """``GET /user/organization_invitations``"""
        result = await self._request("GET", "user/organization_invitations")
        return result if isinstance(result, list) else []

    async def accept_user_org_invitation(self, invitation_id: int) -> None:
        """``PATCH /user/organization_invitations/{id}``"""
        await self._request(
            "PATCH", f"user/organization_invitations/{invitation_id}",
        )

    async def decline_user_org_invitation(self, invitation_id: int) -> None:
        """``DELETE /user/organization_invitations/{id}``"""
        await self._request(
            "DELETE", f"user/organization_invitations/{invitation_id}",
        )

    # -- Org invitations (admin-side) ---------------------------------------

    async def list_org_invitations(self, org: str) -> list[dict]:
        """``GET /orgs/{org}/invitations``"""
        result = await self._request(
            "GET", f"orgs/{quote(org, safe='')}/invitations",
        )
        return result if isinstance(result, list) else []

    async def create_org_invitation(
        self,
        org: str,
        invitee_login: str,
        *,
        role: str = "member",
        team_ids: list[int] | None = None,
        expires_in_days: int | None = None,
    ) -> dict:
        """``POST /orgs/{org}/invitations``"""
        body: dict[str, Any] = {
            "invitee_login": invitee_login,
            "role": role,
        }
        if team_ids:
            body["team_ids"] = team_ids
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        return await self._request(
            "POST",
            f"orgs/{quote(org, safe='')}/invitations",
            body=body,
        )

    async def revoke_org_invitation(
        self, org: str, invitation_id: int,
    ) -> None:
        """``DELETE /orgs/{org}/invitations/{id}``"""
        await self._request(
            "DELETE",
            f"orgs/{quote(org, safe='')}/invitations/{invitation_id}",
        )
