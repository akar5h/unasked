"""Secret redaction and arg summarisation for unasked.

Vendored from kairos_hook.py's redaction approach — kept stdlib-only so the
whole package stays zero-dep.  The pattern list is intentionally identical to
the hook so any value the hook already scrubbed before writing to the spool
will also be caught here on a second pass.

Public API
----------
summarize_args(tool_name, tool_input) -> str
    Short one-line redacted summary of a tool call's inputs.
    Never includes full file contents or raw arg blobs.

redact(text) -> str
    Scrub secret-shaped substrings from an arbitrary string.
"""

from __future__ import annotations

import re
from typing import Any

# ── Secret-pattern list ───────────────────────────────────────────────────────
# Kept identical to kairos_hook.py so both layers redact the same things.

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}", re.ASCII),
    re.compile(r"ghp_[A-Za-z0-9]{36}", re.ASCII),
    re.compile(r"AKIA[0-9A-Z]{16}", re.ASCII),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
    re.compile(
        r"-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----",
        re.DOTALL,
    ),
    # JWT
    re.compile(
        r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+",
        re.ASCII,
    ),
    # DB URLs with embedded credentials
    re.compile(
        r"(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^:\s]+:[^@\s]+@\S+",
        re.IGNORECASE,
    ),
    # Assignment-style secrets: KEY=value or KEY: value
    re.compile(
        r"(?i)(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key)"
        r"(?:\s*[=:]\s*)\S+",
    ),
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}", re.ASCII),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}", re.ASCII),
    # Long hex strings (>=32 chars) — likely keys or hashes
    re.compile(r"\b[0-9a-fA-F]{32,}\b"),
    # Long base64-like blobs (>=40 chars, no spaces) — exclude common paths
    re.compile(r"(?<![/\w])[A-Za-z0-9+/]{40,}={0,2}(?![/\w])"),
]

_REDACTED = "[REDACTED]"
_MAX_LEN = 120  # max chars for any single value in a summary


def redact(text: str) -> str:
    """Scrub secret-shaped substrings from ``text``, return cleaned copy."""
    result = text
    for pat in _SECRET_PATTERNS:
        result = pat.sub(_REDACTED, result)
    return result


def _trunc(s: str, n: int = _MAX_LEN) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _clean(value: Any) -> str:
    """Stringify, truncate, and redact a single value."""
    return redact(_trunc(str(value)))


# ── Per-tool summarisers ──────────────────────────────────────────────────────


def summarize_args(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Return a SHORT redacted one-line summary of a tool call's inputs.

    Surfaces only the signal needed for provenance / scope analysis:
      - Tool name implicit (caller usually prepends it)
      - Key identifying arg(s): file path, command first line, URL, query, etc.
      - Long strings truncated, secret-shaped values replaced with [REDACTED]
      - Never includes full file contents or raw arg blobs

    Examples
    --------
    >>> summarize_args("Edit", {"file_path": "config/db.yaml", ...})
    'file_path=config/db.yaml'
    >>> summarize_args("Bash", {"command": "git push origin main"})
    'git push origin main'
    >>> summarize_args("Read", {"file_path": ".env"})
    '.env'
    """
    t = tool_name

    if not tool_input:
        return "(no args)"

    # ── Tool-specific extractors ──────────────────────────────────────────────

    if t == "Bash":
        cmd = str(tool_input.get("command", ""))
        first_line = cmd.split("\n")[0]
        return redact(_trunc(first_line))

    if t == "Read":
        return _clean(tool_input.get("file_path", ""))

    if t in ("Write", "Edit"):
        fp = _clean(tool_input.get("file_path", ""))
        return f"file_path={fp}"

    if t == "Glob":
        pattern = _clean(tool_input.get("pattern", ""))
        path = tool_input.get("path", "")
        return f"{pattern}" + (f" in {_clean(path)}" if path else "")

    if t == "WebFetch":
        return _clean(tool_input.get("url", ""))

    if t == "WebSearch":
        return _clean(tool_input.get("query", ""))

    if t == "Agent":
        subtype = tool_input.get("subagent_type", "")
        desc = tool_input.get("description", "")
        return _clean(f"{subtype}: {desc}" if subtype else str(desc))

    if t == "SendMessage":
        to = tool_input.get("to", "")
        summary = tool_input.get("summary", "")
        return _clean(f"to={to} {summary}" if to else str(summary))

    if t == "Skill":
        skill = tool_input.get("skill", "")
        args = tool_input.get("args", "")
        return _clean(f"{skill}" + (f" {args}" if args else ""))

    if t == "TaskCreate":
        return _clean(tool_input.get("subject", tool_input.get("description", "")))

    if t == "TaskUpdate":
        task_id = tool_input.get("taskId", "")
        status = tool_input.get("status", "")
        return f"taskId={task_id} status={status}"

    if t == "ToolSearch":
        return _clean(tool_input.get("query", ""))

    # Generic fallback: key=value pairs for first few keys, no blob values
    parts = []
    for k, v in list(tool_input.items())[:4]:
        v_str = str(v)
        # Skip large blobs (file content etc.) silently
        if len(v_str) > 200:
            parts.append(f"{k}=<blob>")
        else:
            parts.append(f"{k}={redact(_trunc(v_str, 80))}")
    return " ".join(parts) if parts else "(no args)"
