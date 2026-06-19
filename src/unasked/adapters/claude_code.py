"""Claude Code adapter for unasked.

Reads one session's spool JSONL (written by kairos_hook.py) and converts
it into the unasked IR: a ``Run`` with one ``Decision`` per PostToolUse /
PostToolUseFailure event.

Spool format recap (one JSON line per event):
  {
    "session_id": "...",
    "event_name": "SessionStart" | "PostToolUse" | "PostToolUseFailure" | "SessionEnd",
    "tool_name": str | null,          # present on PostToolUse / PostToolUseFailure
    "tool_input_redacted": dict | null,  # already redacted by kairos_hook.py
    "is_error": bool | null,
    "occurred_at": "2026-06-18T..."    # ISO-8601
    "payload_redacted": {...}           # full redacted payload; may carry "transcript_path"
  }

Secret redaction is handled upstream by kairos_hook.py.  We vendor the same
patterns here only for the optional ``redact_args_summary()`` helper so
callers that build a summary from raw args can also sanitise on the way in.

Spool path resolution (in priority order):
  1. Explicit ``spool_dir`` argument to ``read_session()``.
  2. ``KAIROS_SPOOL_DIR`` environment variable.
  3. Default: ``~/.kairos/spool``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from unasked.ir import Decision, Run

# ── Secret-redaction patterns ─────────────────────────────────────────────────
# Same patterns as kairos_hook.py — vendored stdlib-only copy so this module
# stays zero-dep and can be called independently of kairos_hook.py.

_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}", re.ASCII),
    re.compile(r"ghp_[A-Za-z0-9]{36}", re.ASCII),
    re.compile(r"AKIA[0-9A-Z]{16}", re.ASCII),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
    re.compile(
        r"-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----",
        re.DOTALL,
    ),
    re.compile(
        r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+",
        re.ASCII,
    ),
    re.compile(
        r"(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^:\s]+:[^@\s]+@\S+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i)(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key)"
        r"(?:\s*[=:]\s*)\S+",
    ),
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}", re.ASCII),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}", re.ASCII),
]

_REDACTED = "[REDACTED]"


def redact_args_summary(text: str) -> str:
    """Scrub secret-shaped strings from a short args summary string.

    Callers that compose a summary from raw (non-hook) tool args should run
    this before storing on a ``Decision``.  Summaries built from
    ``tool_input_redacted`` (already scrubbed by kairos_hook.py) do not
    strictly need this, but a second pass is harmless.
    """
    result = text
    for pat in _SECRET_PATTERNS:
        result = pat.sub(_REDACTED, result)
    return result


# ── Args summariser ───────────────────────────────────────────────────────────

_MAX_VALUE_LEN = 120  # truncate individual values to keep summaries readable


def _truncate(s: str, max_len: int = _MAX_VALUE_LEN) -> str:
    return s if len(s) <= max_len else s[:max_len] + "…"


def _summarise_args(tool_name: str, args: dict[str, Any]) -> str:
    """Produce a short human-readable summary of tool args.

    Rules per tool type — we surface the most meaningful key(s) only:
      Bash        → command (first line, truncated)
      Read        → file_path
      Write/Edit  → file_path
      Glob        → pattern [in path]
      WebFetch    → url
      WebSearch   → query
      Agent       → subagent_type description
      SendMessage → to: summary
      Skill       → skill [args]
      TaskCreate  → subject
      TaskUpdate  → taskId status
      Fallback    → key=value pairs, truncated
    """
    if not args:
        return "(no args)"

    t = tool_name

    if t == "Bash":
        cmd = str(args.get("command", ""))
        first_line = cmd.split("\n")[0]
        return _truncate(first_line)

    if t == "Read":
        return _truncate(str(args.get("file_path", "")))

    if t in ("Write", "Edit"):
        return _truncate(str(args.get("file_path", "")))

    if t == "Glob":
        pattern = str(args.get("pattern", ""))
        path = args.get("path", "")
        return f"{_truncate(pattern)}" + (f" in {_truncate(str(path))}" if path else "")

    if t == "WebFetch":
        return _truncate(str(args.get("url", "")))

    if t == "WebSearch":
        return _truncate(str(args.get("query", "")))

    if t == "Agent":
        subtype = args.get("subagent_type", "")
        desc = args.get("description", "")
        return _truncate(f"{subtype}: {desc}" if subtype else str(desc))

    if t == "SendMessage":
        to = args.get("to", "")
        summary = args.get("summary", "")
        return _truncate(f"to={to} {summary}" if to else str(summary))

    if t == "Skill":
        skill = args.get("skill", "")
        skill_args = args.get("args", "")
        return _truncate(f"{skill}" + (f" {skill_args}" if skill_args else ""))

    if t == "TaskCreate":
        return _truncate(str(args.get("subject", args.get("description", ""))))

    if t == "TaskUpdate":
        task_id = args.get("taskId", "")
        status = args.get("status", "")
        return f"taskId={task_id} status={status}"

    # Generic fallback: key=value pairs, first few keys.
    parts = []
    for k, v in list(args.items())[:4]:
        v_str = _truncate(str(v), 60)
        parts.append(f"{k}={v_str}")
    return " ".join(parts)


# ── Spool path resolution ─────────────────────────────────────────────────────


def _default_spool_dir() -> Path:
    override = os.environ.get("KAIROS_SPOOL_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".kairos" / "spool"


def _spool_path(session_id: str, spool_dir: str | Path | None) -> Path:
    root = Path(spool_dir) if spool_dir is not None else _default_spool_dir()
    return root / f"{session_id}.jsonl"


# ── Task-text extraction ──────────────────────────────────────────────────────


def _extract_task_text(session_start_payload: dict[str, Any]) -> str | None:
    """Best-effort: pull a short task description from SessionStart payload.

    The payload_redacted may carry a ``transcript_path`` pointing to the CC
    JSONL transcript.  The first ``user`` message in that transcript is the
    human's task.  We return only the first 500 chars; full text is rarely
    useful for a receipt.

    Returns None if transcript is absent, unreadable, or has no user message.
    """
    transcript_path = session_start_payload.get("transcript_path")
    if not transcript_path:
        return None
    tp = Path(str(transcript_path))
    if not tp.exists():
        return None
    try:
        for line in tp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") == "user":
                msg = entry.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content[:500]
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text.strip():
                                return text[:500]
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────


class SpoolNotFoundError(FileNotFoundError):
    """Raised when the spool file for a session_id does not exist."""


def read_session(
    session_id: str,
    spool_dir: str | Path | None = None,
) -> Run:
    """Read a Claude Code spool JSONL and return an unasked ``Run``.

    Parameters
    ----------
    session_id:
        The Claude Code session UUID (matches the spool filename stem).
    spool_dir:
        Override the spool directory.  Falls back to ``KAIROS_SPOOL_DIR``
        env var, then ``~/.kairos/spool``.

    Raises
    ------
    SpoolNotFoundError
        When no spool file exists for that session_id.

    Returns
    -------
    Run
        A ``Run`` with one ``Decision`` per PostToolUse / PostToolUseFailure
        event, ordered by occurrence (step_index 0-based).
        ``task_text`` is populated from the CC transcript when available,
        otherwise None.  ``started_at`` comes from the SessionStart event's
        ``occurred_at`` field.
    """
    spool_file = _spool_path(session_id, spool_dir)
    if not spool_file.exists():
        raise SpoolNotFoundError(
            f"No spool file for session {session_id!r}: {spool_file}"
        )

    raw_lines = spool_file.read_text(encoding="utf-8").splitlines()
    events: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # malformed line — skip, don't fail the whole run

    started_at: str | None = None
    task_text: str | None = None
    decisions: list[Decision] = []

    for event in events:
        name = event.get("event_name", "")

        if name == "SessionStart":
            started_at = event.get("occurred_at")
            payload = event.get("payload_redacted") or {}
            task_text = _extract_task_text(payload)

        elif name in ("PostToolUse", "PostToolUseFailure"):
            tool_name = str(event.get("tool_name") or "unknown")
            args: dict[str, Any] = event.get("tool_input_redacted") or {}
            # Build the summary from already-redacted args; apply one more pass.
            raw_summary = _summarise_args(tool_name, args)
            args_summary = redact_args_summary(raw_summary)

            is_error_raw = event.get("is_error")
            # PostToolUseFailure implies error; is_error field may be null/False.
            is_error = bool(is_error_raw) or name == "PostToolUseFailure"

            decisions.append(
                Decision(
                    step_index=len(decisions),
                    ts=event.get("occurred_at"),
                    tool_name=tool_name,
                    tool_args_summary=args_summary,
                    is_error=is_error,
                )
            )
        # SessionEnd: no IR equivalent in F2; ignored.

    return Run(
        run_id=session_id,
        source="claude_code",
        task_text=task_text,
        started_at=started_at,
        decisions=decisions,
    )
