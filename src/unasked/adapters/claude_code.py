"""Claude Code transcript adapter for unasked.

Reads a Claude Code session transcript (``~/.claude/projects/**/<id>.jsonl``)
and converts it into the unasked IR: a ``Run`` with one ``Decision`` per
``tool_use`` content block emitted by the assistant.

Transcript format (one JSON line per record):
  - ``type == "user"``      — user message; may contain ``tool_result`` blocks
  - ``type == "assistant"`` — assistant message; may contain ``tool_use`` blocks
  - Other types (queue-operation, attachment, …) — skipped

Each record:
  {
    "type": "user" | "assistant" | ...,
    "timestamp": "2026-06-04T22:44:07.123Z",
    "sessionId": "f367ec06-...",
    "message": {
      "role": "user" | "assistant",
      "content": str | [
        {"type": "tool_use",    "id": "toolu_...", "name": "...", "input": {...}},
        {"type": "tool_result", "tool_use_id": "toolu_...", "is_error": bool, ...},
        {"type": "text", "text": "..."},
        ...
      ]
    }
  }

Parsing rules
-------------
- ``task_text``: first user message whose content is non-empty text (str or
  joined text blocks). Cap at 500 chars. Tool-result-only messages skipped.
- ``Decision`` per ``tool_use`` block (assistant messages only). Step index is
  0-based over the whole session in order of occurrence.
- ``is_error``: resolved by matching the immediately-following user message's
  ``tool_result`` block whose ``tool_use_id`` == this tool_use ``id``.
  Default False when no matching result found.
- ``parent_step_index``: sequential — each decision's parent is the previous
  one (index - 1), None for the first. Simple causal chain; no uuid graph.
- ``started_at``: timestamp of the first record in the file.
- ``run_id``: the session UUID (filename stem).
- Malformed / unparseable lines are silently skipped (mirror live_normalizer).

Path resolution for ``load_session(source)``
--------------------------------------------
  - If ``source`` is an existing file path → use it directly.
  - Otherwise treat as a session_id and glob
    ``~/.claude/projects/**/<session_id>.jsonl``, take first match.
  - Raises ``FileNotFoundError`` when nothing is found.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unasked.ir import Decision, Run
from unasked.ledger import save_run
from unasked.redact import summarize_args

# Default projects root under which CC stores transcripts.
_PROJECTS_ROOT = Path.home() / ".claude" / "projects"


# ── Path resolution ───────────────────────────────────────────────────────────


def _resolve_transcript(source: str) -> Path:
    """Return the Path to the transcript JSONL for ``source``.

    If ``source`` looks like an existing file, return it directly.
    Otherwise treat it as a session_id and search the CC projects tree.
    """
    p = Path(source)
    if p.exists():
        return p

    # Glob for the session file under ~/.claude/projects/.
    matches = list(_PROJECTS_ROOT.glob(f"**/{source}.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"No transcript found for session {source!r} under {_PROJECTS_ROOT}"
        )
    return matches[0]


# ── Text extraction helper ────────────────────────────────────────────────────


def _extract_text_content(content: Any) -> str | None:
    """Pull plain text from a message content value (str or content-array)."""
    if isinstance(content, str):
        stripped = content.strip()
        return stripped[:500] if stripped else None
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        joined = " ".join(parts)
        return joined[:500] if joined else None
    return None


def _is_pure_tool_result(content: Any) -> bool:
    """True when content is a list composed entirely of tool_result blocks."""
    if not isinstance(content, list) or not content:
        return False
    return all(
        isinstance(b, dict) and b.get("type") == "tool_result"
        for b in content
    )


# ── Main parser ───────────────────────────────────────────────────────────────


def load_session(source: str) -> Run:
    """Parse a Claude Code transcript and return an unasked ``Run``.

    Parameters
    ----------
    source:
        Either a file path to a ``.jsonl`` transcript, or a bare session UUID
        (the filename stem). Session IDs are resolved by globbing
        ``~/.claude/projects/**/<session_id>.jsonl``; first match wins.

    Returns
    -------
    Run
        IR with one Decision per ``tool_use`` block, in occurrence order.
        Provenance fields (provenance, scope_drift, why, feedback) are all
        None — populated by later features.

    Raises
    ------
    FileNotFoundError
        When the file does not exist and no matching session is found.
    """
    transcript = _resolve_transcript(source)
    session_id = transcript.stem

    raw_lines = transcript.read_text(encoding="utf-8").splitlines()

    # First pass: collect parsed records, silently skipping malformed lines.
    records: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # soft-fail

    started_at: str | None = records[0].get("timestamp") if records else None

    # Index tool_results by tool_use_id for is_error lookup.
    # They appear in user messages following the assistant's tool_use blocks.
    tool_results: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec.get("type") != "user":
            continue
        content = rec.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tid = block.get("tool_use_id")
                if tid:
                    tool_results[tid] = block

    # Extract task_text from the first non-empty, non-tool-result user message.
    task_text: str | None = None
    for rec in records:
        if rec.get("type") != "user":
            continue
        content = rec.get("message", {}).get("content")
        if _is_pure_tool_result(content):
            continue
        text = _extract_text_content(content)
        if text:
            task_text = text
            break

    # Walk assistant messages and collect tool_use blocks as Decisions.
    decisions: list[Decision] = []
    for rec in records:
        if rec.get("type") != "assistant":
            continue
        ts: str | None = rec.get("timestamp")
        content = rec.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue

            tool_use_id = str(block.get("id") or "")
            tool_name = str(block.get("name") or "unknown")
            tool_input: dict[str, Any] = block.get("input") or {}

            args_summary = summarize_args(tool_name, tool_input)

            # Resolve is_error from the matching tool_result (default False).
            result_block = tool_results.get(tool_use_id, {})
            is_error = bool(result_block.get("is_error", False))

            step_index = len(decisions)
            parent = step_index - 1 if step_index > 0 else None

            decisions.append(
                Decision(
                    step_index=step_index,
                    ts=ts,
                    tool_name=tool_name,
                    tool_args_summary=args_summary,
                    is_error=is_error,
                    parent_step_index=parent,
                )
            )

    return Run(
        run_id=session_id,
        source="claude_code",
        task_text=task_text,
        started_at=started_at,
        decisions=decisions,
    )


def ingest_session(source: str, db_path: str | None = None) -> Run:
    """Load a transcript and persist it to the ledger in one call.

    ``load_session`` is pure (returns Run, no side effects). This thin wrapper
    adds the ``save_run`` call for callers that want load + persist together.

    Parameters
    ----------
    source:
        File path or session UUID — passed through to ``load_session``.
    db_path:
        Override ledger DB path. Falls back to ``UNASKED_LEDGER_DB`` env or
        ``~/.kairos/ledger.db``.

    Returns
    -------
    Run
        The same ``Run`` that was persisted.
    """
    run = load_session(source)
    save_run(run, path=db_path)
    return run
