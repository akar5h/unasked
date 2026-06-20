"""Claude Code adapters for unasked.

Two source paths:

PRIMARY — native transcript reader (``load_session``)
    Reads ``~/.claude/projects/**/<session_id>.jsonl`` directly.  Works on
    any existing CC session with ZERO setup — no hook required.

SECONDARY — spool reader (``read_session_from_spool``)
    Reads ``~/.kairos/spool/<session_id>.jsonl`` written by kairos_hook.py.
    Opt-in capture-time path; only available after the hook is installed.
    Already-redacted by the hook; a second redaction pass is applied anyway.

``ingest_session(source, db_path, source_kind)`` is the persist wrapper.
Set ``source_kind="spool"`` to use the spool path; default is ``"transcript"``.

──────────────────────────────────────────────────────────────────────────────
Native transcript format (one JSON line per record):
  - ``type == "user"``      — user message; may contain ``tool_result`` blocks
  - ``type == "assistant"`` — assistant message; may contain ``tool_use`` blocks
  - Other types (queue-operation, attachment, isMeta, …) — skipped

Each record:
  {
    "type": "user" | "assistant" | ...,
    "uuid": "...",
    "parentUuid": "...",
    "isMeta": bool | absent,          # metadata lines — skipped for task_text
    "timestamp": "2026-06-04T22:44:07.123Z",
    "sessionId": "f367ec06-...",
    "message": {
      "role": "user" | "assistant",
      "content": str | [
        {"type": "tool_use",    "id": "toolu_...", "name": "...", "input": {...}},
        {"type": "tool_result", "tool_use_id": "toolu_...", "content": ...,
         "is_error": bool},           # is_error often absent → default False
        {"type": "text", "text": "..."},
        ...
      ]
    }
  }

Parsing rules
-------------
- ``task_text``: first ``user`` message that is NOT a tool_result-only message
  AND NOT an isMeta line. Joined text blocks, capped at 500 chars.
- ``Decision`` per ``tool_use`` block (assistant messages only). Step index is
  0-based over the whole session in order of occurrence.
- ``is_error``: resolved by matching ``tool_result.tool_use_id == tool_use.id``.
  Default False when absent.
- ``parent_step_index``: sequential (index - 1), None for the first decision.
- ``started_at``: timestamp of the first record in the file.
- ``run_id``: filename stem.
- Malformed / unparseable lines are silently skipped (soft-fail).

──────────────────────────────────────────────────────────────────────────────
Spool format (written by kairos_hook.py, one JSON line per event):
  {
    "session_id": "...",
    "event_name": "SessionStart" | "PostToolUse" | "PostToolUseFailure" | "SessionEnd",
    "tool_name": str | null,
    "tool_input_redacted": dict | null,   # already redacted by the hook
    "is_error": bool | null,
    "occurred_at": "2026-06-18T...",
    "payload_redacted": {...}             # may carry "transcript_path"
  }

Spool path resolution (``read_session_from_spool``):
  1. Explicit ``spool_dir`` argument.
  2. ``KAIROS_SPOOL_DIR`` env var.
  3. Default: ``~/.kairos/spool``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from unasked.entities import extract_result_entities, extract_targets
from unasked.ir import Decision, Run
from unasked.ledger import save_run
from unasked.redact import redact, summarize_args

# Default roots.
_PROJECTS_ROOT = Path.home() / ".claude" / "projects"
_DEFAULT_SPOOL_ROOT = Path.home() / ".kairos" / "spool"


# ── Path resolution ───────────────────────────────────────────────────────────


def _resolve_transcript(source: str) -> Path:
    """Return the Path to the transcript JSONL for ``source``.

    Resolution order:
    1. Existing file path — returned as-is.
    2. Exact stem match: ``~/.claude/projects/**/<source>.jsonl``.
    3. Prefix match (F4.4): ``~/.claude/projects/**/<source>*.jsonl``.
       When multiple files match, the one with the most recent mtime wins.

    Raises FileNotFoundError when no match is found at any step.
    """
    p = Path(source)
    if p.exists():
        return p

    # Exact stem match (full UUID).
    exact = list(_PROJECTS_ROOT.glob(f"**/{source}.jsonl"))
    if exact:
        return exact[0]

    # Prefix match — e.g. "986bc712" resolves "986bc712-4d6a-...jsonl".
    prefix_matches = list(_PROJECTS_ROOT.glob(f"**/{source}*.jsonl"))
    if not prefix_matches:
        raise FileNotFoundError(
            f"No transcript found for session {source!r} under {_PROJECTS_ROOT}"
        )
    # Newest by mtime wins on ties.
    return max(prefix_matches, key=lambda p: p.stat().st_mtime)


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


# Tags injected by CC into user messages that signal a slash-command artifact
# or hook/system injection rather than a real human instruction.
_COMMAND_ARTIFACT_TAGS = (
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-",
)


def _is_command_artifact(text: str) -> bool:
    """True when text is a slash-command artifact or hook/system injection.

    These messages are injected by Claude Code itself and do not represent
    real human task instructions.  Patterns detected:
    - Text starting with a slash command: ``/clear``, ``/model``, etc.
    - Text containing CC command XML tags: <command-name>, <command-message>,
      <command-args>, or any <local-command-*> tag.
    """
    stripped = text.strip()
    if not stripped:
        return True  # empty — not a real instruction
    # Slash-command prefix: /word (e.g. /clear, /model, /compact)
    if stripped.startswith("/") and len(stripped) > 1 and stripped[1:2].isalpha():
        return True
    # CC-injected command/hook XML tags
    for tag in _COMMAND_ARTIFACT_TAGS:
        if tag in stripped:
            return True
    return False


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

    # Extract task_text from the first real human instruction.  Scan user
    # messages in order; skip a message when it is:
    #   - isMeta (CC internal metadata)
    #   - tool_result-only
    #   - empty
    #   - a command artifact (slash-command or CC hook/system injection)
    # If no real instruction is found, task_text remains None — the classifier
    # will suppress AUTONOMOUS in that case (F4.1 precision fix).
    task_text: str | None = None
    for rec in records:
        if rec.get("type") != "user":
            continue
        if rec.get("isMeta"):
            continue
        content = rec.get("message", {}).get("content")
        if _is_pure_tool_result(content):
            continue
        text = _extract_text_content(content)
        if not text:
            continue
        if _is_command_artifact(text):
            continue
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
            targets = extract_targets(tool_name, tool_input)

            # Resolve is_error and extract result entities from the matching
            # tool_result block.  Raw content is read once, entities extracted,
            # then discarded — never stored on the Decision.
            result_block = tool_results.get(tool_use_id, {})
            is_error = bool(result_block.get("is_error", False))

            # Extract redacted entities from the tool_result content.
            result_content = result_block.get("content", "")
            if isinstance(result_content, list):
                # Content may be a list of text/image blocks — join text parts.
                text_parts = [
                    b.get("text", "")
                    for b in result_content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                result_content = " ".join(text_parts)
            result_entities = extract_result_entities(str(result_content) if result_content else "")
            # Raw result_content is not stored beyond this point.

            step_index = len(decisions)
            parent = step_index - 1 if step_index > 0 else None

            decisions.append(
                Decision(
                    step_index=step_index,
                    ts=ts,
                    tool_name=tool_name,
                    tool_args_summary=args_summary,
                    targets=targets,
                    result_entities=result_entities,
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


def ingest_session(
    source: str,
    db_path: str | None = None,
    source_kind: str = "transcript",
) -> Run:
    """Load a session and persist it to the ledger in one call.

    Parameters
    ----------
    source:
        File path or session UUID.
    db_path:
        Override ledger DB path. Falls back to ``UNASKED_LEDGER_DB`` env or
        ``~/.kairos/ledger.db``.
    source_kind:
        ``"transcript"`` (default) — reads the native CC transcript via
        ``load_session``; works on any existing session with zero setup.
        ``"spool"`` — reads the kairos hook spool via
        ``read_session_from_spool``; requires the hook to be installed.

    Returns
    -------
    Run
        The same ``Run`` that was persisted.
    """
    if source_kind == "spool":
        run = read_session_from_spool(source)
    else:
        run = load_session(source)
    save_run(run, path=db_path)
    return run


# ── Spool reader (opt-in, capture-time path) ──────────────────────────────────


def _spool_path(session_id: str, spool_dir: str | Path | None) -> Path:
    if spool_dir is not None:
        root = Path(spool_dir)
    else:
        override = os.environ.get("KAIROS_SPOOL_DIR", "").strip()
        root = Path(override) if override else _DEFAULT_SPOOL_ROOT
    return root / f"{session_id}.jsonl"


def read_session_from_spool(
    session_id: str,
    spool_dir: str | Path | None = None,
) -> Run:
    """Read a Claude Code spool JSONL (written by kairos_hook.py) → ``Run``.

    This is the OPT-IN capture-time path.  It only works after kairos_hook.py
    is installed.  Prefer ``load_session`` for zero-setup reading of existing
    CC session history.

    Parameters
    ----------
    session_id:
        The session UUID — matches the spool filename stem.
    spool_dir:
        Override the spool root directory.  Falls back to ``KAIROS_SPOOL_DIR``
        env var, then ``~/.kairos/spool``.

    Raises
    ------
    FileNotFoundError
        When no spool file exists for that session_id.
    """
    spool_file = _spool_path(session_id, spool_dir)
    if not spool_file.exists():
        raise FileNotFoundError(
            f"No spool file for session {session_id!r}: {spool_file}"
        )

    raw_lines = spool_file.read_text(encoding="utf-8").splitlines()

    started_at: str | None = None
    task_text: str | None = None
    decisions: list[Decision] = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            event: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue  # soft-fail

        name = event.get("event_name", "")

        if name == "SessionStart":
            started_at = event.get("occurred_at")
            # Best-effort task_text from transcript_path if present.
            if task_text is None:
                tp = (event.get("payload_redacted") or {}).get("transcript_path")
                if tp:
                    tp_path = Path(str(tp))
                    try:
                        for tline in tp_path.read_text(encoding="utf-8").splitlines():
                            tline = tline.strip()
                            if not tline:
                                continue
                            entry = json.loads(tline)
                            if entry.get("type") == "user" and not entry.get("isMeta"):
                                content = entry.get("message", {}).get("content")
                                if not _is_pure_tool_result(content):
                                    text = _extract_text_content(content)
                                    if text and not _is_command_artifact(text):
                                        task_text = text
                                        break
                    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                        pass

        elif name in ("PostToolUse", "PostToolUseFailure"):
            tool_name = str(event.get("tool_name") or "unknown")
            args: dict[str, Any] = event.get("tool_input_redacted") or {}
            # args are already redacted by kairos_hook.py; one more pass is safe.
            raw_summary = summarize_args(tool_name, args)
            args_summary = redact(raw_summary)
            targets = extract_targets(tool_name, args)

            is_error_raw = event.get("is_error")
            is_error = bool(is_error_raw) or name == "PostToolUseFailure"

            step_index = len(decisions)
            parent = step_index - 1 if step_index > 0 else None

            decisions.append(
                Decision(
                    step_index=step_index,
                    ts=event.get("occurred_at"),
                    tool_name=tool_name,
                    tool_args_summary=args_summary,
                    targets=targets,
                    # Spool doesn't carry tool_result content — no result_entities
                    result_entities=[],
                    is_error=is_error,
                    parent_step_index=parent,
                )
            )
        # SessionEnd: no IR equivalent — ignored.

    return Run(
        run_id=session_id,
        source="claude_code",
        task_text=task_text,
        started_at=started_at,
        decisions=decisions,
    )
