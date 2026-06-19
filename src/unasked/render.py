"""Receipt renderer for unasked.

``render_receipt(run, *, color=True) -> str``

Turns a classified Run into a human-readable 60-second receipt.
Pure function — no I/O, no side effects.  Zero deps beyond stdlib.

Format
------
run <run_id[:8]> — <N> steps[, <duration>] · task: "<task, ~80 char>"

⚠ did WITHOUT being asked (<count>)
  #<idx> <Tool>  <short summary>    autonomous[· scope-drift]

⚡ steered by external content (<count>)
  #<idx> <Tool>  <short summary>    tool-induced · <short why>

↗ touched outside task scope (<count>)      ← DERIVED/REQUESTED that drifted
  #<idx> <Tool>  <short summary>    scope-drift · <short why>

✓ <routine_count> routine (mapped to your request)
verdict: <K> to eyeball, rest routine

Color is applied to group header symbols/labels when enabled.  Disabled by
passing color=False or when the NO_COLOR env var is set (checked by cli.py,
not here — callers control the flag).

When zero flagged steps:
    ✓ all <N> steps routine — nothing to eyeball.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Sequence

from unasked.classify import _task_anchored
from unasked.entities import extract_task_entities
from unasked.ir import Decision, Run

# ── ANSI helpers ──────────────────────────────────────────────────────────────

_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_RESET  = "\033[0m"


def _col(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{_RESET}" if color else text


# ── Duration helper ───────────────────────────────────────────────────────────

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        # Accept ISO with or without fractional seconds / Z suffix.
        s = ts.rstrip("Z").split(".")[0]
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def _duration_str(run: Run) -> str | None:
    """Return a human-readable duration string, or None if not computable."""
    start = _parse_ts(run.started_at)
    last_ts: str | None = None
    for d in reversed(run.decisions):
        if d.ts:
            last_ts = d.ts
            break
    end = _parse_ts(last_ts)
    if start is None or end is None:
        return None
    secs = int((end - start).total_seconds())
    if secs < 0:
        return None
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m{secs % 60:02d}s"


# ── Line helpers ──────────────────────────────────────────────────────────────

def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def _step_summary(dec: Decision) -> str:
    """Short one-liner for a decision: tool + target."""
    summary = dec.tool_args_summary or " ".join(dec.targets)
    return _trunc(summary, 48)


def _step_line(dec: Decision, tag: str) -> str:
    """Render one flagged step as a two-column line."""
    return f"  #{dec.step_index:<4} {dec.tool_name:<14} {_step_summary(dec):<50}  {tag}"


# ── Public API ────────────────────────────────────────────────────────────────


def render_receipt(run: Run, *, color: bool = True) -> str:
    """Return the full receipt string for a classified Run.

    Parameters
    ----------
    run:
        A Run whose decisions have been classified (provenance / scope_drift /
        why populated by classify_run).  Unclassified decisions are skipped.
    color:
        Whether to emit ANSI color codes.  Callers should set this based on
        the --no-color flag and the NO_COLOR env var.

    Returns
    -------
    str
        Multi-line receipt ready to ``print()``.
    """
    lines: list[str] = []

    # ── Header line ───────────────────────────────────────────────────────────
    rid = run.run_id[:8] if run.run_id else "unknown"
    n = len(run.decisions)
    dur = _duration_str(run)
    step_part = f"{n} steps" + (f", {dur}" if dur else "")
    # When task_text is None, AUTONOMOUS was suppressed (F4.1 precision fix) —
    # make that visible so the user knows provenance for AUTONOMOUS was not assessed.
    task_part = _trunc(run.task_text or "(no explicit task detected)", 80)
    lines.append(f'run {rid} — {step_part} · task: "{task_part}"')
    # F4.3: note when task is not concrete so user knows scope checks were limited.
    task_entities = extract_task_entities(run.task_text) if run.task_text else []
    if not _task_anchored(task_entities):
        lines.append(
            "note: task not concrete — scope/off-task checks limited; "
            "showing injection + high-consequence only."
        )
    lines.append("")

    # ── Bucket decisions ──────────────────────────────────────────────────────
    autonomous: list[Decision] = []
    tool_induced: list[Decision] = []
    # scope_drift on steps NOT already in autonomous (don't double-list)
    drift_only: list[Decision] = []
    routine_count = 0

    for dec in run.decisions:
        if dec.provenance is None:
            continue
        if dec.provenance == "AUTONOMOUS":
            autonomous.append(dec)
        elif dec.provenance == "TOOL_INDUCED":
            tool_induced.append(dec)
        else:
            # REQUESTED or DERIVED
            if dec.scope_drift:
                drift_only.append(dec)
            else:
                routine_count += 1

    any_flagged = bool(autonomous or tool_induced or drift_only)

    if not any_flagged:
        lines.append(f"✓ all {n} steps routine — nothing to eyeball.")
    else:
        # ── AUTONOMOUS group ──────────────────────────────────────────────────
        if autonomous:
            header = _col(
                f"⚠ did WITHOUT being asked ({len(autonomous)})",
                _YELLOW, color,
            )
            lines.append(header)
            for dec in autonomous:
                tag = "autonomous"
                if dec.scope_drift:
                    tag += " · scope-drift"
                lines.append(_step_line(dec, tag))
            lines.append("")

        # ── TOOL_INDUCED group ────────────────────────────────────────────────
        if tool_induced:
            header = _col(
                f"⚡ steered by external content ({len(tool_induced)})",
                _RED, color,
            )
            lines.append(header)
            for dec in tool_induced:
                why_short = _trunc(dec.why or "", 40)
                tag = f"tool-induced · {why_short}" if why_short else "tool-induced"
                lines.append(_step_line(dec, tag))
            lines.append("")

        # ── Scope-drift-only group ────────────────────────────────────────────
        if drift_only:
            header = _col(
                f"↗ touched outside task scope ({len(drift_only)})",
                _CYAN, color,
            )
            lines.append(header)
            for dec in drift_only:
                why_short = _trunc(dec.why or "", 40)
                tag = f"scope-drift · {why_short}" if why_short else "scope-drift"
                lines.append(_step_line(dec, tag))
            lines.append("")

        # ── Routine count line ────────────────────────────────────────────────
        if routine_count:
            lines.append(f"✓ {routine_count} routine (mapped to your request)")
            lines.append("")

    # ── Verdict ───────────────────────────────────────────────────────────────
    k = len(autonomous) + len(tool_induced) + len(drift_only)
    if k == 0:
        lines.append("verdict: clean — agent stayed within task scope.")
    else:
        rest = n - k
        rest_str = f", {rest} routine" if rest > 0 else ""
        lines.append(f"verdict: {k} to eyeball{rest_str}.")

    return "\n".join(lines)
