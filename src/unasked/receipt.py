"""Receipt formatter for unasked.

``format_receipt(run)`` turns a classified Run into a human-readable
60-second receipt string suitable for printing to a terminal.

Design choices
--------------
- Pure function: takes a Run, returns a str.  No I/O, no side effects.
- Zero dependencies beyond stdlib.
- Sections:
    1. Header — task summary + run metadata.
    2. Flagged steps — AUTONOMOUS and TOOL_INDUCED grouped and labelled.
       Each step shows: step index, tool name, target/summary, scope_drift
       indicator, and the classifier's why explanation.
    3. Scope-drift aside — any REQUESTED/DERIVED steps that still drifted.
    4. Verdict line — one-line summary judgement.
- Unclassified decisions (provenance is None) are skipped gracefully.
- If no decisions are flagged the receipt says so explicitly (clean run).
"""

from __future__ import annotations

from unasked.ir import Decision, Run

# ── Label display config ──────────────────────────────────────────────────────

_LABEL_HEADER: dict[str, str] = {
    "AUTONOMOUS": "AUTONOMOUS — did without being asked",
    "TOOL_INDUCED": "TOOL_INDUCED — steered by external tool output",
}

_LABEL_ORDER = ["TOOL_INDUCED", "AUTONOMOUS"]

_SCOPE_MARKER = " [scope-drift]"

# Width for the horizontal rule
_WIDTH = 72


# ── Helpers ───────────────────────────────────────────────────────────────────


def _rule(char: str = "─") -> str:
    return char * _WIDTH


def _trunc(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def _step_line(dec: Decision) -> str:
    """One display line for a single flagged decision."""
    drift = _SCOPE_MARKER if dec.scope_drift else ""
    # Prefer targets for the display target; fall back to tool_args_summary.
    target_display = (
        ", ".join(dec.targets[:2]) if dec.targets
        else dec.tool_args_summary
    )
    return (
        f"  [{dec.step_index:>3}] {dec.tool_name:<14}"
        f" {_trunc(target_display, 40)}{drift}"
    )


def _why_line(dec: Decision) -> str:
    """Indented why explanation for a decision."""
    why = dec.why or ""
    return f"       why: {_trunc(why, 65)}"


# ── Public API ────────────────────────────────────────────────────────────────


def format_receipt(run: Run) -> str:
    """Return the full receipt string for a classified Run.

    Parameters
    ----------
    run:
        A Run whose decisions have been classified (provenance/scope_drift/why
        populated by classify_run).  Unclassified decisions are skipped.

    Returns
    -------
    str
        Multi-line receipt ready to print.
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(_rule("═"))
    lines.append("unasked — agent run receipt")
    lines.append(_rule("─"))

    task_display = _trunc(run.task_text or "(no task captured)", 68)
    lines.append(f"Task    : {task_display}")
    lines.append(f"Run ID  : {run.run_id}")

    total = len(run.decisions)
    classified = sum(1 for d in run.decisions if d.provenance is not None)
    lines.append(f"Steps   : {total} total, {classified} classified")
    lines.append(_rule("─"))

    # ── Flagged steps ─────────────────────────────────────────────────────────
    flagged: dict[str, list[Decision]] = {label: [] for label in _LABEL_ORDER}
    drift_elsewhere: list[Decision] = []

    for dec in run.decisions:
        if dec.provenance in _LABEL_ORDER:
            flagged[dec.provenance].append(dec)
        elif dec.scope_drift and dec.provenance not in (None,):
            drift_elsewhere.append(dec)

    any_flagged = any(flagged[lbl] for lbl in _LABEL_ORDER)

    if not any_flagged and not drift_elsewhere:
        lines.append("Clean run — no autonomous, tool-induced, or scope-drift steps.")
    else:
        for label in _LABEL_ORDER:
            group = flagged[label]
            if not group:
                continue
            lines.append(f"\n{_LABEL_HEADER[label]} ({len(group)} step{'s' if len(group) != 1 else ''})")
            lines.append(_rule("·"))
            for dec in group:
                lines.append(_step_line(dec))
                lines.append(_why_line(dec))

        if drift_elsewhere:
            lines.append(f"\nScope drift in otherwise-acceptable steps ({len(drift_elsewhere)})")
            lines.append(_rule("·"))
            for dec in drift_elsewhere:
                lines.append(_step_line(dec))
                lines.append(_why_line(dec))

    # ── Verdict ───────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(_rule("─"))

    n_auto = len(flagged["AUTONOMOUS"])
    n_induced = len(flagged["TOOL_INDUCED"])
    n_drift = sum(
        1 for d in run.decisions
        if d.scope_drift and d.provenance is not None
    )

    if n_auto == 0 and n_induced == 0 and n_drift == 0:
        verdict = "CLEAN — agent stayed within task scope."
    else:
        parts: list[str] = []
        if n_induced:
            parts.append(f"{n_induced} tool-induced")
        if n_auto:
            parts.append(f"{n_auto} autonomous")
        if n_drift:
            parts.append(f"{n_drift} scope-drift")
        verdict = "FLAGGED — " + ", ".join(parts) + "."

    lines.append(f"Verdict : {verdict}")
    lines.append(_rule("═"))

    return "\n".join(lines)
