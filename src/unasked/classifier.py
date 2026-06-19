"""Provenance classifier for unasked.

Assigns a provenance label and scope_drift flag to each Decision in a Run
using structural event-graph rules only — no LLM calls, no external I/O.

Labels
------
REQUESTED
    The decision directly implements what the task asked for.  The tool name
    or its target appears in (or is clearly implied by) ``task_text``.

DERIVED
    A natural follow-on from the immediately preceding decision in the same
    run.  E.g. running tests after an edit, reading a file before editing it,
    sending a message after doing research.  The action is expected but was
    not explicitly named in the task.

AUTONOMOUS
    The agent expanded scope without being asked.  Signals:
    - A write/execute/delete tool (Bash, Write, Edit, TaskCreate, …) whose
      target is NOT mentioned in the task AND is NOT a natural follow-on from
      the prior decision.
    - The agent spawned a sub-agent (Agent tool) whose description is not
      derivable from the task.
    - The agent sent a message (SendMessage) or created an issue (TaskCreate)
      not implied by the task.

TOOL_INDUCED
    The decision was steered by content returned from a prior tool call.
    Primary signal: the previous decision was a READ-class tool (Read,
    WebFetch, WebSearch, Glob, ToolSearch) and the current decision's tool or
    target doesn't appear in task_text but does share tokens with the prior
    step's args_summary.  This is inherently imprecise at the structural level
    (we don't see the actual content); we flag it when the pattern is present
    and let the user review.

Scope drift
-----------
``scope_drift = True`` when the decision's ``tool_args_summary`` references a
file path or URL that cannot be derived from ``task_text``.  Heuristic: no
token overlap between the resource identifier in the summary and the task
words.  Always False for decisions with no extractable resource.

Design notes
------------
- Pure function: ``classify(run) -> Run`` returns a NEW Run with provenance
  fields populated.  The original is not mutated.
- Precision over recall for TOOL_INDUCED and AUTONOMOUS: when ambiguous,
  prefer DERIVED or REQUESTED over a false positive flag.
- No runtime deps beyond stdlib.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Sequence

from unasked.ir import Decision, Run

# ── Tool taxonomy ─────────────────────────────────────────────────────────────

# Tools that READ external state; their output may steer the next decision.
_READ_TOOLS: frozenset[str] = frozenset(
    {"Read", "WebFetch", "WebSearch", "Glob", "ToolSearch", "Monitor"}
)

# Tools that WRITE / EXECUTE / MUTATE state.  An unprompted use → AUTONOMOUS.
_WRITE_TOOLS: frozenset[str] = frozenset(
    {"Write", "Edit", "Bash", "NotebookEdit"}
)

# Tools that DELEGATE work to another agent or sub-process.
_DELEGATE_TOOLS: frozenset[str] = frozenset({"Agent"})

# Tools that COMMUNICATE outward (send message, create issue, etc.).
_COMMS_TOOLS: frozenset[str] = frozenset(
    {"SendMessage", "TaskCreate", "TaskUpdate", "PushNotification"}
)

# Tools that are lightweight / purely mechanical: skips, searches, plumbing.
_PLUMBING_TOOLS: frozenset[str] = frozenset(
    {"ExitPlanMode", "EnterWorktree", "ExitWorktree", "Skill",
     "AskUserQuestion", "CronCreate", "CronDelete", "CronList"}
)

# ── Token helpers ─────────────────────────────────────────────────────────────

# Split on anything that isn't alphanumeric, underscore, or dot.
# Notably: / : = - are separators, so paths, URLs, key=value pairs, and
# hyphenated compound words are split into individual components.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_.]+")
_MIN_TOKEN_LEN = 3  # ignore short noise tokens

# Structural tokens from summarize_args key names — always present as noise.
_STOPWORDS: frozenset[str] = frozenset(
    {"file_path", "file", "path", "http", "https", "query", "type", "name"}
)


_FILE_EXT_RE = re.compile(r"\.[a-zA-Z0-9]{1,5}$")
_DOT_SPLIT_RE = re.compile(r"\.")


def _tokens(text: str) -> frozenset[str]:
    """Lower-cased content tokens from a string, length ≥ 3.

    Splits on /  :  =  -  and other non-word chars so that paths, URLs,
    key=value pairs, and hyphenated compounds are individually tokenised.
    Stopwords (structural noise from summarize_args key names) are excluded.

    Additional normalisation for each raw token:
    - File-extension stripping: ``stripe.py`` → also emits ``stripe``.
    - Dot-segment expansion: ``api.stripe.com`` → also emits each ``stripe``,
      ``com``, ``api`` segment so URL components match standalone terms.
    """
    raw = frozenset(
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if len(t) >= _MIN_TOKEN_LEN and t.lower() not in _STOPWORDS
    )
    extras: set[str] = set()
    for tok in raw:
        # Strip trailing file extension → bare stem
        stem = _FILE_EXT_RE.sub("", tok)
        if stem and stem != tok and len(stem) >= _MIN_TOKEN_LEN:
            extras.add(stem)
        # Expand dot-separated segments (domain names, dotted identifiers)
        if "." in tok:
            for seg in _DOT_SPLIT_RE.split(tok):
                seg = seg.lower()
                if len(seg) >= _MIN_TOKEN_LEN and seg not in _STOPWORDS:
                    extras.add(seg)
    return raw | frozenset(extras)


def _overlap(a: str, b: str) -> bool:
    """True when the two strings share at least one meaningful token."""
    return bool(_tokens(a) & _tokens(b))


# ── Resource extraction ───────────────────────────────────────────────────────

# Patterns that pull the primary resource identifier out of an args_summary.
_RESOURCE_PATTERNS: list[re.Pattern[str]] = [
    # file_path=... or just a path-like token at the start
    re.compile(r"file_path=(\S+)"),
    re.compile(r"^(/[\S]+)"),
    # URL
    re.compile(r"(https?://\S+)"),
    # bare relative path-like token containing a slash
    re.compile(r"(\S+/\S+)"),
]


def _extract_resource(summary: str) -> str | None:
    """Pull the most specific resource identifier from an args_summary."""
    for pat in _RESOURCE_PATTERNS:
        m = pat.search(summary)
        if m:
            return m.group(1)
    return None


# ── Scope-drift check ─────────────────────────────────────────────────────────


def _has_scope_drift(decision: Decision, task_text: str | None) -> bool:
    """True when the decision's resource cannot be derived from task_text.

    Returns False when:
    - No task_text is available (can't evaluate).
    - No resource identifier is extractable from the summary.
    - The resource tokens overlap with task tokens (it's in scope).
    """
    if not task_text:
        return False
    resource = _extract_resource(decision.tool_args_summary)
    if not resource:
        return False
    # If the resource shares tokens with the task, it's in scope.
    if _overlap(resource, task_text):
        return False
    # Also check the full summary against task (catches bare filenames).
    return not _overlap(decision.tool_args_summary, task_text)


# ── Provenance rules ──────────────────────────────────────────────────────────


def _classify_one(
    decision: Decision,
    prior: Decision | None,
    task_text: str | None,
    task_tokens: frozenset[str],
) -> tuple[str, str]:
    """Return (provenance_label, why) for one decision.

    Evaluated in priority order: TOOL_INDUCED > AUTONOMOUS > REQUESTED > DERIVED.
    """
    tool = decision.tool_name
    summary = decision.tool_args_summary

    # ── TOOL_INDUCED ─────────────────────────────────────────────────────────
    # Condition: prior step was a READ-class tool AND current tool/target has
    # no direct token overlap with the task but does share tokens with the
    # prior step's summary (suggesting the agent acted on what it just read).
    if (
        prior is not None
        and prior.tool_name in _READ_TOOLS
        and not _overlap(summary, task_text or "")
        and _overlap(summary, prior.tool_args_summary)
    ):
        return (
            "TOOL_INDUCED",
            f"Action follows a {prior.tool_name} call and targets content "
            f"not stated in the task (shares tokens with prior step's args).",
        )

    # ── AUTONOMOUS ────────────────────────────────────────────────────────────
    # Condition: write/execute/delegate/comms tool whose target isn't in the
    # task AND isn't a natural follow-on from a prior write/read of the same
    # resource.
    if tool in _WRITE_TOOLS | _DELEGATE_TOOLS | _COMMS_TOOLS:
        in_task = _overlap(summary, task_text or "")
        prior_related = prior is not None and _overlap(summary, prior.tool_args_summary)
        if not in_task and not prior_related:
            label = "AUTONOMOUS"
            if tool in _DELEGATE_TOOLS:
                reason = (
                    "Agent spawned a sub-agent whose target is not mentioned "
                    "in the task and is not a follow-on from the prior step."
                )
            elif tool in _COMMS_TOOLS:
                reason = (
                    "Agent sent an outbound message / created a task not "
                    "derivable from the stated task or the prior step."
                )
            else:
                reason = (
                    f"{tool} targets a resource not mentioned in the task "
                    "and not directly related to the prior step."
                )
            return label, reason

    # ── REQUESTED ────────────────────────────────────────────────────────────
    # Condition: the tool name or its target shares tokens with task_text.
    if task_text and (
        tool.lower() in task_tokens
        or _overlap(summary, task_text)
    ):
        return (
            "REQUESTED",
            "Tool or target appears in (or is implied by) the task text.",
        )

    # ── DERIVED ──────────────────────────────────────────────────────────────
    # Default: a plausible follow-on from the prior step or a plumbing op.
    if prior is not None:
        reason = (
            f"Natural follow-on from step {prior.step_index} "
            f"({prior.tool_name})."
        )
    elif tool in _PLUMBING_TOOLS:
        reason = "Plumbing or mode-switch tool; no explicit task link needed."
    else:
        reason = "No prior context; treated as derived task setup."
    return "DERIVED", reason


# ── Public API ────────────────────────────────────────────────────────────────


def classify(run: Run) -> Run:
    """Classify the provenance of every Decision in ``run``.

    Returns a NEW Run (deep copy) with ``provenance``, ``scope_drift``, and
    ``why`` set on each Decision.  Decisions that already have a ``feedback``
    annotation are left as-is (operator override takes precedence).

    Parameters
    ----------
    run:
        A Run produced by the adapter layer.  ``task_text`` may be None; in
        that case REQUESTED labels are not assigned and scope_drift is always
        False.

    Returns
    -------
    Run
        Deep-copied Run with provenance fields populated.
    """
    result = deepcopy(run)
    task_text = run.task_text or ""
    task_tokens = _tokens(task_text)

    decisions = result.decisions
    for i, dec in enumerate(decisions):
        # Operator feedback overrides classifier; skip.
        if dec.feedback is not None:
            continue

        prior = decisions[i - 1] if i > 0 else None
        label, why = _classify_one(dec, prior, task_text or None, task_tokens)

        dec.provenance = label
        dec.why = why
        dec.scope_drift = _has_scope_drift(dec, task_text or None)

    return result
