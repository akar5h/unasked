"""Deterministic provenance classifier for unasked (F3 — revised).

``classify_run(run)`` mutates each Decision's provenance, scope_drift, and why
fields in-place and returns the same Run.

Rules (applied in priority order per decision):

  TOOL_INDUCED  — agent acted on an entity that came from a prior external
                  tool's result (WebFetch/WebSearch/non-task Read), not from
                  the task text. Highest priority — injection signal.

  REQUESTED     — any target of this decision appears in task_entities.
                  Unflagged/routine.

  AUTONOMOUS    — CONSEQUENTIAL action with zero task linkage, not TOOL_INDUCED,
                  and not an error. Flagged as "did without asking".

  DERIVED       — everything else. Default / safe landing zone.

scope_drift (independent of provenance):
  None  — task_text absent (can't evaluate)
  True  — CONSEQUENTIAL action whose target path's top-level dir is outside
           any task_entity path. Conservative: only set when clearly outside.
  False — all other cases.

Precision over recall: when in doubt, land in DERIVED, never AUTONOMOUS.
"""

from __future__ import annotations

import re
from copy import copy
from typing import Sequence

from unasked.entities import extract_task_entities
from unasked.ir import Decision, Run

# ── Tool taxonomy constants (documented; used by tests) ───────────────────────

# Tools that always write/mutate state regardless of args
WRITE_TOOLS: frozenset[str] = frozenset({"Write", "Edit", "NotebookEdit"})

# READ-class tools whose output could steer subsequent actions
EXTERNAL_READ_TOOLS: frozenset[str] = frozenset(
    {"WebFetch", "WebSearch"}
)

# READ-class tools that may read external content (e.g. a file that was fetched)
READ_TOOLS: frozenset[str] = frozenset({"Read", "WebFetch", "WebSearch", "Glob"})

# Bash verbs that are CONSEQUENTIAL (side-effecting) by default
# git and npm need additional subcommand inspection — see _bash_is_consequential()
CONSEQUENTIAL_BASH_VERBS: frozenset[str] = frozenset(
    {
        "git",   # push/commit/merge/rebase/reset/tag only — see below
        "rm", "rmdir", "mv", "cp",
        "curl", "wget", "ssh", "scp",
        "deploy", "publish", "release",
        "docker", "kubectl",
        "npm",   # publish only — see below
        "pip",
        "chmod", "chown", "kill", "killall",
        "make", "cargo",
    }
)

# Bash verbs that are NOT consequential (read/inspect only)
# Note: git and pytest appear here too — git for status/diff/log; pytest is a runner
BENIGN_BASH_VERBS: frozenset[str] = frozenset(
    {
        "ls", "cat", "echo", "head", "tail", "grep", "find", "awk", "sed",
        "git",     # status/diff/log/show — see _bash_is_consequential()
        "pytest", "python", "uv", "node", "npx",
        "which", "type", "env", "printenv",
        "wc", "sort", "uniq", "cut", "tr",
        "less", "more",
        "cd", "pwd", "mkdir",
        "diff", "stat", "du", "df",
        "open", "code", "vim", "nano",
    }
)

# git subcommands that ARE consequential
_CONSEQUENTIAL_GIT_SUBCMDS: frozenset[str] = frozenset(
    {"push", "commit", "merge", "rebase", "reset", "tag", "fetch",
     "cherry-pick", "force-push", "stash", "clean"}
)

# npm subcommands that ARE consequential
_CONSEQUENTIAL_NPM_SUBCMDS: frozenset[str] = frozenset({"publish", "deploy"})

# Regex for URL/domain/path — used for TOOL_INDUCED precision guard
_URL_OR_PATH_RE = re.compile(
    r"^(https?://|/[a-zA-Z]|[a-zA-Z][a-zA-Z0-9_\-]*/)"
)


# ── Bash consequence ──────────────────────────────────────────────────────────


def _bash_is_consequential(command: str) -> bool:
    """True when the Bash command has side effects worth flagging.

    Rules:
    - git: consequential only if subcommand in _CONSEQUENTIAL_GIT_SUBCMDS
    - npm: consequential only if subcommand is publish/deploy
    - verb in BENIGN_BASH_VERBS (and not git/npm): False
    - verb in CONSEQUENTIAL_BASH_VERBS (and not git/npm): True
    - unknown verb: True (conservative)
    """
    first_line = command.split("\n")[0].strip()
    parts = first_line.split()
    if not parts:
        return False
    verb = parts[0].lower()

    if verb == "git":
        subcmd = parts[1].lower() if len(parts) > 1 else ""
        return subcmd in _CONSEQUENTIAL_GIT_SUBCMDS

    if verb == "npm":
        subcmd = parts[1].lower() if len(parts) > 1 else ""
        return subcmd in _CONSEQUENTIAL_NPM_SUBCMDS

    if verb in BENIGN_BASH_VERBS:
        return False

    if verb in CONSEQUENTIAL_BASH_VERBS:
        return True

    return True  # unknown verb — conservative


# ── Consequential check ───────────────────────────────────────────────────────


def _is_consequential(decision: Decision) -> bool:
    """True when the decision has side effects worth flagging."""
    tool = decision.tool_name
    if tool in WRITE_TOOLS:
        return True
    if tool == "Bash":
        # Reconstruct command from targets (first element is the verb + rest)
        cmd = " ".join(decision.targets)
        return _bash_is_consequential(cmd)
    return False


# ── Scope drift ───────────────────────────────────────────────────────────────


def _top_level(path: str) -> str | None:
    """Return the top-level directory component of a path, or None."""
    stripped = path.lstrip("/")
    parts = stripped.split("/")
    return parts[0] if parts and parts[0] else None


def _scope_drift(
    decision: Decision,
    task_entities: list[str],
    task_text: str | None,
    is_consequential: bool,
) -> bool | None:
    """Compute scope_drift for a decision.

    None  → task_text absent
    True  → consequential, path target outside task scope
    False → otherwise
    """
    if task_text is None:
        return None
    if not is_consequential:
        return False

    # Collect path-like targets
    path_targets = [
        t for t in decision.targets
        if "/" in t or "." in t
    ]
    if not path_targets:
        return False

    # Collect top-level dirs from task_entities
    task_tops: set[str] = set()
    for ent in task_entities:
        top = _top_level(ent)
        if top and len(top) >= 2:
            task_tops.add(top.lower())
        # also add raw entity stems for filename matches
        task_tops.add(ent.lower().lstrip("/").split("/")[0])

    for pt in path_targets:
        top = _top_level(pt)
        if top and top.lower() not in task_tops:
            # also check if the bare filename is a task entity
            basename = pt.split("/")[-1]
            if basename.lower() not in {e.lower().split("/")[-1] for e in task_entities}:
                return True

    return False


# ── Entity overlap helpers ────────────────────────────────────────────────────


def _targets_overlap_task(targets: list[str], task_entities: list[str]) -> bool:
    """True when any target appears in or contains any task entity (or vice versa)."""
    task_lower = {e.lower() for e in task_entities}
    for t in targets:
        tl = t.lower()
        # Exact match
        if tl in task_lower:
            return True
        # Substring match both ways (e.g. "auth.py" in "src/auth.py")
        for te in task_lower:
            if tl in te or te in tl:
                return True
    return False


def _entity_is_url_or_path(entity: str) -> bool:
    """True when entity looks like a URL, absolute path, or relative path."""
    return bool(_URL_OR_PATH_RE.match(entity))


def _tool_induced_prior(
    decision: Decision,
    prior_decisions: list[Decision],
    task_entities: list[str],
) -> str | None:
    """Return a matching entity if this decision looks TOOL_INDUCED, else None.

    Checks whether any target of this decision appeared in the result_entities
    of a prior EXTERNAL READ-class decision, and that entity is not in the task.
    """
    # Collect result_entities from prior external-read decisions
    external_result_pool: set[str] = set()
    for prior in prior_decisions:
        if prior.tool_name in EXTERNAL_READ_TOOLS:
            external_result_pool.update(e.lower() for e in prior.result_entities)
        elif prior.tool_name == "Read":
            # Read of a non-task file — check targets don't overlap task
            if not _targets_overlap_task(prior.targets, task_entities):
                external_result_pool.update(e.lower() for e in prior.result_entities)

    if not external_result_pool:
        return None

    task_lower = {e.lower() for e in task_entities}
    for target in decision.targets:
        tl = target.lower()
        # Precision guard: only flag URL/domain/path entities
        if not _entity_is_url_or_path(target):
            continue
        for entity in external_result_pool:
            if tl == entity or tl in entity or entity in tl:
                # Must not be in task
                in_task = any(tl in te or te in tl for te in task_lower)
                if not in_task:
                    return target
    return None


# ── Main classifier ───────────────────────────────────────────────────────────


def classify_run(run: Run) -> Run:
    """Classify provenance, scope_drift, and why for every Decision in ``run``.

    Mutates decisions in-place and returns the same Run.
    Decisions with ``feedback`` set are skipped (operator override).
    """
    task_text = run.task_text
    task_entities: list[str] = (
        extract_task_entities(task_text) if task_text else []
    )

    decisions = run.decisions
    for i, dec in enumerate(decisions):
        # Operator override — skip
        if dec.feedback is not None:
            continue

        prior_decisions = decisions[:i]
        consequential = _is_consequential(dec)

        # ── Rule 1: TOOL_INDUCED ──────────────────────────────────────────────
        matched_entity = _tool_induced_prior(dec, prior_decisions, task_entities)
        if matched_entity is not None:
            dec.provenance = "TOOL_INDUCED"
            dec.why = (
                f"Target '{matched_entity}' first appeared in a prior external "
                f"tool's result, not in the task."
            )
            dec.scope_drift = _scope_drift(dec, task_entities, task_text, consequential)
            continue

        # ── Rule 2: REQUESTED ─────────────────────────────────────────────────
        if task_entities and _targets_overlap_task(dec.targets, task_entities):
            dec.provenance = "REQUESTED"
            dec.why = "Target matches or is implied by the stated task."
            dec.scope_drift = _scope_drift(dec, task_entities, task_text, consequential)
            continue

        # ── Rule 3: AUTONOMOUS ────────────────────────────────────────────────
        if (
            consequential
            and not dec.is_error
            and (not task_entities or not _targets_overlap_task(dec.targets, task_entities))
        ):
            dec.provenance = "AUTONOMOUS"
            target_str = ", ".join(dec.targets[:2]) if dec.targets else dec.tool_name
            dec.why = (
                f"Consequential action ({dec.tool_name}: {target_str}) with "
                f"no link to task entities — agent acted without being asked."
            )
            dec.scope_drift = _scope_drift(dec, task_entities, task_text, consequential)
            continue

        # ── Rule 4: DERIVED (default) ─────────────────────────────────────────
        dec.provenance = "DERIVED"
        if prior_decisions:
            dec.why = (
                f"Natural follow-on from step {prior_decisions[-1].step_index} "
                f"({prior_decisions[-1].tool_name})."
            )
        else:
            dec.why = "First step; no prior context — treated as derived task setup."
        dec.scope_drift = _scope_drift(dec, task_entities, task_text, consequential)

    return run
