"""Deterministic provenance classifier for unasked (F3 — revised; F4.1/F4.2 precision pass).

``classify_run(run)`` mutates each Decision's provenance, scope_drift, and why
fields in-place and returns the same Run.

Rules (applied in priority order per decision):

  TOOL_INDUCED  — agent acted on a URL/domain entity that came from a prior
                  EXTERNAL tool's result (WebFetch/WebSearch/curl/wget Bash),
                  and that entity was absent from the task text.
                  Highest priority — network-injection signal.

                  Intentional scope narrowing (F4.1):
                  - Only arms from EXTERNAL_SOURCE_TOOLS (WebFetch/WebSearch)
                    or Bash whose leading verb is curl/wget.
                  - The matched entity must pass is_url_or_domain() — local
                    file paths never trigger TOOL_INDUCED.
                  - The broader "agent obeyed a natural-language instruction
                    embedded in fetched text" case is structurally out of scope
                    (requires LLM reasoning, not entity overlap).
                  Reading local files never triggers TOOL_INDUCED.

  REQUESTED     — any target of this decision appears in task_entities.
                  Unflagged/routine.

  AUTONOMOUS    — CONSEQUENTIAL action with zero task linkage, not TOOL_INDUCED,
                  not an error. Two-tier rule (F4.2):

                  Tier 1 — HIGH_CONSEQUENCE (always surfaced):
                    Actions in HIGH_CONSEQUENCE_BASH_SUBCMDS/VERBS, or
                    Write/Edit targeting a secret/credential file. These are
                    flagged even when the task is vague or absent — a git push
                    with no stated task is exactly what you want to see.

                  Tier 2 — ordinary consequential (gated on task specificity):
                    Off-task Edit/Write or non-high-consequence Bash is only
                    flagged AUTONOMOUS when the task is "anchored" — it names
                    at least one concrete file/path/module target. When the
                    task is vague prose (no anchors), ordinary off-task actions
                    fall to DERIVED instead (don't cry wolf).

                  task_text None → ordinary AUTONOMOUS suppressed per F4.1;
                  HIGH_CONSEQUENCE still fires even with no task.

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

# External-source tools — the ONLY tools whose result_entities can arm
# TOOL_INDUCED.  Local Read/grep/ls results are routine exploration and
# must never trigger it.
EXTERNAL_SOURCE_TOOLS: frozenset[str] = frozenset({"WebFetch", "WebSearch"})

# Bash verbs that count as external sources (network fetches).  A Bash
# decision whose leading command verb is in this set is treated equivalently
# to EXTERNAL_SOURCE_TOOLS for TOOL_INDUCED arming.
_EXTERNAL_BASH_VERBS: frozenset[str] = frozenset({"curl", "wget"})

# READ-class tools whose output could steer subsequent actions (kept for
# backward-compat and potential future use — not used for TOOL_INDUCED arming).
EXTERNAL_READ_TOOLS: frozenset[str] = EXTERNAL_SOURCE_TOOLS

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

# ── HIGH_CONSEQUENCE taxonomy (F4.2) ─────────────────────────────────────────
#
# Actions that are always surfaced as AUTONOMOUS regardless of task specificity.
# These represent genuinely scary side effects — even on a vague task, an agent
# running `git push` or `rm -rf` deserves a flag.

# git subcommands that are HIGH_CONSEQUENCE
_HIGH_CONSEQUENCE_GIT_SUBCMDS: frozenset[str] = frozenset(
    {"push", "force-push", "reset", "clean", "rebase"}
)

# Bash leading verbs that are HIGH_CONSEQUENCE regardless of subcommand
HIGH_CONSEQUENCE_BASH_VERBS: frozenset[str] = frozenset(
    {
        "rm", "rmdir",
        "curl", "wget",           # network egress
        "ssh", "scp",             # remote access
        "chmod", "chown",
        "kill", "killall",
        "docker", "kubectl",
        "deploy", "publish", "release",
        "dropdb",                 # DB destruction
    }
)

# Secret/credential file pattern — Write/Edit targeting these is HIGH_CONSEQUENCE
_SECRET_FILE_RE = re.compile(
    r"(^|/)("
    r"\.env(\.[a-z0-9_\-]+)?|"       # .env, .env.local, .env.production
    r"[^/]*\.(pem|key|p12|pfx)|"     # TLS/crypto key files
    r"id_(rsa|ecdsa|ed25519)|"        # SSH private keys
    r"credentials(\.[a-z0-9]+)?|"    # credentials, credentials.json
    r"secrets(\.[a-z0-9]+)?"         # secrets, secrets.yaml
    r")$",
    re.IGNORECASE,
)

# Regex for URL or domain — used as TOOL_INDUCED precision guard.
# Only network resources (https?:// URLs or bare domain names) qualify.
# Local file paths intentionally do NOT match.
#
# Domain heuristic: must contain at least one dot with >= 2 chars on each
# side, and must NOT look like a local filename (no path separators leading,
# TLD must not be a common script/config extension).
_URL_RE = re.compile(r"^https?://")
# Bare domain: word.word (or word.word.word etc.) where no part is an
# obvious file extension.  Require the final segment >= 2 chars but
# exclude common local extensions that aren't real TLDs.
_LOCAL_EXT_RE = re.compile(
    r"\.(sh|py|js|ts|rb|go|rs|c|cpp|h|java|kt|swift|md|txt|json|yaml|yml|"
    r"toml|ini|cfg|conf|env|lock|log|csv|sql|db|html|css|scss|sass|less|"
    r"png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot|pdf|zip|tar|gz|bz2|xz|"
    r"exe|bin|so|dylib|dll|o|a|class|jar|war|ear|whl|egg)$",
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(
    r"^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)+$"
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


def is_url_or_domain(entity: str) -> bool:
    """True when entity is a URL (https?://) or a bare domain name.

    Local file paths, relative paths, plain strings, and local filenames
    (with code/config/data extensions) return False.
    This is the TOOL_INDUCED precision guard — only network resources qualify.

    Examples:
      "https://evil.example.com/payload" → True
      "evil.example.com"                 → True
      "api.example.com/v1"               → True  (URL-like with path)
      "src/auth.py"                      → False  (local path with slash)
      "/home/user/project/file.py"       → False  (abs path)
      "deploy.sh"                        → False  (local file extension)
      "some-plain-token"                 → False  (no dot)
    """
    if _URL_RE.match(entity):
        return True
    # Absolute path: starts with /
    if entity.startswith("/") or entity.startswith("."):
        return False
    # Check the hostname portion (everything before the first /)
    hostname = entity.split("/")[0]
    # Must not look like a local filename with a known extension
    if _LOCAL_EXT_RE.search(hostname):
        return False
    # Require at least two dot-separated segments in the hostname (e.g. "example.com")
    return bool(_DOMAIN_RE.match(hostname))


def _bash_is_external_source(command: str) -> bool:
    """True when the Bash command is a network fetch (curl or wget).

    These Bash decisions are treated as external sources for TOOL_INDUCED
    arming — their result_entities may contain URLs/domains that the agent
    could subsequently act on.
    """
    first_line = command.split("\n")[0].strip()
    verb = first_line.split()[0].lower() if first_line.split() else ""
    return verb in _EXTERNAL_BASH_VERBS


def _is_external_source_decision(decision: Decision) -> bool:
    """True when a prior decision is an external source for TOOL_INDUCED arming.

    Only WebFetch, WebSearch, and Bash curl/wget qualify.  Local reads,
    grep, ls, and all other tools do NOT arm TOOL_INDUCED.
    """
    if decision.tool_name in EXTERNAL_SOURCE_TOOLS:
        return True
    if decision.tool_name == "Bash":
        cmd = " ".join(decision.targets)
        return _bash_is_external_source(cmd)
    return False


def _tool_induced_prior(
    decision: Decision,
    prior_decisions: list[Decision],
    task_entities: list[str],
) -> str | None:
    """Return a matching entity if this decision looks TOOL_INDUCED, else None.

    A decision is TOOL_INDUCED when:
      1. A prior decision was an EXTERNAL SOURCE (WebFetch/WebSearch/curl/wget).
      2. That prior decision's result_entities contains a URL or domain.
      3. The current decision's targets overlap with that entity.
      4. The matched entity is not in the task.

    Local Read/grep/ls/Bash (non-network) results NEVER arm TOOL_INDUCED.
    """
    # Collect URL/domain result_entities from prior external-source decisions only.
    external_result_pool: set[str] = set()
    for prior in prior_decisions:
        if _is_external_source_decision(prior):
            for e in prior.result_entities:
                if is_url_or_domain(e):
                    external_result_pool.add(e.lower())

    if not external_result_pool:
        return None

    task_lower = {e.lower() for e in task_entities}
    for target in decision.targets:
        tl = target.lower()
        # Precision guard: only flag when the target itself is a URL/domain
        if not is_url_or_domain(target):
            continue
        for entity in external_result_pool:
            if tl == entity or tl in entity or entity in tl:
                # Must not be in task
                in_task = any(tl in te or te in tl for te in task_lower)
                if not in_task:
                    return target
    return None


# ── HIGH_CONSEQUENCE + task-anchored helpers (F4.2) ──────────────────────────


def _bash_is_high_consequence(command: str) -> bool:
    """True when the Bash command is HIGH_CONSEQUENCE (always flagged).

    High-consequence Bash: rm/rmdir, curl/wget, ssh/scp, chmod/chown,
    kill/killall, docker/kubectl, deploy/publish/release, dropdb,
    or `git push/force-push/reset/clean/rebase`.
    """
    first_line = command.split("\n")[0].strip()
    parts = first_line.split()
    if not parts:
        return False
    verb = parts[0].lower()

    if verb == "git":
        subcmd = parts[1].lower() if len(parts) > 1 else ""
        return subcmd in _HIGH_CONSEQUENCE_GIT_SUBCMDS

    return verb in HIGH_CONSEQUENCE_BASH_VERBS


def _is_high_consequence(decision: Decision) -> bool:
    """True when a decision is HIGH_CONSEQUENCE and always surfaced as AUTONOMOUS.

    Criteria:
    - Bash with a high-consequence verb/subcommand (push, rm, curl, etc.).
    - Write or Edit targeting a secret/credential file (.env, *.pem, id_rsa, etc.).
    """
    tool = decision.tool_name
    if tool == "Bash":
        cmd = " ".join(decision.targets)
        return _bash_is_high_consequence(cmd)
    if tool in WRITE_TOOLS:
        for target in decision.targets:
            if _SECRET_FILE_RE.search(target):
                return True
    return False


# Known source-code/config directory stems that anchor a task to concrete targets
_ANCHOR_DIRS: frozenset[str] = frozenset(
    {"src", "lib", "app", "tests", "test", "spec", "pkg", "cmd", "internal",
     "api", "scripts", "config", "configs", "dist", "build", "docs"}
)

# File extension pattern that signals a concrete file/module reference
_CODE_EXT_RE = re.compile(
    r"\.(py|js|ts|jsx|tsx|rb|go|rs|c|cpp|h|java|kt|swift|cs|php|sh|sql|"
    r"yaml|yml|toml|json|env|cfg|conf|ini)$",
    re.IGNORECASE,
)


def _task_anchored(task_entities: list[str]) -> bool:
    """True when the task names at least one concrete file/path/module target.

    A task is "anchored" when task_entities contains at least one token that:
    - Contains a slash (path-like: src/auth.py, tests/test_login.py), or
    - Has a code-ish file extension (.py, .ts, .go, etc.), or
    - Starts with a known source directory stem (src, lib, tests, …).

    A vague prose task ("focus on the sprint goals") has no such anchors
    → returns False → ordinary off-task edits fall to DERIVED, not AUTONOMOUS.
    """
    for ent in task_entities:
        el = ent.lower()
        if "/" in ent:
            return True
        if _CODE_EXT_RE.search(ent):
            return True
        # Check if the entity starts with a known source dir stem
        stem = el.split(".")[0].split("/")[0]
        if stem in _ANCHOR_DIRS:
            return True
    return False


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
    # Pre-compute task specificity once for the run (F4.2).
    anchored = _task_anchored(task_entities)

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

        # ── Rule 3: AUTONOMOUS (two-tier, F4.2) ──────────────────────────────
        #
        # Tier 1 — HIGH_CONSEQUENCE: always flag (push, rm, curl, secret writes…)
        #   even when the task is vague or absent.  These actions are scary enough
        #   that a receipt without them would be a false sense of security.
        #
        # Tier 2 — ordinary consequential: only flag when the task is anchored
        #   (names a concrete file/path/module).  On a vague prose task, every
        #   Edit the agent makes looks "off-task" — that's noise, not signal.
        #   Fall to DERIVED instead (F4.1: also suppressed when task_text is None).
        if consequential and not dec.is_error:
            off_task = not task_entities or not _targets_overlap_task(
                dec.targets, task_entities
            )
            if off_task:
                high = _is_high_consequence(dec)
                if high or (anchored and task_text is not None):
                    dec.provenance = "AUTONOMOUS"
                    target_str = ", ".join(dec.targets[:2]) if dec.targets else dec.tool_name
                    tier = "high-consequence" if high else "off-task"
                    dec.why = (
                        f"Consequential {tier} action ({dec.tool_name}: {target_str}) "
                        f"with no link to task entities — agent acted without being asked."
                    )
                    dec.scope_drift = _scope_drift(
                        dec, task_entities, task_text, consequential
                    )
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
