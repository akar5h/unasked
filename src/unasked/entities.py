"""Entity extraction helpers for unasked provenance classification.

Three pure functions, stdlib only:

  extract_targets(tool_name, tool_input) -> list[str]
      Identifying entities the action operates on, redacted.

  extract_result_entities(text) -> list[str]
      URLs, domains, file paths extracted from raw tool_result content.
      Call with the result text, then DISCARD the raw text — never store it.

  extract_task_entities(task_text) -> list[str]
      File paths, module/dir names, salient nouns from task_text.

All returned values are run through redact() — no secrets leak.
"""

from __future__ import annotations

import re
from typing import Any

from unasked.redact import redact

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_TARGETS = 5
_MAX_RESULT_ENTITIES = 20
_MAX_ENTITY_LEN = 120

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "from", "into", "then",
        "when", "what", "have", "will", "your", "about", "which", "there",
        "where", "their", "been", "than", "make", "also", "more", "fix",
        "add", "edit", "update", "using", "just", "some", "each", "such",
        "its", "our", "all", "any", "out", "but", "not", "are", "was",
        "were", "has", "had", "did", "can", "may", "use", "get", "set",
        "run", "new",
    }
)

# Context verbs — words immediately after these may be task entities
_CONTEXT_VERBS: frozenset[str] = frozenset(
    {"in", "from", "to", "edit", "fix", "add", "update", "the", "for",
     "of", "at", "on", "change", "modify", "read", "write", "create",
     "delete", "remove", "rename", "move", "refactor", "implement"}
)

# ── Regex patterns ────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s\"'<>]{4,}")
_ABS_PATH_RE = re.compile(r"(?:^|[\s\"'])(/[^\s\"'<>]{2,})")
_REL_PATH_RE = re.compile(r"(?:^|[\s\"'])([a-zA-Z][a-zA-Z0-9_\-]*/[^\s\"'<>]{2,})")
_DOMAIN_RE = re.compile(r"\b([a-zA-Z0-9][a-zA-Z0-9\-]*\.[a-zA-Z]{2,6})\b")
_FILE_TOKEN_RE = re.compile(r"\S+\.\w{1,10}")
_SLASH_TOKEN_RE = re.compile(r"\S*/\S+")
_COMMON_DOMAIN_WORDS: frozenset[str] = frozenset(
    {"e.g", "i.e", "etc", "vs", "ie", "eg", "fig", "pp", "no"}
)


# ── extract_targets ───────────────────────────────────────────────────────────


def extract_targets(tool_name: str, tool_input: dict[str, Any]) -> list[str]:
    """Identifying entities the action operates on, redacted.

    Returns at most _MAX_TARGETS elements.  All values go through redact().
    Never includes file contents or raw blobs.
    """
    results: list[str] = []

    if tool_name == "Bash":
        cmd = str(tool_input.get("command", "")).strip()
        first_line = cmd.split("\n")[0].strip()
        parts = first_line.split()
        # Collect verb + meaningful following tokens (URLs, paths, subcommands)
        for part in parts:
            if len(results) >= _MAX_TARGETS:
                break
            cleaned = redact(part)
            if cleaned:
                results.append(cleaned)
        return results[:_MAX_TARGETS]

    if tool_name == "Read":
        fp = str(tool_input.get("file_path", "")).strip()
        if fp:
            results.append(redact(fp))
        return results

    if tool_name in ("Write", "Edit", "NotebookEdit"):
        fp = str(tool_input.get("file_path", "")).strip()
        if fp:
            results.append(redact(fp))
        return results

    if tool_name == "WebFetch":
        url = str(tool_input.get("url", "")).strip()
        if url:
            results.append(redact(url))
        return results

    if tool_name == "WebSearch":
        query = str(tool_input.get("query", "")).strip()
        words = [w for w in query.split() if len(w) >= 3]
        for w in words[:_MAX_TARGETS]:
            results.append(redact(w))
        return results

    if tool_name == "Agent":
        desc = str(tool_input.get("description", "")).strip()
        words = [w for w in desc.split() if len(w) >= 3]
        for w in words[:_MAX_TARGETS]:
            results.append(redact(w))
        return results

    if tool_name == "SendMessage":
        to = str(tool_input.get("to", "")).strip()
        if to:
            results.append(redact(to))
        return results

    if tool_name == "TaskCreate":
        subject = str(tool_input.get("subject", tool_input.get("description", ""))).strip()
        words = [w for w in subject.split() if len(w) >= 3]
        for w in words[:_MAX_TARGETS]:
            results.append(redact(w))
        return results

    # Generic fallback: first non-blob string values
    for k, v in list(tool_input.items())[:_MAX_TARGETS]:
        v_str = str(v)
        if len(v_str) > 300:
            continue  # skip blobs
        results.append(redact(_trunc(v_str, 120)))
    return results[:_MAX_TARGETS]


# ── extract_result_entities ───────────────────────────────────────────────────


def extract_result_entities(text: str) -> list[str]:
    """URLs, domains, and file paths extracted from tool_result content text.

    Call this with the raw result text, then DISCARD the raw text.
    Returns at most _MAX_RESULT_ENTITIES elements, deduplicated, redacted.
    """
    if not text:
        return []

    seen: set[str] = set()
    results: list[str] = []

    def _add(val: str) -> None:
        v = val.strip()
        if len(v) > _MAX_ENTITY_LEN:
            v = v[:_MAX_ENTITY_LEN]
        cleaned = redact(v)
        if cleaned and cleaned not in seen and len(results) < _MAX_RESULT_ENTITIES:
            seen.add(cleaned)
            results.append(cleaned)

    # URLs first (highest signal)
    for m in _URL_RE.finditer(text):
        _add(m.group(0))

    # Absolute paths
    for m in _ABS_PATH_RE.finditer(text):
        candidate = m.group(1)
        if len(candidate) > 2 and not candidate.endswith((".jpg", ".png", ".gif")):
            _add(candidate)

    # Relative paths with slashes (src/foo, tests/bar, etc.)
    for m in _REL_PATH_RE.finditer(text):
        candidate = m.group(1)
        if len(candidate) > 3:
            _add(candidate)

    # Domain names (non-URL context) — skip common words
    for m in _DOMAIN_RE.finditer(text):
        domain = m.group(1).lower()
        if domain not in _COMMON_DOMAIN_WORDS and "." in domain:
            _add(domain)

    return results[:_MAX_RESULT_ENTITIES]


# ── extract_task_entities ─────────────────────────────────────────────────────


def extract_task_entities(task_text: str) -> list[str]:
    """File paths, module/dir names, salient nouns from task_text.

    Returns deduplicated, redacted list.
    """
    if not task_text:
        return []

    seen: set[str] = set()
    results: list[str] = []

    def _add(val: str) -> None:
        cleaned = redact(val.strip())
        if cleaned and cleaned.lower() not in _STOPWORDS and cleaned not in seen:
            seen.add(cleaned)
            results.append(cleaned)

    # File path tokens (contain dot with extension or slash)
    for m in _FILE_TOKEN_RE.finditer(task_text):
        _add(m.group(0))
    for m in _SLASH_TOKEN_RE.finditer(task_text):
        _add(m.group(0))

    # Words after context verbs
    words = task_text.split()
    for i, word in enumerate(words):
        if word.lower().rstrip(",:") in _CONTEXT_VERBS and i + 1 < len(words):
            candidate = words[i + 1].strip(".,;:\"'")
            if len(candidate) >= 3 and candidate.lower() not in _STOPWORDS:
                _add(candidate)

    # Salient nouns: words >= 4 chars not in stopwords
    for word in words:
        w = word.strip(".,;:\"'()[]")
        if len(w) >= 4 and w.lower() not in _STOPWORDS and not w.startswith("-"):
            _add(w)

    return results


# ── Internal helpers ──────────────────────────────────────────────────────────


def _trunc(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[:n] + "…"
