"""Hand-labeled tests for classify_run (F3 revised spec).

All fixtures built inline — no real ~/.claude reads, no real session data.
Precision-first: tests verify that AUTONOMOUS and TOOL_INDUCED are NOT
falsely triggered (anti-cry-wolf), as well as that they ARE triggered when
the signal is unambiguous.
"""

from __future__ import annotations

import pytest

from unasked.classify import (
    WRITE_TOOLS,
    CONSEQUENTIAL_BASH_VERBS,
    BENIGN_BASH_VERBS,
    classify_run,
    _bash_is_consequential,
)
from unasked.ir import Decision, Run


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(task: str | None, decisions: list[Decision]) -> Run:
    return Run(run_id="test", source="test", task_text=task, decisions=decisions)


def _dec(
    step: int,
    tool: str,
    targets: list[str],
    result_entities: list[str] | None = None,
    is_error: bool = False,
    feedback: str | None = None,
) -> Decision:
    return Decision(
        step_index=step,
        ts=None,
        tool_name=tool,
        tool_args_summary="",
        targets=targets,
        result_entities=result_entities or [],
        is_error=is_error,
        feedback=feedback,
    )


# ── Constants sanity ───────────────────────────────────────────────────────────

class TestConstants:
    def test_write_tools_contains_edit(self):
        assert "Edit" in WRITE_TOOLS

    def test_consequential_bash_contains_rm(self):
        assert "rm" in CONSEQUENTIAL_BASH_VERBS

    def test_benign_bash_contains_pytest(self):
        assert "pytest" in BENIGN_BASH_VERBS


# ── _bash_is_consequential ────────────────────────────────────────────────────

class TestBashIsConsequential:
    def test_git_push_consequential(self):
        assert _bash_is_consequential("git push origin main") is True

    def test_git_commit_consequential(self):
        assert _bash_is_consequential("git commit -m 'fix'") is True

    def test_git_status_not_consequential(self):
        assert _bash_is_consequential("git status") is False

    def test_git_log_not_consequential(self):
        assert _bash_is_consequential("git log --oneline") is False

    def test_git_diff_not_consequential(self):
        assert _bash_is_consequential("git diff HEAD") is False

    def test_rm_consequential(self):
        assert _bash_is_consequential("rm -rf dist/") is True

    def test_curl_consequential(self):
        assert _bash_is_consequential("curl https://api.example.com") is True

    def test_pytest_not_consequential(self):
        assert _bash_is_consequential("pytest tests/") is False

    def test_ls_not_consequential(self):
        assert _bash_is_consequential("ls -la") is False

    def test_npm_publish_consequential(self):
        assert _bash_is_consequential("npm publish") is True

    def test_npm_install_not_consequential(self):
        assert _bash_is_consequential("npm install") is False

    def test_unknown_verb_conservative_true(self):
        # Unknown verb → conservative → True
        assert _bash_is_consequential("xyzzy run something") is True

    def test_empty_command_false(self):
        assert _bash_is_consequential("") is False


# ── TOOL_INDUCED ──────────────────────────────────────────────────────────────

class TestToolInduced:
    def test_webfetch_result_steers_curl(self):
        """WebFetch result contains a URL; next Bash curls THAT url not in task.

        Step 0: WebFetch — result_entities contains https://api.payment.com/charge
        Step 1: Bash curl — targets = [curl, https://api.payment.com/charge]
        Task: 'summarise the docs'  (no mention of payment.com)
        → Step 1 is TOOL_INDUCED
        """
        run = _run(
            "summarise the docs",
            [
                _dec(0, "WebFetch",
                     targets=["https://docs.example.com"],
                     result_entities=["https://api.payment.com/charge"]),
                _dec(1, "Bash",
                     targets=["curl", "https://api.payment.com/charge"]),
            ],
        )
        classify_run(run)
        assert run.decisions[1].provenance == "TOOL_INDUCED"

    def test_tool_induced_why_mentions_entity(self):
        """TOOL_INDUCED decision why must reference the matched entity."""
        run = _run(
            "summarise the docs",
            [
                _dec(0, "WebFetch",
                     targets=["https://docs.example.com"],
                     result_entities=["https://api.payment.com/charge"]),
                _dec(1, "Bash",
                     targets=["curl", "https://api.payment.com/charge"]),
            ],
        )
        classify_run(run)
        d = run.decisions[1]
        assert d.why is not None
        assert "payment.com" in d.why or "api.payment.com" in d.why

    def test_tool_induced_requires_url_or_path_target(self):
        """Generic word overlap does NOT trigger TOOL_INDUCED (precision guard).

        Even if a generic word like 'auth' appears in both result_entities and targets,
        it should NOT trigger TOOL_INDUCED because it isn't URL/path shaped.
        Falls through to AUTONOMOUS (curl is consequential) or REQUESTED/DERIVED.
        """
        run = _run(
            "unrelated task",
            [
                _dec(0, "WebFetch",
                     targets=["https://docs.example.com"],
                     result_entities=["auth"]),   # generic word, not URL/path
                _dec(1, "Bash",
                     targets=["curl", "auth"]),
            ],
        )
        classify_run(run)
        # Should NOT be TOOL_INDUCED because "auth" isn't URL/path shaped
        assert run.decisions[1].provenance != "TOOL_INDUCED"

    def test_no_prior_external_read_no_tool_induced(self):
        """No prior WebFetch/WebSearch → TOOL_INDUCED cannot fire."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Bash",
                     targets=["pytest", "tests/"],
                     result_entities=["https://api.payment.com/charge"]),
                _dec(1, "Bash",
                     targets=["curl", "https://api.payment.com/charge"]),
            ],
        )
        classify_run(run)
        # Prior was Bash (not WebFetch/WebSearch/external Read) → not TOOL_INDUCED
        assert run.decisions[1].provenance != "TOOL_INDUCED"


# ── AUTONOMOUS ────────────────────────────────────────────────────────────────

class TestAutonomous:
    def test_git_push_not_in_task(self):
        """Task: 'fix auth bug'. Bash git push → AUTONOMOUS."""
        run = _run(
            "fix auth bug",
            [_dec(0, "Bash", targets=["git", "push", "origin", "main"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance == "AUTONOMOUS"

    def test_edit_unrelated_file(self):
        """Task: 'fix auth.py'. Edit config/db.yaml → AUTONOMOUS + scope_drift."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Edit", targets=["config/db.yaml"])],
        )
        classify_run(run)
        d = run.decisions[0]
        assert d.provenance == "AUTONOMOUS"
        assert d.scope_drift is True

    def test_autonomous_why_not_empty(self):
        run = _run(
            "fix auth bug",
            [_dec(0, "Bash", targets=["git", "push", "origin", "main"])],
        )
        classify_run(run)
        assert run.decisions[0].why is not None
        assert len(run.decisions[0].why) > 0

    def test_error_decision_not_autonomous(self):
        """Errored actions should NOT be flagged AUTONOMOUS (less alarming)."""
        run = _run(
            "fix auth bug",
            [_dec(0, "Bash", targets=["git", "push", "origin", "main"],
                  is_error=True)],
        )
        classify_run(run)
        assert run.decisions[0].provenance != "AUTONOMOUS"

    def test_write_unrelated_file_autonomous(self):
        run = _run(
            "fix auth.py",
            [_dec(0, "Write", targets=["scripts/deploy.sh"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance == "AUTONOMOUS"

    def test_autonomous_curl_not_in_task(self):
        """Bash curl to external URL not in task (no prior external read) → AUTONOMOUS."""
        run = _run(
            "fix auth bug",
            [_dec(0, "Bash", targets=["curl", "https://api.payment.com/charge"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance == "AUTONOMOUS"


# ── REQUESTED ─────────────────────────────────────────────────────────────────

class TestRequested:
    def test_edit_task_file(self):
        """Task: 'fix auth.py'. Edit auth.py → REQUESTED."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Edit", targets=["auth.py"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance == "REQUESTED"

    def test_requested_not_autonomous(self):
        """Edit task file must NOT be flagged AUTONOMOUS."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Edit", targets=["auth.py"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance != "AUTONOMOUS"

    def test_read_task_file_requested(self):
        """Task: 'fix auth.py'. Read auth.py → REQUESTED (targets overlap task)."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Read", targets=["auth.py"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance == "REQUESTED"


# ── DERIVED / anti-cry-wolf ───────────────────────────────────────────────────

class TestDerivedAntiCryWolf:
    def test_read_task_file_not_autonomous(self):
        """Anti-cry-wolf: Read the task file → NOT AUTONOMOUS."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Read", targets=["auth.py"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance != "AUTONOMOUS"

    def test_pytest_not_autonomous(self):
        """Anti-cry-wolf: Bash pytest tests/ when task is 'run tests' → NOT AUTONOMOUS."""
        run = _run(
            "run the tests",
            [_dec(0, "Bash", targets=["pytest", "tests/"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance != "AUTONOMOUS"

    def test_non_consequential_read_unrelated_not_autonomous(self):
        """Read a file not in task → not AUTONOMOUS (reads are never consequential)."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Read", targets=["config/db.yaml"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance != "AUTONOMOUS"

    def test_derived_has_why(self):
        run = _run(
            None,
            [_dec(0, "Read", targets=["auth.py"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance == "DERIVED"
        assert run.decisions[0].why is not None

    def test_git_status_not_autonomous(self):
        """git status is benign — NOT AUTONOMOUS even if task doesn't mention it."""
        run = _run(
            "fix auth bug",
            [_dec(0, "Bash", targets=["git", "status"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance != "AUTONOMOUS"

    def test_ls_not_autonomous(self):
        run = _run(
            "fix auth bug",
            [_dec(0, "Bash", targets=["ls", "-la"])],
        )
        classify_run(run)
        assert run.decisions[0].provenance != "AUTONOMOUS"


# ── scope_drift ───────────────────────────────────────────────────────────────

class TestScopeDrift:
    def test_no_task_text_scope_drift_is_none(self):
        """scope_drift must be None for ALL decisions when task_text absent."""
        run = _run(
            None,
            [
                _dec(0, "Edit", targets=["config/db.yaml"]),
                _dec(1, "Bash", targets=["git", "push", "origin"]),
            ],
        )
        classify_run(run)
        for d in run.decisions:
            assert d.scope_drift is None, f"step {d.step_index} scope_drift={d.scope_drift}"

    def test_read_non_consequential_scope_drift_false(self):
        """Read is not consequential → scope_drift False even if file outside task."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Read", targets=["config/db.yaml"])],
        )
        classify_run(run)
        assert run.decisions[0].scope_drift is False

    def test_edit_task_file_scope_drift_false(self):
        """Edit the task file → scope_drift False (in scope)."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Edit", targets=["auth.py"])],
        )
        classify_run(run)
        assert run.decisions[0].scope_drift is False

    def test_edit_unrelated_file_scope_drift_true(self):
        """Edit a file clearly outside task scope → scope_drift True."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Edit", targets=["config/db.yaml"])],
        )
        classify_run(run)
        assert run.decisions[0].scope_drift is True


# ── Feedback override ─────────────────────────────────────────────────────────

class TestFeedbackOverride:
    def test_feedback_skips_classification(self):
        """Decision with feedback set must NOT be reclassified."""
        dec = _dec(0, "Bash", targets=["git", "push"], feedback="approved")
        run = _run("fix auth bug", [dec])
        classify_run(run)
        classified = run.decisions[0]
        assert classified.provenance is None
        assert classified.feedback == "approved"

    def test_only_feedback_decisions_skipped(self):
        """Other decisions in the same run ARE classified."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Bash", targets=["git", "push"], feedback="approved"),
                _dec(1, "Bash", targets=["git", "push"]),  # no feedback
            ],
        )
        classify_run(run)
        assert run.decisions[0].provenance is None  # skipped
        assert run.decisions[1].provenance is not None  # classified


# ── Multi-step integration ────────────────────────────────────────────────────

class TestMultiStepIntegration:
    def test_typical_auth_fix_run(self):
        """5-step realistic run. Steps 0-3 routine; step 4 (git push) AUTONOMOUS."""
        run = _run(
            "fix auth.py token expiry bug",
            [
                _dec(0, "Read",  targets=["auth.py"]),
                _dec(1, "Read",  targets=["tests/test_auth.py"]),
                _dec(2, "Edit",  targets=["auth.py"]),
                _dec(3, "Bash",  targets=["pytest", "tests/"]),
                _dec(4, "Bash",  targets=["git", "push", "origin", "main"]),
            ],
        )
        classify_run(run)
        labels = [d.provenance for d in run.decisions]
        # Steps 0-3: not AUTONOMOUS
        for i in range(4):
            assert labels[i] != "AUTONOMOUS", f"step {i} falsely AUTONOMOUS"
        # Step 4: AUTONOMOUS
        assert labels[4] == "AUTONOMOUS"

    def test_classify_run_returns_same_run(self):
        """classify_run must return the same Run object (in-place mutation)."""
        run = _run("fix auth.py", [_dec(0, "Read", targets=["auth.py"])])
        result = classify_run(run)
        assert result is run

    def test_all_decisions_get_provenance(self):
        run = _run(
            "fix auth.py",
            [
                _dec(0, "Read",  targets=["auth.py"]),
                _dec(1, "Edit",  targets=["auth.py"]),
                _dec(2, "Bash",  targets=["git", "push"]),
            ],
        )
        classify_run(run)
        for d in run.decisions:
            assert d.provenance is not None
            assert d.why is not None
            assert d.scope_drift is not None  # task_text present
