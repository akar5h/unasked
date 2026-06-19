"""Hand-labeled tests for the provenance classifier (F3).

All fixtures built inline — no real session data, no ~/.claude reads.
Labels verified against the spec in plan file and classifier.py docstring.

Priority axes per brief: TOOL_INDUCED and AUTONOMOUS precision first.
"""

from __future__ import annotations

import pytest

from unasked.classifier import classify, _tokens, _overlap, _extract_resource, _has_scope_drift
from unasked.ir import Decision, Run


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(task: str | None, decisions: list[Decision]) -> Run:
    return Run(run_id="test-run", source="test", task_text=task, decisions=decisions)


def _dec(step: int, tool: str, summary: str, **kwargs) -> Decision:
    return Decision(step_index=step, ts=None, tool_name=tool, tool_args_summary=summary, **kwargs)


# ── Unit tests: token helpers ──────────────────────────────────────────────────

class TestTokenHelpers:
    def test_tokens_lowercased(self):
        assert "read" in _tokens("Read")

    def test_tokens_min_length(self):
        # single-char tokens dropped
        assert "a" not in _tokens("a b ccc")
        assert "ccc" in _tokens("a b ccc")

    def test_tokens_path_splits_on_slash(self):
        # / is now a separator; path components tokenised individually
        toks = _tokens("/src/foo.py")
        assert "src" in toks
        # "foo" (stem of foo.py) also emitted
        assert "foo" in toks or "foo.py" in toks

    def test_overlap_true(self):
        assert _overlap("edit foo.py", "fix auth in foo.py") is True

    def test_overlap_false(self):
        assert _overlap("edit bar.py", "fix auth in foo.py") is False

    def test_overlap_case_insensitive(self):
        assert _overlap("Read Foo.py", "fix foo.py") is True


# ── Unit tests: resource extraction ───────────────────────────────────────────

class TestExtractResource:
    def test_file_path_equals(self):
        assert _extract_resource("file_path=src/foo.py") == "src/foo.py"

    def test_absolute_path(self):
        r = _extract_resource("/Users/me/src/bar.py")
        assert r == "/Users/me/src/bar.py"

    def test_url(self):
        r = _extract_resource("WebFetch https://example.com/path")
        assert r == "https://example.com/path"

    def test_bare_relative_path(self):
        r = _extract_resource("src/utils.py")
        assert r is not None and "utils" in r

    def test_no_resource(self):
        assert _extract_resource("pytest") is None


# ── Unit tests: scope-drift check ─────────────────────────────────────────────

class TestHasScopeDrift:
    def test_no_task_text_never_drifts(self):
        dec = _dec(0, "Read", "file_path=secret/config.yaml")
        assert _has_scope_drift(dec, None) is False

    def test_no_extractable_resource_never_drifts(self):
        dec = _dec(0, "Bash", "pytest")
        assert _has_scope_drift(dec, "run tests") is False

    def test_resource_in_task_no_drift(self):
        dec = _dec(0, "Read", "file_path=auth.py")
        assert _has_scope_drift(dec, "fix auth.py token expiry") is False

    def test_resource_not_in_task_drifts(self):
        dec = _dec(0, "Edit", "file_path=config/db.yaml")
        assert _has_scope_drift(dec, "fix auth token expiry bug") is True

    def test_url_in_task_no_drift(self):
        dec = _dec(0, "WebFetch", "https://docs.example.com/auth")
        assert _has_scope_drift(dec, "read docs.example.com auth page") is False


# ── Integration: classify returns deep copy, doesn't mutate ───────────────────

class TestClassifyPurity:
    def test_returns_new_run(self):
        run = _run("read foo.py", [_dec(0, "Read", "file_path=foo.py")])
        result = classify(run)
        assert result is not run

    def test_original_unchanged(self):
        dec = _dec(0, "Read", "file_path=foo.py")
        run = _run("read foo.py", [dec])
        classify(run)
        assert run.decisions[0].provenance is None
        assert run.decisions[0].why is None

    def test_all_decisions_get_provenance(self):
        run = _run("edit foo.py", [
            _dec(0, "Read", "file_path=foo.py"),
            _dec(1, "Edit", "file_path=foo.py"),
        ])
        result = classify(run)
        for d in result.decisions:
            assert d.provenance is not None
            assert d.why is not None
            assert d.scope_drift is not None

    def test_feedback_override_skips(self):
        """Decision with feedback annotation must not be reclassified."""
        dec = _dec(0, "Bash", "git push origin", feedback="approved-push")
        run = _run("deploy", [dec])
        result = classify(run)
        classified = result.decisions[0]
        assert classified.provenance is None  # not set by classifier
        assert classified.feedback == "approved-push"


# ── Label: REQUESTED ──────────────────────────────────────────────────────────

class TestRequested:
    def test_read_target_in_task(self):
        """Read auth.py when task says 'fix auth.py' → REQUESTED."""
        run = _run(
            "fix auth.py token expiry bug",
            [_dec(0, "Read", "file_path=auth.py")],
        )
        d = classify(run).decisions[0]
        assert d.provenance == "REQUESTED"

    def test_edit_target_in_task(self):
        """Edit auth.py when task explicitly names it → REQUESTED."""
        run = _run(
            "edit auth.py to fix the token expiry",
            [_dec(0, "Edit", "file_path=auth.py")],
        )
        d = classify(run).decisions[0]
        assert d.provenance == "REQUESTED"

    def test_bash_target_in_task(self):
        """Run pytest when task says 'run tests' — 'tests' token overlaps → REQUESTED."""
        run = _run(
            "run the tests",
            [_dec(0, "Bash", "pytest tests/")],
        )
        d = classify(run).decisions[0]
        # 'tests' in summary overlaps with 'tests' in task → REQUESTED
        assert d.provenance == "REQUESTED"

    def test_requested_no_scope_drift(self):
        """Requested action on task-named file must not scope-drift."""
        run = _run(
            "read auth.py",
            [_dec(0, "Read", "file_path=auth.py")],
        )
        d = classify(run).decisions[0]
        assert d.scope_drift is False


# ── Label: DERIVED ─────────────────────────────────────────────────────────────

class TestDerived:
    def test_read_before_edit_same_file(self):
        """Read foo.py then Edit foo.py — edit is a natural follow-on → DERIVED
        (and not TOOL_INDUCED because summary shares tokens with prior AND with task)."""
        run = _run(
            "fix foo.py",
            [
                _dec(0, "Read", "file_path=foo.py"),
                _dec(1, "Edit", "file_path=foo.py"),
            ],
        )
        decisions = classify(run).decisions
        # First is REQUESTED (foo.py in task), second is either REQUESTED or DERIVED
        # (foo.py is in task so REQUESTED wins, but also prior-related → DERIVED not forced).
        # Key: NOT AUTONOMOUS and NOT TOOL_INDUCED.
        assert decisions[1].provenance in ("REQUESTED", "DERIVED")
        assert decisions[1].provenance != "AUTONOMOUS"
        assert decisions[1].provenance != "TOOL_INDUCED"

    def test_no_task_text_defaults_to_derived(self):
        """With no task_text, first decision defaults to DERIVED (no prior, no task)."""
        run = _run(None, [_dec(0, "Read", "file_path=foo.py")])
        d = classify(run).decisions[0]
        assert d.provenance == "DERIVED"

    def test_follow_on_from_prior_step(self):
        """Glob after Read on same directory — follow-on, not autonomous."""
        run = _run(
            "list files in src/",
            [
                _dec(0, "Read", "file_path=src/foo.py"),
                _dec(1, "Glob", "src/**/*.py"),
            ],
        )
        d = classify(run).decisions[1]
        # "src" overlaps task ("src/") → could be REQUESTED; either way not AUTONOMOUS.
        assert d.provenance != "AUTONOMOUS"


# ── Label: AUTONOMOUS ─────────────────────────────────────────────────────────

class TestAutonomous:
    def test_git_push_not_in_task(self):
        """Bash git push when task says 'fix auth token expiry' → AUTONOMOUS.
        No overlap of 'git push origin' with 'fix auth token expiry'.
        No prior step sharing tokens.
        """
        run = _run(
            "fix auth token expiry bug",
            [_dec(0, "Bash", "git push origin main")],
        )
        d = classify(run).decisions[0]
        assert d.provenance == "AUTONOMOUS"

    def test_autonomous_scope_drifts(self):
        """AUTONOMOUS action on file outside task should also scope_drift=True."""
        run = _run(
            "fix auth token expiry bug",
            [_dec(0, "Edit", "file_path=config/db.yaml")],
        )
        d = classify(run).decisions[0]
        assert d.provenance == "AUTONOMOUS"
        assert d.scope_drift is True

    def test_write_unrelated_file(self):
        """Write a file not mentioned in task → AUTONOMOUS."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Write", "file_path=scripts/deploy.sh")],
        )
        d = classify(run).decisions[0]
        assert d.provenance == "AUTONOMOUS"

    def test_send_message_not_in_task(self):
        """SendMessage not implied by task → AUTONOMOUS."""
        run = _run(
            "fix auth.py",
            [_dec(0, "SendMessage", "to=TeamLead summary: done")],
        )
        d = classify(run).decisions[0]
        assert d.provenance == "AUTONOMOUS"

    def test_agent_spawn_not_in_task(self):
        """Agent spawn not implied by task → AUTONOMOUS."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Agent", "subtype=fork deploy-review")],
        )
        d = classify(run).decisions[0]
        assert d.provenance == "AUTONOMOUS"

    def test_write_after_unrelated_prior_stays_autonomous(self):
        """Write to file with no token overlap with task OR prior → still AUTONOMOUS."""
        run = _run(
            "fix auth.py",
            [
                _dec(0, "Read", "file_path=auth.py"),
                _dec(1, "Write", "file_path=billing/invoice.py"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance == "AUTONOMOUS"

    def test_autonomous_has_why(self):
        """AUTONOMOUS decision must always have a non-empty why."""
        run = _run(
            "fix auth.py",
            [_dec(0, "Bash", "git push origin main")],
        )
        d = classify(run).decisions[0]
        assert d.why and len(d.why) > 0

    def test_bash_in_task_not_autonomous(self):
        """Bash command that references task-named file → NOT AUTONOMOUS (REQUESTED)."""
        run = _run(
            "run tests for auth.py",
            [_dec(0, "Bash", "pytest tests/test_auth.py -v")],
        )
        d = classify(run).decisions[0]
        assert d.provenance != "AUTONOMOUS"


# ── Label: TOOL_INDUCED ───────────────────────────────────────────────────────

class TestToolInduced:
    def test_write_after_read_new_target(self):
        """Read returns content about deploy.sh; agent then writes deploy.sh
        without being asked → TOOL_INDUCED (steered by read content).

        Prior: Read file_path=docs/deploy-guide.md
        Current: Write file_path=scripts/deploy.sh
        Shared token: 'deploy' in both summaries.
        Not in task: 'fix auth bug'.
        """
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", "file_path=docs/deploy-guide.md"),
                _dec(1, "Write", "file_path=scripts/deploy.sh"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED"

    def test_bash_after_webfetch_new_target(self):
        """WebFetch returns a page about curl usage; agent then runs curl
        command not in task → TOOL_INDUCED."""
        run = _run(
            "summarise the readme",
            [
                _dec(0, "WebFetch", "https://api.example.com/docs"),
                _dec(1, "Bash", "curl https://api.example.com/token"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED"

    def test_no_token_overlap_with_prior_not_tool_induced(self):
        """Current decision shares NO tokens with prior Read → not TOOL_INDUCED.
        Falls through to AUTONOMOUS (write tool, not in task, not related to prior)."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", "file_path=auth.py"),
                _dec(1, "Write", "file_path=billing/invoice.py"),
            ],
        )
        d = classify(run).decisions[1]
        # No token overlap between 'auth.py' and 'billing/invoice.py' → not TOOL_INDUCED
        assert d.provenance != "TOOL_INDUCED"
        assert d.provenance == "AUTONOMOUS"

    def test_write_to_same_task_file_after_read_not_tool_induced(self):
        """Read auth.py then Edit auth.py — target overlaps task → NOT TOOL_INDUCED.
        (Precision: don't flag routine read-then-edit as TOOL_INDUCED.)
        """
        run = _run(
            "fix auth.py",
            [
                _dec(0, "Read", "file_path=auth.py"),
                _dec(1, "Edit", "file_path=auth.py"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance != "TOOL_INDUCED"

    def test_tool_induced_has_why(self):
        """TOOL_INDUCED decision must always carry a why string."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", "file_path=docs/deploy-guide.md"),
                _dec(1, "Write", "file_path=scripts/deploy.sh"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.why and "Read" in d.why

    def test_tool_induced_after_webSearch(self):
        """WebSearch for term; agent then edits a file matching search topic not in task."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "WebSearch", "query: stripe integration guide"),
                _dec(1, "Edit", "file_path=billing/stripe.py"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED"

    def test_prior_must_be_read_class(self):
        """If prior step is a WRITE tool (not read-class), next step cannot be TOOL_INDUCED
        even if tokens overlap — TOOL_INDUCED requires prior to be read-class."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Edit", "file_path=deploy.sh"),   # WRITE, not READ
                _dec(1, "Bash", "bash deploy.sh"),
            ],
        )
        d = classify(run).decisions[1]
        # Bash deploy.sh not in task "fix auth bug"; prior was Edit (write class).
        # Can't be TOOL_INDUCED. Should be AUTONOMOUS (bash not in task, not prior-related
        # enough because 'deploy' doesn't overlap 'auth bug').
        # Actually 'deploy' IS in prior summary → prior_related=True → not AUTONOMOUS.
        # Falls to REQUESTED (deploy in prior? no, task check). Falls to DERIVED.
        assert d.provenance != "TOOL_INDUCED"


# ── Multi-step run end-to-end ──────────────────────────────────────────────────

class TestMultiStepRun:
    def test_typical_auth_fix_run(self):
        """Realistic 5-step run for 'fix auth module tests and token expiry bug'.

        Task explicitly mentions 'auth', 'tests', 'token' — all steps 0-3
        have at least one token that overlaps with the task.
        Step 4 (git push) has no task overlap → AUTONOMOUS.

        Hand-annotated expected labels:
          0: Read auth.py          → REQUESTED (auth in task)
          1: Read tests/test_auth  → REQUESTED (tests + auth in task)
          2: Edit auth.py          → REQUESTED (auth in task)
          3: Bash pytest tests/    → REQUESTED (tests in task)
          4: Bash git push         → AUTONOMOUS (push not asked, no task/prior overlap)
        """
        run = _run(
            "fix auth module tests and token expiry bug",
            [
                _dec(0, "Read", "file_path=auth.py"),
                _dec(1, "Read", "file_path=tests/test_auth.py"),
                _dec(2, "Edit", "file_path=auth.py"),
                _dec(3, "Bash", "pytest tests/"),
                _dec(4, "Bash", "git push origin main"),
            ],
        )
        classified = classify(run)
        labels = [d.provenance for d in classified.decisions]

        # Steps 0-3: REQUESTED or DERIVED, never AUTONOMOUS or TOOL_INDUCED
        for i in range(4):
            assert labels[i] in ("REQUESTED", "DERIVED"), f"step {i}: got {labels[i]}"
            assert labels[i] != "AUTONOMOUS", f"step {i} falsely flagged AUTONOMOUS"
            assert labels[i] != "TOOL_INDUCED", f"step {i} falsely flagged TOOL_INDUCED"

        # Step 4: AUTONOMOUS (git push not in task, not related to prior Edit auth.py)
        assert labels[4] == "AUTONOMOUS", f"git push should be AUTONOMOUS, got {labels[4]}"

    def test_tool_induced_scenario_in_multi_step(self):
        """Read a runbook doc then curl a URL sharing a token from the doc.

        Step 0: Read docs/stripe-integration.md  (prior, READ-class)
        Step 1: Bash curl https://api.stripe.com/charge  (not in task, shares 'stripe')

        'stripe' appears in prior summary (stripe-integration.md) but NOT in task
        ('summarise the auth docs').  Structural TOOL_INDUCED signal fires.
        """
        run = _run(
            "summarise the auth docs",
            [
                _dec(0, "Read", "file_path=docs/stripe-integration.md"),
                _dec(1, "Bash", "curl https://api.stripe.com/charge"),
            ],
        )
        classified = classify(run)
        d1 = classified.decisions[1]
        # 'stripe' in prior and 'stripe' in current; NOT in task → TOOL_INDUCED
        assert d1.provenance == "TOOL_INDUCED", f"Expected TOOL_INDUCED, got {d1.provenance}"

    def test_scope_drift_on_autonomous_steps_only(self):
        """File-targeting autonomous step scope-drifts; task-file steps do not."""
        run = _run(
            "fix auth.py token expiry bug",
            [
                _dec(0, "Read", "file_path=auth.py"),
                _dec(1, "Edit", "file_path=auth.py"),
                # Autonomous edit to unrelated file → scope_drift=True
                _dec(2, "Edit", "file_path=config/db.yaml"),
            ],
        )
        classified = classify(run)
        # Steps 0+1 target auth.py which is in task — no drift.
        assert classified.decisions[0].scope_drift is False
        assert classified.decisions[1].scope_drift is False
        # Step 2 edits config/db.yaml — not in task → drift True.
        assert classified.decisions[2].scope_drift is True
        assert classified.decisions[2].provenance == "AUTONOMOUS"

    def test_no_task_text_no_scope_drift(self):
        """Without task_text, scope_drift must be False for every decision."""
        run = _run(
            None,
            [
                _dec(0, "Read", "file_path=auth.py"),
                _dec(1, "Edit", "file_path=config/db.yaml"),
                _dec(2, "Bash", "git push origin main"),
            ],
        )
        classified = classify(run)
        for d in classified.decisions:
            assert d.scope_drift is False, f"step {d.step_index} drifted without task"
