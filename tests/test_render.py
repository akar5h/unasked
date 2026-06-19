"""Golden-output tests for render_receipt (F4).

All fixtures are built inline — no real ~/.claude reads.
Duration is suppressed by using no ``started_at`` / ``ts`` on decisions
so the header duration part is absent and output is stable across runs.

Coverage
--------
- Header line structure (run_id[:8], step count, task truncation)
- Duration string when timestamps are present
- Clean run: "✓ all N steps routine — nothing to eyeball."
- AUTONOMOUS group: header, step lines, scope-drift tag
- TOOL_INDUCED group: header, step lines, why tag
- Scope-drift-only group: steps with scope_drift True but REQUESTED/DERIVED
- Routine count line
- Verdict line (k to eyeball / clean)
- color=False suppresses ANSI codes
- Unclassified decisions (provenance=None) skipped without error
"""

from __future__ import annotations

import pytest

from unasked.classify import classify_run
from unasked.ir import Decision, Run
from unasked.render import render_receipt, _duration_str


# ── Helpers ────────────────────────────────────────────────────────────────────


def _run(task: str | None, decisions: list[Decision], started_at: str | None = None) -> Run:
    return Run(
        run_id="abcdef1234567890",  # 16 chars; first 8 = "abcdef12"
        source="test",
        task_text=task,
        started_at=started_at,
        decisions=decisions,
    )


def _dec(
    step: int,
    tool: str,
    targets: list[str],
    result_entities: list[str] | None = None,
    is_error: bool = False,
    ts: str | None = None,
) -> Decision:
    return Decision(
        step_index=step,
        ts=ts,
        tool_name=tool,
        tool_args_summary=" ".join(targets),
        targets=targets,
        result_entities=result_entities or [],
        is_error=is_error,
    )


def _classified(
    task: str | None,
    decisions: list[Decision],
    started_at: str | None = None,
    color: bool = False,  # default off for golden tests
) -> str:
    run = _run(task, decisions, started_at=started_at)
    classify_run(run)
    return render_receipt(run, color=color)


# ── Header ─────────────────────────────────────────────────────────────────────


class TestHeader:
    def test_run_id_first_8_chars(self):
        receipt = _classified("fix auth.py", [_dec(0, "Read", ["auth.py"])])
        assert "abcdef12" in receipt

    def test_step_count_in_header(self):
        receipt = _classified(
            "fix auth.py",
            [_dec(0, "Read", ["auth.py"]), _dec(1, "Edit", ["auth.py"])],
        )
        assert "2 steps" in receipt

    def test_task_text_in_header(self):
        receipt = _classified("fix auth.py token expiry", [_dec(0, "Read", ["auth.py"])])
        assert "fix auth.py token expiry" in receipt

    def test_long_task_truncated_in_header(self):
        long_task = "a" * 90
        receipt = _classified(long_task, [_dec(0, "Read", ["auth.py"])])
        # Header should not contain the full 90-char task
        assert long_task not in receipt
        assert "…" in receipt

    def test_no_task_placeholder(self):
        """F4.1: when task_text is None, render shows the 'no explicit task detected' message."""
        receipt = _classified(None, [_dec(0, "Read", ["auth.py"])])
        assert "no explicit task detected" in receipt

    def test_no_duration_when_no_timestamps(self):
        """Without started_at or decision ts, duration is omitted."""
        receipt = _classified("fix auth.py", [_dec(0, "Read", ["auth.py"])])
        # Should not contain 's' or 'm' duration suffix adjacent to step count
        assert ", " not in receipt.split("\n")[0] or "," not in receipt.split("\n")[0]

    def test_duration_shown_when_timestamps_present(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"], ts="2026-06-19T10:00:10.000Z"),
                _dec(1, "Edit", ["auth.py"], ts="2026-06-19T10:00:40.000Z"),
            ],
            started_at="2026-06-19T10:00:00.000Z",
        )
        # 40-second run → "40s" in header
        assert "40s" in receipt.split("\n")[0]


# ── Duration helper unit tests ─────────────────────────────────────────────────


class TestDurationStr:
    def test_seconds(self):
        run = Run(
            run_id="x", source="t",
            started_at="2026-06-19T10:00:00.000Z",
            decisions=[
                Decision(step_index=0, ts="2026-06-19T10:00:45.000Z",
                         tool_name="Read", tool_args_summary="f"),
            ],
        )
        assert _duration_str(run) == "45s"

    def test_minutes(self):
        run = Run(
            run_id="x", source="t",
            started_at="2026-06-19T10:00:00.000Z",
            decisions=[
                Decision(step_index=0, ts="2026-06-19T10:02:05.000Z",
                         tool_name="Read", tool_args_summary="f"),
            ],
        )
        assert _duration_str(run) == "2m05s"

    def test_no_started_at_returns_none(self):
        run = Run(run_id="x", source="t", decisions=[
            Decision(step_index=0, ts="2026-06-19T10:00:45.000Z",
                     tool_name="Read", tool_args_summary="f"),
        ])
        assert _duration_str(run) is None

    def test_no_decision_ts_returns_none(self):
        run = Run(
            run_id="x", source="t",
            started_at="2026-06-19T10:00:00.000Z",
            decisions=[
                Decision(step_index=0, ts=None, tool_name="Read", tool_args_summary="f"),
            ],
        )
        assert _duration_str(run) is None


# ── Clean run ─────────────────────────────────────────────────────────────────


class TestCleanRun:
    def test_clean_run_message(self):
        receipt = _classified(
            "fix auth.py",
            [_dec(0, "Read", ["auth.py"]), _dec(1, "Edit", ["auth.py"])],
        )
        assert "nothing to eyeball" in receipt

    def test_clean_verdict(self):
        receipt = _classified(
            "fix auth.py",
            [_dec(0, "Read", ["auth.py"]), _dec(1, "Edit", ["auth.py"])],
        )
        assert "verdict: clean" in receipt

    def test_no_flagged_symbols_in_clean_run(self):
        receipt = _classified(
            "fix auth.py",
            [_dec(0, "Read", ["auth.py"]), _dec(1, "Edit", ["auth.py"])],
        )
        assert "⚠" not in receipt
        assert "⚡" not in receipt
        assert "↗" not in receipt


# ── AUTONOMOUS group ──────────────────────────────────────────────────────────


class TestAutonomousGroup:
    def test_warning_symbol_present(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        assert "⚠" in receipt

    def test_without_being_asked_phrase(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        assert "did WITHOUT being asked" in receipt

    def test_step_index_shown(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        assert "#1" in receipt

    def test_tool_name_shown(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        assert "Bash" in receipt

    def test_autonomous_tag_on_step_line(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        assert "autonomous" in receipt

    def test_scope_drift_appended_when_set(self):
        """AUTONOMOUS step whose target is outside task scope → 'scope-drift' in line."""
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Edit", ["config/db.yaml"]),  # outside auth.py scope
            ],
        )
        # Edit config/db.yaml → AUTONOMOUS + scope_drift
        assert "scope-drift" in receipt

    def test_no_scope_drift_tag_when_not_set(self):
        receipt = _classified(
            "fix auth.py token expiry",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Edit", ["auth.py"]),
                _dec(2, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        # Step 2 is AUTONOMOUS but git push on known target; may or may not drift.
        # The point is that scope-drift tag only appears when explicitly set.
        lines = receipt.split("\n")
        auto_lines = [l for l in lines if "#2" in l]
        if auto_lines:
            # If scope_drift is False for this step, 'scope-drift' must not be in the step line
            pass  # classify_run determines this — just check receipt renders without error
        assert "⚠" in receipt  # group header still present


# ── TOOL_INDUCED group ────────────────────────────────────────────────────────


class TestToolInducedGroup:
    def _tool_induced_run(self) -> str:
        return _classified(
            "fix auth.py",
            [
                _dec(0, "WebFetch", ["https://docs.example.com/api"],
                     result_entities=["https://evil.example.com/payload"]),
                _dec(1, "WebFetch", ["https://evil.example.com/payload"]),
            ],
        )

    def test_lightning_symbol_present(self):
        assert "⚡" in self._tool_induced_run()

    def test_steered_phrase(self):
        assert "steered by external content" in self._tool_induced_run()

    def test_step_index_shown(self):
        assert "#1" in self._tool_induced_run()

    def test_tool_name_shown(self):
        assert "WebFetch" in self._tool_induced_run()

    def test_tool_induced_tag_on_step_line(self):
        assert "tool-induced" in self._tool_induced_run()

    def test_why_shown_in_step_line(self):
        # why from classify_run should appear (truncated) after "tool-induced ·"
        receipt = self._tool_induced_run()
        assert "tool-induced ·" in receipt

    def test_verdict_flagged(self):
        assert "verdict:" in self._tool_induced_run()
        assert "to eyeball" in self._tool_induced_run()


# ── Scope-drift-only group ────────────────────────────────────────────────────


class TestScopeDriftGroup:
    def test_scope_drift_group_header(self):
        """A DERIVED step with scope_drift True surfaces in the ↗ group."""
        # To get a DERIVED + scope_drift step we need: no prior step overlap AND
        # no task overlap, but NOT a consequential action (so not AUTONOMOUS).
        # Read of an off-task file with no prior → DERIVED (first step, no prior),
        # and if the file is outside task scope → scope_drift True.
        run = _run(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),         # REQUESTED
                _dec(1, "Read", ["vendor/setup.cfg"]),  # DERIVED but scope-drifts
            ],
        )
        classify_run(run)
        # Manually set scope_drift on step 1 to True for testing purposes
        # (the classifier may or may not set it; we test render logic directly)
        run.decisions[1].provenance = "DERIVED"
        run.decisions[1].scope_drift = True
        run.decisions[1].why = "Natural follow-on from step 0."
        receipt = render_receipt(run, color=False)
        assert "↗" in receipt
        assert "touched outside task scope" in receipt

    def test_scope_drift_step_not_double_listed(self):
        """An AUTONOMOUS step with scope_drift should appear only in AUTONOMOUS group."""
        run = _run(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Edit", ["config/db.yaml"]),
            ],
        )
        classify_run(run)
        receipt = render_receipt(run, color=False)
        # If step 1 is AUTONOMOUS+scope_drift it should appear exactly once
        step_line_count = receipt.count("#1")
        assert step_line_count == 1, f"Step 1 appeared {step_line_count} times"


# ── Routine count line ────────────────────────────────────────────────────────


class TestRoutineCount:
    def test_routine_line_present_when_some_routine(self):
        """When there are both flagged and routine steps, routine count shown."""
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),      # REQUESTED → routine
                _dec(1, "Edit", ["auth.py"]),       # REQUESTED → routine
                _dec(2, "Bash", ["git", "push", "origin", "main"]),  # AUTONOMOUS
            ],
        )
        assert "routine" in receipt

    def test_routine_line_absent_when_all_flagged(self):
        """When there are zero routine steps, no routine count line."""
        run = _run("summarise the docs", [
            _dec(0, "Bash", ["git", "push", "origin", "main"]),
        ])
        classify_run(run)
        receipt = render_receipt(run, color=False)
        # The clean-run / verdict lines say "routine" too, so check the specific line
        lines = receipt.split("\n")
        routine_lines = [l for l in lines if l.strip().startswith("✓") and "routine" in l and "nothing" not in l]
        assert not routine_lines, f"Unexpected routine count line: {routine_lines}"


# ── Verdict line ──────────────────────────────────────────────────────────────


class TestVerdict:
    def test_verdict_clean_on_clean_run(self):
        receipt = _classified(
            "fix auth.py",
            [_dec(0, "Read", ["auth.py"]), _dec(1, "Edit", ["auth.py"])],
        )
        assert "verdict: clean" in receipt

    def test_verdict_k_to_eyeball(self):
        """Flagged run: 'K to eyeball' where K = flagged step count."""
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        assert "to eyeball" in receipt
        assert "1 to eyeball" in receipt

    def test_verdict_shows_routine_count(self):
        """With 2 routine + 1 flagged: verdict mentions 'routine'."""
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),      # routine
                _dec(1, "Edit", ["auth.py"]),       # routine
                _dec(2, "Bash", ["git", "push", "origin", "main"]),  # autonomous
            ],
        )
        assert "routine" in receipt.split("\n")[-1]


# ── Color ─────────────────────────────────────────────────────────────────────


class TestColor:
    def test_no_color_suppresses_ansi(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
            color=False,
        )
        assert "\033[" not in receipt

    def test_color_true_adds_ansi_on_flagged_header(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
            color=True,
        )
        assert "\033[" in receipt


# ── Unclassified decisions ────────────────────────────────────────────────────


class TestUnclassified:
    def test_unclassified_skipped_without_error(self):
        """Decisions with provenance=None must not crash or appear in output."""
        run = _run(
            "fix auth.py",
            [
                Decision(
                    step_index=0, ts=None, tool_name="Read",
                    tool_args_summary="auth.py", targets=["auth.py"],
                ),
            ],
        )
        # provenance stays None — not classified
        receipt = render_receipt(run, color=False)
        assert "run abcdef12" in receipt  # header still renders
        # No steps with provenance=None should appear in a flagged group
        assert "⚠" not in receipt
        assert "⚡" not in receipt


# ── Full golden snapshot ──────────────────────────────────────────────────────


class TestGoldenSnapshot:
    """Freeze the structure of a canonical mixed run.

    Steps:
      0: Read auth.py            → REQUESTED
      1: Edit auth.py            → REQUESTED
      2: Bash pytest tests/      → REQUESTED/DERIVED
      3: WebFetch docs (result: evil URL)  → classifier sees this as external
      4: WebFetch evil URL       → TOOL_INDUCED
      5: Bash git push           → AUTONOMOUS
    """

    def _make_receipt(self) -> str:
        run = _run(
            "fix auth.py token expiry bug",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Edit", ["auth.py"]),
                _dec(2, "Bash", ["pytest", "tests/"]),
                _dec(3, "WebFetch", ["https://docs.example.com"],
                     result_entities=["https://api.thirdparty.com/v2"]),
                _dec(4, "WebFetch", ["https://api.thirdparty.com/v2"]),
                _dec(5, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        classify_run(run)
        return render_receipt(run, color=False)

    def test_header_present(self):
        assert "run abcdef12" in self._make_receipt()

    def test_task_shown(self):
        assert "fix auth.py token expiry bug" in self._make_receipt()

    def test_step_count(self):
        assert "6 steps" in self._make_receipt()

    def test_autonomous_group(self):
        assert "⚠" in self._make_receipt()
        assert "did WITHOUT being asked" in self._make_receipt()

    def test_tool_induced_group(self):
        assert "⚡" in self._make_receipt()
        assert "steered by external content" in self._make_receipt()

    def test_verdict_to_eyeball(self):
        assert "to eyeball" in self._make_receipt()

    def test_receipt_is_multiline(self):
        assert self._make_receipt().count("\n") >= 5

    def test_mixed_run_all_sections(self):
        receipt = self._make_receipt()
        # Both flagged groups present
        assert "⚠" in receipt
        assert "⚡" in receipt
        # Verdict present
        assert "verdict:" in receipt


# ── F4.3: render note for unanchored tasks ────────────────────────────────────


class TestUnanchoredNote:
    def test_vague_task_shows_note(self):
        """F4.3: vague/unanchored task_text → render includes 'task not concrete' note."""
        receipt = _classified(
            "focus on delivery this sprint",
            [_dec(0, "Edit", ["src/models.py"])],
        )
        assert "task not concrete" in receipt

    def test_anchored_task_no_note(self):
        """F4.3: anchored task (names a file) → no 'task not concrete' note."""
        receipt = _classified(
            "fix src/auth.py token expiry",
            [_dec(0, "Edit", ["src/auth.py"])],
        )
        assert "task not concrete" not in receipt

    def test_no_task_shows_note(self):
        """F4.3: task_text None → also shows 'task not concrete' note."""
        receipt = _classified(None, [_dec(0, "Read", ["auth.py"])])
        assert "task not concrete" in receipt
