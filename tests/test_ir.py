"""Tests for the Decision-Event IR (src/unasked/ir.py)."""

from unasked.ir import Decision, Run


def _make_run() -> Run:
    """Build a Run with three Decisions for use across tests."""
    return Run(
        run_id="run-001",
        source="claude_code",
        task_text="Fix the failing test",
        started_at="2026-06-18T10:00:00Z",
        decisions=[
            Decision(
                step_index=0,
                ts="2026-06-18T10:00:01Z",
                tool_name="Read",
                tool_args_summary="path=/src/foo.py",
            ),
            Decision(
                step_index=1,
                ts="2026-06-18T10:00:02Z",
                tool_name="Edit",
                tool_args_summary="path=/src/foo.py lines=10-12",
                is_error=False,
                parent_step_index=0,
            ),
            Decision(
                step_index=2,
                ts="2026-06-18T10:00:03Z",
                tool_name="Bash",
                tool_args_summary="pytest -v",
                is_error=True,
            ),
        ],
    )


class TestDecisionDefaults:
    def test_is_error_defaults_false(self):
        d = Decision(step_index=0, ts=None, tool_name="Read", tool_args_summary="x")
        assert d.is_error is False

    def test_parent_step_index_defaults_none(self):
        d = Decision(step_index=0, ts=None, tool_name="Read", tool_args_summary="x")
        assert d.parent_step_index is None

    def test_provenance_defaults_none(self):
        d = Decision(step_index=0, ts=None, tool_name="Read", tool_args_summary="x")
        assert d.provenance is None

    def test_scope_drift_defaults_none(self):
        d = Decision(step_index=0, ts=None, tool_name="Read", tool_args_summary="x")
        assert d.scope_drift is None

    def test_why_defaults_none(self):
        d = Decision(step_index=0, ts=None, tool_name="Read", tool_args_summary="x")
        assert d.why is None

    def test_feedback_defaults_none(self):
        d = Decision(step_index=0, ts=None, tool_name="Read", tool_args_summary="x")
        assert d.feedback is None


class TestRunDefaults:
    def test_decisions_defaults_empty_list(self):
        r = Run(run_id="r1", source="test")
        assert r.decisions == []

    def test_task_text_defaults_none(self):
        r = Run(run_id="r1", source="test")
        assert r.task_text is None

    def test_started_at_defaults_none(self):
        r = Run(run_id="r1", source="test")
        assert r.started_at is None


class TestRunWithDecisions:
    def test_decisions_count(self):
        run = _make_run()
        assert len(run.decisions) == 3

    def test_decision_fields_preserved(self):
        run = _make_run()
        bash_step = run.decisions[2]
        assert bash_step.tool_name == "Bash"
        assert bash_step.is_error is True
        assert bash_step.parent_step_index is None

    def test_parent_step_index_set(self):
        run = _make_run()
        edit_step = run.decisions[1]
        assert edit_step.parent_step_index == 0

    def test_run_metadata(self):
        run = _make_run()
        assert run.run_id == "run-001"
        assert run.source == "claude_code"
        assert run.task_text == "Fix the failing test"
