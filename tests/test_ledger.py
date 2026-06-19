"""Tests for the SQLite ledger store (src/unasked/ledger.py)."""

import pytest

from unasked.ir import Decision, Run
from unasked.ledger import init_db, list_runs, load_run, save_run


def _make_run(run_id: str = "run-abc") -> Run:
    return Run(
        run_id=run_id,
        source="claude_code",
        task_text="Refactor the auth module",
        started_at="2026-06-18T09:00:00Z",
        decisions=[
            Decision(
                step_index=0,
                ts="2026-06-18T09:00:01Z",
                tool_name="Read",
                tool_args_summary="path=/src/auth.py",
            ),
            Decision(
                step_index=1,
                ts="2026-06-18T09:00:02Z",
                tool_name="Edit",
                tool_args_summary="path=/src/auth.py lines=5-8",
                parent_step_index=0,
                provenance="AUTONOMOUS",
                scope_drift=False,
                why="Agent edited without explicit instruction",
                feedback=None,
            ),
        ],
    )


class TestInitDb:
    def test_creates_db(self, tmp_path):
        db = tmp_path / "test.db"
        conn = init_db(db)
        conn.close()
        assert db.exists()

    def test_idempotent_double_init(self, tmp_path):
        db = tmp_path / "test.db"
        conn = init_db(db)
        conn.close()
        # Should not raise.
        conn2 = init_db(db)
        conn2.close()

    def test_tables_exist(self, tmp_path):
        db = tmp_path / "test.db"
        conn = init_db(db)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "runs" in tables
        assert "decisions" in tables


class TestSaveAndLoad:
    def test_round_trip_run_fields(self, tmp_path):
        db = tmp_path / "test.db"
        run = _make_run()
        save_run(run, db)
        loaded = load_run(run.run_id, db)
        assert loaded is not None
        assert loaded.run_id == run.run_id
        assert loaded.source == run.source
        assert loaded.task_text == run.task_text
        assert loaded.started_at == run.started_at

    def test_round_trip_decision_count(self, tmp_path):
        db = tmp_path / "test.db"
        run = _make_run()
        save_run(run, db)
        loaded = load_run(run.run_id, db)
        assert loaded is not None
        assert len(loaded.decisions) == 2

    def test_round_trip_decision_fields(self, tmp_path):
        db = tmp_path / "test.db"
        run = _make_run()
        save_run(run, db)
        loaded = load_run(run.run_id, db)
        assert loaded is not None
        d = loaded.decisions[1]
        assert d.tool_name == "Edit"
        assert d.parent_step_index == 0
        assert d.provenance == "AUTONOMOUS"
        assert d.scope_drift is False
        assert d.why == "Agent edited without explicit instruction"
        assert d.feedback is None

    def test_load_missing_run_returns_none(self, tmp_path):
        db = tmp_path / "test.db"
        init_db(db).close()
        assert load_run("nonexistent", db) is None

    def test_resave_replaces_decisions(self, tmp_path):
        """Re-saving the same run_id must replace decisions, not duplicate them."""
        db = tmp_path / "test.db"
        run = _make_run()
        save_run(run, db)

        # Modify and re-save.
        run2 = Run(
            run_id=run.run_id,
            source="claude_code",
            task_text=run.task_text,
            started_at=run.started_at,
            decisions=[
                Decision(
                    step_index=0,
                    ts="2026-06-18T09:00:01Z",
                    tool_name="Bash",
                    tool_args_summary="pytest -v",
                )
            ],
        )
        save_run(run2, db)
        loaded = load_run(run.run_id, db)
        assert loaded is not None
        # Must have exactly 1 decision (not 3 = 2 old + 1 new).
        assert len(loaded.decisions) == 1
        assert loaded.decisions[0].tool_name == "Bash"

    def test_scope_drift_true_round_trips(self, tmp_path):
        db = tmp_path / "test.db"
        run = Run(
            run_id="run-drift",
            source="test",
            decisions=[
                Decision(
                    step_index=0,
                    ts=None,
                    tool_name="Write",
                    tool_args_summary="path=/etc/passwd",
                    scope_drift=True,
                )
            ],
        )
        save_run(run, db)
        loaded = load_run("run-drift", db)
        assert loaded is not None
        assert loaded.decisions[0].scope_drift is True


class TestListRuns:
    def test_list_returns_saved_run(self, tmp_path):
        db = tmp_path / "test.db"
        save_run(_make_run("r1"), db)
        runs = list_runs(db)
        assert any(r.run_id == "r1" for r in runs)

    def test_list_decisions_empty(self, tmp_path):
        """list_runs returns summary rows — decisions intentionally not loaded."""
        db = tmp_path / "test.db"
        save_run(_make_run("r1"), db)
        runs = list_runs(db)
        for r in runs:
            assert r.decisions == []

    def test_list_ordered_by_started_at_desc(self, tmp_path):
        db = tmp_path / "test.db"
        r1 = Run(run_id="r1", source="s", started_at="2026-06-18T08:00:00Z")
        r2 = Run(run_id="r2", source="s", started_at="2026-06-18T09:00:00Z")
        save_run(r1, db)
        save_run(r2, db)
        runs = list_runs(db)
        ids = [r.run_id for r in runs]
        assert ids.index("r2") < ids.index("r1")

    def test_list_empty_db(self, tmp_path):
        db = tmp_path / "test.db"
        assert list_runs(db) == []
