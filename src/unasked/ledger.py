"""SQLite ledger store for unasked.

Persists Run + Decision objects to a local SQLite database.  All operations
are idempotent: re-saving a run replaces its decisions rather than duplicating.

DB path resolution (in priority order):
  1. Explicit ``path`` argument passed to each function.
  2. ``UNASKED_LEDGER_DB`` environment variable.
  3. Default: ``~/.kairos/ledger.db`` (parent dir created if absent).

Pass an explicit ``path`` in tests (e.g. pytest ``tmp_path``) so tests never
touch the real home database.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from unasked.ir import Decision, Run

# ── Path resolution ───────────────────────────────────────────────────────────

_DEFAULT_DB_STEM = Path.home() / ".kairos" / "ledger.db"


def _resolve_path(path: str | Path | None) -> Path:
    """Return the DB path to use, creating parent dirs as needed."""
    if path is not None:
        resolved = Path(path)
    elif "UNASKED_LEDGER_DB" in os.environ:
        resolved = Path(os.environ["UNASKED_LEDGER_DB"])
    else:
        resolved = _DEFAULT_DB_STEM
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    source      TEXT,
    task_text   TEXT,
    started_at  TEXT,
    step_count  INTEGER
);
"""

_DDL_DECISIONS = """
CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    step_index        INTEGER,
    ts                TEXT,
    tool_name         TEXT,
    args_summary      TEXT,
    targets           TEXT,
    result_entities   TEXT,
    parent_step_index INTEGER,
    provenance        TEXT,
    scope_drift       INTEGER,
    why               TEXT,
    feedback          TEXT,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
"""

# Migration statements for existing databases that predate targets/result_entities columns.
_MIGRATIONS = [
    "ALTER TABLE decisions ADD COLUMN targets TEXT",
    "ALTER TABLE decisions ADD COLUMN result_entities TEXT",
]

_DDL_IDX_DECISIONS = """
CREATE INDEX IF NOT EXISTS idx_decisions_run_id ON decisions(run_id);
"""


# ── Public API ────────────────────────────────────────────────────────────────


def init_db(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (or create) the SQLite database and apply schema idempotently.

    Returns an open ``sqlite3.Connection`` with foreign keys enabled.
    Caller is responsible for closing it.
    """
    resolved = _resolve_path(path)
    conn = sqlite3.connect(str(resolved))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(_DDL_RUNS)
    conn.execute(_DDL_DECISIONS)
    conn.execute(_DDL_IDX_DECISIONS)
    # Idempotent migrations for pre-existing databases
    for migration in _MIGRATIONS:
        try:
            conn.execute(migration)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn


def save_run(run: Run, path: str | Path | None = None) -> None:
    """Upsert a Run and replace its decisions idempotently.

    If a run with the same ``run_id`` already exists its ``runs`` row is
    replaced and all existing ``decisions`` rows for that run_id are deleted
    before re-inserting.  This makes re-save safe and deterministic.

    ``args_summary`` is already expected to be redacted by the caller; no
    further scrubbing is applied here.
    """
    conn = init_db(path)
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, source, task_text, started_at, step_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.source,
                    run.task_text,
                    run.started_at,
                    len(run.decisions),
                ),
            )
            # Delete existing decisions for idempotent re-save.
            conn.execute("DELETE FROM decisions WHERE run_id = ?", (run.run_id,))
            conn.executemany(
                """
                INSERT INTO decisions
                    (run_id, step_index, ts, tool_name, args_summary,
                     targets, result_entities,
                     parent_step_index, provenance, scope_drift, why, feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run.run_id,
                        d.step_index,
                        d.ts,
                        d.tool_name,
                        d.tool_args_summary,
                        json.dumps(d.targets),
                        json.dumps(d.result_entities),
                        d.parent_step_index,
                        d.provenance,
                        int(d.scope_drift) if d.scope_drift is not None else None,
                        d.why,
                        d.feedback,
                    )
                    for d in run.decisions
                ],
            )
    finally:
        conn.close()


def load_run(run_id: str, path: str | Path | None = None) -> Run | None:
    """Reconstruct a Run (with all its Decisions) from the database.

    Returns ``None`` when no run with that id exists.
    """
    conn = init_db(path)
    try:
        cur = conn.execute(
            "SELECT run_id, source, task_text, started_at FROM runs WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        rid, source, task_text, started_at = row

        dcur = conn.execute(
            """
            SELECT step_index, ts, tool_name, args_summary,
                   targets, result_entities,
                   parent_step_index, provenance, scope_drift, why, feedback
            FROM decisions
            WHERE run_id = ?
            ORDER BY step_index
            """,
            (run_id,),
        )
        decisions = [
            Decision(
                step_index=r[0],
                ts=r[1],
                tool_name=r[2],
                tool_args_summary=r[3],
                targets=json.loads(r[4] or "[]"),
                result_entities=json.loads(r[5] or "[]"),
                parent_step_index=r[6],
                provenance=r[7],
                scope_drift=bool(r[8]) if r[8] is not None else None,
                why=r[9],
                feedback=r[10],
            )
            for r in dcur.fetchall()
        ]
        return Run(
            run_id=rid,
            source=source,
            task_text=task_text,
            started_at=started_at,
            decisions=decisions,
        )
    finally:
        conn.close()


def list_runs(path: str | Path | None = None) -> list[Run]:
    """Return all runs ordered by started_at descending.

    Decisions are NOT loaded for each run in this summary listing —
    ``decisions`` will be an empty list on every returned ``Run``.
    Call ``load_run`` to get decisions for a specific run.
    """
    conn = init_db(path)
    try:
        cur = conn.execute(
            """
            SELECT run_id, source, task_text, started_at
            FROM runs
            ORDER BY started_at DESC
            """
        )
        return [
            Run(run_id=r[0], source=r[1], task_text=r[2], started_at=r[3])
            for r in cur.fetchall()
        ]
    finally:
        conn.close()
