"""Tests for the native CC transcript reader (load_session) and spool reader.

Synthetic fixtures only — no real ~/.claude session data is committed.
Fixtures are built inline via tmp_path so the real schema fields
(type, uuid, parentUuid, isMeta, timestamp, message.content with tool_use
and tool_result blocks) are exercised directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unasked.adapters.claude_code import (
    ingest_session,
    load_session,
    read_session_from_spool,
)
from unasked.ir import Run

_FIXTURES = Path(__file__).parent / "fixtures"


# ─────────────────────────────────────────────────────────────────────────────
# Transcript builder helpers
# ─────────────────────────────────────────────────────────────────────────────


def _user(content, *, uuid="u1", parent=None, ts="2026-06-18T08:00:00.000Z",
          sid="test-native-001", is_meta=False) -> dict:
    rec = {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": ts,
        "sessionId": sid,
        "message": {"role": "user", "content": content},
    }
    if is_meta:
        rec["isMeta"] = True
    return rec


def _assistant(tool_uses: list[dict], *, uuid="a1", parent="u1",
               ts="2026-06-18T08:00:01.000Z", sid="test-native-001") -> dict:
    content = [
        {"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": tu.get("input", {})}
        for tu in tool_uses
    ]
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": ts,
        "sessionId": sid,
        "message": {"role": "assistant", "content": content},
    }


def _tool_results(results: list[dict], *, uuid="u2", parent="a1",
                  ts="2026-06-18T08:00:02.000Z", sid="test-native-001") -> dict:
    content = [
        {
            "type": "tool_result",
            "tool_use_id": r["tool_use_id"],
            "content": r.get("content", "ok"),
            **({"is_error": True} if r.get("is_error") else {}),
        }
        for r in results
    ]
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": ts,
        "sessionId": sid,
        "message": {"role": "user", "content": content},
    }


def _write_transcript(tmp_path: Path, name: str, lines: list[dict]) -> Path:
    p = tmp_path / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# task_text extraction: isMeta and tool_result-only lines are skipped
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskTextExtraction:
    def test_first_real_user_msg_is_task(self, tmp_path):
        """task_text comes from the first non-isMeta, non-tool_result user msg."""
        lines = [
            # isMeta line injected by a hook at session start — must be skipped
            _user("CAVEMAN MODE ACTIVE", uuid="u0", is_meta=True,
                  ts="2026-06-18T08:00:00.000Z"),
            # Real task message
            _user("Fix the auth bug", uuid="u1", parent="u0",
                  ts="2026-06-18T08:00:01.000Z"),
            _assistant([{"id": "t1", "name": "Read", "input": {"file_path": "/auth.py"}}],
                       parent="u1", ts="2026-06-18T08:00:02.000Z"),
            _tool_results([{"tool_use_id": "t1"}],
                          parent="a1", ts="2026-06-18T08:00:03.000Z"),
        ]
        p = _write_transcript(tmp_path, "task-text-test", lines)
        run = load_session(str(p))
        assert run.task_text == "Fix the auth bug"

    def test_tool_result_only_user_msg_skipped(self, tmp_path):
        """A user message that is ONLY tool_result blocks doesn't become task_text."""
        lines = [
            # First user msg: only tool_results (no task yet)
            _tool_results([{"tool_use_id": "t_phantom"}], uuid="u0",
                          parent=None, ts="2026-06-18T08:00:00.000Z"),
            # Actual task
            _user("Deploy to prod", uuid="u1", parent="u0",
                  ts="2026-06-18T08:00:01.000Z"),
            _assistant([{"id": "t2", "name": "Bash", "input": {"command": "git push"}}],
                       parent="u1", ts="2026-06-18T08:00:02.000Z"),
            _tool_results([{"tool_use_id": "t2"}],
                          parent="a1", ts="2026-06-18T08:00:03.000Z"),
        ]
        p = _write_transcript(tmp_path, "tr-only-test", lines)
        run = load_session(str(p))
        assert run.task_text == "Deploy to prod"

    def test_both_is_meta_and_tool_result_before_real_msg(self, tmp_path):
        """Skip both isMeta and tool_result-only lines before reaching task."""
        lines = [
            _user("hook output", uuid="u0", is_meta=True,
                  ts="2026-06-18T08:00:00.000Z"),
            _tool_results([{"tool_use_id": "tx"}], uuid="u_tr",
                          parent="u0", ts="2026-06-18T08:00:01.000Z"),
            _user("Refactor the DB layer", uuid="u1", parent="u_tr",
                  ts="2026-06-18T08:00:02.000Z"),
        ]
        p = _write_transcript(tmp_path, "combined-skip-test", lines)
        run = load_session(str(p))
        assert run.task_text == "Refactor the DB layer"

    def test_content_array_text_blocks_joined(self, tmp_path):
        """task_text from content-array with multiple text blocks is joined."""
        lines = [
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "timestamp": "2026-06-18T08:00:00.000Z",
                "sessionId": "arr-test",
                "message": {"role": "user", "content": [
                    {"type": "text", "text": "Part one."},
                    {"type": "text", "text": "Part two."},
                ]},
            }
        ]
        p = tmp_path / "arr-test.jsonl"
        p.write_text(json.dumps(lines[0]) + "\n")
        run = load_session(str(p))
        assert "Part one" in run.task_text
        assert "Part two" in run.task_text


# ─────────────────────────────────────────────────────────────────────────────
# Decision parsing: tool_use blocks and is_error propagation
# ─────────────────────────────────────────────────────────────────────────────


class TestDecisionParsing:
    def test_decision_count(self, tmp_path):
        lines = [
            _user("Do work", uuid="u1", ts="2026-06-18T08:00:00.000Z"),
            _assistant([
                {"id": "t1", "name": "Read", "input": {"file_path": "/a.py"}},
                {"id": "t2", "name": "Edit", "input": {"file_path": "/a.py",
                                                        "old_string": "x", "new_string": "y"}},
            ], ts="2026-06-18T08:00:01.000Z"),
            _tool_results([{"tool_use_id": "t1"}, {"tool_use_id": "t2"}],
                          ts="2026-06-18T08:00:02.000Z"),
            _assistant([{"id": "t3", "name": "Bash", "input": {"command": "pytest"}}],
                       uuid="a2", parent="u2", ts="2026-06-18T08:00:03.000Z"),
            _tool_results([{"tool_use_id": "t3"}], uuid="u3", parent="a2",
                          ts="2026-06-18T08:00:04.000Z"),
        ]
        p = _write_transcript(tmp_path, "decisions-test", lines)
        run = load_session(str(p))
        assert len(run.decisions) == 3

    def test_tool_names_ordered(self, tmp_path):
        lines = [
            _user("task", uuid="u1", ts="2026-06-18T08:00:00.000Z"),
            _assistant([
                {"id": "t1", "name": "Read", "input": {"file_path": "/x"}},
                {"id": "t2", "name": "Write", "input": {"file_path": "/y", "content": "hi"}},
            ], ts="2026-06-18T08:00:01.000Z"),
            _tool_results([{"tool_use_id": "t1"}, {"tool_use_id": "t2"}],
                          ts="2026-06-18T08:00:02.000Z"),
        ]
        p = _write_transcript(tmp_path, "order-test", lines)
        run = load_session(str(p))
        assert [d.tool_name for d in run.decisions] == ["Read", "Write"]

    def test_is_error_propagated(self, tmp_path):
        lines = [
            _user("task", uuid="u1", ts="2026-06-18T08:00:00.000Z"),
            _assistant([{"id": "t1", "name": "Bash", "input": {"command": "bad cmd"}}],
                       ts="2026-06-18T08:00:01.000Z"),
            _tool_results([{"tool_use_id": "t1", "is_error": True, "content": "Exit code 1"}],
                          ts="2026-06-18T08:00:02.000Z"),
        ]
        p = _write_transcript(tmp_path, "error-test", lines)
        run = load_session(str(p))
        assert run.decisions[0].is_error is True

    def test_is_error_false_when_absent(self, tmp_path):
        lines = [
            _user("task", uuid="u1", ts="2026-06-18T08:00:00.000Z"),
            _assistant([{"id": "t1", "name": "Read", "input": {"file_path": "/f"}}],
                       ts="2026-06-18T08:00:01.000Z"),
            _tool_results([{"tool_use_id": "t1"}],  # no is_error field
                          ts="2026-06-18T08:00:02.000Z"),
        ]
        p = _write_transcript(tmp_path, "no-error-test", lines)
        run = load_session(str(p))
        assert run.decisions[0].is_error is False

    def test_parent_step_index_chain(self, tmp_path):
        lines = [
            _user("task", uuid="u1", ts="2026-06-18T08:00:00.000Z"),
            _assistant([
                {"id": "t1", "name": "Read", "input": {"file_path": "/a"}},
                {"id": "t2", "name": "Bash", "input": {"command": "ls"}},
                {"id": "t3", "name": "Write", "input": {"file_path": "/b", "content": "x"}},
            ], ts="2026-06-18T08:00:01.000Z"),
        ]
        p = _write_transcript(tmp_path, "parent-test", lines)
        run = load_session(str(p))
        assert run.decisions[0].parent_step_index is None
        assert run.decisions[1].parent_step_index == 0
        assert run.decisions[2].parent_step_index == 1

    def test_started_at_from_first_record(self, tmp_path):
        lines = [
            _user("task", uuid="u1", ts="2026-06-18T08:00:00.000Z"),
            _assistant([{"id": "t1", "name": "Read", "input": {}}],
                       ts="2026-06-18T08:00:01.000Z"),
        ]
        p = _write_transcript(tmp_path, "started-test", lines)
        run = load_session(str(p))
        assert run.started_at == "2026-06-18T08:00:00.000Z"

    def test_run_id_is_filename_stem(self, tmp_path):
        lines = [_user("task", uuid="u1")]
        p = _write_transcript(tmp_path, "my-special-session", lines)
        run = load_session(str(p))
        assert run.run_id == "my-special-session"

    def test_non_message_records_ignored(self, tmp_path):
        """queue-operation and attachment records must not produce decisions."""
        lines = [
            {"type": "queue-operation", "operation": "enqueue",
             "timestamp": "2026-06-18T08:00:00.000Z", "sessionId": "ignore-test"},
            _user("real task", uuid="u1", sid="ignore-test",
                  ts="2026-06-18T08:00:01.000Z"),
            {"type": "attachment", "timestamp": "2026-06-18T08:00:01.500Z",
             "sessionId": "ignore-test", "attachment": {"type": "hook_success"}},
            _assistant([{"id": "t1", "name": "Read", "input": {"file_path": "/x"}}],
                       sid="ignore-test", ts="2026-06-18T08:00:02.000Z"),
        ]
        p = tmp_path / "ignore-test.jsonl"
        p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
        run = load_session(str(p))
        assert len(run.decisions) == 1
        assert run.decisions[0].tool_name == "Read"


# ─────────────────────────────────────────────────────────────────────────────
# Redaction of secret-shaped arg values
# ─────────────────────────────────────────────────────────────────────────────


class TestRedactionInSummary:
    def test_secret_in_bash_command_redacted(self, tmp_path):
        secret = "sk-ant-" + "abcdefghijklmnopqrstuvwxyz12345678"
        lines = [
            _user("call api", uuid="u1", ts="2026-06-18T08:00:00.000Z"),
            _assistant([{
                "id": "t1", "name": "Bash",
                "input": {"command": "curl -H 'Authorization: Bearer " + secret + "' https://api.x.com"},
            }], ts="2026-06-18T08:00:01.000Z"),
        ]
        p = _write_transcript(tmp_path, "redact-test", lines)
        run = load_session(str(p))
        summary = run.decisions[0].tool_args_summary
        assert "[REDACTED]" in summary
        assert secret not in summary

    def test_clean_command_not_redacted(self, tmp_path):
        lines = [
            _user("run tests", uuid="u1", ts="2026-06-18T08:00:00.000Z"),
            _assistant([{
                "id": "t1", "name": "Bash",
                "input": {"command": "pytest -v --tb=short"},
            }], ts="2026-06-18T08:00:01.000Z"),
        ]
        p = _write_transcript(tmp_path, "clean-test", lines)
        run = load_session(str(p))
        assert run.decisions[0].tool_args_summary == "pytest -v --tb=short"


# ─────────────────────────────────────────────────────────────────────────────
# Soft-fail on malformed lines
# ─────────────────────────────────────────────────────────────────────────────


class TestSoftFail:
    def test_malformed_line_skipped(self, tmp_path):
        good = json.dumps(_user("do X", uuid="u1", ts="2026-06-18T08:00:00.000Z"))
        bad = "NOT JSON }{{"
        tool = json.dumps(_assistant(
            [{"id": "t1", "name": "Read", "input": {"file_path": "/x"}}],
            ts="2026-06-18T08:00:01.000Z",
        ))
        p = tmp_path / "soft-fail.jsonl"
        p.write_text("\n".join([good, bad, tool]) + "\n")
        run = load_session(str(p))
        # Bad line skipped; 1 decision from the valid assistant line.
        assert len(run.decisions) == 1

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        run = load_session(str(p))
        assert run.decisions == []
        assert run.task_text is None


# ─────────────────────────────────────────────────────────────────────────────
# Spool reader (read_session_from_spool)
# ─────────────────────────────────────────────────────────────────────────────


class TestSpoolReader:
    def _spool_line(self, event_name: str, **kwargs) -> str:
        rec: dict = {"session_id": "spool-test-001", "event_name": event_name,
                     "tool_name": None, "tool_input_redacted": None,
                     "is_error": None, "occurred_at": "2026-06-18T08:00:00+00:00",
                     "payload_redacted": {}}
        rec.update(kwargs)
        return json.dumps(rec)

    def test_spool_basic_round_trip(self, tmp_path):
        lines = [
            self._spool_line("SessionStart",
                             occurred_at="2026-06-18T08:00:00+00:00"),
            self._spool_line("PostToolUse", tool_name="Read",
                             tool_input_redacted={"file_path": "/src/a.py"},
                             is_error=False,
                             occurred_at="2026-06-18T08:00:01+00:00"),
            self._spool_line("PostToolUse", tool_name="Bash",
                             tool_input_redacted={"command": "pytest"},
                             is_error=False,
                             occurred_at="2026-06-18T08:00:02+00:00"),
            self._spool_line("SessionEnd",
                             occurred_at="2026-06-18T08:00:03+00:00"),
        ]
        spool = tmp_path / "spool-test-001.jsonl"
        spool.write_text("\n".join(lines) + "\n")
        run = read_session_from_spool("spool-test-001", spool_dir=tmp_path)
        assert run.run_id == "spool-test-001"
        assert run.source == "claude_code"
        assert len(run.decisions) == 2
        assert run.decisions[0].tool_name == "Read"
        assert run.decisions[1].tool_name == "Bash"
        assert run.started_at == "2026-06-18T08:00:00+00:00"

    def test_spool_post_tool_use_failure_is_error(self, tmp_path):
        lines = [
            self._spool_line("PostToolUseFailure", tool_name="Bash",
                             tool_input_redacted={"command": "bad"},
                             is_error=True),
        ]
        spool = tmp_path / "spool-test-001.jsonl"
        spool.write_text("\n".join(lines) + "\n")
        run = read_session_from_spool("spool-test-001", spool_dir=tmp_path)
        assert run.decisions[0].is_error is True

    def test_spool_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_session_from_spool("no-such-session", spool_dir=tmp_path)

    def test_spool_parent_step_index_chain(self, tmp_path):
        lines = [
            self._spool_line("PostToolUse", tool_name="Read",
                             tool_input_redacted={"file_path": "/a"}),
            self._spool_line("PostToolUse", tool_name="Edit",
                             tool_input_redacted={"file_path": "/a",
                                                  "old_string": "x", "new_string": "y"}),
        ]
        spool = tmp_path / "spool-test-001.jsonl"
        spool.write_text("\n".join(lines) + "\n")
        run = read_session_from_spool("spool-test-001", spool_dir=tmp_path)
        assert run.decisions[0].parent_step_index is None
        assert run.decisions[1].parent_step_index == 0


# ─────────────────────────────────────────────────────────────────────────────
# ingest_session source_kind switch
# ─────────────────────────────────────────────────────────────────────────────


class TestIngestSession:
    def test_ingest_transcript_default(self, tmp_path):
        from unasked.ledger import load_run
        lines = [
            _user("Fix bug", uuid="u1", sid="ingest-native",
                  ts="2026-06-18T08:00:00.000Z"),
            _assistant([{"id": "t1", "name": "Read",
                         "input": {"file_path": "/bug.py"}}],
                       sid="ingest-native", ts="2026-06-18T08:00:01.000Z"),
        ]
        p = tmp_path / "ingest-native.jsonl"
        p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
        db = str(tmp_path / "ledger.db")
        run = ingest_session(str(p), db_path=db)
        loaded = load_run(run.run_id, path=db)
        assert loaded is not None
        assert len(loaded.decisions) == 1

    def test_ingest_spool_source_kind(self, tmp_path):
        from unasked.ledger import load_run
        rec = json.dumps({
            "session_id": "spool-ingest-001",
            "event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input_redacted": {"command": "ls"},
            "is_error": False,
            "occurred_at": "2026-06-18T08:00:00+00:00",
            "payload_redacted": {},
        })
        spool = tmp_path / "spool-ingest-001.jsonl"
        spool.write_text(rec + "\n")
        db = str(tmp_path / "ledger.db")
        import os
        old = os.environ.get("KAIROS_SPOOL_DIR")
        os.environ["KAIROS_SPOOL_DIR"] = str(tmp_path)
        try:
            run = ingest_session("spool-ingest-001", db_path=db, source_kind="spool")
        finally:
            if old is None:
                os.environ.pop("KAIROS_SPOOL_DIR", None)
            else:
                os.environ["KAIROS_SPOOL_DIR"] = old
        loaded = load_run(run.run_id, path=db)
        assert loaded is not None
        assert loaded.decisions[0].tool_name == "Bash"
