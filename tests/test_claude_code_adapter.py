"""Tests for the Claude Code adapter (src/unasked/adapters/claude_code.py).

All tests use the JSONL fixtures under tests/fixtures/ — no live spool files,
no home-directory side effects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unasked.adapters.claude_code import (
    SpoolNotFoundError,
    _summarise_args,
    read_session,
    redact_args_summary,
)
from unasked.ir import Run

_FIXTURES = Path(__file__).parent / "fixtures"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read(session_id: str) -> Run:
    """Read from the fixtures directory."""
    return read_session(session_id, spool_dir=_FIXTURES)


# ─────────────────────────────────────────────────────────────────────────────
# read_session — basic structure
# ─────────────────────────────────────────────────────────────────────────────


class TestReadSessionStructure:
    def test_returns_run(self):
        run = _read("test-session-001")
        assert isinstance(run, Run)

    def test_run_id(self):
        run = _read("test-session-001")
        assert run.run_id == "test-session-001"

    def test_source_is_claude_code(self):
        run = _read("test-session-001")
        assert run.source == "claude_code"

    def test_started_at_from_session_start(self):
        run = _read("test-session-001")
        assert run.started_at == "2026-06-18T08:00:00.000000+00:00"

    def test_task_text_none_when_no_transcript(self):
        # fixture has no transcript_path in payload → task_text is None
        run = _read("test-session-001")
        assert run.task_text is None

    def test_decision_count_excludes_start_end(self):
        # session_normal.jsonl has 3 PostToolUse events; SessionStart/End excluded
        run = _read("test-session-001")
        assert len(run.decisions) == 3

    def test_step_index_is_monotonic(self):
        run = _read("test-session-001")
        indices = [d.step_index for d in run.decisions]
        assert indices == list(range(len(run.decisions)))

    def test_decisions_ordered_by_occurrence(self):
        run = _read("test-session-001")
        tools = [d.tool_name for d in run.decisions]
        assert tools == ["Read", "Edit", "Bash"]

    def test_ts_populated(self):
        run = _read("test-session-001")
        assert run.decisions[0].ts == "2026-06-18T08:00:01.000000+00:00"


# ─────────────────────────────────────────────────────────────────────────────
# read_session — error handling
# ─────────────────────────────────────────────────────────────────────────────


class TestReadSessionErrors:
    def test_raises_spool_not_found(self, tmp_path):
        with pytest.raises(SpoolNotFoundError):
            read_session("no-such-session", spool_dir=tmp_path)

    def test_post_tool_use_failure_is_error(self):
        run = _read("test-session-002")
        # Second decision (index 1) is the PostToolUseFailure
        bash_fail = next(d for d in run.decisions if d.tool_name == "Bash")
        assert bash_fail.is_error is True

    def test_post_tool_use_success_not_error(self):
        run = _read("test-session-002")
        web = next(d for d in run.decisions if d.tool_name == "WebSearch")
        assert web.is_error is False

    def test_normal_session_no_errors(self):
        run = _read("test-session-001")
        assert all(not d.is_error for d in run.decisions)


# ─────────────────────────────────────────────────────────────────────────────
# read_session — multiple tool types
# ─────────────────────────────────────────────────────────────────────────────


class TestReadSessionToolTypes:
    def test_agent_tool_present(self):
        run = _read("test-session-002")
        tools = [d.tool_name for d in run.decisions]
        assert "Agent" in tools

    def test_send_message_present(self):
        run = _read("test-session-002")
        tools = [d.tool_name for d in run.decisions]
        assert "SendMessage" in tools

    def test_decision_count_session2(self):
        # 5 PostToolUse/PostToolUseFailure events
        run = _read("test-session-002")
        assert len(run.decisions) == 5

    def test_write_args_summary_is_filepath(self):
        run = _read("test-session-002")
        write_d = next(d for d in run.decisions if d.tool_name == "Write")
        assert "/tmp/out.txt" in write_d.tool_args_summary

    def test_agent_args_summary_has_subtype(self):
        run = _read("test-session-002")
        agent_d = next(d for d in run.decisions if d.tool_name == "Agent")
        assert "claude" in agent_d.tool_args_summary

    def test_send_message_args_summary_has_to(self):
        run = _read("test-session-002")
        sm = next(d for d in run.decisions if d.tool_name == "SendMessage")
        assert "team-lead" in sm.tool_args_summary


# ─────────────────────────────────────────────────────────────────────────────
# Redaction
# ─────────────────────────────────────────────────────────────────────────────


class TestRedaction:
    def test_secret_scrubbed_from_bash_summary(self):
        run = _read("test-session-003")
        bash_d = next(d for d in run.decisions if d.tool_name == "Bash")
        # The spool fixture has a Bearer + sk- token in the command field.
        assert "[REDACTED]" in bash_d.tool_args_summary

    def test_redact_args_summary_scrubs_bearer(self):
        text = "curl -H 'Bearer " + "sk-ant-" + "abc123xyz456def789ghi012jkl345mno678' https://api"
        out = redact_args_summary(text)
        assert "[REDACTED]" in out
        assert "sk-ant-" not in out

    def test_redact_args_summary_clean_string_unchanged(self):
        text = "pytest -v --tb=short"
        assert redact_args_summary(text) == text


# ─────────────────────────────────────────────────────────────────────────────
# _summarise_args per-tool unit tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSummariseArgs:
    def test_bash_first_line_only(self):
        result = _summarise_args("Bash", {"command": "echo hi\necho bye", "description": "x"})
        assert result == "echo hi"

    def test_bash_truncates_long_command(self):
        long_cmd = "x" * 200
        result = _summarise_args("Bash", {"command": long_cmd})
        assert len(result) <= 121  # _MAX_VALUE_LEN + ellipsis char

    def test_read_returns_filepath(self):
        result = _summarise_args("Read", {"file_path": "/src/foo.py", "limit": 100})
        assert result == "/src/foo.py"

    def test_edit_returns_filepath(self):
        result = _summarise_args("Edit", {"file_path": "/a.py", "old_string": "x", "new_string": "y"})
        assert result == "/a.py"

    def test_write_returns_filepath(self):
        result = _summarise_args("Write", {"file_path": "/b.py", "content": "hello"})
        assert result == "/b.py"

    def test_glob_pattern_and_path(self):
        result = _summarise_args("Glob", {"pattern": "*.py", "path": "/src"})
        assert "*.py" in result
        assert "/src" in result

    def test_webfetch_url(self):
        result = _summarise_args("WebFetch", {"url": "https://example.com", "prompt": "get it"})
        assert result == "https://example.com"

    def test_websearch_query(self):
        result = _summarise_args("WebSearch", {"query": "python sqlite"})
        assert result == "python sqlite"

    def test_agent_subtype_and_desc(self):
        result = _summarise_args("Agent", {"subagent_type": "fork", "description": "audit"})
        assert "fork" in result
        assert "audit" in result

    def test_send_message_to_and_summary(self):
        result = _summarise_args("SendMessage", {"to": "lead", "summary": "done", "message": "..."})
        assert "lead" in result
        assert "done" in result

    def test_skill_name(self):
        result = _summarise_args("Skill", {"skill": "paperclip", "args": "--list"})
        assert "paperclip" in result

    def test_task_create_subject(self):
        result = _summarise_args("TaskCreate", {"subject": "Fix bug", "description": "..."})
        assert result == "Fix bug"

    def test_task_update_fields(self):
        result = _summarise_args("TaskUpdate", {"taskId": "42", "status": "completed"})
        assert "42" in result
        assert "completed" in result

    def test_empty_args(self):
        result = _summarise_args("ExitPlanMode", {})
        assert result == "(no args)"

    def test_unknown_tool_fallback(self):
        result = _summarise_args("UnknownTool", {"foo": "bar", "baz": "qux"})
        assert "foo=bar" in result


# ─────────────────────────────────────────────────────────────────────────────
# Robustness — malformed lines in spool
# ─────────────────────────────────────────────────────────────────────────────


class TestRobustness:
    def test_malformed_lines_skipped(self, tmp_path):
        """A spool with some garbled lines should not raise; valid events are parsed."""
        spool = tmp_path / "bad-session-xyz.jsonl"
        line1 = json.dumps({
            "session_id": "bad-session-xyz",
            "event_name": "SessionStart",
            "tool_name": None,
            "tool_input_redacted": None,
            "is_error": None,
            "occurred_at": "2026-06-18T08:00:00+00:00",
            "payload_redacted": {},
        })
        line2 = "THIS IS NOT JSON }{{"
        line3 = json.dumps({
            "session_id": "bad-session-xyz",
            "event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input_redacted": {"file_path": "/x.py"},
            "is_error": False,
            "occurred_at": "2026-06-18T08:00:01+00:00",
            "payload_redacted": {},
        })
        spool.write_text("\n".join([line1, line2, line3]) + "\n")
        run = read_session("bad-session-xyz", spool_dir=tmp_path)
        # Only the PostToolUse should become a Decision; malformed line skipped.
        assert len(run.decisions) == 1
        assert run.decisions[0].tool_name == "Read"

    def test_empty_spool_returns_empty_run(self, tmp_path):
        spool = tmp_path / "empty-session-abc.jsonl"
        spool.write_text("")
        run = read_session("empty-session-abc", spool_dir=tmp_path)
        assert run.run_id == "empty-session-abc"
        assert run.decisions == []
        assert run.started_at is None

    def test_provenance_defaults_none(self):
        run = _read("test-session-001")
        for d in run.decisions:
            assert d.provenance is None
            assert d.scope_drift is None
            assert d.why is None
            assert d.feedback is None
