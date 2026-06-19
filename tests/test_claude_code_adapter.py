"""Tests for the Claude Code transcript adapter.

All tests use synthetic JSONL fixtures under tests/fixtures/ that mirror the
real CC transcript schema — no real session data, no home-directory reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unasked.adapters.claude_code import (
    ingest_session,
    load_session,
    _is_command_artifact,
)
from unasked.ir import Run
from unasked.redact import redact, summarize_args

_FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> Run:
    """Load a fixture by filename (no extension needed)."""
    return load_session(str(_FIXTURES / f"{name}.jsonl"))


# ─────────────────────────────────────────────────────────────────────────────
# load_session — basic structure (cc-session-basic fixture)
# ─────────────────────────────────────────────────────────────────────────────


class TestBasicSession:
    def test_returns_run(self):
        assert isinstance(_load("cc-session-basic"), Run)

    def test_run_id_is_stem(self):
        run = _load("cc-session-basic")
        assert run.run_id == "cc-session-basic"

    def test_source_is_claude_code(self):
        run = _load("cc-session-basic")
        assert run.source == "claude_code"

    def test_started_at_first_record(self):
        run = _load("cc-session-basic")
        assert run.started_at == "2026-06-18T08:00:00.000Z"

    def test_task_text_from_first_user_message(self):
        run = _load("cc-session-basic")
        assert run.task_text == "Fix the failing auth test in src/auth.py"

    def test_decision_count(self):
        # 3 tool_use blocks: Read, Edit, Bash
        run = _load("cc-session-basic")
        assert len(run.decisions) == 3

    def test_tool_names_in_order(self):
        run = _load("cc-session-basic")
        names = [d.tool_name for d in run.decisions]
        assert names == ["Read", "Edit", "Bash"]

    def test_step_indices_monotonic(self):
        run = _load("cc-session-basic")
        indices = [d.step_index for d in run.decisions]
        assert indices == [0, 1, 2]

    def test_ts_populated(self):
        run = _load("cc-session-basic")
        assert run.decisions[0].ts == "2026-06-18T08:00:01.000Z"

    def test_no_errors_in_basic_session(self):
        run = _load("cc-session-basic")
        assert all(not d.is_error for d in run.decisions)

    def test_parent_step_index_chain(self):
        run = _load("cc-session-basic")
        assert run.decisions[0].parent_step_index is None  # first has no parent
        assert run.decisions[1].parent_step_index == 0
        assert run.decisions[2].parent_step_index == 1

    def test_provenance_fields_are_none(self):
        run = _load("cc-session-basic")
        for d in run.decisions:
            assert d.provenance is None
            assert d.scope_drift is None
            assert d.why is None
            assert d.feedback is None


# ─────────────────────────────────────────────────────────────────────────────
# is_error propagation (cc-session-with-error fixture)
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorPropagation:
    def test_first_decision_is_error(self):
        run = _load("cc-session-with-error")
        # toolu_101 (git push) has is_error=true in its tool_result
        assert run.decisions[0].is_error is True

    def test_second_decision_not_error(self):
        run = _load("cc-session-with-error")
        # toolu_102 (git remote -v) has normal result
        assert run.decisions[1].is_error is False

    def test_third_decision_not_error(self):
        run = _load("cc-session-with-error")
        assert run.decisions[2].is_error is False

    def test_decision_count_with_error(self):
        run = _load("cc-session-with-error")
        assert len(run.decisions) == 3

    def test_task_text_extracted(self):
        run = _load("cc-session-with-error")
        assert run.task_text == "Deploy the app to production"


# ─────────────────────────────────────────────────────────────────────────────
# Multiple tool_use blocks in one assistant message + non-message records
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiToolSession:
    def test_decision_count(self):
        # WebFetch + Agent + Skill + SendMessage = 4
        run = _load("cc-session-multi-tool")
        assert len(run.decisions) == 4

    def test_tool_names(self):
        run = _load("cc-session-multi-tool")
        names = [d.tool_name for d in run.decisions]
        assert names == ["WebFetch", "Agent", "Skill", "SendMessage"]

    def test_attachment_records_ignored(self):
        # The fixture has an "attachment" type record — must not become a Decision
        run = _load("cc-session-multi-tool")
        assert all(d.tool_name != "attachment" for d in run.decisions)

    def test_webfetch_args_summary(self):
        run = _load("cc-session-multi-tool")
        assert "example.com" in run.decisions[0].tool_args_summary

    def test_agent_args_summary_has_subtype(self):
        run = _load("cc-session-multi-tool")
        assert "fork" in run.decisions[1].tool_args_summary

    def test_sendmessage_args_summary(self):
        run = _load("cc-session-multi-tool")
        sm = run.decisions[3]
        assert "team-lead" in sm.tool_args_summary


# ─────────────────────────────────────────────────────────────────────────────
# FileNotFoundError for unknown session
# ─────────────────────────────────────────────────────────────────────────────


class TestMissingSession:
    def test_raises_for_nonexistent_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(str(tmp_path / "no-such-session.jsonl"))

    def test_raises_for_unknown_session_id(self):
        # A UUID-like id that won't match anything in ~/.claude/projects
        with pytest.raises(FileNotFoundError):
            load_session("00000000-0000-0000-0000-000000000000")


# ─────────────────────────────────────────────────────────────────────────────
# Soft-fail on malformed lines
# ─────────────────────────────────────────────────────────────────────────────


class TestRobustness:
    def test_malformed_lines_skipped(self, tmp_path):
        good_line = json.dumps({
            "type": "user",
            "timestamp": "2026-06-18T08:00:00.000Z",
            "sessionId": "robust-session",
            "message": {"role": "user", "content": "Do something"},
        })
        bad_line = "THIS IS NOT JSON }{{"
        tool_line = json.dumps({
            "type": "assistant",
            "timestamp": "2026-06-18T08:00:01.000Z",
            "sessionId": "robust-session",
            "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_x01", "name": "Read",
                 "input": {"file_path": "/x.py"}}
            ]},
        })
        result_line = json.dumps({
            "type": "user",
            "timestamp": "2026-06-18T08:00:02.000Z",
            "sessionId": "robust-session",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_x01", "content": "ok"}
            ]},
        })
        transcript = tmp_path / "robust-session.jsonl"
        transcript.write_text("\n".join([good_line, bad_line, tool_line, result_line]) + "\n")

        run = load_session(str(transcript))
        assert run.task_text == "Do something"
        assert len(run.decisions) == 1
        assert run.decisions[0].tool_name == "Read"

    def test_empty_file_returns_empty_run(self, tmp_path):
        transcript = tmp_path / "empty-session.jsonl"
        transcript.write_text("")
        run = load_session(str(transcript))
        assert run.run_id == "empty-session"
        assert run.decisions == []
        assert run.started_at is None
        assert run.task_text is None

    def test_no_tool_use_returns_empty_decisions(self, tmp_path):
        """Transcript with only text messages → no decisions."""
        transcript = tmp_path / "text-only.jsonl"
        lines = [
            json.dumps({"type": "user", "timestamp": "2026-06-18T08:00:00.000Z",
                        "sessionId": "text-only",
                        "message": {"role": "user", "content": "Hello"}}),
            json.dumps({"type": "assistant", "timestamp": "2026-06-18T08:00:01.000Z",
                        "sessionId": "text-only",
                        "message": {"role": "assistant",
                                    "content": [{"type": "text", "text": "Hi there!"}]}}),
        ]
        transcript.write_text("\n".join(lines))
        run = load_session(str(transcript))
        assert run.decisions == []
        assert run.task_text == "Hello"


# ─────────────────────────────────────────────────────────────────────────────
# ingest_session — round-trips through ledger
# ─────────────────────────────────────────────────────────────────────────────


class TestIngestSession:
    def test_ingest_returns_run(self, tmp_path):
        from unasked.ledger import load_run
        run = ingest_session(
            str(_FIXTURES / "cc-session-basic.jsonl"),
            db_path=str(tmp_path / "ledger.db"),
        )
        assert isinstance(run, Run)

    def test_ingest_persists_to_ledger(self, tmp_path):
        from unasked.ledger import load_run
        db = str(tmp_path / "ledger.db")
        run = ingest_session(str(_FIXTURES / "cc-session-basic.jsonl"), db_path=db)
        loaded = load_run(run.run_id, path=db)
        assert loaded is not None
        assert loaded.run_id == run.run_id
        assert len(loaded.decisions) == len(run.decisions)


# ─────────────────────────────────────────────────────────────────────────────
# summarize_args — per-tool unit tests (src/unasked/redact.py)
# ─────────────────────────────────────────────────────────────────────────────


class TestSummarizeArgs:
    def test_bash_first_line_only(self):
        r = summarize_args("Bash", {"command": "echo hi\necho bye", "description": "x"})
        assert r == "echo hi"

    def test_bash_truncates_long_command(self):
        r = summarize_args("Bash", {"command": "x" * 200})
        assert len(r) <= 122  # 120 + "…" char

    def test_read_returns_filepath(self):
        assert summarize_args("Read", {"file_path": "/src/foo.py", "limit": 100}) == "/src/foo.py"

    def test_edit_has_file_path(self):
        r = summarize_args("Edit", {"file_path": "/a.py", "old_string": "x", "new_string": "y"})
        assert "/a.py" in r

    def test_write_has_file_path(self):
        r = summarize_args("Write", {"file_path": "/b.py", "content": "hello world"})
        assert "/b.py" in r

    def test_glob_pattern_and_path(self):
        r = summarize_args("Glob", {"pattern": "*.py", "path": "/src"})
        assert "*.py" in r and "/src" in r

    def test_webfetch_url(self):
        assert "example.com" in summarize_args("WebFetch", {"url": "https://example.com"})

    def test_websearch_query(self):
        assert "python sqlite" in summarize_args("WebSearch", {"query": "python sqlite"})

    def test_agent_subtype_and_desc(self):
        r = summarize_args("Agent", {"subagent_type": "fork", "description": "audit"})
        assert "fork" in r and "audit" in r

    def test_send_message_to_and_summary(self):
        r = summarize_args("SendMessage", {"to": "lead", "summary": "done", "message": "..."})
        assert "lead" in r and "done" in r

    def test_skill_name(self):
        r = summarize_args("Skill", {"skill": "paperclip", "args": "--list"})
        assert "paperclip" in r

    def test_task_update_fields(self):
        r = summarize_args("TaskUpdate", {"taskId": "42", "status": "completed"})
        assert "42" in r and "completed" in r

    def test_empty_args_fallback(self):
        assert summarize_args("ExitPlanMode", {}) == "(no args)"

    def test_generic_fallback(self):
        r = summarize_args("UnknownTool", {"foo": "bar"})
        assert "foo=bar" in r

    def test_large_content_blob_hidden(self):
        r = summarize_args("Write", {"file_path": "/out.py", "content": "x" * 500})
        # content blob should not appear verbatim
        assert "x" * 200 not in r


# ─────────────────────────────────────────────────────────────────────────────
# redact — secret scrubbing
# ─────────────────────────────────────────────────────────────────────────────


class TestRedact:
    def test_scrubs_sk_key(self):
        key = "sk-ant-" + "abcdefghijklmnopqrstuvwxyz1234"
        assert "[REDACTED]" in redact(key)
        assert "sk-ant-" not in redact(key)

    def test_scrubs_bearer_token(self):
        text = "Authorization: Bearer " + "eyJhbGciOiJIUzI1NiJ9.payload.sig"
        assert "[REDACTED]" in redact(text)

    def test_scrubs_github_pat(self):
        token = "ghp_" + "A" * 36
        assert "[REDACTED]" in redact(token)

    def test_clean_string_unchanged(self):
        text = "pytest -v --tb=short"
        assert redact(text) == text

    def test_secret_in_bash_summary_scrubbed(self, tmp_path):
        """End-to-end: a secret in tool_input is scrubbed in the Decision summary."""
        secret_val = "sk-ant-" + "secretkeyabcdefghijklmnopqrs1234"
        transcript = tmp_path / "secret-session.jsonl"
        lines = [
            json.dumps({"type": "user", "timestamp": "2026-06-18T08:00:00.000Z",
                        "sessionId": "secret-session",
                        "message": {"role": "user", "content": "Call the API"}}),
            json.dumps({"type": "assistant", "timestamp": "2026-06-18T08:00:01.000Z",
                        "sessionId": "secret-session",
                        "message": {"role": "assistant", "content": [
                            {"type": "tool_use", "id": "toolu_s01", "name": "Bash",
                             "input": {"command": "curl -H 'Authorization: Bearer " + secret_val + "' https://api.example.com"}}
                        ]}}),
            json.dumps({"type": "user", "timestamp": "2026-06-18T08:00:02.000Z",
                        "sessionId": "secret-session",
                        "message": {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "toolu_s01", "content": "200 OK"}
                        ]}}),
        ]
        transcript.write_text("\n".join(lines))
        run = load_session(str(transcript))
        summary = run.decisions[0].tool_args_summary
        assert "[REDACTED]" in summary
        assert secret_val not in summary


# ─────────────────────────────────────────────────────────────────────────────
# F4.1: _is_command_artifact helper
# ─────────────────────────────────────────────────────────────────────────────


class TestIsCommandArtifact:
    def test_slash_clear_is_artifact(self):
        assert _is_command_artifact("/clear") is True

    def test_slash_model_is_artifact(self):
        assert _is_command_artifact("/model") is True

    def test_slash_command_with_content_is_artifact(self):
        # e.g. "/compact summarise the session"
        assert _is_command_artifact("/compact summarise the session") is True

    def test_command_name_tag_is_artifact(self):
        text = "<command-name>/clear</command-name>"
        assert _is_command_artifact(text) is True

    def test_command_message_tag_is_artifact(self):
        text = "<command-message>some message</command-message>"
        assert _is_command_artifact(text) is True

    def test_command_args_tag_is_artifact(self):
        text = "<command-args>--verbose</command-args>"
        assert _is_command_artifact(text) is True

    def test_local_command_tag_is_artifact(self):
        text = "<local-command-output>some hook output</local-command-output>"
        assert _is_command_artifact(text) is True

    def test_real_instruction_not_artifact(self):
        assert _is_command_artifact("Fix the failing auth test in src/auth.py") is False

    def test_empty_string_is_artifact(self):
        assert _is_command_artifact("") is True

    def test_whitespace_only_is_artifact(self):
        assert _is_command_artifact("   ") is True

    def test_normal_sentence_starting_with_word_not_artifact(self):
        assert _is_command_artifact("Please update the config file") is False


# ─────────────────────────────────────────────────────────────────────────────
# F4.1: task extraction skips command artifacts
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskExtractionSkipsArtifacts:
    def _make_transcript(self, tmp_path, lines):
        path = tmp_path / "test-session.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))
        return str(path)

    def test_skips_slash_clear_finds_real_instruction(self, tmp_path):
        """F4.1: first message is /clear artifact; second is real instruction.
        task_text must be the real instruction, not the artifact.
        """
        lines = [
            {"type": "user", "timestamp": "2026-06-18T08:00:00.000Z",
             "message": {"role": "user",
                         "content": "<command-name>/clear</command-name>"}},
            {"type": "user", "timestamp": "2026-06-18T08:00:01.000Z",
             "message": {"role": "user",
                         "content": "Fix the failing auth test in src/auth.py"}},
            {"type": "assistant", "timestamp": "2026-06-18T08:00:02.000Z",
             "message": {"role": "assistant", "content": [
                 {"type": "tool_use", "id": "toolu_x01", "name": "Read",
                  "input": {"file_path": "src/auth.py"}}
             ]}},
        ]
        run = load_session(self._make_transcript(tmp_path, lines))
        assert run.task_text == "Fix the failing auth test in src/auth.py"

    def test_skips_slash_model_finds_real_instruction(self, tmp_path):
        """F4.1: first message is /model artifact; real instruction follows."""
        lines = [
            {"type": "user", "timestamp": "2026-06-18T08:00:00.000Z",
             "message": {"role": "user", "content": "/model"}},
            {"type": "user", "timestamp": "2026-06-18T08:00:01.000Z",
             "message": {"role": "user", "content": "Refactor the database module"}},
        ]
        run = load_session(self._make_transcript(tmp_path, lines))
        assert run.task_text == "Refactor the database module"

    def test_only_artifacts_returns_none(self, tmp_path):
        """F4.1: when all user messages are artifacts, task_text must be None."""
        lines = [
            {"type": "user", "timestamp": "2026-06-18T08:00:00.000Z",
             "message": {"role": "user",
                         "content": "<command-name>/clear</command-name>"}},
            {"type": "user", "timestamp": "2026-06-18T08:00:01.000Z",
             "message": {"role": "user", "content": "/compact"}},
        ]
        run = load_session(self._make_transcript(tmp_path, lines))
        assert run.task_text is None

    def test_normal_first_message_still_captured(self, tmp_path):
        """Regression: normal first user message is still captured as task_text."""
        lines = [
            {"type": "user", "timestamp": "2026-06-18T08:00:00.000Z",
             "message": {"role": "user", "content": "Add unit tests for the parser"}},
        ]
        run = load_session(self._make_transcript(tmp_path, lines))
        assert run.task_text == "Add unit tests for the parser"


# ─────────────────────────────────────────────────────────────────────────────
# F4.1: no-task run → zero AUTONOMOUS flags; render shows detection message
# ─────────────────────────────────────────────────────────────────────────────


class TestNoTaskRenderAndClassify:
    def test_no_task_ordinary_action_zero_autonomous_from_transcript(self, tmp_path):
        """F4.1/F4.2: session with only slash-command artifacts → task_text None.
        Ordinary consequential action (Edit to non-secret file) → zero AUTONOMOUS flags.
        (HIGH_CONSEQUENCE actions like git push still fire — tested separately.)
        """
        from unasked.classify import classify_run

        lines = [
            {"type": "user", "timestamp": "2026-06-18T08:00:00.000Z",
             "message": {"role": "user",
                         "content": "<command-name>/clear</command-name>"}},
            {"type": "assistant", "timestamp": "2026-06-18T08:00:01.000Z",
             "message": {"role": "assistant", "content": [
                 {"type": "tool_use", "id": "toolu_n01", "name": "Edit",
                  "input": {"file_path": "src/models.py", "old_string": "x", "new_string": "y"}},
             ]}},
            {"type": "user", "timestamp": "2026-06-18T08:00:02.000Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "toolu_n01", "content": "ok"}
             ]}},
        ]
        path = tmp_path / "notask-session.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        run = load_session(str(path))
        assert run.task_text is None
        classify_run(run)
        autonomous = [d for d in run.decisions if d.provenance == "AUTONOMOUS"]
        assert len(autonomous) == 0, (
            f"F4.1/F4.2: expected 0 AUTONOMOUS flags for ordinary action with no task, "
            f"got {len(autonomous)}"
        )

    def test_no_task_render_shows_detected_message(self, tmp_path):
        """F4.1: render shows '(no explicit task detected)' when task_text is None."""
        from unasked.classify import classify_run
        from unasked.render import render_receipt

        lines = [
            {"type": "user", "timestamp": "2026-06-18T08:00:00.000Z",
             "message": {"role": "user",
                         "content": "<command-name>/clear</command-name>"}},
        ]
        path = tmp_path / "notask2-session.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        run = load_session(str(path))
        classify_run(run)
        receipt = render_receipt(run, color=False)
        assert "no explicit task detected" in receipt
