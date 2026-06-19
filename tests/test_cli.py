"""Tests for the CLI entry point (F4).

Tests main() behaviour via argparse + direct calls where possible,
and via subprocess for end-to-end integration tests.

Coverage
--------
- ``review <session>`` with explicit file path: loads, classifies, prints receipt
- ``review --last``: resolves most recent transcript
- ``--strict``: exit 1 when flagged, exit 0 when clean
- ``--no-color``: suppresses ANSI codes in subprocess output
- ``NO_COLOR`` env var: suppresses ANSI codes
- Unknown command: exit 1
- No args: exit 1 (help printed to stderr)
- Missing session file: exit 2
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ── Subprocess helpers ────────────────────────────────────────────────────────

_UNASKED = ["uv", "run", "unasked"]
# Repo root, derived from this file's location (no hardcoded machine paths).
_CWD = str(Path(__file__).resolve().parent.parent)


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        [*_UNASKED, *args],
        capture_output=True,
        text=True,
        cwd=_CWD,
        env=full_env,
    )


def _make_transcript(tmp_path: Path, session_id: str = "test-session") -> Path:
    """Write a minimal synthetic CC transcript and return its path."""
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "timestamp": "2026-06-19T10:00:00.000Z",
            "sessionId": session_id,
            "message": {"role": "user", "content": "fix auth.py token expiry"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "timestamp": "2026-06-19T10:00:05.000Z",
            "sessionId": session_id,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "Read",
                        "input": {"file_path": "auth.py"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "timestamp": "2026-06-19T10:00:10.000Z",
            "sessionId": session_id,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": "def authenticate(): pass",
                    }
                ],
            },
        },
    ]
    p = tmp_path / f"{session_id}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


def _make_flagged_transcript(tmp_path: Path, session_id: str = "flagged-session") -> Path:
    """Write a transcript that will produce at least one AUTONOMOUS step."""
    records = [
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": None,
            "timestamp": "2026-06-19T10:00:00.000Z",
            "sessionId": session_id,
            "message": {"role": "user", "content": "fix auth.py"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "timestamp": "2026-06-19T10:00:05.000Z",
            "sessionId": session_id,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "Bash",
                        "input": {"command": "git push origin main"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "timestamp": "2026-06-19T10:00:15.000Z",
            "sessionId": session_id,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": "Everything up-to-date",
                    }
                ],
            },
        },
    ]
    p = tmp_path / f"{session_id}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


# ── Basic usage ───────────────────────────────────────────────────────────────


class TestBasicUsage:
    def test_no_args_exits_nonzero(self):
        result = _run_cli()
        assert result.returncode != 0

    def test_unknown_command_exits_nonzero(self):
        result = _run_cli("frobnicate")
        assert result.returncode != 0

    def test_help_flag_exits_0(self):
        result = _run_cli("--help")
        assert result.returncode == 0

    def test_review_help_exits_0(self):
        result = _run_cli("review", "--help")
        assert result.returncode == 0

    def test_missing_session_exits_2(self, tmp_path):
        result = _run_cli("review", str(tmp_path / "nonexistent.jsonl"))
        assert result.returncode == 2
        assert "error" in result.stderr.lower()

    def test_nonexistent_session_id_exits_2(self):
        result = _run_cli("review", "no-such-session-id-xyz-abc")
        assert result.returncode == 2


# ── review <file> ─────────────────────────────────────────────────────────────


class TestReviewFile:
    def test_clean_transcript_exit_0(self, tmp_path):
        p = _make_transcript(tmp_path)
        result = _run_cli("review", str(p))
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_receipt_printed_to_stdout(self, tmp_path):
        p = _make_transcript(tmp_path)
        result = _run_cli("review", str(p))
        assert "task:" in result.stdout
        assert "verdict:" in result.stdout

    def test_task_text_in_output(self, tmp_path):
        p = _make_transcript(tmp_path)
        result = _run_cli("review", str(p))
        assert "fix auth.py token expiry" in result.stdout

    def test_step_count_in_output(self, tmp_path):
        p = _make_transcript(tmp_path)
        result = _run_cli("review", str(p))
        assert "1 steps" in result.stdout or "step" in result.stdout


# ── --strict ──────────────────────────────────────────────────────────────────


class TestStrictFlag:
    def test_strict_exits_0_on_clean_run(self, tmp_path):
        p = _make_transcript(tmp_path)
        result = _run_cli("review", str(p), "--strict")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_strict_exits_1_on_flagged_run(self, tmp_path):
        p = _make_flagged_transcript(tmp_path)
        result = _run_cli("review", str(p), "--strict")
        # git push should be AUTONOMOUS → strict exits 1
        assert result.returncode == 1, (
            f"Expected exit 1 for flagged run, got {result.returncode}. "
            f"stdout: {result.stdout}"
        )

    def test_strict_still_prints_receipt(self, tmp_path):
        p = _make_flagged_transcript(tmp_path)
        result = _run_cli("review", str(p), "--strict")
        assert "verdict:" in result.stdout


# ── --no-color / NO_COLOR ─────────────────────────────────────────────────────


class TestColor:
    def test_no_color_flag_suppresses_ansi(self, tmp_path):
        p = _make_flagged_transcript(tmp_path)
        result = _run_cli("review", str(p), "--no-color")
        assert "\033[" not in result.stdout

    def test_no_color_env_suppresses_ansi(self, tmp_path):
        p = _make_flagged_transcript(tmp_path)
        result = _run_cli("review", str(p), env={"NO_COLOR": "1"})
        assert "\033[" not in result.stdout


# ── --last ────────────────────────────────────────────────────────────────────


class TestLastFlag:
    def test_last_finds_most_recent(self, tmp_path, monkeypatch):
        """--last resolves via mtime; point PROJECTS_ROOT at a tmp dir."""
        import unasked.cli as cli_mod

        proj_dir = tmp_path / "projects" / "proj"
        proj_dir.mkdir(parents=True)

        old = proj_dir / "old-session.jsonl"
        new = proj_dir / "new-session.jsonl"

        # Write both transcripts
        for p, content_task in [(old, "old task"), (new, "fix auth.py")]:
            records = [
                {
                    "type": "user", "uuid": "u1", "parentUuid": None,
                    "timestamp": "2026-06-19T10:00:00.000Z",
                    "sessionId": p.stem,
                    "message": {"role": "user", "content": content_task},
                }
            ]
            p.write_text("\n".join(json.dumps(r) for r in records))

        # Set mtime so new > old
        os.utime(old, (1000000, 1000000))
        os.utime(new, (2000000, 2000000))

        # Patch PROJECTS_ROOT inside the cli module
        monkeypatch.setattr(cli_mod, "_PROJECTS_ROOT", tmp_path / "projects")

        from unasked.adapters.claude_code import load_session
        from unasked.classify import classify_run
        from unasked.render import render_receipt

        # Call _most_recent_transcript directly to check it picks the right file
        result = cli_mod._most_recent_transcript()
        assert result == new

    def test_last_raises_when_no_transcripts(self, tmp_path, monkeypatch):
        import unasked.cli as cli_mod
        monkeypatch.setattr(cli_mod, "_PROJECTS_ROOT", tmp_path / "empty")
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            cli_mod._most_recent_transcript()


# ── --save ────────────────────────────────────────────────────────────────────


class TestSaveFlag:
    def test_save_persists_to_ledger(self, tmp_path):
        """--save writes the run to the ledger without error."""
        p = _make_transcript(tmp_path)
        db = tmp_path / "ledger.db"
        # We can't pass --save with a custom db path from CLI, but we can
        # check that --save doesn't crash and receipt is still printed.
        result = _run_cli(
            "review", str(p), "--save",
            env={"UNASKED_LEDGER_DB": str(db)},
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "verdict:" in result.stdout
        # Ledger DB should have been created
        assert db.exists()
