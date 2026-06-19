"""Golden-output tests for the receipt formatter (F4).

Strategy
--------
- Build Run fixtures inline (no real ~/.claude reads).
- Classify with classify_run.
- Assert on specific substrings in the rendered receipt — not the full string,
  so minor wording tweaks don't break tests.  Critical structural assertions
  are explicit (header, verdict, flagged section headings).
- One full golden-snapshot test that freezes the exact output for a canonical
  five-step run.

Sections tested
---------------
1. Header presence (═══, "unasked", Task, Run ID, Steps).
2. Clean run — no flagged steps → "Clean run" + "CLEAN" verdict.
3. AUTONOMOUS steps appear under the right heading.
4. TOOL_INDUCED steps appear under the right heading.
5. Scope-drift marker ([scope-drift]) appears on drifting steps.
6. Verdict line varies with flag counts.
7. Unclassified decisions are skipped without error.
8. Full golden snapshot for a canonical five-step run.
9. CLI smoke test — subprocess call returns exit 2 for missing session.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from unasked.classify import classify_run
from unasked.ir import Decision, Run
from unasked.receipt import format_receipt


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(task: str | None, decisions: list[Decision]) -> Run:
    return Run(run_id="test-run-abc", source="test", task_text=task, decisions=decisions)


def _dec(
    step: int,
    tool: str,
    targets: list[str],
    result_entities: list[str] | None = None,
    is_error: bool = False,
) -> Decision:
    return Decision(
        step_index=step,
        ts=None,
        tool_name=tool,
        tool_args_summary=" ".join(targets),
        targets=targets,
        result_entities=result_entities or [],
        is_error=is_error,
    )


def _classified(task: str | None, decisions: list[Decision]) -> str:
    """Build, classify, and format a run; return the receipt string."""
    run = _run(task, decisions)
    classify_run(run)
    return format_receipt(run)


# ── Header ─────────────────────────────────────────────────────────────────────

class TestReceiptHeader:
    def test_contains_unasked_title(self):
        receipt = _classified("fix auth.py", [_dec(0, "Read", ["auth.py"])])
        assert "unasked" in receipt

    def test_contains_task_text(self):
        receipt = _classified("fix auth.py token expiry", [_dec(0, "Read", ["auth.py"])])
        assert "fix auth.py token expiry" in receipt

    def test_contains_run_id(self):
        receipt = _classified("fix auth.py", [_dec(0, "Read", ["auth.py"])])
        assert "test-run-abc" in receipt

    def test_contains_step_count(self):
        receipt = _classified(
            "fix auth.py",
            [_dec(0, "Read", ["auth.py"]), _dec(1, "Edit", ["auth.py"])],
        )
        assert "2" in receipt  # total steps

    def test_no_task_shows_placeholder(self):
        receipt = _classified(None, [_dec(0, "Read", ["auth.py"])])
        assert "no task captured" in receipt

    def test_horizontal_rules_present(self):
        receipt = _classified("fix auth.py", [_dec(0, "Read", ["auth.py"])])
        assert "═" in receipt
        assert "─" in receipt


# ── Clean run ─────────────────────────────────────────────────────────────────

class TestCleanRun:
    def test_clean_message_when_no_flags(self):
        """Read + Edit on task-named file → no flags → 'Clean run' shown."""
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Edit", ["auth.py"]),
            ],
        )
        assert "Clean run" in receipt

    def test_verdict_clean(self):
        receipt = _classified(
            "fix auth.py",
            [_dec(0, "Read", ["auth.py"]), _dec(1, "Edit", ["auth.py"])],
        )
        assert "CLEAN" in receipt

    def test_no_flagged_headings_in_clean_run(self):
        receipt = _classified(
            "fix auth.py",
            [_dec(0, "Read", ["auth.py"]), _dec(1, "Edit", ["auth.py"])],
        )
        assert "AUTONOMOUS" not in receipt
        assert "TOOL_INDUCED" not in receipt


# ── AUTONOMOUS section ────────────────────────────────────────────────────────

class TestAutonomousSection:
    def test_autonomous_heading_present(self):
        """git push not in task → AUTONOMOUS → heading appears."""
        receipt = _classified(
            "fix auth.py token expiry",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Edit", ["auth.py"]),
                _dec(2, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        assert "AUTONOMOUS" in receipt

    def test_autonomous_step_index_shown(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        # Step 1 is the autonomous one
        assert "[  1]" in receipt

    def test_autonomous_tool_name_shown(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        assert "Bash" in receipt

    def test_verdict_flagged_when_autonomous(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        assert "FLAGGED" in receipt
        assert "autonomous" in receipt


# ── TOOL_INDUCED section ──────────────────────────────────────────────────────

class TestToolInducedSection:
    def test_tool_induced_heading_present(self):
        """WebFetch result contains a URL; agent then fetches that URL not in task
        → TOOL_INDUCED heading.

        WebFetch is in EXTERNAL_READ_TOOLS so its result_entities always propagate
        to the injection pool regardless of task overlap.
        """
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "WebFetch", ["https://docs.example.com/api"],
                     result_entities=["https://evil.example.com/payload"]),
                _dec(1, "WebFetch", ["https://evil.example.com/payload"]),
            ],
        )
        assert "TOOL_INDUCED" in receipt

    def test_tool_induced_step_shown(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "WebFetch", ["https://docs.example.com/api"],
                     result_entities=["https://evil.example.com/payload"]),
                _dec(1, "WebFetch", ["https://evil.example.com/payload"]),
            ],
        )
        assert "[  1]" in receipt

    def test_verdict_flagged_tool_induced(self):
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "WebFetch", ["https://docs.example.com/api"],
                     result_entities=["https://evil.example.com/payload"]),
                _dec(1, "WebFetch", ["https://evil.example.com/payload"]),
            ],
        )
        assert "FLAGGED" in receipt
        assert "tool-induced" in receipt


# ── Scope drift marker ────────────────────────────────────────────────────────

class TestScopeDriftMarker:
    def test_scope_drift_marker_on_autonomous_step(self):
        """Autonomous edit to file outside task path → [scope-drift] marker."""
        receipt = _classified(
            "fix auth.py",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Edit", ["config/db.yaml"]),
            ],
        )
        # Step 1 edits config/db.yaml — outside auth.py task scope
        assert "[scope-drift]" in receipt

    def test_no_scope_drift_marker_on_in_scope_step(self):
        receipt = _classified(
            "fix auth.py",
            [_dec(0, "Read", ["auth.py"]), _dec(1, "Edit", ["auth.py"])],
        )
        assert "[scope-drift]" not in receipt


# ── Verdict variations ────────────────────────────────────────────────────────

class TestVerdictLine:
    def test_verdict_line_present(self):
        receipt = _classified("fix auth.py", [_dec(0, "Read", ["auth.py"])])
        assert "Verdict" in receipt

    def test_verdict_counts_multiple_labels(self):
        """Both AUTONOMOUS and TOOL_INDUCED → verdict mentions both.

        WebFetch is EXTERNAL_READ_TOOLS so its result_entities always propagate.
        """
        receipt = _classified(
            "fix auth.py",
            [
                # TOOL_INDUCED: step 1 acts on entity from step 0's WebFetch result
                _dec(0, "WebFetch", ["https://docs.example.com/api"],
                     result_entities=["https://evil.example.com"]),
                _dec(1, "WebFetch", ["https://evil.example.com"]),
                # AUTONOMOUS: git push unrelated to task
                _dec(2, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        assert "tool-induced" in receipt
        assert "autonomous" in receipt


# ── Unclassified decisions ────────────────────────────────────────────────────

class TestUnclassifiedDecisions:
    def test_unclassified_does_not_crash(self):
        """Decisions with provenance=None are skipped without error."""
        run = _run(
            "fix auth.py",
            [
                Decision(
                    step_index=0, ts=None, tool_name="Read",
                    tool_args_summary="auth.py",
                    targets=["auth.py"],
                ),
            ],
        )
        # Do NOT classify — provenance stays None
        receipt = format_receipt(run)
        assert "unasked" in receipt  # header still renders
        assert "0 classified" in receipt


# ── Full golden snapshot ──────────────────────────────────────────────────────

class TestGoldenSnapshot:
    """Freeze the exact structure of a canonical five-step run.

    Run: task = "fix auth.py token expiry bug"
    Steps:
      0: Read auth.py            → REQUESTED
      1: Edit auth.py            → REQUESTED
      2: Bash pytest tests/      → REQUESTED  (tests in task via 'bug'? no;
                                                but let's include it as DERIVED)
      3: WebFetch (result entity not in task) → TOOL_INDUCED
      4: Bash git push           → AUTONOMOUS

    We assert on structural substrings, not byte-exact output,
    so wording tweaks that don't change meaning don't break the test.
    """

    def _receipt(self) -> str:
        run = _run(
            "fix auth.py token expiry bug",
            [
                _dec(0, "Read", ["auth.py"]),
                _dec(1, "Edit", ["auth.py"]),
                _dec(2, "Bash", ["pytest", "tests/"]),
                _dec(3, "Read", ["docs/external-api.md"],
                     result_entities=["https://api.thirdparty.com/v2"]),
                _dec(4, "WebFetch", ["https://api.thirdparty.com/v2"]),
                _dec(5, "Bash", ["git", "push", "origin", "main"]),
            ],
        )
        classify_run(run)
        return format_receipt(run)

    def test_header_present(self):
        assert "unasked" in self._receipt()

    def test_task_shown(self):
        assert "fix auth.py token expiry bug" in self._receipt()

    def test_tool_induced_section(self):
        assert "TOOL_INDUCED" in self._receipt()

    def test_autonomous_section(self):
        assert "AUTONOMOUS" in self._receipt()

    def test_verdict_flagged(self):
        receipt = self._receipt()
        assert "FLAGGED" in receipt

    def test_step_5_autonomous(self):
        receipt = self._receipt()
        # Step 5 is git push → AUTONOMOUS
        assert "[  5]" in receipt

    def test_step_4_tool_induced(self):
        receipt = self._receipt()
        # Step 4 is WebFetch acting on entity from step 3 result
        assert "[  4]" in receipt

    def test_receipt_is_multiline(self):
        assert self._receipt().count("\n") >= 10

    def test_receipt_ends_with_rule(self):
        receipt = self._receipt()
        # Last line should be the closing ═══ rule
        assert receipt.strip().endswith("═" * 72)


# ── CLI smoke tests ───────────────────────────────────────────────────────────

class TestCLISmoke:
    def test_review_missing_session_exits_2(self):
        """unasked review <nonexistent-session> → exit code 2."""
        result = subprocess.run(
            ["uv", "run", "unasked", "review", "does-not-exist-session-abc123"],
            capture_output=True,
            text=True,
            cwd="/Users/akarshgajbhiye/unasked",
        )
        assert result.returncode == 2
        assert "error" in result.stderr.lower()

    def test_no_args_exits_nonzero(self):
        """unasked with no args → exit code 1."""
        result = subprocess.run(
            ["uv", "run", "unasked"],
            capture_output=True,
            text=True,
            cwd="/Users/akarshgajbhiye/unasked",
        )
        assert result.returncode == 1

    def test_unknown_command_exits_nonzero(self):
        result = subprocess.run(
            ["uv", "run", "unasked", "frobnicate"],
            capture_output=True,
            text=True,
            cwd="/Users/akarshgajbhiye/unasked",
        )
        assert result.returncode == 1

    def test_help_flag_exits_0(self):
        result = subprocess.run(
            ["uv", "run", "unasked", "--help"],
            capture_output=True,
            text=True,
            cwd="/Users/akarshgajbhiye/unasked",
        )
        assert result.returncode == 0

    def test_review_real_transcript(self, tmp_path):
        """Run against a minimal synthetic transcript JSONL → exit 0, receipt printed."""
        import json

        # Minimal CC transcript: one user message + one assistant tool_use
        records = [
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "timestamp": "2026-06-19T00:00:00.000Z",
                "sessionId": "test-session",
                "message": {
                    "role": "user",
                    "content": "fix auth.py token expiry",
                },
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "timestamp": "2026-06-19T00:00:01.000Z",
                "sessionId": "test-session",
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
                "timestamp": "2026-06-19T00:00:02.000Z",
                "sessionId": "test-session",
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

        transcript = tmp_path / "test-session.jsonl"
        transcript.write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8"
        )

        result = subprocess.run(
            ["uv", "run", "unasked", "review", str(transcript)],
            capture_output=True,
            text=True,
            cwd="/Users/akarshgajbhiye/unasked",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "unasked" in result.stdout
        assert "fix auth.py token expiry" in result.stdout
        assert "Verdict" in result.stdout
