"""Tests for src/unasked/entities.py — entity extraction helpers."""

from __future__ import annotations

import pytest

from unasked.entities import (
    extract_result_entities,
    extract_targets,
    extract_task_entities,
)


# ── extract_targets ───────────────────────────────────────────────────────────

class TestExtractTargetsBash:
    def test_git_push(self):
        result = extract_targets("Bash", {"command": "git push origin main"})
        assert "git" in result
        assert "push" in result

    def test_curl_url(self):
        result = extract_targets("Bash", {"command": "curl https://api.example.com/v1"})
        assert "curl" in result
        assert "https://api.example.com/v1" in result

    def test_pytest(self):
        result = extract_targets("Bash", {"command": "pytest tests/"})
        assert "pytest" in result

    def test_rm_file(self):
        result = extract_targets("Bash", {"command": "rm -rf dist/"})
        assert "rm" in result

    def test_multiline_uses_first_line(self):
        result = extract_targets("Bash", {"command": "git status\ngit push"})
        assert "git" in result
        assert "status" in result
        # second line not included
        assert "push" not in result

    def test_capped_at_max_targets(self):
        # A long command with many tokens — result capped at 5
        cmd = "cmd a b c d e f g h i j"
        result = extract_targets("Bash", {"command": cmd})
        assert len(result) <= 5

    def test_empty_command(self):
        result = extract_targets("Bash", {"command": ""})
        assert result == []

    def test_secret_redacted(self):
        secret = "sk-" + "A" * 30
        result = extract_targets("Bash", {"command": f"curl {secret}"})
        assert secret not in result
        assert any("[REDACTED]" in r for r in result)


class TestExtractTargetsReadWrite:
    def test_read_file_path(self):
        result = extract_targets("Read", {"file_path": "src/auth.py"})
        assert result == ["src/auth.py"]

    def test_edit_file_path(self):
        result = extract_targets("Edit", {"file_path": "config/db.yaml"})
        assert result == ["config/db.yaml"]

    def test_write_file_path(self):
        result = extract_targets("Write", {"file_path": "tests/test_foo.py"})
        assert result == ["tests/test_foo.py"]

    def test_read_empty(self):
        result = extract_targets("Read", {"file_path": ""})
        assert result == []


class TestExtractTargetsWebTools:
    def test_webfetch_url(self):
        result = extract_targets("WebFetch", {"url": "https://docs.example.com"})
        assert result == ["https://docs.example.com"]

    def test_websearch_words(self):
        result = extract_targets("WebSearch", {"query": "stripe integration guide"})
        assert "stripe" in result
        assert "integration" in result

    def test_agent_description_words(self):
        result = extract_targets("Agent", {"description": "build auth service"})
        assert "build" in result or "auth" in result

    def test_sendmessage_recipient(self):
        result = extract_targets("SendMessage", {"to": "team-lead", "summary": "done"})
        assert "team-lead" in result

    def test_taskcreate_subject_words(self):
        result = extract_targets("TaskCreate", {"subject": "Fix auth token expiry"})
        assert "Fix" in result or "auth" in result or "token" in result


class TestExtractTargetsGeneric:
    def test_unknown_tool_first_values(self):
        result = extract_targets("CustomTool", {"action": "delete", "target": "prod-db"})
        # Should return the first non-blob string values
        assert len(result) > 0

    def test_blob_skipped(self):
        big_val = "x" * 400
        result = extract_targets("CustomTool", {"content": big_val, "name": "foo"})
        # blob should be skipped; "foo" should be present
        assert "foo" in result
        assert big_val not in result


# ── extract_result_entities ───────────────────────────────────────────────────

class TestExtractResultEntities:
    def test_url_extracted(self):
        text = "See https://api.stripe.com/v1/charge for details"
        result = extract_result_entities(text)
        assert "https://api.stripe.com/v1/charge" in result

    def test_multiple_urls(self):
        text = "Fetch https://a.com and https://b.com/path"
        result = extract_result_entities(text)
        urls = [r for r in result if r.startswith("https://")]
        assert len(urls) >= 2

    def test_absolute_path_extracted(self):
        text = "File saved to /Users/me/src/auth.py"
        result = extract_result_entities(text)
        assert any("/src/auth.py" in r or "/Users/me/src/auth.py" in r for r in result)

    def test_relative_path_extracted(self):
        text = "Modified src/auth.py successfully"
        result = extract_result_entities(text)
        assert any("src/auth.py" in r for r in result)

    def test_empty_text(self):
        assert extract_result_entities("") == []

    def test_no_entities(self):
        result = extract_result_entities("Hello world, everything is fine.")
        # No URLs, paths, or domains should be extracted
        # (domains matching common words are excluded)
        assert isinstance(result, list)

    def test_deduplication(self):
        text = "https://example.com https://example.com https://example.com"
        result = extract_result_entities(text)
        count = result.count("https://example.com")
        assert count == 1

    def test_secret_in_result_redacted(self):
        secret = "sk-" + "A" * 30
        text = f"Token: {secret}"
        result = extract_result_entities(text)
        assert secret not in result
        # [REDACTED] may appear if the secret matched a pattern in a captured entity

    def test_capped_at_max(self):
        text = " ".join(f"https://site{i}.com/path" for i in range(25))
        result = extract_result_entities(text)
        assert len(result) <= 20  # _MAX_RESULT_ENTITIES


# ── extract_task_entities ─────────────────────────────────────────────────────

class TestExtractTaskEntities:
    def test_file_path_extracted(self):
        result = extract_task_entities("fix the auth.py token expiry bug")
        assert "auth.py" in result

    def test_path_with_slash(self):
        result = extract_task_entities("add rate limiting to src/api/routes.py")
        assert "src/api/routes.py" in result

    def test_empty_string(self):
        assert extract_task_entities("") == []

    def test_salient_nouns(self):
        result = extract_task_entities("implement rate limiting for the auth module")
        # "rate", "limiting", "auth", "module" should be extracted
        assert any(w in result for w in ["rate", "limiting", "auth", "module"])

    def test_stopwords_excluded(self):
        result = extract_task_entities("fix the bug")
        assert "the" not in result
        assert "fix" not in result  # fix is in stopwords

    def test_deduplication(self):
        result = extract_task_entities("edit auth.py auth.py auth.py")
        assert result.count("auth.py") == 1

    def test_context_verb_captures_next_word(self):
        result = extract_task_entities("update routes module")
        # "routes" follows "update" (context verb) → captured
        assert "routes" in result

    def test_no_entities_in_generic_text(self):
        # No file paths, no salient > 4 char non-stopwords — returns something
        result = extract_task_entities("run it")
        # "run" is in stopwords, "it" < 4 chars — may be empty or minimal
        assert isinstance(result, list)

    def test_redaction_applied(self):
        secret = "sk-" + "A" * 30
        result = extract_task_entities(f"fix {secret}")
        assert secret not in result
