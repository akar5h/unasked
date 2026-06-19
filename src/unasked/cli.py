"""Command-line entry point for unasked.

Usage
-----
    unasked review <session>

    <session> is either:
      - A file path to a Claude Code ``.jsonl`` transcript, OR
      - A bare session UUID (searched under ~/.claude/projects/).

Pipeline
--------
    load_session  →  classify_run  →  format_receipt  →  stdout

Exit codes
----------
    0  — receipt printed (even if flagged steps were found)
    1  — unrecognised command or missing argument
    2  — session / file not found
"""

from __future__ import annotations

import sys

from unasked.adapters.claude_code import load_session
from unasked.classify import classify_run
from unasked.receipt import format_receipt


def _usage() -> None:
    print("usage: unasked review <session-file-or-uuid>", file=sys.stderr)


def main() -> None:
    """Entry point wired by pyproject.toml ``[project.scripts]``."""
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _usage()
        sys.exit(0 if args and args[0] in ("-h", "--help") else 1)

    cmd = args[0]

    if cmd == "review":
        if len(args) < 2:
            print("error: 'review' requires a session argument", file=sys.stderr)
            _usage()
            sys.exit(1)
        source = args[1]
        try:
            run = load_session(source)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)

        classify_run(run)
        receipt = format_receipt(run)
        print(receipt)

    else:
        print(f"error: unknown command '{cmd}'", file=sys.stderr)
        _usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
