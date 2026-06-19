"""Command-line entry point for unasked.

Usage
-----
    unasked review <session>       # path to .jsonl OR bare session UUID
    unasked review --last          # most recent transcript under ~/.claude/projects/

Flags
-----
    --save          Persist the classified Run to the ledger (default off).
    --strict        Exit 1 when any flagged steps exist (for CI / pre-merge).
    --no-color      Suppress ANSI colour codes (also honoured: NO_COLOR env var).
    -h / --help     Print usage and exit 0.

Exit codes
----------
    0  — receipt printed, no flagged steps (or --strict not set)
    1  — flagged steps found with --strict; or bad args / unknown command
    2  — session / file not found
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from unasked.adapters.claude_code import load_session
from unasked.classify import classify_run
from unasked.render import render_receipt


# ── --last resolution ─────────────────────────────────────────────────────────

_PROJECTS_ROOT = Path.home() / ".claude" / "projects"


def _most_recent_transcript() -> Path:
    """Return the most recently modified .jsonl under ~/.claude/projects/."""
    candidates = list(_PROJECTS_ROOT.glob("**/*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No transcripts found under {_PROJECTS_ROOT}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ── Color detection ───────────────────────────────────────────────────────────

def _want_color(no_color_flag: bool) -> bool:
    """True when ANSI colour should be emitted."""
    if no_color_flag:
        return False
    if os.environ.get("NO_COLOR", ""):
        return False
    return sys.stdout.isatty()


# ── review subcommand ─────────────────────────────────────────────────────────

def _cmd_review(args: argparse.Namespace) -> int:
    """Execute `unasked review ...`. Returns exit code."""
    color = _want_color(args.no_color)

    # Resolve source.
    if args.last:
        try:
            source = str(_most_recent_transcript())
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif args.session:
        source = args.session
    else:
        print("error: supply a <session> argument or --last", file=sys.stderr)
        return 1

    # Load.
    try:
        run = load_session(source)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Classify.
    classify_run(run)

    # Optionally persist.
    if args.save:
        from unasked.ledger import save_run
        save_run(run)

    # Render.
    receipt = render_receipt(run, color=color)
    print(receipt)

    # Strict mode: exit 1 when any step was flagged.
    if args.strict:
        flagged = any(
            d.provenance in ("AUTONOMOUS", "TOOL_INDUCED") or d.scope_drift
            for d in run.decisions
            if d.provenance is not None
        )
        if flagged:
            return 1

    return 0


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unasked",
        description="See what your agent did without being asked.",
    )
    sub = parser.add_subparsers(dest="command")

    review = sub.add_parser(
        "review",
        help="Print a receipt for one agent run.",
    )
    review.add_argument(
        "session",
        nargs="?",
        metavar="<session>",
        help="Path to a .jsonl transcript or bare session UUID.",
    )
    review.add_argument(
        "--last",
        action="store_true",
        help="Use the most recently modified transcript under ~/.claude/projects/.",
    )
    review.add_argument(
        "--save",
        action="store_true",
        help="Persist the classified run to the ledger.",
    )
    review.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any flagged steps exist (for CI / pre-merge hooks).",
    )
    review.add_argument(
        "--no-color",
        dest="no_color",
        action="store_true",
        help="Disable ANSI colour codes.",
    )

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Console script entry point wired by pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "review":
        sys.exit(_cmd_review(args))
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
