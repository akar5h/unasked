# unasked

**See what your agent did *without being asked*.**

`unasked` reads one AI coding-agent run and prints a 60-second receipt that surfaces only the steps worth your attention — the ones the agent did on its own authority, the ones steered by external content, and the ones that wandered outside the task — so you don't have to read all 40 steps to trust it.

Local-first. Zero runtime dependencies. Nothing leaves your machine.

```
run 986bc712 — 172 steps · task: "fix the auth token-expiry bug"

⚠ did WITHOUT being asked (3)
  #28  Bash   git push origin main                    autonomous
  #31  Edit   config/db.yaml                          autonomous · scope-drift
  #12  Read   .env                                    autonomous · scope-drift

⚡ steered by external content (1)
  #19  Bash   curl https://api.thirdparty.com/v2      tool-induced

✓ 167 routine (mapped to your request)

verdict: 4 to eyeball, 167 routine.
```

---

## Why this exists

In 2026 the bottleneck of agentic coding isn't writing code — it's *verifying* it.

- 96% of developers don't fully trust AI-generated code; only ~48% verify it ("the verification gap").
- 38% say reviewing AI output takes **more** effort than reviewing a human's; AI-heavy PRs take ~26% longer to review.
- The pain has names now: *verification debt*, *comprehension debt*, *the babysitting problem*.

Agents shifted the job from "check each suggestion" to "oversee an entire autonomous sequence." Existing tools answer **what happened** (trace viewers) or **block by policy** (enterprise guardrails). Almost nobody answers the question a developer actually has after a run:

> Of everything the agent did, **which did it do because I asked — and which did it decide on its own, or get talked into by something it read?**

That's the gap `unasked` fills. Reviewing a *receipt* — the few flagged steps with a one-line why — is faster than re-reading the whole session, and it works for runs you weren't watching.

## What it flags (authorization provenance)

Every tool call the agent made is classified by its **trigger**:

| Class | Meaning | Shown? |
|---|---|---|
| `REQUESTED` | maps to your explicit instruction | no (routine) |
| `DERIVED` | a natural follow-on from the task | no (routine) |
| **`AUTONOMOUS`** | the agent did it on its own authority | **flagged** — "did without being asked" |
| **`TOOL_INDUCED`** | steered by content returned from a tool (web/MCP) | **flagged** — injection / lethal-trifecta signal |
| **`scope_drift`** | touched files/resources outside the stated task | **flagged** — "touched outside task scope" |

Classification is **deterministic and precision-first**: when uncertain, it stays quiet. A receipt that cries wolf is worthless, so ambiguous steps fall through to "routine" rather than getting flagged.

## Install & quickstart

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/akar5h/unasked.git
cd unasked
uv sync

# review your most recent Claude Code session
uv run unasked review --last

# review a specific session (path or session-id)
uv run unasked review ~/.claude/projects/<project>/<session>.jsonl
uv run unasked review 986bc712-...

# CI / pre-merge gate: non-zero exit if anything is flagged
uv run unasked review --last --strict
```

Flags: `--last`, `--save` (persist to a local ledger), `--strict` (exit 1 when flagged), `--no-color` (also honors `NO_COLOR`).

## How it works

```
~/.claude transcript ─┐
                      ├─ adapter ─► Decision IR ─► classifier ─► receipt
(LangChain, soon)  ───┘  (redacted)   (Run/Decision)  (provenance + scope-drift)
```

1. **Adapter** reads the native Claude Code transcript (`~/.claude/projects/**/*.jsonl`) — zero setup, works on your existing history. (An opt-in capture-time path reads a hook spool; a LangChain adapter is on the roadmap.)
2. **Redaction at the boundary** — only short, redacted argument summaries enter the IR. Secrets (API keys, tokens, DB URLs, PEM/JWT, etc.) are masked; raw file contents and tool outputs are never stored.
3. **Classifier** assigns provenance + scope-drift using structural rules over the event graph — no LLM, no network.
4. **Receipt** groups the flagged steps with a one-line why and a verdict.

### Privacy

`unasked` runs entirely locally and makes no network calls. The optional local ledger (SQLite) stores only redacted summaries — never secrets or raw content. The forthcoming LLM-`why` enrichment (opt-in) will send only tool names + redacted summaries + the stated task, never raw arguments or file contents.

## Status

**v0.1.0 — early. `unasked review` works end-to-end on real Claude Code sessions (240 passing tests).** Treat it as a preview.

Known limitations (precision tuning in progress):

- **TOOL_INDUCED can over-fire** when the agent greps then reads files in its own repo — local file paths surfaced by a local tool shouldn't count as "external content." The signal is being narrowed to genuinely external sources (WebFetch/WebSearch, URLs/domains).
- **Task detection is naive.** If a session's first message is a slash-command artifact (e.g. `/clear`, `/model`) rather than a real instruction, the task is misread and flag counts inflate. Real-instruction extraction is being hardened.

If you hit a false flag, that's the bug we're hunting — open an issue with the (redacted) receipt.

## Roadmap

- **Precision pass** — narrow TOOL_INDUCED to external sources; robust task extraction.
- **LLM `why` enrichment** (opt-in) — a sharper one-line rationale per flagged step (redacted inputs only).
- **LangChain / framework adapters** — capture the real reasoning (ReAct thought) where we own the loop; one IR, many adapters.
- **Minimal local UI** — scan a run, mark a step "fine" / "not allowed."
- **Feedback → personal policy** — the tool learns *your* sense of what's out-of-bounds and pre-flags similar steps next time.
- **Future** — browser-agent workflows; a mirroring partner that learns your workflow over time.

The throughline: make trusting an autonomous agent run cheap, local, and framework-agnostic — without enforcement, without a server, without your data leaving your machine.

## Development

```bash
uv sync
uv run pytest -v
```

Zero runtime dependencies (stdlib only). Tests use synthetic fixtures — no real session data is committed.

## License

MIT
