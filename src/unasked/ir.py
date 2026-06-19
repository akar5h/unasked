"""Decision-Event IR for unasked.

Two lean dataclasses represent one agent run as an ordered list of tool decisions.
No OTel/span machinery — just enough shape to feed the ledger store and,
later, the classifier and CLI.

Intentionally zero-dependency (stdlib dataclasses only).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    """One tool invocation decision made by the agent during a run.

    Fields
    ------
    step_index : int
        Monotonic position of this decision within the run (0-based).
    ts : str | None
        ISO-8601 timestamp when the tool call was emitted, if known.
    tool_name : str
        Name of the tool the agent chose to invoke (e.g. "Bash", "Read").
    tool_args_summary : str
        Short, redacted description of the arguments — NOT the raw args.
        Callers are responsible for summarising/redacting before building
        a Decision; raw secrets must never appear here.
    is_error : bool
        True when the tool call resulted in an error response.
    parent_step_index : int | None
        step_index of the causal parent decision (e.g. the LLM call that
        produced this tool call). None when no parent is known or applicable.

    Entity fields (populated by the adapter):

    result_entities : list[str]
        Redacted URLs, domains, and file paths extracted from THIS decision's
        tool_result content.  Raw result text is never stored.  Used by the
        classifier to detect TOOL_INDUCED steps (the agent acted on something
        that originated from a prior tool's output, not from the user).

    Provenance / classification fields (populated by later features):

    provenance : str | None
        Classification label assigned by the classifier, e.g. "AUTONOMOUS",
        "TOOL_INDUCED", "USER_DIRECTED". None until classified.
    scope_drift : bool | None
        True when the decision touched a file or resource outside the stated
        task scope. None until scope analysis runs.
    why : str | None
        Human-readable explanation of the provenance classification.
    feedback : str | None
        Operator or user override annotation applied after review.
    """

    step_index: int
    ts: str | None
    tool_name: str
    tool_args_summary: str
    # Entity fields — populated by the adapter.
    # targets: redacted identifying entities this action operates on (file paths,
    # command verbs, URLs).  Extracted from tool_input at parse time.
    targets: list[str] = field(default_factory=list)
    # result_entities: redacted URLs, domains, and file paths found in THIS
    # decision's tool_result content.  Raw result text is never stored — only
    # the extracted, redacted entity strings.  Used by the classifier to detect
    # TOOL_INDUCED decisions (agent acted on something from a prior tool output).
    result_entities: list[str] = field(default_factory=list)
    is_error: bool = False
    parent_step_index: int | None = None
    # Provenance fields — populated by later features, nullable by design.
    provenance: str | None = None
    scope_drift: bool | None = None
    why: str | None = None
    feedback: str | None = None


@dataclass
class Run:
    """An ordered sequence of tool decisions from one agent run.

    Fields
    ------
    run_id : str
        Stable unique identifier for this run (e.g. a trace-id or UUID).
    source : str
        Which agent runtime produced this run (e.g. "claude_code", "openai").
    task_text : str | None
        The task or prompt the user gave the agent at the start of the run.
        None when not captured.
    started_at : str | None
        ISO-8601 timestamp of when the run began. None when not captured.
    decisions : list[Decision]
        Ordered list of tool decisions, ascending by step_index.
    """

    run_id: str
    source: str
    task_text: str | None = None
    started_at: str | None = None
    decisions: list[Decision] = field(default_factory=list)
