"""
Pydantic data models for the Optexity action cache.

Each `CachedAction` represents one deterministic step derived from a
browser-use agent run — it stores the best Playwright locator command,
the Optexity action type, and enough metadata to fall back gracefully.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class CachedAction(BaseModel):
    """One deterministic step extracted from a browser-use agent history."""

    step_index: int
    """Original position in the agent history (0-based)."""

    browser_use_action_type: str
    """The raw browser-use tool name (e.g. 'click', 'input', 'navigate')."""

    optexity_action_type: str
    """The Optexity InteractionAction field name (e.g. 'click_element', 'input_text')."""

    command: str
    """Playwright locator expression WITHOUT the 'page.' prefix.
    Stored this way because browser.get_locator_from_command(cmd) evals f'page.{cmd}'.
    Example: "locator(\"input[name='01___title']\")"
    """

    prompt_instructions: str
    """Human-readable description for LLM fallback if the command fails."""

    action_params: dict[str, Any] = Field(default_factory=dict)
    """Extra action parameters beyond the locator (e.g. {'text': 'myname'} for input)."""

    source_url: str = ""
    """Page URL at the time this action was recorded."""

    element_description: str = ""
    """Short human-readable element identifier for logs (tag + id/name + ax_name)."""

    locator_score: int = 0
    """Stability score from LocatorExtraction (higher = more stable locator)."""

    locator_kind: str = ""
    """Kind of locator used (e.g. 'name', 'id', 'role+name', 'xpath')."""

    success_count: int = 1
    """Number of times this cached step has executed successfully."""

    failure_count: int = 0
    """Number of times this cached step has failed."""

    upgrade_locked: bool = False
    """Set True once a locator upgrade attempt found no improvement.
    Prevents repeated expensive locator_from_playwright() JS evaluations
    on subsequent cache hits for locators that are already optimal (e.g.
    xpath elements with no stable id/role/href alternative)."""


class RunMetrics(BaseModel):
    """Performance metrics for a single execution run."""

    elapsed_seconds: float
    """Wall-clock time for this run."""

    llm_calls: int = 0
    """Number of LLM API calls made (0 for cache hits)."""

    step_count: int = 0
    """Number of agent/action steps executed."""

    raw_step_count: int | None = None
    """Raw actions before pruning (agentic runs only)."""

    pruned_count: int | None = None
    """Steps removed by the optimizer (agentic runs only)."""

    was_cache_hit: bool = False
    """True if this run replayed from cache rather than running the agent."""

    locators_upgraded: int = 0
    """Number of locators upgraded to more stable versions during replay."""


class AgenticTaskCache(BaseModel):
    """Full cache entry for one agentic task + URL combination."""

    task_hash: str
    """SHA-256 prefix of '{task_text}:{base_url}' used as the cache key."""

    url_pattern: str
    """Normalized base URL the task runs against."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_run_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    run_count: int = 1
    """Total number of times this cache entry has been used (cache hits)."""

    original_task: str = ""
    """The original agentic task text, stored for reference."""

    actions: list[CachedAction] = Field(default_factory=list)
    """Ordered list of deterministic steps to replay."""

    # ── Performance metrics ──────────────────────────────────────────────────
    agentic_metrics: RunMetrics | None = None
    """Metrics from the first (agentic) run that established this cache."""

    cache_hit_metrics: list[RunMetrics] = Field(default_factory=list)
    """Metrics for each subsequent cache hit run — tracks improvement over time."""

    raw_history_summary: list[dict[str, Any]] | None = None
    """Serialized summary of the original agent history steps for debugging."""

    def summary(self) -> dict[str, Any]:
        """Return a human-readable comparison of agentic vs cache hit performance."""
        if not self.agentic_metrics or not self.cache_hit_metrics:
            return {"status": "insufficient data"}

        avg_hit_time = sum(m.elapsed_seconds for m in self.cache_hit_metrics) / len(
            self.cache_hit_metrics
        )
        speedup = (
            self.agentic_metrics.elapsed_seconds / avg_hit_time
            if avg_hit_time > 0
            else 0
        )

        return {
            "task_hash": self.task_hash,
            "agentic": {
                "time_s": round(self.agentic_metrics.elapsed_seconds, 2),
                "llm_calls": self.agentic_metrics.llm_calls,
                "steps": self.agentic_metrics.step_count,
                "raw_steps": self.agentic_metrics.raw_step_count,
                "pruned": self.agentic_metrics.pruned_count,
            },
            "cache_hit": {
                "avg_time_s": round(avg_hit_time, 2),
                "llm_calls": 0,
                "steps": len(self.actions),
                "runs": len(self.cache_hit_metrics),
            },
            "speedup": f"{speedup:.1f}x faster",
            "llm_savings": f"100% ({self.agentic_metrics.llm_calls} calls → 0)",
        }
