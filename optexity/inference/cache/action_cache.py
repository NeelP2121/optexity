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

    raw_history_summary: list[dict[str, Any]] | None = None
    """Serialized summary of the original agent history steps for debugging."""
