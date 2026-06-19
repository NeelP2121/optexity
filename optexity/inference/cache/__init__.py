"""
Optexity Memory / Caching Layer

Converts the first successful browser-use agentic run into a sequence of
deterministic Playwright `command`-based ActionNodes. Subsequent runs replay
those cached steps directly — zero LLM calls — and fall back to the full
agentic loop only when a cached step fails.

Submodules (imported lazily to avoid circular settings imports):
    action_cache      — CachedAction, AgenticTaskCache Pydantic models
    cache_manager     — load / save / invalidate / record_failure
    history_converter — AgentHistoryList → list[CachedAction]
    cache_runner      — deterministic replay via run_interaction_action
    schema_builder    — [Bonus 1] LLM-based schema builder with self-healing
    optimizer         — [Bonus 2] iterative pruning + convergence loop
"""
