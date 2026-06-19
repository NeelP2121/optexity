import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from browser_use import Agent, BrowserSession, ChatGoogle, Tools
from browser_use.agent.views import AgentHistoryList

from optexity.inference.cache.action_cache import AgenticTaskCache, RunMetrics
from optexity.inference.cache.cache_manager import (
    compute_cache_key,
    load_cache,
    record_failure,
    save_cache,
)
from optexity.inference.cache.cache_runner import run_cached_actions
from optexity.inference.cache.history_converter import convert_history_to_cached_actions
from optexity.inference.cache.optimizer import prune_redundant_actions
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import (
    AgenticTask,
    CloseOverlayPopupAction,
)
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


def _write_cached_automation_json(
    task: Task,
    cache: AgenticTaskCache,
    output_path: str = "test_automation_cached.json",
) -> None:
    """Write a deterministic automation JSON from the cache.

    Each CachedAction becomes an ActionNode with its command field set so
    execution takes the fast 'command-first' path (zero LLM calls).
    Validates with Pydantic before writing so the file is always runnable.
    """
    from optexity.schema.automation import Automation

    nodes = []
    for ca in cache.actions:
        t = ca.optexity_action_type
        p = ca.action_params

        if t == "click_element":
            ia = {
                "click_element": {
                    "command": ca.command,
                    "prompt_instructions": ca.prompt_instructions,
                }
            }
        elif t == "input_text":
            text = p.get("text") or p.get("value") or ""
            ia = {
                "input_text": {
                    "command": ca.command,
                    "prompt_instructions": ca.prompt_instructions,
                    "input_text": text,
                }
            }
        elif t == "select_option":
            value = p.get("value") or p.get("option") or ""
            ia = {
                "select_option": {
                    "command": ca.command,
                    "prompt_instructions": ca.prompt_instructions,
                    "select_values": [value] if value else [],
                }
            }
        elif t == "go_to_url":
            url = p.get("url") or task.automation.url
            ia = {"go_to_url": {"url": url}}
        elif t == "go_back":
            ia = {"go_back": {}}
        elif t == "scroll":
            ia = {
                "scroll": {
                    "down": p.get("direction", "down") != "up",
                    "amount": int(p.get("amount", -1)),
                }
            }
        elif t == "key_press":
            key = p.get("key") or p.get("keys") or "Enter"
            ia = {
                "key_press": {
                    "command": ca.command,
                    "prompt_instructions": ca.prompt_instructions,
                    "type": key,
                }
            }
        else:
            logger.debug(f"[cache] Skipping unknown action type in JSON export: {t}")
            continue

        nodes.append({"type": "action_node", "interaction_action": ia})

    automation_dict = {
        "url": task.automation.url,
        "parameters": {"input_parameters": {}, "generated_parameters": {}},
        "nodes": nodes,
    }

    try:
        Automation.model_validate(automation_dict)
    except Exception as exc:
        logger.error(f"[cache] Generated automation failed validation: {exc}")
        logger.error(f"[cache] Attempted JSON: {json.dumps(automation_dict, indent=2)}")
        return

    Path(output_path).write_text(json.dumps(automation_dict, indent=2))
    logger.info(f"[cache] Wrote {len(nodes)}-node deterministic automation → {output_path}")


def _run_bonus1_schema_builder(
    cache: AgenticTaskCache,
    task: Task,
    output_path: str = "test_automation_cached_llm.json",
) -> None:
    """[Bonus 1] Blocking call — run in executor so it doesn't block the event loop."""
    try:
        from optexity.inference.cache.schema_builder import build_automation_from_cache
        import os
        provider = task.llm_provider or "gemini"
        model = task.llm_model_name or "gemini-2.5-flash"
        automation = build_automation_from_cache(
            cache=cache,
            starting_url=task.automation.url,
            output_path=output_path,
            llm_provider=provider,
            llm_model_name=model,
        )
        logger.info(
            f"[bonus1] LLM schema builder generated {len(automation.nodes)}-node "
            f"automation → {output_path}"
        )
    except Exception as exc:
        logger.error(f"[bonus1] Schema builder failed: {exc}", exc_info=True)


async def handle_agentic_task(
    agentic_task_action: AgenticTask | CloseOverlayPopupAction,
    task: Task,
    memory: Memory,
    browser: Browser,
):

    if agentic_task_action.backend == "browser_use":

        if isinstance(agentic_task_action, CloseOverlayPopupAction):
            tools = Tools(
                exclude_actions=[
                    "search",
                    "navigate",
                    "go_back",
                    "upload_file",
                    "scroll",
                    "find_text",
                    "send_keys",
                    "evaluate",
                    "switch",
                    "close",
                    "extract",
                    "dropdown_options",
                    "select_dropdown",
                    "write_file",
                    "read_file",
                    "replace_file",
                ]
            )
        else:
            tools = Tools()

        # ── Cache check ──────────────────────────────────────────────────────
        cache_key: str | None = None
        if not isinstance(agentic_task_action, CloseOverlayPopupAction):
            base_url = task.automation.url
            # Normalize cache key: replace actual param values with {{names}}
            # so "fill name as myname" and "fill name as John" share a cache entry
            input_params = {
                k: [str(v) for v in vs]
                for k, vs in task.input_parameters.items()
            } if task.input_parameters else None
            cache_key = compute_cache_key(
                agentic_task_action.task, base_url, input_params
            )
            existing_cache = load_cache(cache_key)

            if existing_cache and existing_cache.actions:
                logger.info(
                    f"[cache] HIT — replaying {len(existing_cache.actions)} deterministic "
                    f"steps for task (key={cache_key})"
                )
                t0 = time.perf_counter()
                success, last_step, any_upgraded = await run_cached_actions(
                    existing_cache.actions, task, memory, browser, input_params
                )
                elapsed = time.perf_counter() - t0
                if success:
                    existing_cache.run_count += 1
                    existing_cache.last_run_at = datetime.now(timezone.utc)

                    # Record cache hit metrics
                    existing_cache.cache_hit_metrics.append(RunMetrics(
                        elapsed_seconds=round(elapsed, 2),
                        llm_calls=0,
                        step_count=len(existing_cache.actions),
                        was_cache_hit=True,
                        locators_upgraded=1 if any_upgraded else 0,
                    ))

                    # ── [Bonus 2] Iterative optimizer — prune on every cache hit ──
                    pre_count = len(existing_cache.actions)
                    pruned = prune_redundant_actions(list(existing_cache.actions))
                    post_count = len(pruned)
                    if post_count < pre_count:
                        logger.info(
                            f"[bonus2] Optimizer pruned {pre_count - post_count} redundant "
                            f"steps ({pre_count} → {post_count}). Updating cache."
                        )
                        existing_cache.actions = pruned
                    else:
                        logger.info(
                            f"[bonus2] Optimizer: cache already optimal ({pre_count} steps, "
                            "nothing to prune)"
                        )
                    # ─────────────────────────────────────────────────────────

                    # Save if pruned or locators were upgraded
                    if any_upgraded or post_count < pre_count:
                        _write_cached_automation_json(
                            task, existing_cache, "test_automation_cached.json"
                        )
                    save_cache(cache_key, existing_cache)
                    summary = existing_cache.summary()
                    logger.info(
                        f"[cache] Replay succeeded in {elapsed:.2f}s — "
                        f"0 LLM calls used. (run_count={existing_cache.run_count})"
                    )
                    if "speedup" in summary:
                        logger.info(
                            f"[metrics] Speedup: {summary['speedup']} | "
                            f"LLM savings: {summary['llm_savings']} | "
                            f"Agentic: {summary['agentic']['time_s']}s / "
                            f"Cache: {summary['cache_hit']['avg_time_s']}s"
                        )
                    return
                else:
                    logger.warning(
                        f"[cache] Replay failed at step {last_step}. "
                        "Falling back to full agentic run and refreshing cache."
                    )
                    record_failure(cache_key, last_step)
        # ─────────────────────────────────────────────────────────────────────

        import os
        google_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        llm = ChatGoogle(model="gemini-flash-latest", api_key=google_api_key) if google_api_key else ChatGoogle(model="gemini-flash-latest")
        browser_session = BrowserSession(
            cdp_url=browser.cdp_url, keep_alive=agentic_task_action.keep_alive
        )

        step_directory = (
            task.logs_directory / f"step_{str(memory.automation_state.step_index)}"
        )
        step_directory.mkdir(parents=True, exist_ok=True)

        agent = Agent(
            task=agentic_task_action.task,
            llm=llm,
            browser_session=browser_session,
            use_vision=agentic_task_action.use_vision,
            tools=tools,
            calculate_cost=True,
            save_conversation_path=step_directory,
        )
        logger.debug(f"Starting browser session for agentic task {browser.cdp_url} ")
        await agent.browser_session.start()
        logger.debug(f"Finally running agentic task on browser_use {browser.cdp_url} ")

        t0_agentic = time.perf_counter()
        agent_history: AgentHistoryList = await agent.run(
            max_steps=agentic_task_action.max_steps
        )
        elapsed_agentic = time.perf_counter() - t0_agentic
        logger.info(
            f"[cache] Agentic run completed in {elapsed_agentic:.2f}s "
            f"({len(agent_history.history)} steps)"
        )
        logger.debug(f"Agentic task completed on browser_use {browser.cdp_url} ")

        # ── Cache save + Bonus 1 + Bonus 2 ──────────────────────────────────
        if (
            cache_key is not None
            and not isinstance(agentic_task_action, CloseOverlayPopupAction)
            and agent_history.is_done()
        ):
            try:
                cached_actions = convert_history_to_cached_actions(
                    agent_history, task.automation.url, input_params
                )
                if cached_actions:
                    raw_count = len(cached_actions)

                    # ── [Bonus 2] Prune redundant steps from raw history immediately ──
                    pruned_actions = prune_redundant_actions(list(cached_actions))
                    pruned_count = raw_count - len(pruned_actions)
                    if pruned_count > 0:
                        logger.info(
                            f"[bonus2] Post-run optimizer pruned {pruned_count} redundant "
                            f"steps from raw history ({raw_count} → {len(pruned_actions)})"
                        )
                    else:
                        logger.info(
                            f"[bonus2] Post-run optimizer: all {raw_count} steps are necessary"
                        )
                    # ─────────────────────────────────────────────────────────

                    raw_summary = []
                    for flat in agent_history.model_actions():
                        entry = {k: str(v) for k, v in flat.items() if k != "interacted_element"}
                        el = flat.get("interacted_element")
                        if el is not None:
                            entry["element"] = {
                                "tag": getattr(el, "node_name", ""),
                                "id": (getattr(el, "attributes", None) or {}).get("id", ""),
                                "xpath": getattr(el, "x_path", ""),
                            }
                        raw_summary.append(entry)

                    new_cache = AgenticTaskCache(
                        task_hash=cache_key,
                        url_pattern=task.automation.url,
                        created_at=datetime.now(timezone.utc),
                        last_run_at=datetime.now(timezone.utc),
                        run_count=1,
                        original_task=agentic_task_action.task,
                        actions=pruned_actions,
                        raw_history_summary=raw_summary,
                        agentic_metrics=RunMetrics(
                            elapsed_seconds=round(elapsed_agentic, 2),
                            llm_calls=len(agent_history.history),
                            step_count=len(agent_history.history),
                            raw_step_count=raw_count,
                            pruned_count=pruned_count,
                            was_cache_hit=False,
                        ),
                    )
                    save_cache(cache_key, new_cache)
                    _write_cached_automation_json(task, new_cache)
                    logger.info(
                        f"[cache] Saved {len(pruned_actions)} actions (key={cache_key})"
                    )

                    # ── [Bonus 1] LLM schema builder — run in thread pool ────
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(
                        None,
                        _run_bonus1_schema_builder,
                        new_cache,
                        task,
                        "test_automation_cached_llm.json",
                    )
                    logger.info("[bonus1] LLM schema builder dispatched (background)")
                    # ─────────────────────────────────────────────────────────

                else:
                    logger.warning("[cache] No replayable actions extracted from history")
            except Exception as exc:
                logger.error(f"[cache] Failed to save cache: {exc}", exc_info=True)
        # ─────────────────────────────────────────────────────────────────────

        agent.stop()
        if agent.browser_session:
            await agent.browser_session.stop()
            await agent.browser_session.reset()

    elif agentic_task_action.backend == "browserbase":
        raise NotImplementedError("Browserbase is not supported yet")
