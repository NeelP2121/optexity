"""
Deterministic cache replay: convert CachedAction entries into InteractionAction
objects and execute them through the existing Optexity dispatch pipeline.

This intentionally reuses `run_interaction_action` so all existing retry logic,
screenshot capture, and error handling applies unchanged.

Self-improving locators: before each step is executed, we query the live page
via LocatorExtraction.locator_from_playwright() and upgrade the stored command
if a more stable locator is available.  This means xpath-based entries (score=10)
discovered during the first agentic run progressively improve to role/id/name
locators (score 72-92) as the cache accumulates replay experience.
"""

import logging

from optexity.inference.cache.action_cache import CachedAction
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    GoBackAction,
    GoToUrlAction,
    InputTextAction,
    InteractionAction,
    KeyPressAction,
    ScrollAction,
    SelectOptionAction,
)
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


def _resolve_param(value: str, input_parameters: dict[str, list] | None) -> str:
    """Replace {{param_name}} with the actual runtime value from input_parameters."""
    if not value or not input_parameters:
        return value
    if value.startswith("{{") and value.endswith("}}"):
        param_name = value[2:-2]
        values = input_parameters.get(param_name, [""])
        return str(values[0]) if values else value
    return value


def build_interaction_action(
    ca: CachedAction,
    input_parameters: dict[str, list] | None = None,
) -> InteractionAction | None:
    """Convert a CachedAction into the appropriate InteractionAction subtype.

    Returns None if the action cannot be converted (e.g., unsupported type).
    The command field is set so execution takes the fast 'command-first' path
    in handle_command.py (zero LLM calls).  prompt_instructions is preserved
    as a natural-language fallback label.
    """
    t = ca.optexity_action_type
    p = ca.action_params

    try:
        if t == "click_element":
            action = ClickElementAction(
                command=ca.command,
                prompt_instructions=ca.prompt_instructions,
            )
            return InteractionAction(click_element=action)

        elif t == "input_text":
            # Resolve {{param_name}} placeholders with runtime input_parameters
            text = p.get("text") or p.get("value") or ""
            text = _resolve_param(text, input_parameters)
            action = InputTextAction(
                command=ca.command,
                prompt_instructions=ca.prompt_instructions,
                input_text=text,
            )
            return InteractionAction(input_text=action)

        elif t == "select_option":
            value = p.get("value") or p.get("option") or ""
            action = SelectOptionAction(
                command=ca.command,
                prompt_instructions=ca.prompt_instructions,
                select_values=[value] if value else None,
            )
            return InteractionAction(select_option=action)

        elif t == "key_press":
            key = p.get("key") or p.get("keys") or "Enter"
            action = KeyPressAction(
                command=ca.command,
                prompt_instructions=ca.prompt_instructions,
                type=key,
            )
            return InteractionAction(key_press=action)

        elif t == "go_to_url":
            url = p.get("url") or ca.source_url
            return InteractionAction(go_to_url=GoToUrlAction(url=url))

        elif t == "go_back":
            return InteractionAction(go_back=GoBackAction())

        elif t == "scroll":
            direction = p.get("direction", "down")
            amount = int(p.get("amount", -1))
            down = direction != "up"
            return InteractionAction(scroll=ScrollAction(down=down, amount=amount))

        else:
            logger.warning(f"[cache_runner] Unsupported action type: {t}")
            return None

    except Exception as exc:
        logger.warning(f"[cache_runner] Failed to build action for step {ca.step_index} ({t}): {exc}")
        return None


async def _try_upgrade_locator(ca: CachedAction, browser: Browser) -> bool:
    """Check the live page for a better locator than what is stored.

    Called BEFORE the action is executed so the target element is still present.
    Returns True if the stored command was upgraded to a higher-scored locator.

    Only actions with a command (i.e. element-based) can be upgraded.
    Navigation actions (go_to_url, go_back, scroll) are skipped.
    """
    if not ca.command or ca.optexity_action_type in ("go_to_url", "go_back", "scroll"):
        return False

    try:
        from optexity.inference.core.interaction.utils import LocatorExtraction

        locator = await browser.get_locator_from_command(ca.command)
        if locator is None:
            return False

        candidates = await LocatorExtraction.locator_from_playwright(locator, "")
        if not candidates:
            return False

        # Try candidates in score order — skip any that resolve to != 1 element
        for candidate in candidates:
            if candidate["score"] <= ca.locator_score:
                break  # remaining candidates are no better than current

            raw = candidate["locator"]
            new_command = raw[5:] if raw.startswith("page.") else raw

            # Verify uniqueness — LocatorExtraction ranks by stability but
            # does not check uniqueness; ambiguous locators cause strict-mode errors
            try:
                test_locator = await browser.get_locator_from_command(new_command)
                if test_locator is None:
                    continue
                count = await test_locator.count()
                if count != 1:
                    logger.debug(
                        f"[cache_runner] Skipping candidate '{candidate['kind']}' "
                        f"for step {ca.step_index}: resolves to {count} elements"
                    )
                    continue
            except Exception:
                continue

            logger.info(
                f"[cache_runner] ⬆ Locator upgraded for step {ca.step_index}: "
                f"{ca.locator_kind}(score={ca.locator_score}) "
                f"→ {candidate['kind']}(score={candidate['score']}) | {new_command[:60]}"
            )
            ca.command = new_command
            ca.locator_score = candidate["score"]
            ca.locator_kind = candidate["kind"]
            return True

        return False  # no better unique locator found

    except Exception as exc:
        logger.debug(f"[cache_runner] Locator upgrade skipped for step {ca.step_index}: {exc}")
        return False


async def run_cached_actions(
    cached_actions: list[CachedAction],
    task: Task,
    memory: Memory,
    browser: Browser,
    input_parameters: dict[str, list] | None = None,
) -> tuple[bool, int, bool]:
    """Replay all cached actions deterministically.

    Before each step, attempts to upgrade the stored locator to a more stable
    one by querying the live element (self-improving cache).

    Returns:
        (all_succeeded: bool, last_successful_step: int, any_upgraded: bool)
        - all_succeeded: True if every step executed without error
        - last_successful_step: index of last step that ran (or failed)
        - any_upgraded: True if at least one locator was improved this run
    """
    any_upgraded = False

    for i, ca in enumerate(cached_actions):
        # Attempt locator upgrade before executing (element still on page)
        if await _try_upgrade_locator(ca, browser):
            any_upgraded = True

        interaction = build_interaction_action(ca, input_parameters)
        if interaction is None:
            logger.warning(f"[cache_runner] Could not build action for step {i}, aborting replay")
            return False, i, any_upgraded

        logger.debug(
            f"[cache_runner] Step {i}/{len(cached_actions)-1}: "
            f"{ca.optexity_action_type} via {ca.locator_kind!r} "
            f"(score={ca.locator_score}) → {ca.command[:60]}"
        )

        try:
            # Lazy import to break circular dependency:
            # handle_agentic_task → cache_runner → run_interaction → handle_agentic_task
            from optexity.inference.core.run_interaction import run_interaction_action
            await run_interaction_action(
                interaction, task, memory, browser, retries_left=2
            )
        except Exception as exc:
            logger.warning(f"[cache_runner] Step {i} raised: {exc}")
            return False, i, any_upgraded

    return True, len(cached_actions) - 1, any_upgraded
