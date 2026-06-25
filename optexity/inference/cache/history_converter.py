"""
Convert a browser-use AgentHistoryList into a list of CachedAction entries.

Key insight: AgentHistoryList.model_actions() zips each ActionModel with its
corresponding DOMInteractedElement (already resolved during the step) and
returns a flat list of dicts.  We build an element proxy from the
DOMInteractedElement that satisfies the duck-typed interface required by
LocatorExtraction.locator_candidates() — no live browser session needed.
"""

import logging
from types import SimpleNamespace

from browser_use.agent.views import AgentHistoryList

from optexity.inference.cache.action_cache import CachedAction
from optexity.inference.core.interaction.utils import LocatorExtraction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Browser-use tool name → Optexity InteractionAction field name
# ---------------------------------------------------------------------------
# Verified from browser_use/tools/service.py — the registered function names
# are: click, input, navigate, go_back, scroll, send_keys, select_dropdown.
BROWSER_USE_TO_OPTEXITY: dict[str, str] = {
    "click": "click_element",
    "input": "input_text",
    "navigate": "go_to_url",
    "go_back": "go_back",
    "scroll": "scroll",
    "send_keys": "key_press",
    "select_dropdown": "select_option",
}

# Actions that don't produce replayable nodes (terminal / diagnostic)
_SKIP_ACTIONS = frozenset(
    {"done", "extract", "search", "find_text", "wait", "evaluate", "screenshot",
     "switch", "close", "upload_file", "write_file", "read_file", "replace_file",
     "dropdown_options"}
)

# Implicit ARIA roles for common HTML tags (mirrors the JS in locator_from_playwright)
_IMPLICIT_ROLE: dict[str, str] = {
    "button": "button",
    "select": "combobox",
    "textarea": "textbox",
    "a": "link",
    "checkbox": "checkbox",
    "radio": "radio",
}


def _make_element_proxy(el) -> SimpleNamespace:
    """Build a SimpleNamespace that satisfies LocatorExtraction._scored_candidates().

    Required interface (duck-typed):
        .attributes     dict[str, str]
        .tag_name       str (lowercase)
        .ax_node        object with .role and .name
        .xpath          str
        .get_meaningful_text_for_llm  callable → str
    """
    attrs = el.attributes or {}
    tag = (el.node_name or "").lower().strip() or "*"

    # Input[type=checkbox|radio] should be treated as their type role
    input_type = attrs.get("type", "").lower()
    if tag == "input" and input_type in _IMPLICIT_ROLE:
        implicit_role = _IMPLICIT_ROLE[input_type]
    else:
        implicit_role = _IMPLICIT_ROLE.get(tag)

    role = attrs.get("role") or implicit_role
    ax_name = (getattr(el, "ax_name", None) or "").strip()
    node_value = (getattr(el, "node_value", None) or "").strip()

    # Best-effort accessible name: prefer ax_name, then fall back to
    # common text-bearing attributes (title, alt, placeholder, value, aria-label)
    # then node_value. This ensures elements like <a>humor</a> (where ax_name
    # is empty but node_value="humor") get a role+name locator instead of
    # a generic CSS class selector that matches dozens of elements.
    best_name = (
        ax_name
        or attrs.get("aria-label", "").strip()
        or attrs.get("title", "").strip()
        or attrs.get("alt", "").strip()
        or attrs.get("placeholder", "").strip()
        or attrs.get("value", "").strip()
        or node_value
    ) or None

    return SimpleNamespace(
        tag_name=tag,
        attributes=attrs,
        xpath=getattr(el, "x_path", "") or "",
        ax_node=SimpleNamespace(role=role, name=best_name),
        get_meaningful_text_for_llm=lambda: node_value,
    )


def _best_command(el) -> tuple[str, int, str]:
    """Return (command, score, kind) for the best locator derived from *el*.

    The command is suitable for storage in ActionNode.command — it does NOT
    include the 'page.' prefix because browser.get_locator_from_command(cmd)
    evaluates f'page.{cmd}'.

    For anchor elements without a unique semantic identifier (id/name/aria-label),
    we use the href attribute to build a specific attribute selector. This avoids
    generic class-based locators like locator("a.tag") that match dozens of elements.
    """
    proxy = _make_element_proxy(el)
    candidates = LocatorExtraction.locator_candidates(proxy, "")
    attrs = el.attributes or {}
    tag = (el.node_name or "").lower().strip()

    if candidates:
        best = candidates[0]

        # For anchors: if the best locator is a bare CSS class (score < 60,
        # no has_text or role+name), try an href-based locator instead.
        # href is highly specific (each tag/page has a unique URL) and avoids
        # the "40 elements matched" problem with shared class names like "a.tag".
        if (
            tag == "a"
            and best["score"] < 60
            and not attrs.get("id")
            and not attrs.get("name")
        ):
            href = attrs.get("href", "").strip()
            if href and not href.startswith(("javascript:", "#", "mailto:")):
                # Escape single quotes in href for use inside a double-quoted locator.
                # Use .first so that if the same href appears multiple times on the page
                # (e.g. a tag link in both a quote row and the Top Ten Tags sidebar),
                # we always click the first match instead of failing strict-mode.
                safe_href = href.replace("'", "\\'")
                return f"locator(\"a[href='{safe_href}']\").first", 65, "href"

        raw = best["locator"]
        command = raw[5:] if raw.startswith("page.") else raw
        return command, best["score"], best["kind"]

    # Absolute fallback: raw XPath from DOMInteractedElement
    xpath = getattr(el, "x_path", "") or ""
    return f"locator(\"xpath={xpath}\")", 0, "xpath"


def _element_description(el) -> str:
    """Short human-readable identifier for logs."""
    attrs = el.attributes or {}
    parts = [f"<{(el.node_name or '?').lower()}>"]
    if el_id := attrs.get("id", ""):
        parts.append(f"id={el_id}")
    if name := attrs.get("name", ""):
        parts.append(f"name={name}")
    if ax_name := getattr(el, "ax_name", None):
        parts.append(f'ax="{ax_name}"')
    return " ".join(parts)


def _parameterize_value(
    value: str,
    input_parameters: dict[str, list] | None,
) -> str:
    """Replace a literal value with its {{param_name}} placeholder if it matches
    any input parameter value. Returns the original value if no match found."""
    if not input_parameters or not value:
        return value
    for param_name, values in input_parameters.items():
        param_values = values if isinstance(values, list) else [values]
        if value in [str(v) for v in param_values]:
            return f"{{{{{param_name}}}}}"
    return value


def convert_history_to_cached_actions(
    history: AgentHistoryList,
    base_url: str,
    input_parameters: dict[str, list] | None = None,
) -> list[CachedAction]:
    """Convert a completed AgentHistoryList into a list of CachedAction entries.

    Uses model_actions() which zips each ActionModel with the already-resolved
    DOMInteractedElement — no live browser session required.

    Only successful, replayable actions are included. The 'done' action and
    other diagnostic tools are filtered out.
    """
    cached: list[CachedAction] = []
    raw_history_log: list[dict] = []

    for flat_action in history.model_actions():
        # model_actions() injects 'interacted_element' as an extra key
        el = flat_action.pop("interacted_element", None)

        # First remaining key is the browser-use tool name
        tool_name = next((k for k in flat_action if k != "interacted_element"), None)
        if tool_name is None:
            continue

        action_params: dict = flat_action.get(tool_name) or {}

        raw_history_log.append({
            "tool": tool_name,
            "params": {k: v for k, v in (action_params.items() if isinstance(action_params, dict) else {})
                       if k != "index"},
            "element": _element_description(el) if el else None,
        })

        if tool_name in _SKIP_ACTIONS:
            logger.debug(f"[converter] Skipping non-replayable action: {tool_name}")
            continue

        optexity_type = BROWSER_USE_TO_OPTEXITY.get(tool_name)
        if optexity_type is None:
            logger.debug(f"[converter] Unknown browser-use tool '{tool_name}', skipping")
            continue

        # --- Build locator command ---
        if el is not None:
            command, score, kind = _best_command(el)
            description = _element_description(el)
        elif tool_name == "navigate":
            # go_to_url — the URL is the "locator"
            url_target = action_params.get("url", "") if isinstance(action_params, dict) else ""
            command = f"goto({url_target!r})"
            score, kind = 100, "url"
            description = f"navigate → {url_target}"
        elif tool_name == "go_back":
            command = "go_back()"
            score, kind = 100, "built-in"
            description = "go_back"
        else:
            logger.debug(f"[converter] No element for action '{tool_name}', skipping")
            continue

        # --- Extract extra action-specific params (exclude index) ---
        # For input_text actions, replace literal values with {{param_name}}
        # so the cache entry works across different parameter values.
        extra_params: dict = {}
        if isinstance(action_params, dict):
            for k, v in action_params.items():
                if k == "index":
                    continue
                if k in ("text", "value") and isinstance(v, str):
                    v = _parameterize_value(v, input_parameters)
                extra_params[k] = v

        prompt = description or f"{optexity_type} on {command}"

        cached.append(CachedAction(
            step_index=len(cached),
            browser_use_action_type=tool_name,
            optexity_action_type=optexity_type,
            command=command,
            prompt_instructions=prompt,
            action_params=extra_params,
            source_url=base_url,
            element_description=description,
            locator_score=score,
            locator_kind=kind,
        ))

    logger.info(
        f"[converter] Extracted {len(cached)} replayable actions from "
        f"{len(history.history)} agent steps (skipped "
        f"{len(history.model_actions()) - len(cached)})"
    )
    return cached
