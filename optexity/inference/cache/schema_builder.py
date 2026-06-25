"""
[Bonus 1] LLM-based schema builder with self-healing Pydantic validation.

Takes a populated AgenticTaskCache and automatically generates a complete,
valid test_automation_cached.json using the LLM.  If the LLM output fails
Pydantic validation, the error is fed back to the LLM for correction — up to
max_heal_attempts times.

Uses the Optexity LLMModel abstraction so the same provider/model config
applies consistently (defaults to gemini / gemini-2.5-flash).
"""

import json
import logging
import re
from pathlib import Path

from pydantic import ValidationError

from optexity.inference.cache.action_cache import AgenticTaskCache
from optexity.inference.models import get_llm_model_with_fallback
from optexity.schema.automation import Automation

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTION = """\
You are a precise JSON generator for the Optexity automation platform.
You must output ONLY valid JSON — no markdown, no code blocks, no explanation.
The JSON must be a complete Optexity Automation object that passes Pydantic validation.
"""

_GENERATION_PROMPT_TEMPLATE = """\
Convert the cached browser actions below into an Optexity Automation JSON.

## Original task
{original_task}

## Starting URL
{url}

## Cached actions to convert (EVERY action must become a node)
{cache_json}

## Output format — copy this structure EXACTLY, filling in values from the cache:
{{
  "url": "<starting URL above>",
  "parameters": {{"input_parameters": {{}}, "generated_parameters": {{}}}},
  "nodes": [
    {{
      "type": "action_node",
      "interaction_action": {{
        "<optexity_action_type from cache>": {{
          "command": "<command from cache>",
          "prompt_instructions": "<element_description from cache>",
          "input_text": "<action_params.text if input_text action, else omit>"
        }}
      }}
    }}
  ]
}}

## Rules
- "nodes" must contain exactly ONE entry per cached action — {node_count} total.
- Use the "optexity_action_type" field as the key inside "interaction_action".
- For "input_text" actions include the "input_text" field with value from action_params.text.
- For "click_element" actions omit "input_text".
- For "go_to_url" actions use {{"go_to_url": {{"url": "<url from action_params>"}}}}.
- For "go_back" actions use {{"go_back": {{}}}}.
- Do NOT include "agentic_task" nodes.
- Output ONLY the JSON object, no markdown, no explanation.
"""

_HEAL_PROMPT_TEMPLATE = """\
The JSON you generated failed Pydantic validation with these errors:

{errors}

Here is the JSON you generated:
{previous_json}

Fix ALL validation errors and return the corrected JSON object only.
"""


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences if the LLM wrapped the JSON anyway."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def build_automation_from_cache(
    cache: AgenticTaskCache,
    starting_url: str,
    output_path: Path | str = "test_automation_cached.json",
    max_heal_attempts: int = 3,
    llm_provider: str = "gemini",
    llm_model_name: str = "gemini-2.5-flash",
) -> Automation:
    """Use the LLM to convert a cache entry into a full Automation JSON.

    Validates the result with Pydantic and re-prompts the LLM with the
    validation error on failure (self-healing loop).

    Returns the validated Automation object and writes it to output_path.
    """
    llm = get_llm_model_with_fallback(llm_provider, llm_model_name, False)

    # Simplify cache to only the fields the LLM needs
    cache_summary = [
        {
            "step": a.step_index,
            "optexity_action_type": a.optexity_action_type,
            "command": a.command,
            "prompt_instructions": a.prompt_instructions,
            "action_params": a.action_params,
            "element_description": a.element_description,
        }
        for a in cache.actions
    ]

    # Provide a simplified schema — the full schema is very large, so we
    # extract just the InteractionAction and ActionNode portions
    full_schema = Automation.model_json_schema()
    schema_str = json.dumps(full_schema, indent=2)
    # Truncate to ~8k chars to stay within prompt budget
    if len(schema_str) > 8000:
        schema_str = schema_str[:8000] + "\n... (truncated)"

    prompt = _GENERATION_PROMPT_TEMPLATE.format(
        original_task=cache.original_task,
        url=starting_url,
        cache_json=json.dumps(cache_summary, indent=2),
        schema_json=schema_str,
        node_count=len(cache.actions),
    )

    raw_output: str = ""
    last_error: str = ""

    for attempt in range(max_heal_attempts):
        if attempt == 0:
            logger.info(f"[schema_builder] Generating automation JSON (attempt 1/{max_heal_attempts})")
            raw_output, _ = llm.get_model_response(prompt, _SYSTEM_INSTRUCTION)
        else:
            logger.info(f"[schema_builder] Healing attempt {attempt + 1}/{max_heal_attempts}")
            heal_prompt = _HEAL_PROMPT_TEMPLATE.format(
                errors=last_error,
                previous_json=raw_output,
            )
            raw_output, _ = llm.get_model_response(heal_prompt, _SYSTEM_INSTRUCTION)

        cleaned = _strip_json_fences(raw_output)

        try:
            automation = Automation.model_validate_json(cleaned)
            output_path = Path(output_path)
            output_path.write_text(
                json.dumps(json.loads(cleaned), indent=2)
            )
            logger.info(
                f"[schema_builder] Valid automation written to {output_path} "
                f"({len(automation.nodes)} nodes, attempt {attempt + 1})"
            )
            return automation

        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            logger.warning(
                f"[schema_builder] Attempt {attempt + 1} failed validation: "
                f"{last_error[:200]}"
            )

    raise RuntimeError(
        f"[schema_builder] Could not generate valid automation after "
        f"{max_heal_attempts} attempts. Last error: {last_error}"
    )
