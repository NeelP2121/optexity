"""
[Bonus 2] Iterative optimization loop.

Starts from a populated cache, runs the deterministic steps, measures
performance, prunes redundant actions, then re-saves and re-runs — converging
toward the fastest possible deterministic automation.

The loop stops when:
  - no more steps can be pruned (convergence), OR
  - a deterministic run fails (needs re-learning), OR
  - max_iterations is reached.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from optexity.inference.cache.action_cache import AgenticTaskCache, CachedAction
from optexity.inference.cache.cache_manager import save_cache
from optexity.inference.cache.cache_runner import run_cached_actions
from optexity.inference.infra.browser import Browser
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    iteration: int
    step_count: int
    elapsed_seconds: float
    succeeded: bool
    pruned_count: int = 0
    notes: list[str] = field(default_factory=list)


def _is_redundant_scroll(action: CachedAction, idx: int, all_actions: list[CachedAction]) -> bool:
    """A scroll is redundant if it is immediately followed by another scroll
    in the same direction — the second one supersedes it."""
    if action.optexity_action_type != "scroll":
        return False
    if idx + 1 < len(all_actions):
        nxt = all_actions[idx + 1]
        if nxt.optexity_action_type == "scroll":
            same_dir = action.action_params.get("direction") == nxt.action_params.get("direction")
            if same_dir:
                return True
    return False


def _is_duplicate_navigate(action: CachedAction, idx: int, all_actions: list[CachedAction]) -> bool:
    """A navigate action is redundant if it targets a URL that was already
    reached by the immediately preceding navigate."""
    if action.optexity_action_type != "go_to_url":
        return False
    if idx == 0:
        return False
    prev = all_actions[idx - 1]
    if prev.optexity_action_type == "go_to_url":
        prev_url = prev.action_params.get("url", "")
        curr_url = action.action_params.get("url", "")
        if prev_url and prev_url == curr_url:
            return True
    return False


def _is_consecutive_same_click(action: CachedAction, idx: int, all_actions: list[CachedAction]) -> bool:
    """Two consecutive clicks on the exact same locator are usually an
    agent exploration artefact — keep only the first."""
    if action.optexity_action_type != "click_element":
        return False
    if idx == 0:
        return False
    prev = all_actions[idx - 1]
    if prev.optexity_action_type == "click_element" and prev.command == action.command:
        return True
    return False


def _find_repeated_sequence(actions: list[CachedAction]) -> set[int]:
    """Detect indices that are part of a repeated action sub-sequence.

    Finds blocks of actions that repeat an earlier identical block
    (same action_type + command + action_params). This catches agent
    retry loops like: [input(A), input(B), click(C), input(A), input(B), click(C)]
    where the second triplet is redundant exploration.

    Correctness guarantees:
    - Only CONSECUTIVE repetitions are removed — a single different action
      in between makes both surrounding blocks immune.
    - Blocks are only built from positions NOT already in to_remove (no
      ghost positions): this prevents pattern-matching against actions that
      won't exist in the final sequence, which could cause false positives.
    - All param values are normalised to str before comparison to prevent
      bool/string type drift across serialisation boundaries.
    - Length=1 is included to catch consecutive duplicate single steps.

    Returns the set of indices to remove (the later duplicate blocks).
    """
    to_remove: set[int] = set()
    n = len(actions)
    if n < 2:
        return to_remove

    def sig(a: CachedAction) -> tuple:
        # Normalise all param values to str — prevents bool vs "True" drift
        normalized = str(sorted((k, str(v)) for k, v in a.action_params.items()))
        return (a.optexity_action_type, a.command, normalized)

    sigs = [sig(a) for a in actions]

    for length in range(1, n // 2 + 1):
        i = 0
        while i + length <= n:
            # Skip positions already marked for removal
            if i in to_remove:
                i += 1
                continue

            # Only build a block from clean (non-removed) positions.
            # Using ghost positions (already in to_remove) would mean
            # pattern-matching against actions that won't be in the final
            # sequence, which can cause false positives.
            if any(k in to_remove for k in range(i, i + length)):
                i += 1
                continue

            block = tuple(sigs[i:i + length])

            # Greedily consume ALL consecutive repetitions of this block.
            # Earlier algorithm jumped past one pair and missed further
            # repeats — e.g. [A,B]×5 was reduced to [A,B,A,B,A,B] instead
            # of [A,B]. The inner while loop now runs until the chain breaks.
            j = i + length
            while j + length <= n:
                # Candidate block must also be free of ghost positions
                if any(k in to_remove for k in range(j, j + length)):
                    break
                if tuple(sigs[j:j + length]) == block:
                    for k in range(j, j + length):
                        to_remove.add(k)
                    j += length
                else:
                    break

            i += 1

    return to_remove


def prune_redundant_actions(actions: list[CachedAction]) -> list[CachedAction]:
    """Remove demonstrably redundant steps from a cached action list.

    Pruning rules (conservative — only remove what's provably safe):
    1. Consecutive scrolls in the same direction → keep last only.
    2. Consecutive navigates to the same URL → keep first only.
    3. Consecutive clicks on the identical locator → keep first only.
    4. Repeated action sub-sequences → remove the duplicate block.
       Catches agent retry loops (e.g. login tried twice because browser-use
       misread the post-submit page state).
    """
    pruned: list[CachedAction] = []
    removed: list[str] = []

    # Rule 4: detect repeated sub-sequences first
    repeated_indices = _find_repeated_sequence(actions)

    for i, action in enumerate(actions):
        if i in repeated_indices:
            removed.append(f"step {action.step_index}: repeated sub-sequence")
            continue
        if _is_redundant_scroll(action, i, actions):
            removed.append(f"step {action.step_index}: redundant scroll")
            continue
        if _is_duplicate_navigate(action, i, actions):
            removed.append(f"step {action.step_index}: duplicate navigate to same URL")
            continue
        if _is_consecutive_same_click(action, i, actions):
            removed.append(f"step {action.step_index}: duplicate click on {action.command}")
            continue
        pruned.append(action)

    if removed:
        logger.info(f"[optimizer] Pruned {len(removed)} steps: {removed}")

    # Re-index step_index to remain contiguous
    for new_idx, a in enumerate(pruned):
        a.step_index = new_idx

    return pruned


async def iterative_optimize(
    cache: AgenticTaskCache,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_iterations: int = 3,
) -> tuple[AgenticTaskCache, list[IterationResult]]:
    """Run the deterministic cache in a loop, pruning after each successful run.

    Returns the final (most optimised) cache and a list of per-iteration results
    for performance comparison.
    """
    results: list[IterationResult] = []
    current_cache = cache

    for i in range(max_iterations):
        logger.info(
            f"[optimizer] ── Iteration {i + 1}/{max_iterations} "
            f"({len(current_cache.actions)} steps) ──"
        )

        t0 = time.perf_counter()
        success, last_step, _ = await run_cached_actions(
            current_cache.actions, task, memory, browser
        )
        elapsed = time.perf_counter() - t0

        result = IterationResult(
            iteration=i + 1,
            step_count=len(current_cache.actions),
            elapsed_seconds=round(elapsed, 2),
            succeeded=success,
        )

        if not success:
            result.notes.append(f"Failed at step {last_step}, stopping optimization")
            results.append(result)
            logger.warning(
                f"[optimizer] Iteration {i + 1} failed at step {last_step}. "
                "Stopping — cache needs re-learning from an agentic run."
            )
            break

        logger.info(
            f"[optimizer] Iteration {i + 1} succeeded in {elapsed:.2f}s "
            f"({len(current_cache.actions)} steps)"
        )

        # Prune and check for convergence
        pruned_actions = prune_redundant_actions(list(current_cache.actions))
        pruned_count = len(current_cache.actions) - len(pruned_actions)
        result.pruned_count = pruned_count
        results.append(result)

        if pruned_count == 0:
            result.notes.append("Converged — no more steps to prune")
            logger.info(f"[optimizer] Converged after {i + 1} iteration(s)")
            break

        # Update cache with pruned actions and persist
        current_cache.actions = pruned_actions
        current_cache.last_run_at = datetime.now(timezone.utc)
        current_cache.run_count += 1
        save_cache(current_cache.task_hash, current_cache)

        logger.info(
            f"[optimizer] Pruned {pruned_count} steps → "
            f"{len(pruned_actions)} remain. Saved."
        )

    # Log performance summary
    logger.info("[optimizer] ── Optimization summary ──")
    for r in results:
        logger.info(
            f"  Iteration {r.iteration}: {r.step_count} steps, "
            f"{r.elapsed_seconds}s, pruned={r.pruned_count}, "
            f"ok={r.succeeded}"
            + (f" | {'; '.join(r.notes)}" if r.notes else "")
        )

    return current_cache, results
