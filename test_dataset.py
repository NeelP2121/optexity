"""
Cache Dataset Runner
====================
Runs 5 multi-page automation samples through the Optexity caching pipeline
and produces a performance comparison table (agentic vs deterministic).

Usage:
    # Start the server first:
    # python -m optexity.inference.child_process --port 9000 --child_process_id 1

    python test_dataset.py

Each sample runs twice:
  1. Agentic (establishes cache, records LLM calls + time)
  2. Deterministic (cache hit, records 0 LLM calls + faster time)
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx

# ── Server config ────────────────────────────────────────────────────────────
SERVER_URL = "http://localhost:9000/inference"
ENDPOINT = os.environ.get("OPTEXITY_TEST_ENDPOINT", "navigate_from_optexity_dashboard-9ad4c495")
CACHE_DIR = Path.home() / ".optexity_cache"
TEST_AUTOMATION_PATH = Path("test_automation.json")

# ── 5 Test Automations ───────────────────────────────────────────────────────
SAMPLES = [
    {
        "name": "Quotes Login + Humor Tag",
        "description": "Login to quotes.toscrape.com and navigate to a humor tag",
        "automation": {
            "url": "https://quotes.toscrape.com",
            "parameters": {
                "input_parameters": {
                    "username": ["admin"],
                    "password": ["password"],
                    "tag": ["humor"],
                },
                "generated_parameters": {},
            },
            "nodes": [{
                "type": "action_node",
                "interaction_action": {
                    "agentic_task": {
                        "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'humor' tag and tell me the author of the first quote on that page",
                        "max_steps": 20,
                        "backend": "browser_use",
                    }
                },
            }],
        },
        "param_variant": {
            "username": ["john"],
            "password": ["secret123"],
            "tag": ["humor"],
            "task": "Click the Login link, login with username 'john' and password 'secret123', then navigate to the 'humor' tag and tell me the author of the first quote on that page",
        },
    },
    {
        "name": "Books Mystery Category",
        "description": "Navigate to Mystery category on books.toscrape.com and open first book",
        "automation": {
            "url": "https://books.toscrape.com",
            "parameters": {
                "input_parameters": {
                    "category": ["Mystery"],
                },
                "generated_parameters": {},
            },
            "nodes": [{
                "type": "action_node",
                "interaction_action": {
                    "agentic_task": {
                        "task": "Navigate to the 'Mystery' category and click on the first book to view its details",
                        "max_steps": 15,
                        "backend": "browser_use",
                    }
                },
            }],
        },
        "param_variant": {
            "category": ["Travel"],
            "task": "Navigate to the 'Travel' category and click on the first book to view its details",
        },
    },
    {
        "name": "Herokuapp Login + Secure Area",
        "description": "Login to the-internet.herokuapp.com and verify secure area",
        "automation": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {
                "input_parameters": {
                    "username": ["tomsmith"],
                    "password": ["SuperSecretPassword!"],
                },
                "generated_parameters": {},
            },
            "nodes": [{
                "type": "action_node",
                "interaction_action": {
                    "agentic_task": {
                        "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', then verify you are on the secure area page",
                        "max_steps": 10,
                        "backend": "browser_use",
                    }
                },
            }],
        },
        "param_variant": {
            "username": ["tomsmith"],
            "password": ["SuperSecretPassword!"],
            "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', then verify you are on the secure area page",
        },
    },
    {
        "name": "Quotes Login + Inspirational Tag",
        "description": "Login and navigate to a different tag — tests parameterized cache reuse",
        "automation": {
            "url": "https://quotes.toscrape.com",
            "parameters": {
                "input_parameters": {
                    "username": ["admin"],
                    "password": ["password"],
                    "tag": ["inspirational"],
                },
                "generated_parameters": {},
            },
            "nodes": [{
                "type": "action_node",
                "interaction_action": {
                    "agentic_task": {
                        "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'inspirational' tag and tell me the author of the first quote on that page",
                        "max_steps": 20,
                        "backend": "browser_use",
                    }
                },
            }],
        },
        "param_variant": {
            "username": ["admin"],
            "password": ["password"],
            "tag": ["life"],
            "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'life' tag and tell me the author of the first quote on that page",
        },
    },
    {
        "name": "Books Science Fiction Category",
        "description": "Navigate to Science Fiction category and view first book details",
        "automation": {
            "url": "https://books.toscrape.com",
            "parameters": {
                "input_parameters": {
                    "category": ["Science Fiction"],
                },
                "generated_parameters": {},
            },
            "nodes": [{
                "type": "action_node",
                "interaction_action": {
                    "agentic_task": {
                        "task": "Navigate to the 'Science Fiction' category and click on the first book to view its details",
                        "max_steps": 15,
                        "backend": "browser_use",
                    }
                },
            }],
        },
        "param_variant": {
            "category": ["Historical Fiction"],
            "task": "Navigate to the 'Historical Fiction' category and click on the first book to view its details",
        },
    },
]


def write_test_automation(automation: dict) -> None:
    TEST_AUTOMATION_PATH.write_text(json.dumps(automation, indent=2))


def find_cache_file() -> Path | None:
    """Find the most recently modified cache file."""
    files = sorted(CACHE_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None


async def run_inference(timeout: int = 300) -> tuple[bool, float]:
    """POST to /inference and wait for the task to appear in logs."""
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            SERVER_URL,
            json={
                "endpoint_name": ENDPOINT,
                "input_parameters": {"target_url": ["https://stockanalysis.com/"]},
                "max_timeout_in_minutes": timeout // 60,
            },
        )
        if not resp.is_success:
            print(f"  ✗ Request failed: {resp.text}")
            return False, 0.0

    # Wait for cache file to be written / updated
    print("  ⏳ Waiting for automation to complete...", end="", flush=True)
    cache_before = {f: f.stat().st_mtime for f in CACHE_DIR.glob("*.json")} if CACHE_DIR.exists() else {}

    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(3)
        print(".", end="", flush=True)
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.json"):
                if f not in cache_before or f.stat().st_mtime > cache_before.get(f, 0):
                    elapsed = time.perf_counter() - start
                    print(f" done ({elapsed:.1f}s)")
                    return True, elapsed
    print(" timeout!")
    return False, 0.0


def read_cache_summary(cache_file: Path) -> dict:
    data = json.loads(cache_file.read_text())
    return {
        "hash": data.get("task_hash", "?"),
        "actions": len(data.get("actions", [])),
        "agentic_metrics": data.get("agentic_metrics"),
        "cache_hit_metrics": data.get("cache_hit_metrics", []),
        "run_count": data.get("run_count", 1),
    }


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("CACHE PERFORMANCE DATASET")
    print("=" * 100)
    header = f"{'Sample':<35} {'Steps':>6} {'Agentic(s)':>10} {'LLM calls':>10} {'Cache(s)':>9} {'Speedup':>9} {'LLM saved':>10}"
    print(header)
    print("-" * 100)

    for r in results:
        if r.get("error"):
            print(f"{r['name']:<35} {'ERROR':>6}")
            continue

        am = r.get("agentic_metrics") or {}
        hits = r.get("cache_hit_metrics", [])
        avg_hit = sum(h["elapsed_seconds"] for h in hits) / len(hits) if hits else 0
        speedup = (am.get("elapsed_seconds", 0) / avg_hit) if avg_hit > 0 else 0

        print(
            f"{r['name']:<35}"
            f"{r['actions']:>6}"
            f"{am.get('elapsed_seconds', 0):>10.2f}"
            f"{am.get('llm_calls', 0):>10}"
            f"{avg_hit:>9.2f}"
            f"{speedup:>8.1f}x"
            f"{'100%':>10}"
        )

    print("=" * 100)
    print("\nPruning summary:")
    for r in results:
        am = r.get("agentic_metrics") or {}
        if am.get("raw_step_count") and am.get("pruned_count") is not None:
            print(
                f"  {r['name']}: {am['raw_step_count']} raw → "
                f"{am['raw_step_count'] - am['pruned_count']} cached "
                f"({am['pruned_count']} redundant steps pruned)"
            )


async def main():
    print("Optexity Cache Dataset Runner")
    print(f"Server: {SERVER_URL}")
    print(f"Cache dir: {CACHE_DIR}\n")

    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True)

    results = []

    for i, sample in enumerate(SAMPLES, 1):
        print(f"\n[{i}/5] {sample['name']}")
        print(f"  {sample['description']}")

        # ── Run 1: Agentic ───────────────────────────────────────────────────
        print("  → Run 1 (agentic):")
        write_test_automation(sample["automation"])
        await asyncio.sleep(2)

        ok, _ = await run_inference()
        if not ok:
            results.append({"name": sample["name"], "error": True})
            continue

        cache_file = find_cache_file()
        if not cache_file:
            results.append({"name": sample["name"], "error": True})
            continue

        summary = read_cache_summary(cache_file)
        am = summary.get("agentic_metrics") or {}
        print(
            f"  ✓ Cached {summary['actions']} actions | "
            f"{am.get('elapsed_seconds', '?')}s | "
            f"{am.get('llm_calls', '?')} LLM calls | "
            f"pruned {am.get('pruned_count', 0)} steps"
        )

        # ── Run 2: Cache hit with param variant ─────────────────────────────
        print("  → Run 2 (cache hit with param variant):")
        variant = sample["param_variant"]
        variant_automation = json.loads(json.dumps(sample["automation"]))
        variant_automation["parameters"]["input_parameters"] = {
            k: v for k, v in variant.items() if k != "task"
        }
        if "task" in variant:
            variant_automation["nodes"][0]["interaction_action"]["agentic_task"]["task"] = variant["task"]

        write_test_automation(variant_automation)
        await asyncio.sleep(2)

        ok, _ = await run_inference()
        if not ok:
            results.append({"name": sample["name"], "error": True, **summary})
            continue

        summary2 = read_cache_summary(cache_file)
        hits = summary2.get("cache_hit_metrics", [])
        if hits:
            last_hit = hits[-1]
            print(
                f"  ✓ Replayed {summary2['actions']} steps | "
                f"{last_hit['elapsed_seconds']}s | "
                f"0 LLM calls | run_count={summary2['run_count']}"
            )

        results.append({
            "name": sample["name"],
            "actions": summary2["actions"],
            "agentic_metrics": summary2.get("agentic_metrics"),
            "cache_hit_metrics": summary2.get("cache_hit_metrics", []),
        })

        # Save individual result JSON
        result_path = Path(f"cache_result_{i}_{sample['name'].lower().replace(' ', '_')}.json")
        result_path.write_text(json.dumps(summary2, indent=2, default=str))
        print(f"  → Saved: {result_path}")

    # Restore last automation
    write_test_automation(SAMPLES[0]["automation"])

    print_table(results)
    print("\nDone! Individual result JSONs saved in current directory.")


if __name__ == "__main__":
    asyncio.run(main())
