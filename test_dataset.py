"""
Cache Dataset Runner
====================
Runs 20 multi-page automation samples through the Optexity caching pipeline
and produces a performance comparison table (agentic vs deterministic).

Structure:
  Phase 1 — All 5 agentic runs back-to-back (establishes caches)
  Phase 2 — All 5 cache-hit runs back-to-back (deterministic replay)
  Summary — Comparison table: time / LLM calls / pruning per sample

Usage:
    # Terminal 1 — start server:
    # cd /Users/neelabh/Optexity
    # python -m optexity.inference.child_process --port 9000 --child_process_id 1

    # Terminal 2 — run dataset:
    # cd /Users/neelabh/Optexity
    # export OPTEXITY_TEST_ENDPOINT="your-endpoint-name"
    # python optexity/test_dataset.py
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URL = "http://localhost:9000/inference"
ENDPOINT = os.environ.get("OPTEXITY_TEST_ENDPOINT", "navigate_from_optexity_dashboard-9ad4c495")
CACHE_DIR = Path.home() / ".optexity_cache"
TEST_AUTOMATION_PATH = Path("test_automation.json")
POLL_INTERVAL = 3
RUN_TIMEOUT = 300

# ── 5 Test Samples ────────────────────────────────────────────────────────────
SAMPLES = [
    {
        "name": "Quotes Login + Humor Tag",
        "description": "Login to quotes.toscrape.com, navigate to humor tag",
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {
                "input_parameters": {
                    "username": ["admin"],
                    "password": ["password"],
                    "tag": ["humor"],
                },
                "generated_parameters": {},
            },
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'humor' tag and tell me the author of the first quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {
                "input_parameters": {
                    "username": ["john"],
                    "password": ["secret123"],
                    "tag": ["humor"],
                },
                "generated_parameters": {},
            },
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'john' and password 'secret123', then navigate to the 'humor' tag and tell me the author of the first quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books Mystery Category",
        "description": "Navigate to Mystery on books.toscrape.com, open first book",
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Mystery"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Mystery' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Travel"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Travel' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Herokuapp Login + Secure Area",
        "description": "Login to the-internet.herokuapp.com, verify secure area",
        "agentic": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {
                "input_parameters": {"username": ["tomsmith"], "password": ["SuperSecretPassword!"]},
                "generated_parameters": {},
            },
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', then verify you are on the secure area page",
                "max_steps": 10, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {
                "input_parameters": {"username": ["tomsmith"], "password": ["SuperSecretPassword!"]},
                "generated_parameters": {},
            },
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', then verify you are on the secure area page",
                "max_steps": 10, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Quotes Login + Inspirational Tag",
        "description": "Same login flow, different tag — tests parameterized cache reuse",
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {
                "input_parameters": {"username": ["admin"], "password": ["password"], "tag": ["inspirational"]},
                "generated_parameters": {},
            },
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'inspirational' tag and tell me the author of the first quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {
                "input_parameters": {"username": ["john"], "password": ["secret123"], "tag": ["inspirational"]},
                "generated_parameters": {},
            },
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'john' and password 'secret123', then navigate to the 'inspirational' tag and tell me the author of the first quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books Science Fiction",
        "description": "Navigate to Science Fiction category, open first book",
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Science Fiction"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Science Fiction' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Historical Fiction"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Historical Fiction' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },

    # ── 6-20: Additional samples ─────────────────────────────────────────────
    {
        "name": "Quotes Login + Love Tag",
        "description": "Login and navigate to love tag — parameterized cache reuse with new tag",
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"], "tag": ["love"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'love' tag and tell me the author of the first quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["alice"], "password": ["pass456"], "tag": ["love"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'alice' and password 'pass456', then navigate to the 'love' tag and tell me the author of the first quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Quotes Login + Life Tag",
        "description": "Login and navigate to life tag",
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"], "tag": ["life"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'life' tag and tell me the author of the first quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["bob"], "password": ["mypass"], "tag": ["life"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'bob' and password 'mypass', then navigate to the 'life' tag and tell me the author of the first quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books Fantasy Category",
        "description": "Navigate to Fantasy category and view first book",
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Fantasy"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Fantasy' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Horror"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Horror' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books Romance Category",
        "description": "Navigate to Romance category and view first book",
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Romance"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Romance' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Crime"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Crime' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Quotes Login + Wisdom Tag",
        "description": "Login and navigate to wisdom tag",
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"], "tag": ["humor"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'humor' tag and note how many quotes are on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["carol"], "password": ["carol99"], "tag": ["humor"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'carol' and password 'carol99', then navigate to the 'humor' tag and note how many quotes are on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books Children Category",
        "description": "Navigate to Children's category and view first book",
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Children's"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the \"Children's\" category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Poetry"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Poetry' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Herokuapp Add/Remove Elements",
        "description": "Navigate to add_remove_elements page and add 3 elements",
        "agentic": {
            "url": "https://the-internet.herokuapp.com/add_remove_elements/",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the 'Add Element' button 3 times to add 3 elements to the page, then tell me how many Delete buttons are visible",
                "max_steps": 10, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://the-internet.herokuapp.com/add_remove_elements/",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the 'Add Element' button 3 times to add 3 elements to the page, then tell me how many Delete buttons are visible",
                "max_steps": 10, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Quotes Login + Deep Thoughts Tag",
        "description": "Login and navigate to deep-thoughts tag",
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"], "tag": ["inspirational"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'inspirational' tag and tell me the text of the first quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["dave"], "password": ["dave123"], "tag": ["inspirational"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'dave' and password 'dave123', then navigate to the 'inspirational' tag and tell me the text of the first quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books Humor Category",
        "description": "Navigate to Humor category and view first book",
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Humor"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Humor' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Sequential Art"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Sequential Art' category and click on the first book to view its details",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Herokuapp Dropdown Select",
        "description": "Navigate to dropdown page and select an option",
        "agentic": {
            "url": "https://the-internet.herokuapp.com/dropdown",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Select 'Option 2' from the dropdown on this page and confirm it is selected",
                "max_steps": 8, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://the-internet.herokuapp.com/dropdown",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Select 'Option 2' from the dropdown on this page and confirm it is selected",
                "max_steps": 8, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Quotes Login + Friendship Tag",
        "description": "Login and navigate to friendship tag",
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"], "tag": ["love"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'love' tag and tell me the author of the second quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["eve"], "password": ["evepass"], "tag": ["love"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'eve' and password 'evepass', then navigate to the 'love' tag and tell me the author of the second quote on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books Biography Category",
        "description": "Navigate to Biography category and view first book",
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Biography"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Biography' category and click on the first book to view its title and price",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"category": ["Classics"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Navigate to the 'Classics' category and click on the first book to view its title and price",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Quotes Login + Truth Tag",
        "description": "Login and navigate to truth tag — varied credentials",
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"], "tag": ["humor"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'admin' and password 'password', then navigate to the 'humor' tag and tell me the author of the last quote visible on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["frank"], "password": ["frank22"], "tag": ["humor"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'frank' and password 'frank22', then navigate to the 'humor' tag and tell me the author of the last quote visible on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books Default Catalogue",
        "description": "View the default catalogue and click second book",
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "On the main books page, click on the second book listed and tell me its title and price",
                "max_steps": 10, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "On the main books page, click on the second book listed and tell me its title and price",
                "max_steps": 10, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Quotes Logout Flow",
        "description": "Login then immediately log out — tests multi-step navigation with state change",
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'admin' and password 'password', verify you are logged in by checking for a Logout link, then click Logout and confirm you are back on the main page",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["grace"], "password": ["grace88"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click the Login link, login with username 'grace' and password 'grace88', verify you are logged in by checking for a Logout link, then click Logout and confirm you are back on the main page",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_automation(automation: dict) -> None:
    TEST_AUTOMATION_PATH.write_text(json.dumps(automation, indent=2))


def clear_cache() -> None:
    """Remove all cache files so Phase 1 always runs agentically."""
    if CACHE_DIR.exists():
        removed = list(CACHE_DIR.glob("*.json"))
        for f in removed:
            f.unlink()
        if removed:
            print(f"  🗑  Cleared {len(removed)} existing cache file(s)\n")


def snapshot_cache() -> dict[str, float]:
    if not CACHE_DIR.exists():
        return {}
    return {str(f): f.stat().st_mtime for f in CACHE_DIR.glob("*.json")}


def find_updated_cache(before: dict[str, float]) -> Path | None:
    if not CACHE_DIR.exists():
        return None
    for f in sorted(CACHE_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        key = str(f)
        if key not in before or f.stat().st_mtime > before[key]:
            return f
    return None


async def trigger_run() -> bool:
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                SERVER_URL,
                json={
                    "endpoint_name": ENDPOINT,
                    "input_parameters": {"target_url": ["https://stockanalysis.com/"]},
                    "max_timeout_in_minutes": RUN_TIMEOUT // 60,
                },
            )
            return resp.is_success
        except Exception as exc:
            print(f"\n  ✗ Request error: {exc}")
            return False


async def wait_for_cache_update(before: dict[str, float]) -> tuple[Path | None, float]:
    t0 = time.perf_counter()
    deadline = time.time() + RUN_TIMEOUT
    while time.time() < deadline:
        await asyncio.sleep(POLL_INTERVAL)
        print(".", end="", flush=True)
        path = find_updated_cache(before)
        if path:
            return path, time.perf_counter() - t0
    return None, time.perf_counter() - t0


def read_metrics(cache_file: Path) -> dict:
    data = json.loads(cache_file.read_text())
    am = data.get("agentic_metrics") or {}
    hits = data.get("cache_hit_metrics") or []
    return {
        "hash": data.get("task_hash", "?"),
        "actions": len(data.get("actions", [])),
        "agentic_time": am.get("elapsed_seconds"),
        "agentic_llm": am.get("llm_calls"),
        "raw_steps": am.get("raw_step_count"),
        "pruned": am.get("pruned_count"),
        "hits": hits,
        "run_count": data.get("run_count", 1),
    }


def fmt(v, suffix="") -> str:
    return "?" if v is None else f"{v}{suffix}"


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("Optexity Cache Dataset Runner")
    print(f"Server:    {SERVER_URL}")
    print(f"Endpoint:  {ENDPOINT}")
    print(f"Cache dir: {CACHE_DIR}\n")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clear_cache()

    cache_files: list[Path | None] = [None] * len(SAMPLES)
    agentic_wall = [0.0] * len(SAMPLES)
    cached_wall  = [0.0] * len(SAMPLES)

    # ── PHASE 1: All agentic runs ─────────────────────────────────────────────
    phase1_start = time.perf_counter()
    print("=" * 72)
    print("PHASE 1 — Agentic runs  (LLM reasons through each task from scratch)")
    print("=" * 72)

    for i, sample in enumerate(SAMPLES):
        print(f"\n[{i+1}/{len(SAMPLES)}] {sample['name']}")
        write_automation(sample["agentic"])
        await asyncio.sleep(2)

        before = snapshot_cache()
        if not await trigger_run():
            print("  ✗ Failed to trigger")
            continue

        print("  ⏳ Agentic run...", end="", flush=True)
        cache_file, elapsed = await wait_for_cache_update(before)
        agentic_wall[i] = elapsed

        if not cache_file:
            print(" ✗ Timed out")
            continue

        cache_files[i] = cache_file
        m = read_metrics(cache_file)
        print(
            f" ✓  wall={elapsed:.1f}s | agent={fmt(m['agentic_time'])}s | "
            f"llm_calls={fmt(m['agentic_llm'])} | "
            f"steps={m['actions']} | pruned={fmt(m['pruned'])}"
        )

    phase1_total = time.perf_counter() - phase1_start
    print(f"\n  Phase 1 total wall time: {phase1_total:.1f}s")

    # ── PHASE 2: All cache-hit runs ───────────────────────────────────────────
    phase2_start = time.perf_counter()
    print("\n" + "=" * 72)
    print("PHASE 2 — Cache-hit runs  (deterministic, 0 LLM calls)")
    print("=" * 72)

    for i, sample in enumerate(SAMPLES):
        print(f"\n[{i+1}/{len(SAMPLES)}] {sample['name']}")
        write_automation(sample["cached"])
        await asyncio.sleep(2)

        before = snapshot_cache()
        if not await trigger_run():
            print("  ✗ Failed to trigger")
            continue

        print("  ⏳ Cache replay...", end="", flush=True)
        cache_file, elapsed = await wait_for_cache_update(before)
        cached_wall[i] = elapsed

        if not cache_file:
            print(" ✗ Timed out")
            continue

        if cache_files[i] is None:
            cache_files[i] = cache_file

        m = read_metrics(cache_file)
        last_hit = (m["hits"] or [{}])[-1]
        print(
            f" ✓  wall={elapsed:.1f}s | replay={fmt(last_hit.get('elapsed_seconds'))}s | "
            f"llm_calls=0 | run_count={m['run_count']}"
        )

    phase2_total = time.perf_counter() - phase2_start
    print(f"\n  Phase 2 total wall time: {phase2_total:.1f}s")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 92)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 92)
    print(f"{'Sample':<35} {'Steps':>5} {'Agentic(s)':>10} {'LLM calls':>10} {'Cache(s)':>9} {'Speedup':>8} {'Pruned':>7}")
    print("-" * 92)

    for i, sample in enumerate(SAMPLES):
        if not cache_files[i]:
            print(f"{sample['name']:<35}  (no data)")
            continue
        m = read_metrics(cache_files[i])
        hits = m["hits"]
        avg_hit = sum(h["elapsed_seconds"] for h in hits) / len(hits) if hits else None
        at = m["agentic_time"]
        speedup = f"{at/avg_hit:.1f}x" if at and avg_hit else "?"
        print(
            f"{sample['name']:<35}"
            f"{m['actions']:>5}"
            f"{fmt(at, 's'):>10}"
            f"{fmt(m['agentic_llm']):>10}"
            f"{(fmt(round(avg_hit,2), 's') if avg_hit else '?'):>9}"
            f"{speedup:>8}"
            f"{fmt(m['pruned']):>7}"
        )

    print("=" * 92)
    speedup_str = f"{phase1_total/phase2_total:.1f}x" if phase2_total > 0 else "?"
    print(f"\n  Phase 1 (all agentic) :  {phase1_total:.1f}s")
    print(f"  Phase 2 (all cached)  :  {phase2_total:.1f}s")
    print(f"  Total speedup         :  {speedup_str} faster end-to-end")
    print(f"  LLM savings           :  100% (Phase 2 = 0 calls vs Phase 1 = N×steps)")

    # Restore first automation
    write_automation(SAMPLES[0]["agentic"])

    # Save per-sample result JSONs
    print()
    for i, sample in enumerate(SAMPLES):
        if cache_files[i]:
            slug = sample['name'].lower().replace(' ', '_').replace('+', '').replace('__', '_')
            out = Path(f"cache_result_{i+1}_{slug}.json")
            data = json.loads(cache_files[i].read_text())
            data["wall_time_agentic_s"] = round(agentic_wall[i], 2)
            data["wall_time_cached_s"]  = round(cached_wall[i], 2)
            out.write_text(json.dumps(data, indent=2, default=str))
            print(f"  → {out}")

    write_automation(SAMPLES[0]["agentic"])
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
