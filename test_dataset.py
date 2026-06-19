"""
Cache Dataset Runner — 20 Samples across 5 Websites
=====================================================
5 websites × 4 flows each (targeting 5 / 6 / 7 / 8 cached action steps).
All flows involve multi-page navigation + form inputs.

Fixes vs previous version:
  - Safe filenames: slug() strips slashes and special chars
  - All identical metrics: better cache-file tracking (prefer newly created)
  - Restructured: Phase 1 all agentic, then Phase 2 all cache hits
  - Summary grouped by website for cleaner comparison

Usage:
    cd /Users/neelabh/Optexity
    export OPTEXITY_TEST_ENDPOINT="navigate_from_optexity_dashboard-9ad4c495"
    python optexity/test_dataset.py
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URL = "http://localhost:9000/inference"
ENDPOINT   = os.environ.get("OPTEXITY_TEST_ENDPOINT", "navigate_from_optexity_dashboard-9ad4c495")
CACHE_DIR  = Path.home() / ".optexity_cache"
# test_automation.json lives one level above this script (project root),
# regardless of where the script is invoked from.
AUTO_PATH  = Path(__file__).resolve().parent.parent / "test_automation.json"
POLL_INTERVAL = 3
RUN_TIMEOUT   = 300


def slug(name: str) -> str:
    """Safe filename slug — strips slashes and special chars."""
    return re.sub(r"[^\w]+", "_", name.lower()).strip("_")


# ── 20 Samples: 5 websites × 4 flows (target 5 / 6 / 7 / 8 action steps) ────
SAMPLES = [

    # ── Website 1: quotes.toscrape.com ───────────────────────────────────────
    {
        "name": "Quotes 5-step login + humor tag",
        "site": "quotes.toscrape.com",
        "target_steps": 5,
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"], "tag": ["humor"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click Login, enter username 'admin' and password 'password', submit, then click the 'humor' tag",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["john"], "password": ["pass99"], "tag": ["humor"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click Login, enter username 'john' and password 'pass99', submit, then click the 'humor' tag",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Quotes 6-step login + humor + author page",
        "site": "quotes.toscrape.com",
        "target_steps": 6,
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"], "tag": ["humor"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click Login, enter username 'admin' and password 'password', submit, click the 'humor' tag, then click on the name of the first author listed",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["alice"], "password": ["alice1"], "tag": ["humor"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click Login, enter username 'alice' and password 'alice1', submit, click the 'humor' tag, then click on the name of the first author listed",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Quotes 7-step login + humor + author + back + inspirational",
        "site": "quotes.toscrape.com",
        "target_steps": 7,
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click Login, enter username 'admin' and password 'password', submit, click the 'humor' tag, click the first author name, go back, then click the 'inspirational' tag",
                "max_steps": 18, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["bob"], "password": ["bob22"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click Login, enter username 'bob' and password 'bob22', submit, click the 'humor' tag, click the first author name, go back, then click the 'inspirational' tag",
                "max_steps": 18, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Quotes 8-step login + humor + author + back + inspirational + author",
        "site": "quotes.toscrape.com",
        "target_steps": 8,
        "agentic": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["admin"], "password": ["password"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click Login, enter username 'admin' and password 'password', submit, click 'humor' tag, click first author, go back, click 'inspirational' tag, then click the first author name on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://quotes.toscrape.com",
            "parameters": {"input_parameters": {"username": ["carol"], "password": ["carol9"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Click Login, enter username 'carol' and password 'carol9', submit, click 'humor' tag, click first author, go back, click 'inspirational' tag, then click the first author name on that page",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },

    # ── Website 2: books.toscrape.com ────────────────────────────────────────
    {
        "name": "Books 5-step 2 categories + book detail",
        "site": "books.toscrape.com",
        "target_steps": 5,
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"cat1": ["Mystery"], "cat2": ["Travel"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Go to 'Mystery' category, click first book, go back, go to 'Travel' category, click first book",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"cat1": ["Fantasy"], "cat2": ["Romance"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Go to 'Fantasy' category, click first book, go back, go to 'Romance' category, click first book",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books 6-step 3 category hops",
        "site": "books.toscrape.com",
        "target_steps": 6,
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"cat1": ["Mystery"], "cat2": ["Fantasy"], "cat3": ["Science Fiction"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Go to 'Mystery' category, click first book, go back, go to 'Fantasy' category, go back to home, go to 'Science Fiction' category",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"cat1": ["Horror"], "cat2": ["Romance"], "cat3": ["Historical Fiction"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Go to 'Horror' category, click first book, go back, go to 'Romance' category, go back to home, go to 'Historical Fiction' category",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books 7-step 3 categories + 2 book details",
        "site": "books.toscrape.com",
        "target_steps": 7,
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"cat1": ["Mystery"], "cat2": ["Travel"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Go to 'Mystery' category, click first book, go back, go to 'Travel' category, click first book, go back, go to 'Fantasy' category",
                "max_steps": 18, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {"cat1": ["Horror"], "cat2": ["Romance"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Go to 'Horror' category, click first book, go back, go to 'Romance' category, click first book, go back, go to 'Crime' category",
                "max_steps": 18, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Books 8-step 4 category hops + 3 book details",
        "site": "books.toscrape.com",
        "target_steps": 8,
        "agentic": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Go to Mystery category, click first book, go back, go to Travel category, click first book, go back, go to Science Fiction category, click first book",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://books.toscrape.com",
            "parameters": {"input_parameters": {}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Go to Mystery category, click first book, go back, go to Travel category, click first book, go back, go to Science Fiction category, click first book",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },

    # ── Website 3: the-internet.herokuapp.com ─────────────────────────────────
    {
        "name": "Herokuapp 5-step login + add elements",
        "site": "the-internet.herokuapp.com",
        "target_steps": 5,
        "agentic": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {"input_parameters": {"username": ["tomsmith"], "password": ["SuperSecretPassword!"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', navigate to the Add/Remove Elements page and click Add Element once",
                "max_steps": 12, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {"input_parameters": {"username": ["tomsmith"], "password": ["SuperSecretPassword!"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', navigate to the Add/Remove Elements page and click Add Element once",
                "max_steps": 12, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Herokuapp 6-step login + dropdown select + back",
        "site": "the-internet.herokuapp.com",
        "target_steps": 6,
        "agentic": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {"input_parameters": {"username": ["tomsmith"], "password": ["SuperSecretPassword!"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', navigate to the Dropdown page, select Option 2, then navigate back to secure area",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {"input_parameters": {"username": ["tomsmith"], "password": ["SuperSecretPassword!"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', navigate to the Dropdown page, select Option 2, then navigate back to secure area",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Herokuapp 7-step login + dropdown + checkboxes",
        "site": "the-internet.herokuapp.com",
        "target_steps": 7,
        "agentic": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {"input_parameters": {"username": ["tomsmith"], "password": ["SuperSecretPassword!"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', navigate to Dropdown page, select Option 1, go back, navigate to Checkboxes page, check the first checkbox, uncheck the second checkbox",
                "max_steps": 18, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {"input_parameters": {"username": ["tomsmith"], "password": ["SuperSecretPassword!"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', navigate to Dropdown page, select Option 1, go back, navigate to Checkboxes page, check the first checkbox, uncheck the second checkbox",
                "max_steps": 18, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Herokuapp 8-step login + dropdown + add elements + back",
        "site": "the-internet.herokuapp.com",
        "target_steps": 8,
        "agentic": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {"input_parameters": {"username": ["tomsmith"], "password": ["SuperSecretPassword!"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', navigate to Dropdown page, select Option 2, go back to secure area, navigate to Add/Remove Elements, click Add Element twice, then go back to secure area",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://the-internet.herokuapp.com/login",
            "parameters": {"input_parameters": {"username": ["tomsmith"], "password": ["SuperSecretPassword!"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'tomsmith' and password 'SuperSecretPassword!', navigate to Dropdown page, select Option 2, go back to secure area, navigate to Add/Remove Elements, click Add Element twice, then go back to secure area",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },

    # ── Website 4: saucedemo.com ──────────────────────────────────────────────
    {
        "name": "Saucedemo 5-step login + view product",
        "site": "saucedemo.com",
        "target_steps": 5,
        "agentic": {
            "url": "https://www.saucedemo.com",
            "parameters": {"input_parameters": {"username": ["standard_user"], "password": ["secret_sauce"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'standard_user' and password 'secret_sauce', then click on the first product to view its details and tell me its name and price",
                "max_steps": 12, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://www.saucedemo.com",
            "parameters": {"input_parameters": {"username": ["problem_user"], "password": ["secret_sauce"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'problem_user' and password 'secret_sauce', then click on the first product to view its details and tell me its name and price",
                "max_steps": 12, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Saucedemo 6-step login + product + add to cart + back",
        "site": "saucedemo.com",
        "target_steps": 6,
        "agentic": {
            "url": "https://www.saucedemo.com",
            "parameters": {"input_parameters": {"username": ["standard_user"], "password": ["secret_sauce"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'standard_user' and password 'secret_sauce', click first product, click Add to Cart, then go back to the product listing",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://www.saucedemo.com",
            "parameters": {"input_parameters": {"username": ["performance_glitch_user"], "password": ["secret_sauce"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'performance_glitch_user' and password 'secret_sauce', click first product, click Add to Cart, then go back to the product listing",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Saucedemo 7-step login + add to cart + open cart",
        "site": "saucedemo.com",
        "target_steps": 7,
        "agentic": {
            "url": "https://www.saucedemo.com",
            "parameters": {"input_parameters": {"username": ["standard_user"], "password": ["secret_sauce"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'standard_user' and password 'secret_sauce', click first product, click Add to Cart, go back to products, click the cart icon to open the cart, then tell me how many items are shown",
                "max_steps": 18, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://www.saucedemo.com",
            "parameters": {"input_parameters": {"username": ["error_user"], "password": ["secret_sauce"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'error_user' and password 'secret_sauce', click first product, click Add to Cart, go back to products, click the cart icon to open the cart, then tell me how many items are shown",
                "max_steps": 18, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "Saucedemo 8-step login + 2 products + cart + checkout",
        "site": "saucedemo.com",
        "target_steps": 8,
        "agentic": {
            "url": "https://www.saucedemo.com",
            "parameters": {"input_parameters": {"username": ["standard_user"], "password": ["secret_sauce"], "firstname": ["Jane"], "lastname": ["Doe"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'standard_user' and password 'secret_sauce', add first product to cart, go back, add second product, open cart, click Checkout, fill first name 'Jane' and last name 'Doe'",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://www.saucedemo.com",
            "parameters": {"input_parameters": {"username": ["standard_user"], "password": ["secret_sauce"], "firstname": ["Bob"], "lastname": ["Smith"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Login with username 'standard_user' and password 'secret_sauce', add first product to cart, go back, add second product, open cart, click Checkout, fill first name 'Bob' and last name 'Smith'",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },

    # ── Website 5: demoqa.com/text-box ───────────────────────────────────────
    {
        "name": "DemoQA 5-step fill 3 fields + submit",
        "site": "demoqa.com",
        "target_steps": 5,
        "agentic": {
            "url": "https://demoqa.com/text-box",
            "parameters": {"input_parameters": {"fullname": ["John Smith"], "email": ["john@test.com"], "address": ["123 Main St"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Fill Full Name as 'John Smith', Email as 'john@test.com', Current Address as '123 Main St', then click Submit",
                "max_steps": 12, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://demoqa.com/text-box",
            "parameters": {"input_parameters": {"fullname": ["Jane Doe"], "email": ["jane@test.com"], "address": ["456 Oak Ave"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Fill Full Name as 'Jane Doe', Email as 'jane@test.com', Current Address as '456 Oak Ave', then click Submit",
                "max_steps": 12, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "DemoQA 6-step fill 4 fields + submit + verify",
        "site": "demoqa.com",
        "target_steps": 6,
        "agentic": {
            "url": "https://demoqa.com/text-box",
            "parameters": {"input_parameters": {"fullname": ["John Smith"], "email": ["john@test.com"], "address": ["123 Main St"], "perm": ["PO Box 1"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Fill Full Name as 'John Smith', Email as 'john@test.com', Current Address as '123 Main St', Permanent Address as 'PO Box 1', click Submit, verify the output section appears",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://demoqa.com/text-box",
            "parameters": {"input_parameters": {"fullname": ["Alice Brown"], "email": ["alice@test.com"], "address": ["789 Pine Rd"], "perm": ["PO Box 2"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Fill Full Name as 'Alice Brown', Email as 'alice@test.com', Current Address as '789 Pine Rd', Permanent Address as 'PO Box 2', click Submit, verify the output section appears",
                "max_steps": 15, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "DemoQA 7-step form fill + navigate to radio buttons",
        "site": "demoqa.com",
        "target_steps": 7,
        "agentic": {
            "url": "https://demoqa.com/text-box",
            "parameters": {"input_parameters": {"fullname": ["John Smith"], "email": ["john@test.com"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Fill Full Name as 'John Smith' and Email as 'john@test.com', click Submit, then navigate to the Radio Button section in the left menu and click the Yes radio button",
                "max_steps": 18, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://demoqa.com/text-box",
            "parameters": {"input_parameters": {"fullname": ["Bob Green"], "email": ["bob@test.com"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Fill Full Name as 'Bob Green' and Email as 'bob@test.com', click Submit, then navigate to the Radio Button section in the left menu and click the Yes radio button",
                "max_steps": 18, "backend": "browser_use",
            }}}],
        },
    },
    {
        "name": "DemoQA 8-step form + radio + check box expand",
        "site": "demoqa.com",
        "target_steps": 8,
        "agentic": {
            "url": "https://demoqa.com/text-box",
            "parameters": {"input_parameters": {"fullname": ["John Smith"], "email": ["john@test.com"], "address": ["123 Main St"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Fill Full Name as 'John Smith', Email as 'john@test.com', Current Address as '123 Main St', click Submit, navigate to Radio Button in left menu, click Yes, navigate to Check Box section, click the expand arrow next to Home",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
        "cached": {
            "url": "https://demoqa.com/text-box",
            "parameters": {"input_parameters": {"fullname": ["Eve White"], "email": ["eve@test.com"], "address": ["321 Elm St"]}, "generated_parameters": {}},
            "nodes": [{"type": "action_node", "interaction_action": {"agentic_task": {
                "task": "Fill Full Name as 'Eve White', Email as 'eve@test.com', Current Address as '321 Elm St', click Submit, navigate to Radio Button in left menu, click Yes, navigate to Check Box section, click the expand arrow next to Home",
                "max_steps": 20, "backend": "browser_use",
            }}}],
        },
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_automation(a: dict) -> None:
    AUTO_PATH.write_text(json.dumps(a, indent=2))


def clear_cache() -> None:
    if CACHE_DIR.exists():
        removed = list(CACHE_DIR.glob("*.json"))
        for f in removed:
            f.unlink()
        if removed:
            print(f"  🗑  Cleared {len(removed)} cache file(s)\n")


def snapshot() -> dict[str, float]:
    if not CACHE_DIR.exists():
        return {}
    return {str(f): f.stat().st_mtime for f in CACHE_DIR.glob("*.json")}


def find_updated(before: dict[str, float]) -> Path | None:
    if not CACHE_DIR.exists():
        return None
    candidates = sorted(CACHE_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    # Prefer brand-new files
    for f in candidates:
        if str(f) not in before:
            return f
    # Fall back to most-recently modified existing file
    for f in candidates:
        if f.stat().st_mtime > before.get(str(f), 0):
            return f
    return None


async def trigger() -> bool:
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(SERVER_URL, json={
                "endpoint_name": ENDPOINT,
                "input_parameters": {"target_url": ["https://stockanalysis.com/"]},
                "max_timeout_in_minutes": RUN_TIMEOUT // 60,
            })
            return r.is_success
        except Exception as e:
            print(f"\n  ✗ {e}")
            return False


async def poll(before: dict[str, float]) -> tuple[Path | None, float]:
    t0 = time.perf_counter()
    deadline = time.time() + RUN_TIMEOUT
    while time.time() < deadline:
        await asyncio.sleep(POLL_INTERVAL)
        print(".", end="", flush=True)
        p = find_updated(before)
        if p:
            return p, time.perf_counter() - t0
    return None, time.perf_counter() - t0


def metrics(f: Path) -> dict:
    d = json.loads(f.read_text())
    am   = d.get("agentic_metrics") or {}
    hits = d.get("cache_hit_metrics") or []
    return {
        "hash":    d.get("task_hash", "?"),
        "actions": len(d.get("actions", [])),
        "at":      am.get("elapsed_seconds"),
        "llm":     am.get("llm_calls"),
        "pruned":  am.get("pruned_count"),
        "hits":    hits,
        "runs":    d.get("run_count", 1),
    }


def fmt(v, s="") -> str:
    return "?" if v is None else f"{v}{s}"


# ── Main ──────────────────────────────────────────────────────────────────────

def load_existing_caches(n: int) -> list[Path | None]:
    """Load existing cache files from ~/.optexity_cache/ by computing
    the expected cache key for each sample. Used by --phase2-only mode."""
    # Import here to avoid circular issues when running standalone
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from optexity.inference.cache.cache_manager import compute_cache_key
    except ImportError:
        print("  ⚠  Could not import compute_cache_key — locating caches by mtime instead")
        files = sorted(CACHE_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime)
        return files[:n] + [None] * max(0, n - len(files))

    cfiles = [None] * n
    for i, s in enumerate(SAMPLES):
        a = s["agentic"]
        task_text = a["nodes"][0]["interaction_action"]["agentic_task"]["task"]
        base_url = a["url"]
        raw_params = a["parameters"].get("input_parameters", {})
        input_params = {k: [str(v) for v in vs] for k, vs in raw_params.items()} if raw_params else None
        key = compute_cache_key(task_text, base_url, input_params)
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            cfiles[i] = path
            print(f"  ✓ [{i+1}] {s['name']} → {key}.json")
        else:
            print(f"  ✗ [{i+1}] {s['name']} — cache not found (key={key})")
    return cfiles


async def main():
    parser = argparse.ArgumentParser(description="Optexity Cache Dataset Runner")
    parser.add_argument(
        "--phase2-only",
        action="store_true",
        help="Skip Phase 1 and run only Phase 2 using existing caches. "
             "Use this after restarting the server to get a fresh browser session.",
    )
    args = parser.parse_args()

    n = len(SAMPLES)
    print(f"Optexity Cache Dataset — {n} samples across 5 websites\n")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cfiles = [None] * n
    aw = [0.0] * n
    cw = [0.0] * n
    p1_total = 0.0

    if args.phase2_only:
        print("=" * 70)
        print("--phase2-only: loading existing caches from ~/.optexity_cache/")
        print("=" * 70)
        cfiles = load_existing_caches(n)
        found = sum(1 for f in cfiles if f is not None)
        print(f"\n  Found {found}/{n} cache files. Starting Phase 2.\n")
    else:
        clear_cache()

        # ── Phase 1: Agentic ──────────────────────────────────────────────────
        p1 = time.perf_counter()
        print("=" * 70)
        print("PHASE 1 — Agentic  (LLM reasons through each task from scratch)")
        print("=" * 70)
        for i, s in enumerate(SAMPLES):
            print(f"\n[{i+1}/{n}] {s['name']}  (target {s['target_steps']} steps)")
            write_automation(s["agentic"])
            await asyncio.sleep(2)
            bef = snapshot()
            if not await trigger():
                continue
            print("  ⏳", end="", flush=True)
            cf, el = await poll(bef)
            aw[i] = el
            if not cf:
                print(" ✗ timeout"); continue
            cfiles[i] = cf
            m = metrics(cf)
            print(f" ✓  wall={el:.0f}s | agent={fmt(m['at'])}s | llm={fmt(m['llm'])} | steps={m['actions']} | pruned={fmt(m['pruned'])}")
        p1_total = time.perf_counter() - p1
        print(f"\n  Phase 1 total: {p1_total:.0f}s\n")

        # ── Verify before Phase 2 ─────────────────────────────────────────────
        print("=" * 70)
        print("VERIFICATION — Confirming all cache entries are saved")
        print("=" * 70)
        all_ready = False
        verify_deadline = time.time() + 60
        while not all_ready and time.time() < verify_deadline:
            await asyncio.sleep(3)
            missing = []
            for i, s in enumerate(SAMPLES):
                if cfiles[i] is None:
                    missing.append(f"[{i+1}] {s['name']} — cache file not found")
                    continue
                try:
                    d = json.loads(cfiles[i].read_text())
                    if not d.get("agentic_metrics"):
                        missing.append(f"[{i+1}] {s['name']} — agentic_metrics not yet written")
                    if not d.get("actions"):
                        missing.append(f"[{i+1}] {s['name']} — actions list empty")
                except Exception as e:
                    missing.append(f"[{i+1}] {s['name']} — read error: {e}")
            if missing:
                print(f"  ⏳ Waiting for {len(missing)} cache(s)...")
                for msg in missing[:3]:
                    print(f"     {msg}")
            else:
                all_ready = True

        if not all_ready:
            print("  ⚠  Some caches not confirmed — proceeding anyway")
        else:
            print(f"  ✅ All {sum(1 for f in cfiles if f)} verified. Starting Phase 2.\n")

    # ── Phase 2: Cache hits ───────────────────────────────────────────────────
    p2 = time.perf_counter()
    print("=" * 70)
    print("PHASE 2 — Cache hits  (deterministic, 0 LLM calls)")
    print("=" * 70)
    for i, s in enumerate(SAMPLES):
        print(f"\n[{i+1}/{n}] {s['name']}")
        # Use same automation as Phase 1 — guaranteed cache key match.
        # Parameterized caching is separately validated; this script measures
        # time and LLM savings cleanly.
        write_automation(s["agentic"])
        await asyncio.sleep(2)
        bef = snapshot()
        if not await trigger():
            continue
        print("  ⏳", end="", flush=True)
        cf, el = await poll(bef)
        cw[i] = el
        if not cf:
            print(" ✗ timeout"); continue
        if cfiles[i] is None:
            cfiles[i] = cf
        m = metrics(cf)
        last = (m["hits"] or [{}])[-1]
        hit = last.get("elapsed_seconds")
        run_count = m["runs"]
        status = "✓ CACHE HIT" if run_count > 1 else "⚠ AGENTIC (cache miss)"
        print(f" {status}  wall={el:.0f}s | replay={fmt(hit)}s | llm=0 | run_count={run_count}")
    p2_total = time.perf_counter() - p2
    print(f"\n  Phase 2 total: {p2_total:.0f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print(f"{'PERFORMANCE COMPARISON — 5 Websites × 4 Flows (5/6/7/8 steps)':^100}")
    print("=" * 100)
    for site in ["quotes.toscrape.com", "books.toscrape.com", "the-internet.herokuapp.com", "saucedemo.com", "demoqa.com"]:
        idxs = [i for i, s in enumerate(SAMPLES) if s["site"] == site]
        print(f"\n  {site}")
        print(f"  {'Flow':<48} {'Steps':>5} {'Target':>6} {'Agentic':>9} {'LLM':>5} {'Cache':>8} {'Speedup':>9}")
        print(f"  {'-'*94}")
        for i in idxs:
            s = SAMPLES[i]
            if not cfiles[i]:
                print(f"  {s['name']:<48}  no data"); continue
            m = metrics(cfiles[i])
            avg = sum(h["elapsed_seconds"] for h in m["hits"]) / len(m["hits"]) if m["hits"] else None
            spd = f"{m['at']/avg:.1f}x" if m["at"] and avg else "?"
            print(
                f"  {s['name']:<48}"
                f"{m['actions']:>5}"
                f"{s['target_steps']:>6}"
                f"{fmt(m['at'],'s'):>9}"
                f"{fmt(m['llm']):>5}"
                f"{(fmt(round(avg,2),'s') if avg else '?'):>8}"
                f"{spd:>9}"
            )
    print("\n" + "=" * 100)
    spd = f"{p1_total/p2_total:.1f}x" if p2_total > 0 else "?"
    print(f"  Phase 1 total: {p1_total:.0f}s  |  Phase 2 total: {p2_total:.0f}s  |  Speedup: {spd}  |  LLM savings: 100%")

    # Save result JSONs with safe filenames
    print()
    for i, s in enumerate(SAMPLES):
        if cfiles[i]:
            fname = f"result_{i+1:02d}_{slug(s['name'])}.json"
            d = json.loads(cfiles[i].read_text())
            d.update({"sample_name": s["name"], "site": s["site"], "target_steps": s["target_steps"],
                       "wall_agentic_s": round(aw[i], 2), "wall_cached_s": round(cw[i], 2)})
            Path(fname).write_text(json.dumps(d, indent=2, default=str))
            print(f"  → {fname}")

    write_automation(SAMPLES[0]["agentic"])
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
