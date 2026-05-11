# tools/__init__.py
"""
Tool registry: auto-load every module under tools/.

Each module should export:
  TOOL_DEFINITIONS : list[dict]   — Ollama/OpenAI function-calling schemas (English descriptions recommended)
  TOOL_FUNCTIONS   : dict[str, callable]  — tool name → Python implementation

To add tools:
  1. Create tools/my_new_tools.py
  2. Define TOOL_DEFINITIONS + TOOL_FUNCTIONS
  3. No other edits required — this registry picks them up on import.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Optional, Callable
from colorama import Fore

# ── Global registry ───────────────────────────────────────────────────────────
ALL_TOOL_DEFINITIONS: list[dict] = []
ALL_TOOL_FUNCTIONS: dict[str, callable] = {}


def _load_all_tools() -> None:
    """Import each tools.* module and register exported definitions."""
    package_dir = Path(__file__).parent
    package_name = __name__  # "tools"

    for finder, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if module_name.startswith("_"):
            continue  # bỏ qua __init__, __pycache__, ...

        full_module = f"{package_name}.{module_name}"
        try:
            mod = importlib.import_module(full_module)

            defs: list[dict] = getattr(mod, "TOOL_DEFINITIONS", [])
            funcs: dict = getattr(mod, "TOOL_FUNCTIONS", {})

            if not defs or not funcs:
                print(
                    f"{Fore.LIGHTYELLOW_EX}[ToolRegistry] {module_name}: "
                    f"missing TOOL_DEFINITIONS or TOOL_FUNCTIONS, skipping.{Fore.RESET}"
                )
                continue

            ALL_TOOL_DEFINITIONS.extend(defs)
            ALL_TOOL_FUNCTIONS.update(funcs)

            print(
                f"{Fore.LIGHTGREEN_EX}[ToolRegistry] Loaded '{module_name}': "
                f"{len(defs)} tool(s) → {list(funcs.keys())}{Fore.RESET}"
            )

        except Exception as e:
            print(
                f"{Fore.LIGHTRED_EX}[ToolRegistry] Failed to load '{full_module}': {e}{Fore.RESET}"
            )


# Tự động load ngay khi import package
_load_all_tools()


def get_tool_definitions() -> list[dict]:
    """Return all registered tool schemas."""
    return ALL_TOOL_DEFINITIONS


def get_tool_function(name: str) -> Optional[Callable]:
    """Return the implementation for a tool name, or None if unknown."""
    return ALL_TOOL_FUNCTIONS.get(name)
