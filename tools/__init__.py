# tools/__init__.py
"""
Tool Registry – tự động load tất cả module trong thư mục tools/.

Mỗi module phải export:
  TOOL_DEFINITIONS : list[dict]   – mô tả tool theo chuẩn Ollama/OpenAI function-calling
  TOOL_FUNCTIONS   : dict[str, callable]  – ánh xạ tên tool → hàm thực thi

Để thêm tool mới:
  1. Tạo file tools/my_new_tools.py
  2. Định nghĩa TOOL_DEFINITIONS + TOOL_FUNCTIONS
  3. Không cần sửa bất kỳ file nào khác – registry tự nhận.
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Optional, Callable
from colorama import Fore

# ── Registry toàn cục ─────────────────────────────────────────────────────────
ALL_TOOL_DEFINITIONS: list[dict] = []   # tập hợp tất cả schema mô tả tool
ALL_TOOL_FUNCTIONS: dict[str, callable] = {}  # tên tool → hàm Python


def _load_all_tools() -> None:
    """Duyệt qua tất cả module trong tools/ và đăng ký tool."""
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
                    f"thiếu TOOL_DEFINITIONS hoặc TOOL_FUNCTIONS, bỏ qua.{Fore.RESET}"
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
                f"{Fore.LIGHTRED_EX}[ToolRegistry] Lỗi load '{full_module}': {e}{Fore.RESET}"
            )


# Tự động load ngay khi import package
_load_all_tools()


def get_tool_definitions() -> list[dict]:
    """Trả về danh sách schema tất cả tool đã đăng ký."""
    return ALL_TOOL_DEFINITIONS


def get_tool_function(name: str) -> Optional[Callable]:
    """Trả về hàm thực thi của tool theo tên, hoặc None nếu không tìm thấy."""
    return ALL_TOOL_FUNCTIONS.get(name)
