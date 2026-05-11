# tool_executor.py
"""
Tool-Calling Engine cho AI-Datanalysis.

Luồng hoạt động:
  1. Nhận câu hỏi + DataFrame hiện tại
  2. Build system prompt mô tả vai trò + tool schemas từ registry
  3. Gọi Ollama /api/chat với danh sách tools (function-calling format)
  4. Parse response → nếu LLM muốn gọi tool, thực thi hàm Python tương ứng
  5. Gửi lại kết quả tool cho LLM để tổng hợp insight cuối cùng
  6. Trả về chuỗi insight tiếng Việt

Điểm mở rộng:
  - Thêm tool mới: chỉ cần thêm vào tools/
  - Thêm chủ đề mới (ví dụ: tools/stock_tools.py): registry tự nhận
"""

import json
import requests
import traceback
import pandas as pd
from colorama import Fore
from typing import Any, Optional, List

import tools  # auto-load tất cả tool modules qua registry
from tools import get_tool_definitions, get_tool_function, inject_dataframe_into_all_tool_modules

TOOL_TIMEOUT_FIRST = 90   # giây cho round đầu (cold start + model load)
TOOL_TIMEOUT_RETRY = 60   # giây cho các round tiếp theo (model đã warm)
TOOL_TIMEOUT = 90         # fallback / legacy alias
MAX_TOOL_ROUNDS = 3       # giảm từ 5 → 3: tránh tích lũy timeout, đủ cho EDA thông thường

MAX_TOOL_MESSAGE_JSON_CHARS = 22_000
MAX_TOOL_LIST_ITEMS = 40
MAX_TOOL_STRING_CHARS = 4_000

# ── Tool routing: keyword → tool names ────────────────────────────────────────
# Mỗi "nhóm" gồm các tools đủ để trả lời một loại câu hỏi.
# Câu hỏi khớp nhiều nhóm → union. Không khớp nhóm nào → fallback 5 tools cốt lõi.
_TOOL_ROUTE_TABLE: list[tuple[list[str], list[str]]] = [
    (["tỉnh", "thành phố", "province", "city", "khu vực", "vùng", "phân bố tỉnh"],
     ["get_province_ranking", "compare_mean_by_group"]),

    (["giá", "price", "price_per_m2", "giá/m2", "đắt", "rẻ", "tốn", "chi phí"],
     ["describe_numeric_column", "compare_mean_by_group", "filter_and_summarize", "detect_outliers"]),

    (["diện tích", "area", "lớn", "nhỏ", "m2", "rộng", "hẹp"],
     ["price_vs_size_summary", "analyze_bigger_house_premium", "describe_numeric_column"]),

    (["phòng ngủ", "bedroom", "phòng tắm", "bathroom", "tầng", "floor", "mặt tiền", "frontage",
      "cấu trúc", "structure", "lớn hơn", "to hơn"],
     ["structure_group_price_compare", "analyze_property_structure_price_impact", "compare_mean_by_group"]),

    (["yếu tố", "ảnh hưởng", "tác động", "driver", "tương quan", "correlation", "quan hệ"],
     ["analyze_price_drivers", "numeric_correlation_pairs", "analyze_property_structure_price_impact"]),

    (["lọc", "filter", "tìm", "search", "điều kiện", "condition", "bao nhiêu nhà"],
     ["filter_rows", "filter_and_summarize", "count_by_category"]),

    (["thống kê", "mô tả", "describe", "trung bình", "mean", "trung vị", "median",
      "min", "max", "phân phối", "distribution"],
     ["describe_numeric_column", "profile_column", "get_data_profile"]),

    (["pháp lý", "legal", "sổ đỏ", "chứng nhận", "certificate", "nội thất", "furniture",
      "hướng", "direction", "loại", "category", "tỷ lệ", "proportion", "mix"],
     ["count_by_category", "compare_mean_by_group", "filter_and_summarize"]),

    (["ngoại lệ", "outlier", "bất thường", "extreme", "cao bất thường", "thấp bất thường"],
     ["detect_outliers", "profile_column"]),

    (["cột", "column", "schema", "dataset", "dữ liệu gồm", "có những gì", "overview"],
     ["get_data_profile"]),

    (["so sánh", "compare", "khác nhau", "difference", "hơn", "kém"],
     ["compare_mean_by_group", "compare_two_groups_stat_test", "pivot_summary"]),
]

_FALLBACK_TOOLS = [
    "describe_numeric_column",
    "compare_mean_by_group",
    "filter_and_summarize",
    "get_data_profile",
    "analyze_price_drivers",
]

_MAX_TOOLS_PER_REQUEST = 5  # không gửi quá 5 tools / request


def _route_tools(question: str) -> list[dict]:
    """
    Chọn subset tool definitions phù hợp với câu hỏi dựa trên keyword matching.
    Trả về tối đa _MAX_TOOLS_PER_REQUEST tools theo thứ tự ưu tiên.
    """
    q_lower = question.lower()
    matched: list[str] = []

    for keywords, tool_names in _TOOL_ROUTE_TABLE:
        if any(kw in q_lower for kw in keywords):
            for t in tool_names:
                if t not in matched:
                    matched.append(t)

    if not matched:
        matched = list(_FALLBACK_TOOLS)

    matched = matched[:_MAX_TOOLS_PER_REQUEST]

    all_defs = get_tool_definitions()
    name_to_def = {d["function"]["name"]: d for d in all_defs if "function" in d}
    selected = [name_to_def[n] for n in matched if n in name_to_def]

    # Nếu routing không match tool nào tồn tại → fallback toàn bộ capped
    if not selected:
        selected = all_defs[:_MAX_TOOLS_PER_REQUEST]

    return selected


# ── Inject DataFrame vào tất cả tool modules ──────────────────────────────────
def _inject_dataframe(df: pd.DataFrame) -> None:
    """
    Set DataFrame cho mọi module tools đã đăng ký hàm set_dataframe (registry).
    """
    inject_dataframe_into_all_tool_modules(df)


def _dataset_brief(df: pd.DataFrame) -> str:
    lines = [
        f"Shape: {len(df):,} rows × {len(df.columns)} columns.",
        "Columns (name: dtype):",
    ]
    dtype_map = df.dtypes.astype(str).to_dict()
    for i, col in enumerate(df.columns):
        if i >= 48:
            lines.append(f"  … +{len(df.columns) - 48} more")
            break
        lines.append(f"  - {col}: {dtype_map[col]}")
    low = {str(c).lower() for c in df.columns}
    hints: list[str] = []
    if "province" in low:
        hints.append("Province column → `get_province_ranking` / `compare_mean_by_group` when relevant.")
    if "price" in low or "price_per_m2" in low:
        hints.append("Price-like columns → domain tools `analyze_price_drivers`, `compare_mean_by_group`, etc.")
    if hints:
        lines.append("Hints: " + " ".join(hints))
    return "\n".join(lines)


def _truncate_tool_payload(obj: Any, depth: int = 0) -> Any:
    """Shrink tool JSON so LLM context stays bounded (drops tracebacks)."""
    if depth > 12:
        return "<truncated: max depth>"
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "traceback":
                continue
            out[k] = _truncate_tool_payload(v, depth + 1)
        return out
    if isinstance(obj, list):
        if len(obj) > MAX_TOOL_LIST_ITEMS:
            head = [_truncate_tool_payload(x, depth + 1) for x in obj[:MAX_TOOL_LIST_ITEMS]]
            head.append({"_truncated_items": len(obj) - MAX_TOOL_LIST_ITEMS})
            return head
        return [_truncate_tool_payload(x, depth + 1) for x in obj]
    if isinstance(obj, str):
        if len(obj) > MAX_TOOL_STRING_CHARS:
            return obj[: MAX_TOOL_STRING_CHARS - 3] + "..."
        return obj
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    s = str(obj)
    if len(s) > MAX_TOOL_STRING_CHARS:
        return s[: MAX_TOOL_STRING_CHARS - 3] + "..."
    return s


def _tool_result_to_llm_json(tool_result: Any) -> str:
    payload = _truncate_tool_payload(tool_result)
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= MAX_TOOL_MESSAGE_JSON_CHARS:
        return text
    insight = ""
    if isinstance(tool_result, dict):
        insight = str(tool_result.get("insight", ""))
    shrink = {
        "truncated": True,
        "original_json_chars": len(text),
        "preview": text[: MAX_TOOL_MESSAGE_JSON_CHARS - 800],
        "insight_preserved": insight[:2000],
    }
    return json.dumps(shrink, ensure_ascii=False, default=str)


# ── Build system prompt ────────────────────────────────────────────────────────
def _build_system_prompt(df: Optional[pd.DataFrame] = None, tool_names: Optional[list[str]] = None) -> str:
    """Compact system prompt. Tool list passed separately to avoid repetition."""
    col_names = list(df.columns) if df is not None else []
    shape = f"{len(df):,}r×{len(df.columns)}c" if df is not None else "unknown"

    names_str = ", ".join(tool_names) if tool_names else "see tools"

    return (
        f"You are a data analyst. Dataset: {shape}, columns: {col_names}.\n"
        f"Available tools: {names_str}.\n"
        "Call the most relevant tool(s) for the question, then reply in Vietnamese (~150 words), "
        "citing key numbers in **bold**. Never invent numbers — only use tool results."
    )


# ── Gọi Ollama chat API ────────────────────────────────────────────────────────
def _call_ollama(
    messages: List[dict],
    tools: Optional[List[dict]],
    ollama_base_url: str,
    model: str,
    temperature: float = 0.1,
    verbose: bool = False,
    timeout: int = TOOL_TIMEOUT_FIRST,
) -> dict:
    url = f"{ollama_base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 1024},
    }
    if tools:
        payload["tools"] = tools

    if verbose:
        print(f"\n{Fore.LIGHTCYAN_EX}[ToolExecutor] POST {url} | tools={len(tools or [])} | timeout={timeout}s{Fore.RESET}")

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Parse và thực thi tool calls từ response ──────────────────────────────────
def _execute_tool_calls(tool_calls: list[dict], verbose: bool = False) -> list[dict]:
    """
    Nhận list tool_calls từ Ollama response, thực thi từng tool,
    trả về list message tool_result để gửi lại LLM.
    """
    results = []
    for tc in tool_calls:
        func_info = tc.get("function", {})
        tool_name = func_info.get("name", "")
        raw_args = func_info.get("arguments", {})

        # Ollama có thể trả arguments dưới dạng string JSON
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args

        if verbose:
            print(
                f"\n{Fore.LIGHTMAGENTA_EX}[ToolExecutor] Gọi tool: {tool_name}"
                f"\n  args: {json.dumps(args, ensure_ascii=False)}{Fore.RESET}"
            )

        tool_fn = get_tool_function(tool_name)
        if tool_fn is None:
            tool_result = {"error": f"Tool '{tool_name}' không tồn tại trong registry."}
            print(f"{Fore.LIGHTRED_EX}[ToolExecutor] Tool không tìm thấy: {tool_name}{Fore.RESET}")
        else:
            try:
                tool_result = tool_fn(**args)
                if verbose:
                    insight = tool_result.get("insight", "")
                    print(f"{Fore.LIGHTGREEN_EX}[ToolExecutor] Kết quả: {insight}{Fore.RESET}")
            except Exception as e:
                tb = traceback.format_exc()
                if verbose:
                    print(f"{Fore.LIGHTRED_EX}[ToolExecutor] Traceback:\n{tb}{Fore.RESET}")
                tool_result = {
                    "error": f"Lỗi khi thực thi tool '{tool_name}': {str(e)}",
                }
                print(f"{Fore.LIGHTRED_EX}[ToolExecutor] Lỗi tool: {e}{Fore.RESET}")

        # Chuẩn format Ollama tool result message
        results.append({
            "role": "tool",
            "content": _tool_result_to_llm_json(tool_result),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def run_tool_insight(
    user_question: str,
    df: pd.DataFrame,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "qwen2.5:7b",
    verbose: bool = False,
) -> str:
    """
    Run tool-calling rounds and return a natural-language insight for the user.

    The UI is Vietnamese-only: the returned text must be Vietnamese even though
    prompts to the model are written in English.
    """
    # ── Inject df vào tất cả tool modules ────────────────────────────────────
    _inject_dataframe(df)

    # ── Route: chọn tools phù hợp với câu hỏi (tối đa _MAX_TOOLS_PER_REQUEST) ──
    tool_defs = _route_tools(user_question)
    tool_names_selected = [t["function"]["name"] for t in tool_defs if "function" in t]

    print(
        f"{Fore.LIGHTCYAN_EX}[ToolExecutor] Routed {len(tool_defs)}/{len(get_tool_definitions())} tools: "
        f"{tool_names_selected}{Fore.RESET}"
    )

    if not tool_defs:
        return "⚠️ Không có tool nào được đăng ký."

    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(df, tool_names_selected)},
        {
            "role": "user",
            "content": f"{user_question}\n\nAnswer in Vietnamese only.",
        },
    ]

    # ── Vòng lặp tool-calling (multi-round) ──────────────────────────────────
    for round_idx in range(MAX_TOOL_ROUNDS):
        if verbose:
            print(f"\n{Fore.LIGHTBLUE_EX}[ToolExecutor] Round {round_idx + 1}/{MAX_TOOL_ROUNDS}{Fore.RESET}")

        try:
            timeout = TOOL_TIMEOUT_FIRST if round_idx == 0 else TOOL_TIMEOUT_RETRY
            response = _call_ollama(
                messages=messages,
                tools=tool_defs,
                ollama_base_url=ollama_base_url,
                model=model,
                verbose=verbose,
                timeout=timeout,
            )
        except requests.exceptions.ConnectionError:
            return f"⚠️ Không thể kết nối Ollama tại '{ollama_base_url}'."
        except requests.exceptions.Timeout:
            return f"⚠️ Timeout khi chờ LLM phản hồi (>{TOOL_TIMEOUT}s)."
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            try:
                body = e.response.json().get("error", "")
            except Exception:
                body = ""
            return f"⚠️ Ollama HTTP {status}: {body}"
        except Exception as e:
            return f"⚠️ Lỗi không xác định: {e}"

        msg = response.get("message", {})
        assistant_content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Thêm assistant message vào history
        messages.append({
            "role": "assistant",
            "content": assistant_content,
            **({"tool_calls": tool_calls} if tool_calls else {}),
        })

        if verbose:
            print(
                f"{Fore.LIGHTYELLOW_EX}[ToolExecutor] tool_calls={len(tool_calls)} "
                f"| content_len={len(assistant_content)}{Fore.RESET}"
            )

        # Nếu không có tool call → LLM đã trả lời xong
        if not tool_calls:
            final_text = assistant_content.strip()
            return final_text if final_text else "⚠️ LLM không trả về nội dung."

        # Thực thi tool calls → thêm kết quả vào messages
        tool_result_messages = _execute_tool_calls(tool_calls, verbose=verbose)
        messages.extend(tool_result_messages)

    # Nếu hết round vẫn còn tool calls → lấy content cuối cùng
    last_content = messages[-1].get("content", "").strip() if messages else ""
    return (
        last_content
        if last_content
        else "⚠️ Đã đạt giới hạn vòng lặp tool-calling mà không có insight cuối cùng."
    )