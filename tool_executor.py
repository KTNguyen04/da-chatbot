# tool_executor.py
"""
Tool-Calling Engine cho AI-Datanalysis.

Luồng hoạt động:
  1. Nhận câu hỏi + DataFrame hiện tại
  2. Build system prompt mô tả vai trò + tool schemas từ registry
  3. Gọi LLM qua LiteLLM (hỗ trợ Ollama, Gemini, OpenAI) với danh sách tools
  4. Parse response → nếu LLM muốn gọi tool, thực thi hàm Python tương ứng
  5. Gửi lại kết quả tool cho LLM để tổng hợp insight cuối cùng
  6. Trả về chuỗi insight tiếng Việt

Điểm mở rộng:
  - Thêm tool mới: chỉ cần thêm vào tools/
  - Thêm chủ đề mới (ví dụ: tools/stock_tools.py): registry tự nhận
"""

import json
import traceback
import pandas as pd
from colorama import Fore
from typing import Any, Optional, List

import litellm

import tools  # auto-load tất cả tool modules qua registry
from tools import get_tool_definitions, get_tool_function, inject_dataframe_into_all_tool_modules

# ── Timeout / vòng lặp ────────────────────────────────────────────────────────
TOOL_TIMEOUT_FIRST = 90   # giây cho round đầu (cold start + model load)
TOOL_TIMEOUT_RETRY = 60   # giây cho các round tiếp theo (model đã warm)
MAX_TOOL_ROUNDS = 3       # tránh tích lũy timeout, đủ cho EDA thông thường

# ── Giới hạn payload trả về cho LLM ──────────────────────────────────────────
MAX_TOOL_MESSAGE_JSON_CHARS = 22_000
MAX_TOOL_LIST_ITEMS = 40
MAX_TOOL_STRING_CHARS = 4_000

# ── Tool routing: keyword → tool names ────────────────────────────────────────
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

_MAX_TOOLS_PER_REQUEST = 5


# ── Tool routing ──────────────────────────────────────────────────────────────
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

    if not selected:
        selected = all_defs[:_MAX_TOOLS_PER_REQUEST]

    return selected


# ── Inject DataFrame vào tất cả tool modules ──────────────────────────────────
def _inject_dataframe(df: pd.DataFrame) -> None:
    inject_dataframe_into_all_tool_modules(df)


# ── System prompt ──────────────────────────────────────────────────────────────
def _build_system_prompt(df: Optional[pd.DataFrame] = None, tool_names: Optional[list[str]] = None) -> str:
    col_names = list(df.columns) if df is not None else []
    shape = f"{len(df):,}r×{len(df.columns)}c" if df is not None else "unknown"
    names_str = ", ".join(tool_names) if tool_names else "see tools"
    return (
        f"You are a data analyst. Dataset: {shape}, columns: {col_names}.\n"
        f"Available tools: {names_str}.\n"
        "Call the most relevant tool(s) for the question, then reply in Vietnamese (~150 words), "
        "citing key numbers in **bold**. Never invent numbers — only use tool results."
    )


# ── Payload helpers ────────────────────────────────────────────────────────────
def _truncate_tool_payload(obj: Any, depth: int = 0) -> Any:
    """Shrink tool JSON so LLM context stays bounded."""
    if depth > 12:
        return "<truncated: max depth>"
    if isinstance(obj, dict):
        return {k: _truncate_tool_payload(v, depth + 1) for k, v in obj.items() if k != "traceback"}
    if isinstance(obj, list):
        if len(obj) > MAX_TOOL_LIST_ITEMS:
            head = [_truncate_tool_payload(x, depth + 1) for x in obj[:MAX_TOOL_LIST_ITEMS]]
            head.append({"_truncated_items": len(obj) - MAX_TOOL_LIST_ITEMS})
            return head
        return [_truncate_tool_payload(x, depth + 1) for x in obj]
    if isinstance(obj, str):
        return obj[:MAX_TOOL_STRING_CHARS - 3] + "..." if len(obj) > MAX_TOOL_STRING_CHARS else obj
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    s = str(obj)
    return s[:MAX_TOOL_STRING_CHARS - 3] + "..." if len(s) > MAX_TOOL_STRING_CHARS else s


def _tool_result_to_llm_str(tool_result: Any) -> str:
    payload = _truncate_tool_payload(tool_result)
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) <= MAX_TOOL_MESSAGE_JSON_CHARS:
        return text
    insight = str(tool_result.get("insight", "")) if isinstance(tool_result, dict) else ""
    shrink = {
        "truncated": True,
        "original_json_chars": len(text),
        "preview": text[:MAX_TOOL_MESSAGE_JSON_CHARS - 800],
        "insight_preserved": insight[:2000],
    }
    return json.dumps(shrink, ensure_ascii=False, default=str)


# ── LiteLLM call (provider-agnostic) ──────────────────────────────────────────
def _call_llm(
    messages: List[dict],
    tool_defs: Optional[List[dict]],
    litellm_model: str,
    api_base: Optional[str],
    temperature: float = 0.1,
    verbose: bool = False,
    timeout: int = TOOL_TIMEOUT_FIRST,
) -> dict:
    """
    Gọi LLM qua LiteLLM — hỗ trợ Ollama, Gemini, OpenAI và mọi provider LiteLLM.
    Trả về dict chuẩn hoá: {"content": str, "tool_calls": list}.
    """
    kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if tool_defs:
        kwargs["tools"] = tool_defs
        kwargs["tool_choice"] = "auto"

    if verbose:
        print(
            f"\n{Fore.LIGHTCYAN_EX}[ToolExecutor] LiteLLM call | model={litellm_model} "
            f"| tools={len(tool_defs or [])} | timeout={timeout}s{Fore.RESET}"
        )

    response = litellm.completion(**kwargs)
    msg = response.choices[0].message

    # Chuẩn hoá tool_calls về list[dict] bất kể provider
    raw_tool_calls = getattr(msg, "tool_calls", None) or []
    tool_calls = []
    for tc in raw_tool_calls:
        # LiteLLM trả về objects hoặc dict tuỳ provider
        if isinstance(tc, dict):
            tool_calls.append(tc)
        else:
            # OpenAI / Gemini / Ollama đều có .function.name + .function.arguments
            func = getattr(tc, "function", None)
            if func:
                args = getattr(func, "arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args)
                tool_calls.append({
                    "id": getattr(tc, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(func, "name", ""),
                        "arguments": args,
                    },
                })

    return {
        "content": msg.content or "",
        "tool_calls": tool_calls,
    }


# ── Thực thi tool calls ────────────────────────────────────────────────────────
def _execute_tool_calls(tool_calls: list[dict], verbose: bool = False) -> list[dict]:
    """
    Nhận list tool_calls đã chuẩn hoá, thực thi từng tool,
    trả về list message tool_result để gửi lại LLM.
    """
    results = []
    for tc in tool_calls:
        func_info = tc.get("function", {})
        tool_name = func_info.get("name", "")
        raw_args = func_info.get("arguments", {})

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
                    print(f"{Fore.LIGHTGREEN_EX}[ToolExecutor] Kết quả: {tool_result.get('insight', '')}{Fore.RESET}")
            except Exception as e:
                if verbose:
                    print(f"{Fore.LIGHTRED_EX}[ToolExecutor] Traceback:\n{traceback.format_exc()}{Fore.RESET}")
                tool_result = {"error": f"Lỗi khi thực thi tool '{tool_name}': {str(e)}"}
                print(f"{Fore.LIGHTRED_EX}[ToolExecutor] Lỗi tool: {e}{Fore.RESET}")

        # tool_call_id bắt buộc cho OpenAI; Ollama/Gemini bỏ qua nếu rỗng
        tool_msg: dict[str, Any] = {
            "role": "tool",
            "content": _tool_result_to_llm_str(tool_result),
        }
        call_id = tc.get("id", "")
        if call_id:
            tool_msg["tool_call_id"] = call_id

        results.append(tool_msg)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def run_tool_insight(
    user_question: str,
    df: pd.DataFrame,
    litellm_model: str = "ollama/qwen2.5:7b",
    api_base: Optional[str] = "http://localhost:11434",
    temperature: float = 0.1,
    verbose: bool = False,
) -> str:
    """
    Chạy vòng lặp tool-calling và trả về insight ngôn ngữ tự nhiên tiếng Việt.

    Args:
        user_question : Câu hỏi của người dùng.
        df            : DataFrame đang phân tích.
        litellm_model : Tên model theo định dạng LiteLLM
                        (vd: "ollama/qwen2.5:7b", "gemini/gemini-2.0-flash", "gpt-4o-mini").
        api_base      : Base URL cho Ollama local; None với Gemini/OpenAI.
        temperature   : Nhiệt độ sinh text.
        verbose       : In log chi tiết.
    """
    _inject_dataframe(df)

    tool_defs = _route_tools(user_question)
    tool_names_selected = [t["function"]["name"] for t in tool_defs if "function" in t]

    print(
        f"{Fore.LIGHTCYAN_EX}[ToolExecutor] model={litellm_model} | "
        f"Routed {len(tool_defs)}/{len(get_tool_definitions())} tools: "
        f"{tool_names_selected}{Fore.RESET}"
    )

    if not tool_defs:
        return "⚠️ Không có tool nào được đăng ký."

    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(df, tool_names_selected)},
        {"role": "user", "content": f"{user_question}\n\nAnswer in Vietnamese only."},
    ]

    for round_idx in range(MAX_TOOL_ROUNDS):
        if verbose:
            print(f"\n{Fore.LIGHTBLUE_EX}[ToolExecutor] Round {round_idx + 1}/{MAX_TOOL_ROUNDS}{Fore.RESET}")

        timeout = TOOL_TIMEOUT_FIRST if round_idx == 0 else TOOL_TIMEOUT_RETRY

        try:
            result = _call_llm(
                messages=messages,
                tool_defs=tool_defs,
                litellm_model=litellm_model,
                api_base=api_base,
                temperature=temperature,
                verbose=verbose,
                timeout=timeout,
            )
        except litellm.exceptions.APIConnectionError:
            return f"⚠️ Không thể kết nối tới provider (model={litellm_model})."
        except litellm.exceptions.Timeout:
            return f"⚠️ Timeout khi chờ LLM phản hồi (>{timeout}s)."
        except litellm.exceptions.AuthenticationError:
            return "⚠️ API key không hợp lệ hoặc chưa được thiết lập."
        except litellm.exceptions.BadRequestError as e:
            return f"⚠️ Model không hỗ trợ tool-calling hoặc request lỗi: {e}"
        except Exception as e:
            return f"⚠️ Lỗi không xác định: {e}"

        assistant_content = result["content"]
        tool_calls = result["tool_calls"]

        # Thêm assistant message (kèm tool_calls nếu có để OpenAI không lỗi)
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
        }
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if verbose:
            print(
                f"{Fore.LIGHTYELLOW_EX}[ToolExecutor] tool_calls={len(tool_calls)} "
                f"| content_len={len(assistant_content)}{Fore.RESET}"
            )

        # Không có tool call → LLM đã trả lời xong
        if not tool_calls:
            final_text = assistant_content.strip()
            return final_text if final_text else "⚠️ LLM không trả về nội dung."

        # Thực thi tool calls → gửi kết quả lại LLM
        tool_result_messages = _execute_tool_calls(tool_calls, verbose=verbose)
        messages.extend(tool_result_messages)

    last_content = messages[-1].get("content", "").strip() if messages else ""
    return (
        last_content
        if last_content
        else "⚠️ Đã đạt giới hạn vòng lặp tool-calling mà không có insight cuối cùng."
    )