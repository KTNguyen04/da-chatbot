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

TOOL_TIMEOUT = 30  # giây chờ LLM phản hồi (fail fast; tăng nếu model chậm)
MAX_TOOL_ROUNDS = 5  # số vòng tool-calling tối đa (EDA thường cần nhiều bước)

MAX_TOOL_MESSAGE_JSON_CHARS = 22_000
MAX_TOOL_LIST_ITEMS = 40
MAX_TOOL_STRING_CHARS = 4_000


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
        lines.append("Dataset hints: " + " ".join(hints))
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
def _build_system_prompt(user_question: str = "", df: Optional[pd.DataFrame] = None) -> str:
    """English system instructions; model must still answer end users in Vietnamese."""
    tool_names = [
        t["function"]["name"]
        for t in get_tool_definitions()
        if "function" in t
    ]

    brief = ""
    if df is not None and len(df.columns):
        brief = f"<dataset>\n{_dataset_brief(df)}\n</dataset>\n\n"

    cols_lower = set()
    if df is not None:
        cols_lower = {str(c).lower() for c in df.columns}

    # Keyword hints (user questions are often Vietnamese)
    province_keywords = [
        "tỉnh nào", "thành phố nào", "province", "xuất hiện nhiều nhất",
        "phân bổ theo tỉnh", "bao nhiêu bất động sản ở mỗi tỉnh",
        "top tỉnh", "xếp hạng tỉnh", "tỉnh nào nhiều nhất", "tỉnh nào ít nhất",
    ]
    province_note = ""
    qlow = user_question.lower()
    if "province" in cols_lower and any(kw in qlow for kw in province_keywords):
        province_note = (
            "\n⚡ Province/city frequency → call `get_province_ranking` for exact counts.\n"
        )

    price_driver_keywords = [
        "ảnh hưởng", "tác động", "liệu", "có luôn", "nhà to hơn",
        "diện tích", "phòng ngủ", "phòng tắm", "số tầng", "mặt tiền", "cấu trúc",
    ]
    price_driver_note = ""
    if (
        ("price" in cols_lower or "price_per_m2" in cols_lower)
        and any(kw in qlow for kw in price_driver_keywords)
        and ("giá" in qlow or "price" in qlow)
    ):
        price_driver_note = (
            "\n⚡ Price-driver style question → prefer `analyze_property_structure_price_impact` "
            "and `analyze_bigger_house_premium`; cross-check with `analyze_price_drivers`, "
            "`price_vs_size_summary`, `structure_group_price_compare` when useful.\n"
        )

    return (
        "You are a senior data analyst working on the user's currently loaded tabular dataset.\n"
        f"{brief}"
        "You may call the following tools to obtain exact numbers from the dataframe before answering:\n"
        f"{', '.join(tool_names)}\n\n"
        f"{province_note}{price_driver_note}"
        "Rules:\n"
        "- When column names, dtypes, or shape are uncertain, call `get_data_profile` first.\n"
        "- Always call tools when specific figures are needed; do not guess.\n"
        "- For generic exploration use `profile_column`, `filter_rows`, `pivot_summary`, "
        "`numeric_correlation_pairs`, `compare_two_groups_stat_test`, `chi_square_categorical_association`.\n"
        "- If a Province column exists and the user asks ranking by province → `get_province_ranking`.\n"
        "- For comparing a numeric metric across one categorical dimension → `compare_mean_by_group`.\n"
        "- You may chain multiple tool calls when useful.\n"
        "- After tool results, write a concise insight in Vietnamese for the end user (max ~200 words).\n"
        "- Stay focused on the user's question.\n"
        "- Present numbers clearly; emphasize key values (e.g. markdown bold).\n"
    )


# ── Gọi Ollama chat API ────────────────────────────────────────────────────────
def _call_ollama(
    messages: List[dict],
    tools: Optional[List[dict]],
    ollama_base_url: str,
    model: str,
    temperature: float = 0.1,
    verbose: bool = False,
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
        print(f"\n{Fore.LIGHTCYAN_EX}[ToolExecutor] POST {url} | tools={len(tools or [])}{Fore.RESET}")

    resp = requests.post(url, json=payload, timeout=TOOL_TIMEOUT)
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

    tool_defs = get_tool_definitions()
    if not tool_defs:
        return "⚠️ Không có tool nào được đăng ký."

    messages: list[dict] = [
        {"role": "system", "content": _build_system_prompt(user_question, df)},
        {
            "role": "user",
            "content": (
                f"User question (Vietnamese): {user_question}\n\n"
                "Call the appropriate tool(s) to fetch exact figures from the dataset, "
                "then reply with a short insight. Respond in Vietnamese only."
            ),
        },
    ]

    # ── Vòng lặp tool-calling (multi-round) ──────────────────────────────────
    for round_idx in range(MAX_TOOL_ROUNDS):
        if verbose:
            print(f"\n{Fore.LIGHTBLUE_EX}[ToolExecutor] Round {round_idx + 1}/{MAX_TOOL_ROUNDS}{Fore.RESET}")

        try:
            response = _call_ollama(
                messages=messages,
                tools=tool_defs,
                ollama_base_url=ollama_base_url,
                model=model,
                verbose=verbose,
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


