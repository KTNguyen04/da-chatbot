# prompt.py
"""
Prompt builder with RAG docs (ChromaDB).

Tối ưu so với phiên bản gốc:
- Metadata được cache vào st.session_state theo id(df) → bỏ df.info() + df.sample() lặp lại
- df.info() đầy đủ được thay bằng bảng col|dtype|nulls gọn hơn (~50% tokens)
- N_SAMPLES giảm từ 5 → 2 rows
- STRICT_PROMPT_RULES và SELF_CHECK_INSTRUCTIONS rút gọn
- Lịch sử hội thoại chỉ gửi 5 câu gần nhất thay vì toàn bộ
- classify_intent (LLM call trong process_prompt) vẫn giữ nguyên
"""

import traceback
import streamlit as st

from io import StringIO
from colorama import Fore

from rag_docs_manager import build_rag_context

# ─────────────────────────────────────────────────────────────────────────────


VIETNAMESE_USER_FACING_OUTPUT = (
    "CRITICAL: The `analysis` string MUST be written in Vietnamese only. No English in `analysis`. "
    "All user-visible text (chart titles, axis labels, legends, `result` strings) → Vietnamese. "
    "Variable names → English."
    "Code comments → Vietnamese."
)

# ── Rút gọn rules: bỏ các chú thích dài, giữ lại điều cốt lõi ──────────────
STRICT_PROMPT_RULES = """<rules>
- Use only columns in <meta>. Missing col → Vietnamese string in result.
- matplotlib/seaborn only; NO plt.show(); fig.tight_layout() before result.
- Price vs drivers: cite numbers, flag outliers, note correlation≠causation.
</rules>"""

SELF_CHECK_INSTRUCTIONS = """<check>columns exist; no hardcoded values; result type correct; no plt.show(); no I/O.</check>"""

# Số dòng sample tối đa gửi vào prompt (giảm từ 5 → 2)
_PROMPT_SAMPLE_ROWS = 2

# Số câu hỏi lịch sử tối đa gửi vào prompt
_MAX_HISTORY_TURNS = 5


def _matplotlib_graph_hints() -> str:
    # Rút gọn so với bản gốc (~40% ít token hơn), giữ đủ thông tin để LLM chọn chart
    return """<chart_hints>
fig,ax=plt.subplots(figsize=(...)); fig.tight_layout() before result.
cat-dist→bar(horiz>8); num-dist→hist/violin; 2num→scatter; cat-num→box/violin;
part-whole→pie(≤6); time→line; corr→heatmap(annot=True,fmt=".2f").
x-labels 45° if crowded; titles/axes/legends Vietnamese; annotate bars if ≤15.
</chart_hints>"""


def _build_slim_metadata(df, df_name: str) -> tuple[str, str]:
    """
    Thay thế df.info() (dài ~30 dòng) bằng bảng col|dtype|nulls compact.
    Trả về (metadata_block, context_line).
    """
    n_rows, n_cols = df.shape
    null_counts = df.isnull().sum()

    # Header ngắn
    lines = [f"shape=({n_rows},{n_cols})"]
    lines.append("col | dtype | nulls")
    lines.append("--- | ----- | -----")
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        nulls = int(null_counts[col])
        lines.append(f"{col} | {dtype_str} | {nulls}")
    meta_table = "\n".join(lines)

    # Sample nhỏ (2 dòng)
    sample_csv = df.head(_PROMPT_SAMPLE_ROWS).to_csv(index=False)

    metadata_block = (
        f"\n<{df_name}>\n"
        f"[SCHEMA {df_name}]:\n{meta_table}\n"
        f"[SAMPLE {df_name}]:\n{sample_csv}"
        f"</{df_name}>\n"
    )
    context_line = f"- {df_name}: columns={list(df.columns)}"
    return metadata_block, context_line


def _get_cached_metadata(df, df_name: str) -> tuple[str, str]:
    """
    Cache metadata trong st.session_state theo id(df).
    Khi df không đổi (cùng object), bỏ qua df.info() + df.sample() hoàn toàn.
    """
    cache_key = f"_prompt_meta_{id(df)}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached
    result = _build_slim_metadata(df, df_name)
    st.session_state[cache_key] = result
    return result


def classify_intent(llm: object, question: str, history: str) -> dict:
    """LLM routing: intent, soft graph_type hint, analysis tag."""
    # Prompt rút gọn: bỏ phần giải thích dài của từng graph_type label
    prompt = f"""Real-estate chatbot router. ONE JSON only, no extra text.
Fields: intent(data_analysis|metadata_query|general_chat),
graph_type(scatter_2d_plot|bubble_plot|scatter_3d_plot|bar_plot|line_plot|histogram_plot|pie_plot|box_plot|area_plot|heatmap|violin_plot|density_contour_plot|polar_plot|surface_plot|candle_plot|treemap_plot|sunburst_plot|choroplethmap_plot|densitymap_plot|scattermap_plot|table|none),
target_col(column name or "none"), analysis_intent(physical_structure_vs_price|bigger_equals_more_expensive|ambiguous|none).
History:{history}
Question:{question}"""
    try:
        import json
        import re
        response = llm.invoke(prompt).content
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {"intent": "data_analysis", "graph_type": "none", "target_col": "none", "analysis_intent": "none"}


def define_graph_type(llm: object, question_user: str, hist_questions: str) -> str:
    res = classify_intent(llm, question_user, hist_questions)
    return res.get("graph_type", "none") or "none"


def process_prompt(
    session_msgs: list[dict], user_question: str, data: dict, llm: object
) -> str:
    """
    Build metadata, RAG routing, semantic RAG from rag_docs/, and the final code-generation prompt.
    Tối ưu: metadata cache + slim schema + lịch sử cắt ngắn.
    """

    # ── Build metadata (cached per df object) ────────────────────────────────
    output_parts = []
    context_parts = []
    try:
        for index, (name, df) in enumerate(data.items(), start=1):
            df_name = f"DF_{index}"
            meta_block, ctx_line = _get_cached_metadata(df, df_name)
            output_parts.append(meta_block)
            context_parts.append(ctx_line)
        metadata = "\n".join(output_parts)
        data_context = "\n".join(context_parts)
    except Exception as e:
        exception_name = type(e).__name__
        track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
        st.error(f"WARNING! Metadata error <{exception_name}: {track_line}>")
        # Fallback: chỉ gửi tên cột
        cols = list(next(iter(data.values())).columns) if data else []
        metadata = f"columns={cols}"
        data_context = metadata

    # ── Last code context ─────────────────────────────────────────────────────
    if st.session_state.get("context_code_var"):
        st.session_state["context_code_var"] = False
        context_code = st.session_state.get("last_code") or "# (none)"
    else:
        context_code = "# (none)"

    # ── Lịch sử: chỉ lấy N câu hỏi gần nhất ─────────────────────────────────
    user_questions = [item for item in session_msgs if item["role"] == "user"]
    recent_questions = user_questions[-_MAX_HISTORY_TURNS:]
    questions_text = "\n".join([item["question"] for item in recent_questions])

    # ── Graph type routing ────────────────────────────────────────────────────
    graph_type = define_graph_type(llm, user_question, questions_text)
    params_plot = _matplotlib_graph_hints()

    # ── RAG context ───────────────────────────────────────────────────────────
    try:
        rag_docs_context = build_rag_context(user_question, top_k=3, min_similarity=0.25)
        if rag_docs_context:
            print(f"\n{Fore.LIGHTMAGENTA_EX}[RAGDocs] Context injected ({len(rag_docs_context)} chars){Fore.RESET}")
        else:
            print(f"\n{Fore.LIGHTBLACK_EX}[RAGDocs] No relevant docs (similarity < 0.25){Fore.RESET}")
    except Exception as e:
        rag_docs_context = ""
        print(f"\n{Fore.LIGHTRED_EX}[RAGDocs] Error: {e}{Fore.RESET}")

    # ── Result contract tuỳ graph type ───────────────────────────────────────
    chart_types = {
        "scatter_2d_plot", "bubble_plot", "scatter_3d_plot", "bar_plot", "line_plot",
        "histogram_plot", "pie_plot", "box_plot", "area_plot", "choroplethmap_plot",
        "densitymap_plot", "scattermap_plot", "polar_plot", "surface_plot", "heatmap",
        "candle_plot", "violin_plot", "density_contour_plot", "sunburst_plot", "treemap_plot",
    }
    if graph_type in chart_types:
        result_instruction = (
            "# result = {'figure': fig, 'analysis': analysis}  ← dict always\n"
            "# analysis: Vietnamese, ≤3 sentences, cite ≥1 number."
        )
        prompt_context = f"Chart intent. {params_plot}\nanalysis Vietnamese ≤3 sentences ≥1 number."
    elif graph_type == "table":
        result_instruction = "# result = <pd.DataFrame>"
        prompt_context = "Table intent. Filter/aggregate df → result. No plot unless asked."
    else:
        result_instruction = (
            "# result = '<Vietnamese answer>'\n"
            "# Chart if it adds insight: result = {'figure': fig, 'analysis': '<Vietnamese>'}"
        )
        prompt_context = f"Text/stats intent. Vietnamese string in result. {params_plot}"

    print(f"\n\n{Fore.LIGHTGREEN_EX}STARTING RUNTIME...{Fore.RESET}")
    print(f"\n{Fore.LIGHTBLUE_EX}GRAPH TYPE:{Fore.RESET} {graph_type}")

    # ── Final prompt ─────────────────────────────────────────────────────────
    prompt_main = f"""Python data-analysis agent — Vietnamese real-estate dataset.
Write ONE self-contained python block answering <q>. Executed in sandbox.

<meta>
{metadata}
</meta>
{rag_docs_context}{STRICT_PROMPT_RULES}
{SELF_CHECK_INSTRUCTIONS}

Sandbox vars: DF_1, DF_2,… (default DF_1). Skeleton:
```python
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
df = DF_1
# your code
{result_instruction}
result = None
```
<q> overrides history/last_code. Round ≤2 decimals.
On error (<code_error>/<message_error>): rewrite root cause.

<guidelines>{prompt_context}</guidelines>
<history>{questions_text}</history>
<q>{user_question}</q>
<last_code>
```python
{context_code}
```
</last_code>

ONE fenced python block only. No text outside.
{VIETNAMESE_USER_FACING_OUTPUT}
"""
    return prompt_main