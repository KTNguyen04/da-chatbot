# prompt.py
"""
Prompt builder with RAG docs (ChromaDB).

Uses rag_docs_manager in process_prompt(); chart code is guided with short,
task-agnostic rules so the model picks matplotlib/seaborn visuals from the
question and dtypes.
"""

import traceback
import streamlit as st

from io import StringIO
from colorama import Fore

from rag_docs_manager import build_rag_context

# ─────────────────────────────────────────────────────────────────────────────

# End-user outputs are Vietnamese; instructions are English for model clarity.
VIETNAMESE_USER_FACING_OUTPUT = (
    "All natural-language output for the end user (variable `analysis`, string `result`, "
    "chart titles, axis labels, legend text, and any dataframe presentation text) "
    "MUST be written in Vietnamese. Every user-visible chart string must be Vietnamese."
)

STRICT_PROMPT_RULES = """
<strict_rules>
- Do NOT invent data: only use columns that exist in <metadata>.
- Do NOT call the network or read files outside the current context.
- Do NOT use fake or illustrative numbers as if they were real query results.
- If a required column is missing, set `result` to a Vietnamese string explaining the missing data.
- Charts: matplotlib and seaborn only; no plotly/bokeh. Do not call plt.show(); build a Figure and assign per contract below.
- Prefer brief Vietnamese comments on important logic blocks in generated code.
</strict_rules>
"""


def _matplotlib_graph_hints(_graph_type: str) -> str:
    """
    Short, chart-agnostic plotting hints (router may still set _graph_type for logging).
    The model chooses chart types from <main_question> and column dtypes.
    """
    return """
<code_ref>
- Use `fig, ax = plt.subplots(figsize=(10, 6))` (or more axes if needed); call `fig.tight_layout()` before assigning `result`.
- Pick sns/plt APIs that fit the task (counts/ranking → bar; distribution of one numeric → hist/KDE; numeric vs numeric → scatter; numeric by category → box/violin; shares of few categories → pie with cap + "Khác"; time series → line; two numeric densities → kdeplot; correlations → heatmap on numeric columns only).
- If the user names an exotic chart (treemap, sunburst, candlestick, true map), approximate readably with supported primitives and state the limitation in `analysis`.
- Rotate crowded tick labels (~45°); `sns.set_theme(style="whitegrid")` when it helps. All title/axis/legend text in Vietnamese.
</code_ref>
"""

SELF_CHECK_INSTRUCTIONS = """
<self_check_before_return>
Before answering, self-check:
1) Every column name exists in the sample data.
2) Every calculation uses the real dataframe, not hard-coded figures.
3) `result` has the correct type (figure+analysis / dataframe / string) as required.
4) No code accesses the internet or resources outside this task.
</self_check_before_return>
"""


def classify_intent(llm: object, question: str, history: str) -> dict:
    """
    LLM-based intent + optional default chart tag (soft hint; codegen still decides details).
    """
    prompt = f"""
You route questions about the user's tabular (housing) dataset. Return one JSON object only.

Intents:
- data_analysis: statistics, filters, comparisons, or visuals from the data.
- metadata_query: column meanings, units, definitions.
- general_chat: greetings or off-topic.

analysis_intent (only when intent=data_analysis): physical_structure_vs_price | bigger_equals_more_expensive | ambiguous | none — semantic tag for written analysis, not a command to force one chart type.

graph_type: optional single internal label when a chart is clearly implied and one type stands out; otherwise "none".
Allowed labels: scatter_2d_plot, bubble_plot, scatter_3d_plot, bar_plot, line_plot, histogram_plot, pie_plot, box_plot, area_plot, heatmap, violin_plot, density_contour_plot, polar_plot, surface_plot, candle_plot, treemap_plot, sunburst_plot, choroplethmap_plot, densitymap_plot, scattermap_plot, table, none.

Also set "target_col" to one relevant column name from the question or "none".

Conversation history:
{history}

User question:
{question}
"""
    try:
        import json
        import re

        response = llm.invoke(prompt).content
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return {
        "intent": "data_analysis",
        "graph_type": "none",
        "target_col": "none",
        "analysis_intent": "none",
    }


def _graph_type_from_keywords(question_user: str) -> str | None:
    """
    Fast graph routing from phrasing — avoids an extra LLM round when confident.
    More specific patterns are checked first.
    """
    q = question_user.lower()

    if "sunburst" in q:
        return "sunburst_plot"
    if "treemap" in q:
        return "treemap_plot"
    if "violin" in q:
        return "violin_plot"
    if "ohlc" in q or "candlestick" in q:
        return "candle_plot"
    if "heatmap" in q or "ma trận tương quan" in q or "correlation matrix" in q:
        return "heatmap"
    if "bubble" in q:
        return "bubble_plot"
    if ("scatter" in q or "phân tán" in q) and ("3d" in q or "3-d" in q):
        return "scatter_3d_plot"
    if "surface" in q and ("3d" in q or "3-d" in q):
        return "surface_plot"
    if "density" in q and "contour" in q:
        return "density_contour_plot"
    if "polar" in q or "radar" in q or "nhện" in q:
        return "polar_plot"
    if "choropleth" in q:
        return "choroplethmap_plot"
    if "density" in q and "map" in q:
        return "densitymap_plot"
    if ("scatter" in q and "map" in q) or (
        "bản đồ" in q and ("điểm" in q or "scatter" in q)
    ):
        return "scattermap_plot"

    if "pie" in q or "tròn" in q or "tỉ lệ %" in q:
        return "pie_plot"
    if "box" in q or "hộp" in q or "quartile" in q or "tứ phân vị" in q:
        return "box_plot"
    if "histogram" in q or "hist" in q or ("phân phối" in q and "tần suất" in q):
        return "histogram_plot"
    if "line" in q or (
        "đường" in q
        and ("xu hướng" in q or "trend" in q or "theo thời gian" in q)
    ):
        return "line_plot"
    if "area chart" in q or ("area" in q and ("chart" in q or "đồ thị" in q or "biểu đồ" in q)):
        return "area_plot"
    if "bar" in q or "cột" in q or "bar chart" in q:
        return "bar_plot"
    if "scatter" in q or "phân tán" in q:
        return "scatter_2d_plot"
    if "table_plotly" in q or ("plotly" in q and "table" in q):
        return "table"
    if ("dataframe" in q and "table" in q) or ("bảng" in q and "dữ liệu" in q):
        return "table"

    return None


def define_graph_type(llm: object, question_user: str, hist_questions: str) -> str:
    """Prefer explicit chart words in the question; otherwise ask the router LLM once."""
    keyword_graph = _graph_type_from_keywords(question_user)
    if keyword_graph is not None:
        return keyword_graph

    res = classify_intent(llm, question_user, hist_questions)
    return res.get("graph_type", "none") or "none"


def process_prompt(
    session_msgs: list[dict], user_question: str, data: dict, llm: object
) -> str:
    """
    Build metadata, RAG routing, semantic RAG from rag_docs/, and the final code-generation prompt.
    """

    last_df = None
    try:
        output_parts = []
        context_parts = []
        for index, (name, df) in enumerate(data.items(), start=1):
            last_df = df
            buffer = StringIO()
            df.info(buf=buffer)
            df_info = buffer.getvalue()
            df_sample = df.sample(st.session_state["sample_var"]).to_csv(
                path_or_buf=None, index=False
            )
            df_context_sample = df.head(2).to_string(index=False)
            df_name = f"DF_{index}"
            output_parts.append(
                f"\n<{df_name}>\n[INFO {df_name}]:\n{df_info}[SAMPLES {df_name}]:\n{df_sample}</{df_name}>\n"
            )
            context_parts.append(
                f"- {df_name}: columns={list(df.columns)}\n  sample:\n{df_context_sample}"
            )
        metadata = "\n".join(output_parts)
        data_context = "\n".join(context_parts)
    except Exception as e:
        data = {}
        exception_name = type(e).__name__
        track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
        message_ = "WARNING! Sample data error, provided only columns as samples"
        st.error(f"{message_}  <{exception_name}: {track_line}>")
        cols = list(last_df.columns) if last_df is not None else []
        metadata = str(cols)
        data_context = f"columns_only={metadata}"

    if st.session_state["context_code_var"]:
        st.session_state["context_code_var"] = False
        context_code = st.session_state["last_code"]
    else:
        context_code = "No code returned for context."

    user_questions = [item for item in session_msgs if item["role"] == "user"]
    questions_text = "\n".join([item["question"] for item in user_questions])

    graph_type = define_graph_type(llm, user_question, questions_text)
    params_plot = _matplotlib_graph_hints(graph_type)

    try:
        rag_docs_context = build_rag_context(
            user_question, top_k=3, min_similarity=0.25
        )
        if rag_docs_context:
            print(
                f"\n{Fore.LIGHTMAGENTA_EX}[RAGDocs] Context injected from rag_docs/{Fore.RESET}"
            )
        else:
            print(
                f"\n{Fore.LIGHTBLACK_EX}[RAGDocs] No relevant docs found (similarity < 0.25){Fore.RESET}"
            )
    except Exception as e:
        rag_docs_context = ""
        print(f"\n{Fore.LIGHTRED_EX}[RAGDocs] Error: {e}{Fore.RESET}")

    price_driver_keywords = [
        "ảnh hưởng",
        "tác động",
        "liệu",
        "có luôn",
        "nhà to hơn",
        "diện tích",
        "phòng ngủ",
        "phòng tắm",
        "số tầng",
        "mặt tiền",
    ]
    q_lower = user_question.lower()
    is_price_driver_question = any(kw in q_lower for kw in price_driver_keywords) and (
        "giá" in q_lower or "price" in q_lower
    )

    if graph_type in [
        "scatter_2d_plot",
        "bubble_plot",
        "scatter_3d_plot",
        "bar_plot",
        "line_plot",
        "histogram_plot",
        "pie_plot",
        "box_plot",
        "area_plot",
        "choroplethmap_plot",
        "densitymap_plot",
        "scattermap_plot",
        "polar_plot",
        "surface_plot",
        "heatmap",
        "candle_plot",
        "violin_plot",
        "density_contour_plot",
        "sunburst_plot",
        "treemap_plot",
    ]:
        result_instruction = """
# For plot-type answers:
# - Draw with matplotlib/seaborn; capture the active figure as `fig = plt.gcf()` (or use the `fig` from plt.subplots).
# - `fig` MUST be a matplotlib.figure.Figure (not pyplot module).
# - Create a short Vietnamese analysis/insight string in a variable named `analysis`.
# - The `analysis` MUST reference the computed values used in the chart (e.g., top-1 entity, highest/lowest value, or main comparison).
# - Set `result = {"figure": fig, "analysis": analysis}`.
# Do NOT set `result` to only the figure.
"""
        price_driver_hint = ""
        if is_price_driver_question:
            price_driver_hint = """
        # Giá vs cấu trúc/diện tích: trong `analysis` hãy dẫn số liệu từ biểu đồ/bảng, nêu ngoại lệ nếu có, tránh khẳng định nhân quả tuyệt đối (tương quan không phải nguyên nhân) — viết bằng tiếng Việt.
"""

        prompt_context = f"""
        {price_driver_hint}
        Matplotlib/seaborn only; gán Figure theo contract ở trên. Tiêu đề/trục/chú giải bằng tiếng Việt.
        Xoay nhãn trục khi đông category hoặc datetime. Bám <main_question> và dtype trong <metadata>.
        Tham khảo gợi ý kỹ thuật ngắn trong <code_ref> (chọn kiểu biểu đồ phù hợp bạn tự quyết).
        {params_plot}"""
    elif graph_type == "table":
        result_instruction = """
# For table requests:
# - Return ONLY a pandas dataframe in `result`.
"""
        prompt_context = """
        For any table or list request in <main_question>, return ONLY a dataframe in `result` and NEVER a graph.
        """
    else:
        result_instruction = """
# For non-plot requests:
# - Set `result` to the final answer text (string) in Vietnamese.
"""
        prompt_context = f"""
        If <main_question> only needs explanation or numbers without a chart, return a Vietnamese string in `result` and do not import plotting libraries unnecessarily.
        If a chart is still appropriate, use matplotlib/seaborn and the dict contract with `fig` and `analysis` as above.
        {params_plot}"""

    print(f"\n\n{Fore.LIGHTGREEN_EX}STARTING RUNTIME...{Fore.RESET}")
    print(f"\n{Fore.LIGHTBLUE_EX}GRAPH TYPE BASE:{Fore.RESET} {graph_type}")

    prompt_main = f"""

        <metadata>
        {metadata}
        </metadata>
        <data_context>
        {data_context}
        </data_context>
        {rag_docs_context}
        {STRICT_PROMPT_RULES}
        {SELF_CHECK_INSTRUCTIONS}

        Map dataframes to DF_* names according to <main_question>, using variables already declared as <DF_1, DF_2, ...>.
        Unless the question specifies otherwise, treat the primary dataframe as DF_1.

        ```python
        # TODO: import the necessary dependencies.
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        ...

        df = pd.DataFrame(DF_*)

        # Complete your code here.
        ...

        # Assign the final payload here.
        {result_instruction}
        result = None
        ```


        Answer <main_question> concisely and accurately. Follow <guidelines> and <last_code> only as supporting context;
        the request in <main_question> and the tables in <metadata> are authoritative.
        Format numeric values in natural-language summaries with at most two decimal places unless an integer is clearer.
        On execution errors, debug using <message_error> and the broken code in <code_error>.
        {"For price-driver questions, the final `analysis` (and any string `result`) MUST include: (1) quantitative evidence, (2) at least one exception or counter-example, (3) a conditional conclusion that avoids absolute causality claims — all in Vietnamese." if is_price_driver_question else ""}


        <guidelines>
        Follow the guidelines below. Use <messages_history> and <last_code> only for context, not as overrides:
        {prompt_context}
        </guidelines>


        <messages_history>
        {questions_text}
        </messages_history>


        <main_question>
        {user_question}
        </main_question>


        <last_code>
        ```python
        {context_code}
        ```
        </last_code>



        Generate complete Python code and return the full updated code in a single fenced ```python block.

        {VIETNAMESE_USER_FACING_OUTPUT}

        """

    return prompt_main
