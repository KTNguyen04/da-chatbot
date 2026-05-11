# prompt.py
"""
Prompt builder with RAG docs (ChromaDB).

Changes from the original baseline:
  1. Import rag_docs_manager at the top.
  2. In process_prompt(): call build_rag_context() and inject into prompt_main.
  3. define_graph_type() extended for province / ranking style questions.
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
- For charts use matplotlib and seaborn only. Do NOT use plotly, plotly express, or bokeh.
- Do NOT call plt.show(). Build a matplotlib.figure.Figure and pass it in `result` as specified.
- Prefer brief Vietnamese comments on important logic blocks in generated code.
</strict_rules>
"""


def _matplotlib_graph_hints(graph_type: str) -> str:
    """
    Inline matplotlib/seaborn patterns per routed graph type (replaces ragdata/*.txt).
    """
    common = """
- Use fig, ax = plt.subplots(figsize=(10, 6)) unless a multi-panel layout needs otherwise; fig.tight_layout() before assigning `result`.
- Rotate crowded category labels (~45°). Prefer sns.set_theme(style=\"whitegrid\") when it helps readability.
"""
    hints: dict[str, str] = {
        "bar_plot": "sns.barplot(data=df, x=..., y=..., ax=ax) or grouped counts: s = df[col].value_counts(); s.plot(kind='bar', ax=ax). For rankings, sort values descending first.",
        "scatter_2d_plot": "sns.scatterplot(data=df, x=..., y=..., ax=ax, alpha=0.3-0.6) for dense clouds; cite trend/outliers in analysis.",
        "bubble_plot": "sns.scatterplot(..., size=..., sizes=(20, 400), ax=ax) or ax.scatter with s= array scaled from a third numeric column.",
        "scatter_3d_plot": "from mpl_toolkits.mplot3d import Axes3D  # use fig.add_subplot(projection='3d') then ax.scatter3D(xs, ys, zs).",
        "line_plot": "sns.lineplot(data=df, x=..., y=..., ax=ax) or df.sort_values(...).plot(x=..., y=..., ax=ax).",
        "histogram_plot": "sns.histplot(data=df, x=..., ax=ax, kde=False) or df[col].plot.hist(ax=ax, bins=...).",
        "pie_plot": "df[col].value_counts().plot.pie(ax=ax, autopct='%1.1f%%') or ax.pie(...); limit slices (e.g. top N + Other).",
        "box_plot": "sns.boxplot(data=df, x=..., y=..., ax=ax) or sns.boxplot(x=group, y=value, data=df, ax=ax).",
        "area_plot": "df.pivot_table(...).plot.area(ax=ax) or fill_between for stacked trends.",
        "heatmap": "sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax, center=0).",
        "violin_plot": "sns.violinplot(data=df, x=..., y=..., ax=ax).",
        "density_contour_plot": "sns.kdeplot(data=df, x=..., y=..., fill=True, ax=ax) for 2D density.",
        "polar_plot": "ax = fig.add_subplot(projection='polar'); ax.plot(theta, r).",
        "surface_plot": "from mpl_toolkits.mplot3d import Axes3D; plot_surface on 3D axes for grid Z.",
        "candle_plot": "matplotlib does not have native OHLC; use line plot of close or bar chart of range per period, and state limitation in analysis.",
        "treemap_plot": "No native treemap; use horizontal bar of top categories by value, or stacked bar — note the mapping in analysis.",
        "sunburst_plot": "No native sunburst; use nested bar or grouped bar for top two levels — note the mapping in analysis.",
        "choroplethmap_plot": "Without geopandas, approximate with bar chart by Province/region column; mention simplification in analysis.",
        "densitymap_plot": "Approximate with 2D kdeplot on lon/lat if columns exist; else bar by region.",
        "scattermap_plot": "If lat/lon exist use scatterplot colored by metric; else bar by province.",
        "base_ref": "Pick sns or ax API consistent with <main_question>; keep layout readable and titled in Vietnamese.",
    }
    body = hints.get(graph_type, hints["base_ref"])
    return f"\n\n<code_ref>\n{common}\n{body}\n</code_ref>\n"

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
    LLM-based intent classification (replaces rigid keyword matching).
    """
    prompt = f"""
Analyze the user's question about the Vietnam housing dataset and return one JSON object only.

Intent types:
1. 'data_analysis': Needs computation, charts, or statistics (e.g. plot price, average area).
2. 'metadata_query': Asks about data meaning, column definitions, or units (e.g. what is the address column).
3. 'general_chat': Greetings or off-topic chat.

## Analytical intent recognition

The user may phrase questions in different ways; map the underlying meaning to one of the two
analytical intents below when applicable. Do not rely on exact keyword matching.

### Predefined analytical intents

Intent 1 — Physical structure vs. price
The user wants to understand how measurable physical attributes (area, floors, bedrooms,
bathrooms, frontage) relate to or influence selling price.

Intent 2 — Bigger = more expensive?
The user wants to verify whether larger homes are always priced higher, including exceptions,
nuances, or non-linear relationships.

### Ambiguous input
If the message could map to either analytical intent or clearly matches neither, set
analysis_intent to 'ambiguous' so the assistant can ask a short clarifying question.

Return exactly one JSON object with this shape:
{{
    "intent": "data_analysis" | "metadata_query" | "general_chat",
    "graph_type": "scatter_2d_plot" | "box_plot" | "bar_plot" | "treemap_plot" | "none",
    "target_col": "related column name or none",
    "analysis_intent": "physical_structure_vs_price" | "bigger_equals_more_expensive" | "ambiguous" | "none"
}}

Rules for graph_type when intent is data_analysis:
- analysis_intent='physical_structure_vs_price' → prefer "scatter_2d_plot" (area vs price),
  or "box_plot"/"bar_plot" if the question emphasizes comparison by floors/bedrooms/bathrooms/frontage.
- analysis_intent='bigger_equals_more_expensive' → prefer "box_plot" by area groups,
  or "scatter_2d_plot" to surface outliers.
- Otherwise use "none".

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

    province_ranking_markers = [
        "tỉnh nào",
        "thành phố nào",
        "province",
        "xuất hiện nhiều nhất",
        "nhiều nhất trong dataset",
        "phân bổ theo tỉnh",
        "top tỉnh",
        "xếp hạng tỉnh",
        "tỉnh nào nhiều",
        "tỉnh nào ít",
        "thành nào",
    ]
    if any(m in q for m in province_ranking_markers) and (
        "bar" in q
        or "cột" in q
        or "biểu đồ" in q
        or "vẽ" in q
        or "đồ thị" in q
        or "nhất" in q
        or "rank" in q
    ):
        return "bar_plot"

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
    q = question_user.lower()
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
    if any(kw in q for kw in price_driver_keywords) and (
        "giá" in q or "price" in q
    ):
        return "scatter_2d_plot"

    keyword_graph = _graph_type_from_keywords(question_user)
    if keyword_graph is not None:
        return keyword_graph

    res = classify_intent(llm, question_user, hist_questions)
    analysis_intent = res.get("analysis_intent", "none")
    graph_type = res.get("graph_type", "none")

    if analysis_intent == "physical_structure_vs_price":
        if any(
            k in q
            for k in [
                "số tầng",
                "tầng",
                "phòng ngủ",
                "phòng tắm",
                "mặt tiền",
                "frontage",
            ]
        ):
            return "box_plot"
        return "scatter_2d_plot"

    if analysis_intent == "bigger_equals_more_expensive":
        if any(
            k in q
            for k in [
                "luôn",
                "always",
                "ngoại lệ",
                "exception",
                "không phải lúc nào",
            ]
        ):
            return "box_plot"
        return "scatter_2d_plot"

    return graph_type


def _deprecated_define_graph_type(
    llm: object, question_user: str, hist_questions: str
) -> str:
    """
    Legacy LLM router for graph type (RAG routing). Kept for reference; prefer define_graph_type().
    """

    prompt_template = f"""
    <question>
    Answer the following question with a single specific word.
    {question_user}
    Answer with just one word.
    </question>

    <hist_questions>
    {hist_questions}
    </hist_questions>

    Analyze the question within the <question> tags and determine which type of graph is most appropriate for
    the question, according to the options listed in <allowed_words>.

    Analyze message history within the <hist_questions> tags for more context in response.

    <allowed_words>
    answer "treemap_plot" if has in question (treemap, or Vietnamese "biểu đồ treemap")
    answer "sunburst_plot" if has in question (sunburst, or Vietnamese layered pie chart phrasing)
    answer "density_contour_plot" if has in question (density contour, contour plot phrasing)
    answer "violin_plot" if has in question (violin plot phrasing in EN or VN)
    answer "candle_plot" if has in question (candle, OHLC, candlestick)
    answer "heatmap" if has in question (heatmap, correlation matrix phrasing in EN or VN)
    answer "scatter_2d_plot" if has in question (scatter, scatter plot phrasing in EN or VN)
    answer "bubble_plot" if has in question (bubble chart phrasing in EN or VN)
    answer "scatter_3d_plot" if has in question (scatter 3d phrasing)
    answer "surface_plot" if has in question (3d surface phrasing)
    answer "bar_plot" if has in question (bar chart phrasing in EN or VN)
                     OR if the question asks about province/city frequency or ranking, e.g. Vietnamese:
                     which province appears most/least, distribution by province, top provinces, ranking.
    answer "line_plot" if has in question (line chart, trend over time phrasing)
    answer "histogram_plot" if has in question (histogram, distribution, frequency phrasing)
    answer "pie_plot" if has in question (pie chart, percentage breakdown phrasing)
    answer "box_plot" if has in question (box plot, quartile phrasing)
    answer "area_plot" if has in question (area chart phrasing; not the column "Area" alone)
    answer "choroplethmap_plot" if has in question (choropleth map phrasing)
    answer "densitymap_plot" if has in question (density map phrasing)
    answer "scattermap_plot" if has in question (scatter map / points on map phrasing)
    answer "polar_plot" if has in question (radar / polar / spider chart phrasing)
    answer "table" if has in question (dataframe table, show as table phrasing)
    answer "table" if has in question (dataframe / table phrasing including former plotly-table wording)
    answer "base_ref" if there is nothing related to the question.
    </allowed_words>

    Use context information for accuracy in the answer.
    DECLARE ONLY one WORD in response according with <allowed_words> tags.

    Response examples:
    question: 'vẽ biểu đồ cột theo tỉnh thành'
    response: 'bar_plot'

    question: 'tỉnh nào xuất hiện nhiều nhất trong dataset?'
    response: 'bar_plot'

    question: 'tỉnh nào có nhiều bất động sản nhất?'
    response: 'bar_plot'

    question: 'thống kê số lượng bất động sản theo từng tỉnh'
    response: 'bar_plot'

    question: 'phân phối giá nhà theo histogram'
    response: 'histogram_plot'

    """

    response = llm.invoke(prompt_template)
    return response.content.strip()


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
        province_hint = ""
        province_keywords = [
            "tỉnh nào",
            "thành phố nào",
            "province",
            "xuất hiện nhiều nhất",
            "phân bổ theo tỉnh",
            "bao nhiêu bất động sản",
            "top tỉnh",
            "xếp hạng tỉnh",
        ]
        if any(kw in user_question.lower() for kw in province_keywords):
            province_hint = """
        # PROVINCE RANKING HINT:
        # The question is about counting listings by province/city.
        # Use column 'Province' (province or city name).
        # Standard pattern:
        #   province_counts = df['Province'].value_counts().reset_index()
        #   province_counts.columns = ['Province', 'Count']
        #   province_counts['Percentage'] = (province_counts['Count'] / len(df) * 100).round(2)
        # Prefer a horizontal bar chart (orientation='h') sorted high to low.
        # Show count labels on each bar. Use Vietnamese labels for the chart.
"""

        price_driver_hint = ""
        intent_hint = ""
        if is_price_driver_question:
            price_driver_hint = """
        # PRICE DRIVER HINT:
        # The question asks how area/structure relates to price.
        # You MUST give quantitative evidence, cite exceptions, and state a conditional conclusion.
        # Do not claim absolute causality; explicitly note that correlation is not causation (in Vietnamese in `analysis`).
        # Prefer scatter (area vs price) plus group comparisons by Floors/Bedrooms/Bathrooms/Frontage.
"""
            intent_hint = """
        # INTENT RECOGNITION:
        # Map the question semantically to one of two intents:
        # (1) Physical structure vs. price
        # (2) Bigger = more expensive?
        # If ambiguous between the two, ask the user a clarifying question before drawing a firm conclusion.
        #
        # CHART SELECTION GUIDELINES:
        # - Intent 1:
        #   + Scatter 2D (area vs price) for continuous trend.
        #   + Box/bar by Floors/Bedrooms/Bathrooms/Frontage for structural group comparison.
        # - Intent 2:
        #   + Box plot by area buckets to show price overlap between smaller vs larger homes.
        #   + Scatter 2D to highlight exceptions (small expensive vs large cheap).
        # Do not use a single chart alone for an absolute conclusion when exceptions exist.
"""

        prompt_context = f"""
        {province_hint}
        {price_driver_hint}
        {intent_hint}
        For plots, use matplotlib and seaborn only; assign a matplotlib Figure per the result contract above.
        All visible text in titles, axis labels, legends, etc., MUST be in Vietnamese.
        Set the legend title from the data when not specified by the user.
        On the x and y axes, rotate long date or category labels to about 45 degrees when it improves readability.
        Derive axis label text from <main_question> where sensible.
        Derive the chart title from <main_question>.
        Follow the blocks below exactly, always driven by <main_question>.
        Use patterns in <code_ref> as a baseline for <main_question>.
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
