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
    "User-visible text (`analysis`, `result`, chart labels/titles/legends, dataframe display) "
    "must be Vietnamese."
)

STRICT_PROMPT_RULES = """
<strict_rules>
- Use only columns in <metadata>; no invented numbers, no network/files outside context.
- Missing column → Vietnamese string in `result` explaining it.
- Plots: matplotlib/seaborn only; no plt.show(); assign Figure per contract.
- For price vs drivers/structure: cite chart/table numbers, note exceptions; correlation ≠ causation.
</strict_rules>
"""


def _matplotlib_graph_hints() -> str:
    """Task-agnostic plotting hints; model chooses chart type from question + dtypes."""
    return """
<code_ref>
- `fig, ax = plt.subplots(...)`; `fig.tight_layout()` before `result`.
- Choose sns/plt by task.
- PRE-IMPORTED: `import pandas as pd`, `import numpy as np`, `import matplotlib.pyplot as plt`, `import seaborn as sns`, `from matplotlib.figure import Figure`.
- DATA: `df` and `DF_1` refer to the main dataframe.
- TITLES/AXES: Vietnamese.
- RESULTS: `result = {"figure": fig, "analysis": "..."}` or `result = df`.
</code_ref>
"""

SELF_CHECK_INSTRUCTIONS = """
<self_check>
Columns exist; values from dataframe; `result` type matches contract; no external I/O.
</self_check>
"""


def classify_intent(llm: object, question: str, history: str) -> dict:
    """
    LLM routing: intent, soft graph_type hint, analysis tag (codegen picks final chart).
    """
    prompt = f"""Return one JSON only. Tabular housing dataset.

Fields:
- intent: data_analysis | metadata_query | general_chat
- analysis_intent (if data_analysis): physical_structure_vs_price | bigger_equals_more_expensive | ambiguous | none
- graph_type: best single label if obvious else "none" — soft hint only; codegen chooses matplotlib details.
  Labels: scatter_2d_plot, bubble_plot, scatter_3d_plot, bar_plot, line_plot, histogram_plot,
  pie_plot, box_plot, area_plot, heatmap, violin_plot, density_contour_plot, polar_plot,
  surface_plot, candle_plot, treemap_plot, sunburst_plot, choroplethmap_plot, densitymap_plot,
  scattermap_plot, table, none
- target_col: column name from question or "none"

History:
{history}

Question:
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


def define_graph_type(llm: object, question_user: str, hist_questions: str) -> str:
    """LLM-only routing for plot/table vs non-plot branch (no keyword fast-path)."""
    res = classify_intent(llm, question_user, hist_questions)
    return res.get("graph_type", "none") or "none"


def process_prompt(
    session_msgs: list[dict], user_question: str, data: dict, llm: object, intent: str = None, graph_type: str = "none"
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

    # intent and graph_type are now passed from app.py to avoid redundant LLM calls
    params_plot = _matplotlib_graph_hints()

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
# Plot: matplotlib/seaborn; `fig` = Figure; Vietnamese `analysis` citing chart numbers;
# `result = {"figure": fig, "analysis": analysis}` (not figure alone).
"""
        prompt_context = f"""
        Matplotlib/seaborn; Figure contract above; Vietnamese labels. Pick chart type from
        <main_question> + dtypes in <metadata>. {params_plot}"""
    elif graph_type == "table":
        result_instruction = "# Table: `result` = pandas DataFrame only."
        prompt_context = "Table/list request → dataframe in `result` only, no plot."
    else:
        result_instruction = "# Non-plot: `result` = Vietnamese string."
        prompt_context = (
            "Text/numbers only → Vietnamese `result`; skip plotting imports unless a chart helps. "
            f"If you plot, use dict contract. {params_plot}"
        )

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

Use DF_* from <metadata>; default primary = DF_1 unless question says otherwise.

# df = DF_1 (already available)
# No need to import pandas, matplotlib, seaborn, or numpy. They are already in the environment.
{result_instruction}
result = None
```

<main_question> is authoritative over history/last_code. Summaries: ≤2 decimals unless int clearer.
Errors: fix with <code_error>/<message_error>.

<guidelines>
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

Return the full solution in one fenced python code block.

{VIETNAMESE_USER_FACING_OUTPUT}
"""

    return prompt_main
