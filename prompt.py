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

# All model instructions are in English for clarity.
# Only end-user–visible output (chart labels, analysis text, result strings) must be Vietnamese.
VIETNAMESE_USER_FACING_OUTPUT = (
    "All user-visible text (chart titles, axis labels, legends, `analysis`, `result`) → Vietnamese. "
    "Variable names and code comments → English."
)

STRICT_PROMPT_RULES = """<strict_rules>
- Use only columns in <metadata>; never invent values or access network/files.
  Missing column → Vietnamese string in `result` explaining it.
- matplotlib/seaborn only; NEVER plt.show(); call fig.tight_layout() before result.
- For price vs. drivers: cite chart numbers, flag outliers, note correlation ≠ causation.
</strict_rules>"""


def _matplotlib_graph_hints() -> str:
    return """<chart_hints>
fig, ax = plt.subplots(figsize=(...)); fig.tight_layout() before result.
Pick by dtype+intent: cat-dist→bar(horiz if >8); num-dist→hist/violin;
2-num→scatter; cat-vs-num→box/violin; part-whole→pie(≤6,+Khác); time→line;
corr-matrix→heatmap(annot=True,fmt=".2f"); unsupported→nearest+note in analysis.
Style: x-labels ~45° if crowded; titles/axes/legends Vietnamese; annotate bars if ≤15.
</chart_hints>"""


SELF_CHECK_INSTRUCTIONS = """<self_check>
Verify: columns exist in <metadata>; no hardcoded values; result type matches contract;
no plt.show(); no file/network I/O.
</self_check>"""


def classify_intent(llm: object, question: str, history: str) -> dict:
    """LLM routing: intent, soft graph_type hint, analysis tag."""
    prompt = f"""Real-estate data chatbot router. Return ONE JSON only, no extra text.
Fields:
- intent: data_analysis | metadata_query | general_chat
- analysis_intent (if data_analysis): physical_structure_vs_price | bigger_equals_more_expensive | ambiguous | none
- graph_type: soft hint — best label or "none". Labels: scatter_2d_plot, bubble_plot, scatter_3d_plot,
  bar_plot, line_plot, histogram_plot, pie_plot, box_plot, area_plot, heatmap, violin_plot,
  density_contour_plot, polar_plot, surface_plot, candle_plot, treemap_plot, sunburst_plot,
  choroplethmap_plot, densitymap_plot, scattermap_plot, table, none
- target_col: column name from question or "none"

History: {history}
Question: {question}"""
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
        "scatter_2d_plot", "bubble_plot", "scatter_3d_plot", "bar_plot", "line_plot",
        "histogram_plot", "pie_plot", "box_plot", "area_plot", "choroplethmap_plot",
        "densitymap_plot", "scattermap_plot", "polar_plot", "surface_plot", "heatmap",
        "candle_plot", "violin_plot", "density_contour_plot", "sunburst_plot", "treemap_plot",
    ]:
        result_instruction = (
            "# result = {'figure': fig, 'analysis': analysis}  ← dict always, never bare Figure\n"
            "# analysis: Vietnamese, ≤3 sentences, cite ≥1 number from chart."
        )
        prompt_context = f"Chart intent. {params_plot}\nanalysis: Vietnamese, ≤3 sentences, ≥1 chart number; note correlation ≠ causation if price vs. driver."

    elif graph_type == "table":
        result_instruction = "# result = <pd.DataFrame>  ← no plot"
        prompt_context = "Table intent. Filter/aggregate df; assign DataFrame to result. Skip matplotlib unless table + chart explicitly requested."

    else:
        result_instruction = (
            "# result = '<Vietnamese answer string>'\n"
            "# If a chart genuinely helps: result = {'figure': fig, 'analysis': '<Vietnamese>'}"
        )
        prompt_context = f"Text/stats intent. Return Vietnamese string in result. Plot only if it materially adds insight; if so use dict contract. {params_plot}"

    print(f"\n\n{Fore.LIGHTGREEN_EX}STARTING RUNTIME...{Fore.RESET}")
    print(f"\n{Fore.LIGHTBLUE_EX}GRAPH TYPE BASE:{Fore.RESET} {graph_type}")

    prompt_main = f"""Python data-analysis agent — Vietnamese real-estate dataset.
Write ONE self-contained python code block answering <main_question>. Executed in sandbox.

<metadata>
{metadata}
</metadata>
<data_context>
{data_context}
</data_context>
{rag_docs_context}
{STRICT_PROMPT_RULES}
{SELF_CHECK_INSTRUCTIONS}

Sandbox vars: DF_1, DF_2, … (default: DF_1). Skeleton:
```python
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
df = DF_1
# your code
{result_instruction}
result = None
```
<main_question> overrides history/last_code. Round numbers ≤2 decimals.
On error (<code_error>/<message_error>): rewrite root cause, don't just wrap try/except.

<guidelines>{prompt_context}</guidelines>
<messages_history>{questions_text}</messages_history>
<main_question>{user_question}</main_question>
<last_code>
```python
{context_code}
```
</last_code>

Return COMPLETE solution in ONE fenced python block. No text outside it.
{VIETNAMESE_USER_FACING_OUTPUT}
"""
    return prompt_main