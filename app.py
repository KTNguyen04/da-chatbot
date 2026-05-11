# App
import os
from pathlib import Path
import json
import re
import traceback
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

from colorama import Fore
from agent import AgentAI
from prompt import build_json_intent_prompt
from styles import process_styles

from chart_pipeline import (
    PLOTLY_RENDER_CONFIG,
    eval_table_query,
    extract_json_object,
    new_chart_message_id,
    normalize_intent_payload,
    optimize_plotly_figure,
    store_in_exec_cache,
    take_from_exec_cache,
    _stable_code_fingerprint,
)

from langchain_community.chat_models import ChatOllama
from tool_executor import run_tool_insight
from rag_docs_manager import build_rag_context, warmup_rag_embedder


# Parameters
WHITELIST_ENV = ["json", "statsmodels", "scipy", "datetime"]
N_SAMPLES = 5  # Number of samples sent to the prompt in non-private mode
MAX_ATTEMPTS = 10  # Number of calls to LLM in case of execution error
VERBOSE = False  # Show final prompt and errors

# Hardcoded dataset/model (UI chỉ còn Temperature)
DATASET_PATH = (
    Path(__file__).resolve().parent / "datasets" / "vietnam_housing_dataset_cleaned.csv"
)
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Optional smaller/faster Ollama model for intent routing only (see classify_intent_llm fallback).
OLLAMA_ROUTER_MODEL = os.getenv("OLLAMA_ROUTER_MODEL", "").strip()

# Tool-calling insight: LLM gọi tool để tính số liệu thực → sinh insight
ENABLE_TOOL_INSIGHT = True  # Tắt/bật tính năng tool-calling insight


@st.cache_data
def load_main_dataset() -> pd.DataFrame:
    # Keep encoding logic identical to uploaded csv handling.
    return pd.read_csv(DATASET_PATH, encoding="utf-8", encoding_errors="replace")


def _get_main_df(data: dict) -> pd.DataFrame:
    # This app hardcodes a single dataset, so the first value is the one we need.
    return next(iter(data.values()))


def _is_greeting(text: str) -> bool:
    t = text.strip().lower()
    # Common Vietnamese greetings (keep it simple to avoid false positives).
    return bool(
        re.search(
            r"(^(xin chào|chào|hello|hi|hey)\b)|(\b(xin chào|chào bạn|chào bạn|hello|hi|hey)\b)",
            t,
        )
    )


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def _is_price_driver_question(text: str) -> bool:
    q = text.lower()
    driver_keywords = [
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
        "cấu trúc",
    ]
    return any(kw in q for kw in driver_keywords) and ("giá" in q or "price" in q)


def _is_dataset_followup_for_analysis(
    user_question: str, messages: list[dict] | None
) -> bool:
    if not messages:
        return False

    last_assistant_response = ""
    for item in reversed(messages):
        if item.get("role") == "assistant":
            resp = item.get("response", "")
            if isinstance(resp, str):
                last_assistant_response = resp.lower()
            elif isinstance(resp, dict):
                last_assistant_response = str(resp.get("analysis", "")).lower()
            break

    if "vẽ biểu đồ/xem thống kê mô tả cho cột nào" not in last_assistant_response:
        return False

    q = user_question.strip().lower()
    followup_markers = [
        "cột ",
        "cot ",
        "column",
        "price",
        "giá",
        "area",
        "diện tích",
        "bedroom",
        "bathroom",
        "province",
        "district",
    ]
    return any(marker in q for marker in followup_markers) or len(q.split()) <= 3


def _rewrite_followup_question_for_analysis(
    user_question: str, messages: list[dict] | None
) -> str:
    """
    Nếu user trả lời ngắn sau câu hỏi "chọn cột để vẽ/thống kê",
    rewrite câu hỏi để graph router nhận diện rõ ý định phân tích.
    """
    if not _is_dataset_followup_for_analysis(user_question, messages):
        return user_question

    q = user_question.strip()
    return (
        "Vẽ biểu đồ phù hợp và thống kê mô tả cho cột "
        f"'{q}'. Nếu là cột phân loại thì ưu tiên bar chart."
    )


def should_answer_normally(
    user_question: str, llm: object, messages: list[dict] | None = None
) -> tuple[bool, str]:
    """
    Return (should_short_circuit, response_text).
    If should_short_circuit is True, caller should NOT run the Agent/code path.
    """
    q = user_question.strip()

    # Follow-up sau dataset overview: ưu tiên đi vào luồng phân tích/chart.
    if _is_dataset_followup_for_analysis(q, messages):
        return (False, "")

    # Fast-path greetings to reduce latency.
    if _is_greeting(q):
        return (
            True,
            "Chào bạn! Mình có thể giúp bạn:\n"
            "- Xem thông tin/tên các cột trong dataset\n"
            "- Vẽ biểu đồ và phân tích quan hệ giữa các biến\n"
            "- Trả lời các câu hỏi liên quan đến dữ liệu\n\n"
            "Bạn muốn bắt đầu với cột nào hoặc kiểu biểu đồ nào?",
        )

    dataset_keywords = [
        "dataset",
        "tập dữ liệu",
        "các cột",
        "có những cột",
        "dữ liệu gồm",
        "cột nào",
        "features",
        "danh sách cột",
        "liệt kê cột",
        "tên các cột",
        "schema",
        "dtypes",
        "kiểu dữ liệu",
        "bao nhiêu dòng",
        "số dòng",
        "số cột",
        "kích thước",
        "shape của",
    ]
    # Câu hỏi về ý nghĩa / ngữ cảnh của cột → nên đi qua RAG, không phải overview
    context_keywords = [
        "có ý nghĩa gì",
        "ý nghĩa",
        "nghĩa là gì",
        "là gì",
        "giải thích",
        "câu chuyện",
        "phía sau",
        "nguồn gốc",
        "mô tả",
        "thuộc tính",
        "trường ",
        "field",
        "column",
        "describe",
        "what is",
        "what does",
        "explain",
    ]
    help_keywords = [
        "bạn làm gì",
        "bạn có thể",
        "giúp",
        "hướng dẫn",
        "cách dùng",
        "tính năng",
    ]

    if _contains_any(q, help_keywords):
        return (
            True,
            "Mình có thể hỗ trợ phân tích dữ liệu theo các cách:\n"
            "- Giới thiệu dataset (số dòng, số cột, tên cột)\n"
            "- Vẽ các biểu đồ (bar/line/scatter/hist/heatmap/mapbox, ...)\n"
            "- Trả lời câu hỏi mô tả theo dữ liệu\n\n"
            "Bạn hãy nói rõ bạn muốn xem biểu đồ gì hoặc bạn quan tâm cột nào nhé.",
        )

    # Câu hỏi về ý nghĩa cột / ngữ cảnh → ưu tiên RAG trước khi check dataset_keywords
    if _contains_any(q, context_keywords):
        return (True, "__RAG_CONTEXT__")

    # High-confidence visualization keywords -> go through the plot flow.
    visualization_keywords = [
        "vẽ",
        "biểu đồ",
        "đồ thị",
        "plot",
        "chart",
        "scatter",
        "histogram",
        "hist",
        "heatmap",
        "map",
        "bản đồ",
        "bar",
        "line",
        "pie",
        "box",
        "violin",
        "treemap",
        "sunburst",
        "ohlc",
        "candlestick",
        "surface",
        # ✅ Thêm ranking/comparison keywords:
        "nhiều nhất",
        "ít nhất",
        "cao nhất",
        "thấp nhất",
        "top ",
        "xếp hạng",
        "so sánh",
        "nhiều hơn",
        "ít hơn",
        "tỉnh nào",
        "thành nào",
        "cái nào",
        "loại nào",
        "phân phối",
        "thống kê",
        "ranking",
        "ảnh hưởng",
        "tác động",
        "nhà to hơn",
        "đồ họa",
        "thống kê mô tả",
        "phân tích dữ liệu",
        "xu hướng giá",
        "theo tỉnh",
        "theo thành phố",
        "theo loại nhà",
    ]
    if _contains_any(q, visualization_keywords):
        return (False, "")

    if _contains_any(q, dataset_keywords):
        return (True, "__DATASET_OVERLAY__")

    # Numeric / statistical questions → full analysis path (no extra classifier LLM).
    analysis_stats_keywords = [
        "trung bình",
        "trung vi",
        # Avoid bare "mean"/"median"/"correlation" — substring false positives (e.g. "meaning", "decorrelation").
        "trung bình giá",
        "tbc ",
        "độ lệch chuẩn",
        "phương sai",
        "variance",
        "tương quan",
        "covariance",
        "hồi quy",
        "regression",
        "ngoại lệ",
        "outlier",
        "quantile",
        "phân vị",
        "tứ phân vị",
        "min ",
        " max ",
        "nhỏ nhất",
        "lớn nhất",
        "tổng số",
        "đếm số",
        "bao nhiêu căn",
        "bao nhiêu nhà",
        "chênh lệch",
        "khác biệt",
        "tỉ lệ",
        "phần trăm",
        "phân trăm",
        "so sánh giá",
        "tb giá",
        "giá tb",
        "giá trung bình",
        "giá trung vi",
        "số căn",
        "số bản ghi",
        "số lượng nhà",
        "đếm theo",
        "tần suất",
        "frequency",
    ]
    if _contains_any(q, analysis_stats_keywords):
        return (False, "")

    # Fallback: use LLM classification (handles typos/wording variations).
    try:
        intent = classify_intent_llm(q, llm=llm)
        if intent == "greeting":
            return (
                True,
                "Chào bạn! Mình có thể giúp bạn:\n"
                "- Xem thông tin/tên các cột trong dataset\n"
                "- Vẽ biểu đồ và phân tích quan hệ giữa các biến\n"
                "- Trả lời các câu hỏi liên quan đến dữ liệu\n\n"
                "Bạn muốn bắt đầu với cột nào hoặc kiểu biểu đồ nào?",
            )
        if intent == "dataset_info":
            return (True, "__DATASET_OVERLAY__")
        if intent == "context_query":
            return (True, "__RAG_CONTEXT__")
        if intent == "direct_chat":
            return (True, "__DIRECT_LLM_CHAT__")
        if intent == "visualization_request":
            return (False, "")
    except Exception:
        # Last fallback: treat as direct chat.
        return (True, "__DIRECT_LLM_CHAT__")

    return (True, "__DIRECT_LLM_CHAT__")


def answer_dataset_overview(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    n_rows, n_cols = df.shape
    preview = ", ".join(cols[:20])
    suffix = f" (và {n_cols - 20} cột nữa)" if n_cols > 20 else ""
    return (
        f"Dataset hiện có {n_rows:,} dòng và {n_cols} cột.\n"
        f"Các cột (một phần): {preview}{suffix}.\n\n"
        "Bạn muốn mình vẽ biểu đồ/xem thống kê mô tả cho cột nào?"
    )


def classify_intent_llm(user_question: str, llm: object) -> str:
    """
    Use LLM to classify user intent.
    Output must be one of:
      - greeting
      - dataset_info
      - context_query
      - visualization_request
      - direct_chat
    """
    intent_labels = [
        "greeting",
        "dataset_info",
        "context_query",
        "visualization_request",
        "direct_chat",
    ]

    classifier_prompt = f"""
You are an intent classifier for a Vietnamese data analysis assistant.
Given the user's message (may contain typos), choose exactly one intent label.

Definitions:
- greeting: user is saying hello/hi/xin chao and is not requesting charts/code.
- dataset_info: user asks what columns/features exist in the dataset as a list, wants to see all column names, or asks "dataset gom nhung gi". Does NOT include questions about the meaning of a specific column/field.
- context_query: user asks about the meaning, description, or background of a specific column/field/attribute, OR asks about the story/origin/context behind the dataset. Examples: "truong address co y nghia gi?", "bathroom la gi?", "cau chuyen phia sau dataset", "giai thich cot price".
- visualization_request: user asks for charts/plots/figures/maps/tables based on data, or says "vẽ/biểu đồ/plot/chart/graph" etc.
  Also treat ranking/comparison questions as visualization_request even if the user does NOT explicitly say "chart", for example:
  "cái nào nhiều nhất", "tỉnh/thành nào chiếm nhiều nhất", "top", "so sánh", "cao nhất/thấp nhất", "nhiều/ít hơn", "xếp hạng".
- direct_chat: everything else (general questions, wording not clearly requesting plots or dataset description).

Examples:
- "Tỉnh thành nào có nhiều bất động sản nhất?" → visualization_request
- "Cái nào cao nhất?" → visualization_request
- "Top 5 tỉnh có giá cao nhất" → visualization_request
- "So sánh giá theo loại nhà" → visualization_request
- "truong address co y nghia gi?" → context_query
- "bathroom la gi?" → context_query
- "cau chuyen phia sau dataset" → context_query
- "dataset co nhung cot nao?" → dataset_info
...

Return ONLY a JSON object with this exact schema:
{{"intent": "<one_of_intent_labels>"}}
where <one_of_intent_labels> MUST be one of: {intent_labels}

User message:
{user_question}
""".strip()

    # Make classification deterministic (minimize randomness).
    prev_temp = getattr(llm, "temperature", None)
    try:
        if hasattr(llm, "temperature"):
            llm.temperature = 0.0
        resp_obj = llm.invoke(classifier_prompt)
        content = resp_obj.content if hasattr(resp_obj, "content") else str(resp_obj)

        # Extract first JSON object found.
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON found in classifier output.")
        payload = json.loads(m.group(0))
        intent = payload.get("intent")
        if intent not in intent_labels:
            raise ValueError(f"Unexpected intent: {intent}")
        return intent
    finally:
        if prev_temp is not None and hasattr(llm, "temperature"):
            llm.temperature = prev_temp


def _get_or_create_chat_ollama(model: str, base_url: str, temperature: float) -> ChatOllama:
    """
    Reuse one ChatOllama across Streamlit reruns; only sync temperature (and model/base_url if changed).
    """
    key = "_cached_chat_ollama"
    llm = st.session_state.get(key)
    if llm is not None:
        same_model = getattr(llm, "model", None) == model
        same_base = getattr(llm, "base_url", None) == base_url
        if same_model and same_base:
            if hasattr(llm, "temperature"):
                llm.temperature = temperature
            return llm
    llm = ChatOllama(model=model, temperature=temperature, base_url=base_url)
    st.session_state[key] = llm
    return llm


def _get_classifier_llm(main_llm: object) -> object:
    """
    Optional dedicated router model (OLLAMA_ROUTER_MODEL) for classify_intent_llm
    to avoid loading the full chat model for a tiny JSON classification task.
    """
    if not OLLAMA_ROUTER_MODEL:
        return main_llm
    key = "_cached_ollama_router_llm"
    llm = st.session_state.get(key)
    if llm is not None:
        same_model = getattr(llm, "model", None) == OLLAMA_ROUTER_MODEL
        same_base = getattr(llm, "base_url", None) == OLLAMA_BASE_URL
        if same_model and same_base:
            if hasattr(llm, "temperature"):
                llm.temperature = 0.0
            return llm
    llm = ChatOllama(
        model=OLLAMA_ROUTER_MODEL,
        temperature=0.0,
        base_url=OLLAMA_BASE_URL,
    )
    st.session_state[key] = llm
    return llm


# ── Streaming helpers ─────────────────────────────────────────────────────────


def _stream_llm_response(llm: object, prompt: str) -> str:
    """
    Stream LLM response token-by-token into the current st.chat_message context.
    Returns the full accumulated text.
    Falls back to llm.invoke() if streaming is not available.
    """
    full_text = ""
    try:
        placeholder = st.empty()
        for chunk in llm.stream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except (AttributeError, NotImplementedError):
        # Fallback: model does not support streaming
        resp_obj = llm.invoke(prompt)
        full_text = resp_obj.content if hasattr(resp_obj, "content") else str(resp_obj)
        st.markdown(full_text)
    return full_text


def _stream_text(text: str) -> None:
    """
    Show a pre-built string (used for tool insight / static responses).
    Word-by-word animation was removed: it added O(n) sleeps and dominated latency.
    """
    st.markdown(text)


# Session_state reruns variables
if "last_code" not in st.session_state:
    st.session_state["last_code"] = ""
if "context_code_var" not in st.session_state:
    st.session_state["context_code_var"] = False
if "count_var" not in st.session_state:
    st.session_state["count_var"] = 1
if "response_error_var" not in st.session_state:
    st.session_state["response_error_var"] = ""
if "sample_var" not in st.session_state:
    st.session_state["sample_var"] = N_SAMPLES
if "agent_var" not in st.session_state:
    st.session_state["agent_var"] = None
if "geojson_var" not in st.session_state:
    st.session_state["geojson_var"] = None


def clear_chat_history() -> None:
    """
    Clear chat history | reset session variables | stop runtime agent.
    """

    # Reset session_state vars
    st.session_state["last_code"] = ""
    st.session_state["context_code_var"] = False
    st.session_state["response_error_var"] = ""
    st.session_state["messages"] = []

    # Stop agent runtime
    agent = st.session_state.get("agent_var")
    if agent is not None:
        agent.chat_stop()
    st.session_state["agent_var"] = None


def _execute_pending_chart_msg(msg_id: str) -> None:
    """Deferred Plotly execution (chatbot.py-style); reuses session exec cache."""
    df = st.session_state.get("_chart_main_df")
    agent: AgentAI | None = st.session_state.get("agent_var")
    code_key = f"chart_code_{msg_id}"
    code = (st.session_state.get(code_key) or "").strip()

    target: dict | None = None
    for m in st.session_state.messages:
        r = m.get("response")
        if isinstance(r, dict) and r.get("chart_pipeline") and r.get("msg_id") == msg_id:
            target = r
            break

    if target is None or df is None or agent is None:
        return

    if not code:
        target["state"] = "error"
        target["error"] = "Chưa có mã Python để thực thi."
        return

    target.pop("error", None)
    target["state"] = "pending"
    target.pop("figure", None)
    target.pop("post_chart_analysis", None)

    fp = _stable_code_fingerprint(code, df)
    cached = take_from_exec_cache(fp)
    if cached is not None:
        fig = cached["figure"]
        exec_analysis = cached.get("analysis")
    else:
        try:
            agent.last_code = code
            result = agent.run_code(code)
        except Exception as e:
            target["state"] = "error"
            target["error"] = f"{type(e).__name__}: {e}"
            return
        if not isinstance(result, dict) or result.get("figure") is None:
            target["state"] = "error"
            target["error"] = (
                "Code chạy xong nhưng không trả về `result` dict có key 'figure'. "
                "Hãy gán `result = {'figure': fig, 'analysis': '...'}`."
            )
            return
        raw_fig = result["figure"]
        exec_analysis = result.get("analysis")
        fig = optimize_plotly_figure(raw_fig)
        store_in_exec_cache(fp, fig, exec_analysis)

    target["state"] = "ready"
    target["figure"] = fig
    target["post_chart_analysis"] = (exec_analysis or "").strip()
    st.session_state["last_code"] = code

    uq = target.get("_user_question", "")
    if ENABLE_TOOL_INSIGHT and uq:
        try:
            tool_insight_text = run_tool_insight(
                user_question=uq,
                df=df,
                ollama_base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                verbose=VERBOSE,
            )
        except Exception:
            tool_insight_text = None
        if tool_insight_text:
            sep = "\n\n---\n" if target.get("post_chart_analysis") else ""
            target["post_chart_analysis"] = (
                (target.get("post_chart_analysis") or "")
                + sep
                + "**🔧 Phân tích từ Tool Insight:**\n\n"
                + tool_insight_text
            ).strip()


def _render_chart_pipeline_message(message: dict, main_df: pd.DataFrame) -> None:
    """Render one assistant message produced by the JSON chart pipeline."""
    r = message["response"]
    if r.get("idea"):
        st.info(f"💡 **Ý tưởng:** {r['idea']}")
    if r.get("explanation"):
        st.markdown(r["explanation"])

    state = r.get("state", "pending")
    if state == "error":
        st.error(r.get("error", "Lỗi không xác định"))
    if state == "ready" and r.get("figure") is not None:
        st.plotly_chart(r["figure"], config=PLOTLY_RENDER_CONFIG)
        if r.get("post_chart_analysis"):
            st.markdown(r["post_chart_analysis"])
        return

    rid = r.get("msg_id")
    if not rid:
        return
    code_key = f"chart_code_{rid}"
    if code_key not in st.session_state:
        st.session_state[code_key] = r.get("code", "") or ""

    st.markdown("**📝 Mã Plotly (chỉnh sửa nếu cần, rồi bấm thực thi):**")
    st.text_area(
        "chart_code_editor",
        height=240,
        key=code_key,
        label_visibility="collapsed",
    )
    st.button(
        "✅ Thực thi biểu đồ",
        key=f"chart_exec_btn_{rid}",
        on_click=_execute_pending_chart_msg,
        args=(rid,),
        type="primary",
    )


def process_chat(llm_agent: AgentAI, llm: object, data: dict) -> None:
    """
    This function creates and processes the chat engine.

    Args:
        llm_agent: AgenteAI object.
        llm: llm langchain object.
        data: dictionary of dataframes
    """

    with st.chat_message("assistant"):
        st.write("Chào bạn! Mình sẵn sàng giúp bạn khám phá dữ liệu. Bắt đầu thôi?")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    main_df = _get_main_df(data)
    st.session_state["_chart_main_df"] = main_df

    # Show messages from chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "question" in message:
                st.markdown(message["question"])
            elif "response" in message:
                resp = message["response"]
                if isinstance(resp, str):
                    st.write(resp)
                elif isinstance(resp, go.Figure):
                    st.plotly_chart(resp, config=PLOTLY_RENDER_CONFIG)
                elif isinstance(resp, dict) and resp.get("chart_pipeline"):
                    _render_chart_pipeline_message(message, main_df)
                elif isinstance(resp, dict) and "table" in resp and isinstance(
                    resp["table"], pd.DataFrame
                ):
                    if resp.get("caption"):
                        st.markdown(resp["caption"])
                    st.dataframe(resp["table"])
                elif isinstance(resp, dict) and "figure" in resp:
                    st.plotly_chart(resp["figure"], config=PLOTLY_RENDER_CONFIG)
                    analysis_text = resp.get("analysis", "")
                    if analysis_text:
                        st.markdown(analysis_text)
                elif isinstance(resp, list) or isinstance(resp, tuple):
                    for i in range(len(resp)):
                        if isinstance(resp[i], go.Figure):
                            st.plotly_chart(resp[i], config=PLOTLY_RENDER_CONFIG)
                        else:
                            st.write(resp[i])
                elif isinstance(resp, dict):
                    for key_, value_ in resp.items():
                        if key_ in ("chart_pipeline", "table", "msg_id"):
                            continue
                        if isinstance(value_, go.Figure):
                            st.plotly_chart(value_, config=PLOTLY_RENDER_CONFIG)
                        elif isinstance(value_, dict) and "figure" in value_:
                            st.plotly_chart(
                                value_["figure"], config=PLOTLY_RENDER_CONFIG
                            )
                            if value_.get("analysis"):
                                st.markdown(value_["analysis"])
                        else:
                            st.write(value_)
                else:
                    st.write(resp)
            elif "error" in message:
                st.text(message["error"])

    # Display the expander with code
    _, col1, _ = st.columns([1, 8, 1])
    if st.session_state["last_code"]:
        with col1:
            with st.expander("Python code"):
                st.code(st.session_state["last_code"], language="python")
    elif st.session_state["response_error_var"] != "":
        with col1:
            with st.expander("Lỗi phản hồi"):
                st.error(st.session_state["response_error_var"])

    # Check if message in chat_input
    if user_question := st.chat_input(
        "Nhập câu hỏi để mình phân tích dữ liệu cho bạn..."
    ):
        # Reset session_state vars
        st.session_state["response_error_var"] = ""

        user_question = user_question.strip()
        st.session_state.messages.append({"role": "user", "question": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        effective_user_question = _rewrite_followup_question_for_analysis(
            user_question, st.session_state.messages
        )

        with st.chat_message("assistant"):
            # ── BƯỚC 1: Phân loại intent ──────────────────────────────────────
            print(f"\n{Fore.CYAN}{'='*60}{Fore.RESET}")
            print(f"{Fore.CYAN}[STEP 1] PHÂN LOẠI INTENT{Fore.RESET}")
            print(f"{Fore.WHITE}  Câu hỏi: {user_question!r}{Fore.RESET}")
            if effective_user_question != user_question:
                print(
                    f"{Fore.WHITE}  rewritten_question={effective_user_question!r}{Fore.RESET}"
                )

            # If the user message is not asking for visualization/data plots,
            # answer normally to avoid slow "generate+execute code" flow.
            should_short_circuit, hint_response = should_answer_normally(
                effective_user_question,
                _get_classifier_llm(llm),
                st.session_state.messages,
            )

            print(
                f"{Fore.WHITE}  short_circuit={should_short_circuit} | hint={hint_response!r:.60}{Fore.RESET}"
            )

            if should_short_circuit:
                if hint_response == "__DATASET_OVERLAY__":
                    print(
                        f"{Fore.YELLOW}  → Trả lời nhanh: DATASET_OVERVIEW{Fore.RESET}"
                    )
                    response = answer_dataset_overview(_get_main_df(data))
                    _stream_text(response)
                elif hint_response == "__RAG_CONTEXT__":
                    print(f"{Fore.YELLOW}  → Trả lời nhanh: RAG_CONTEXT{Fore.RESET}")
                    try:
                        rag_context = build_rag_context(
                            effective_user_question, top_k=3, min_similarity=0.2
                        )
                        rag_section = rag_context if rag_context else ""
                        rag_prompt = f"""You are an expert on the Vietnam housing dataset.
The user is asking about column meaning, definitions, or broader dataset context.
Answer concisely and accurately in Vietnamese only.
Do NOT generate code; reply in plain language only.

{rag_section}

User question:
{effective_user_question}"""
                        response = _stream_llm_response(llm, rag_prompt)
                        if not rag_context:
                            print(
                                f"{Fore.YELLOW}  ⚠ RAG không tìm thấy doc liên quan — LLM tự trả lời{Fore.RESET}"
                            )
                        else:
                            print(
                                f"{Fore.GREEN}  ✓ RAG context injected ({len(rag_context)} chars){Fore.RESET}"
                            )
                    except Exception as e:
                        response = "Mình chưa thể tìm thấy thông tin ngữ cảnh. Bạn thử hỏi lại nhé."
                        st.markdown(response)
                        print(f"{Fore.RED}  ✗ RAG context FAILED: {e}{Fore.RESET}")
                elif hint_response == "__DIRECT_LLM_CHAT__":
                    print(
                        f"{Fore.YELLOW}  → Trả lời nhanh: DIRECT_LLM_CHAT{Fore.RESET}"
                    )
                    # Keep the instruction text in English; force Vietnamese output.
                    chat_prompt = f"""
You are a helpful data assistant for Vietnamese users.
Do NOT generate any code and do NOT request the system to draw plots.
Answer in Vietnamese only.
User message: {effective_user_question}
"""
                    try:
                        response = _stream_llm_response(llm, chat_prompt)
                        print(
                            f"{Fore.GREEN}  ✓ LLM direct chat OK ({len(response)} chars){Fore.RESET}"
                        )
                    except Exception:
                        response = "Mình chưa thể trả lời ngay lúc này. Bạn thử hỏi lại theo cách khác nhé."
                        st.markdown(response)
                        print(f"{Fore.RED}  ✗ LLM direct chat FAILED{Fore.RESET}")
                else:
                    print(f"{Fore.YELLOW}  → Trả lời nhanh: GREETING/HELP{Fore.RESET}")
                    response = hint_response
                    _stream_text(response)

                st.session_state["last_code"] = ""
                st.session_state["context_code_var"] = False
                st.session_state.messages.append(
                    {"role": "assistant", "response": response}
                )
                # Response was already rendered via streaming above
                print(f"{Fore.CYAN}{'='*60}{Fore.RESET}\n")
                st.rerun()

            # ── BƯỚC 2: JSON intent (chatbot.py-style) — một vòng LLM, không Agent.retry ──
            print(f"\n{Fore.CYAN}[STEP 2] JSON INTENT PROMPT (single LLM){Fore.RESET}")
            st.session_state["_chart_main_df"] = _get_main_df(data)

            prompt = build_json_intent_prompt(
                st.session_state.messages, effective_user_question, data, llm
            )
            print(f"{Fore.WHITE}  Prompt built ({len(prompt)} chars){Fore.RESET}")

            raw_text = ""
            with st.spinner("Đang phân tích (sinh JSON)..."):
                try:
                    resp_obj = llm.invoke(prompt)
                    raw_text = (
                        resp_obj.content
                        if hasattr(resp_obj, "content")
                        else str(resp_obj)
                    )
                except Exception as e:
                    print(f"{Fore.RED}  ✗ LLM invoke FAILED: {e}{Fore.RESET}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "response": "Mình không gọi được model lúc này. Bạn thử lại sau nhé.",
                        }
                    )
                    st.session_state["last_code"] = ""
                    print(f"{Fore.CYAN}{'='*60}{Fore.RESET}\n")
                    st.rerun()

            parsed = extract_json_object(raw_text)
            if not parsed:
                repair_prompt = (
                    prompt
                    + "\n\nLần trước bạn không trả về JSON hợp lệ. "
                    "Trả về DUY NHẤT một JSON object đúng schema, không markdown, không giải thích ngoài JSON."
                )
                try:
                    resp2 = llm.invoke(repair_prompt)
                    raw2 = (
                        resp2.content
                        if hasattr(resp2, "content")
                        else str(resp2)
                    )
                    parsed = extract_json_object(raw2)
                except Exception:
                    parsed = None

            if not parsed:
                print(f"{Fore.RED}  ✗ JSON parse FAILED{Fore.RESET}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "response": (
                            "Mình không đọc được phản hồi có cấu trúc từ model. "
                            "Bạn thử hỏi lại hoặc diễn đạt rõ hơn nhé."
                        ),
                    }
                )
                st.session_state["last_code"] = ""
                print(f"{Fore.CYAN}{'='*60}{Fore.RESET}\n")
                st.rerun()

            payload = normalize_intent_payload(parsed)
            intent = payload["intent"]
            st.session_state["last_code"] = payload["code"] or ""
            st.session_state["context_code_var"] = True
            print(f"\n{Fore.CYAN}[STEP 3] INTENT={intent}{Fore.RESET}")

            if intent == "TEXT":
                tool_insight_text: str | None = None
                response_text = payload.get("explanation", "").strip() or (
                    payload.get("idea", "").strip()
                    or "Mình chưa có nội dung trả lời từ model."
                )
                if ENABLE_TOOL_INSIGHT:
                    print(f"\n{Fore.CYAN}[STEP 4] TOOL INSIGHT (TEXT){Fore.RESET}")
                    with st.spinner("🔧 Đang gọi tool phân tích số liệu..."):
                        try:
                            tool_insight_text = run_tool_insight(
                                user_question=effective_user_question,
                                df=_get_main_df(data),
                                ollama_base_url=OLLAMA_BASE_URL,
                                model=OLLAMA_MODEL,
                                verbose=VERBOSE,
                            )
                        except Exception as e:
                            print(f"{Fore.YELLOW}  ⚠ tool insight: {e}{Fore.RESET}")
                            tool_insight_text = None
                    if tool_insight_text:
                        response_text = (
                            f"{response_text}\n\n---\n"
                            f"**🔧 Phân tích từ Tool Insight:**\n\n{tool_insight_text}"
                        )
                _stream_text(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "response": response_text}
                )

            elif intent == "TABLE":
                tq = (payload.get("table_query") or "").strip()
                if not tq:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "response": "Model trả về TABLE nhưng thiếu `table_query`. Bạn thử lại nhé.",
                        }
                    )
                else:
                    try:
                        df_by_name = {
                            f"DF_{i + 1}": dfi for i, dfi in enumerate(data.values())
                        }
                        result_df = eval_table_query(tq, df_by_name)
                        cap = (payload.get("explanation") or "").strip()
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "response": {"table": result_df, "caption": cap},
                            }
                        )
                    except Exception as e:
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "response": f"Lỗi khi tạo bảng: {type(e).__name__}: {e}",
                            }
                        )

            elif intent == "CHART":
                rid = new_chart_message_id()
                code = (payload.get("code") or "").strip()
                st.session_state[f"chart_code_{rid}"] = code
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "response": {
                            "chart_pipeline": True,
                            "state": "pending",
                            "msg_id": rid,
                            "idea": payload.get("idea", ""),
                            "explanation": payload.get("explanation", ""),
                            "code": code,
                            "_user_question": effective_user_question,
                        },
                    }
                )
                print(
                    f"{Fore.GREEN}  ✓ Pending chart message | msg_id={rid}{Fore.RESET}"
                )

            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "response": payload.get("explanation", "")
                        or "Không xử lý được intent.",
                    }
                )

            print(f"\n{Fore.CYAN}[STEP 5] LƯU SESSION & RERUN{Fore.RESET}")
            print(f"{Fore.CYAN}{'='*60}{Fore.RESET}\n")
            st.rerun()


def get_llm_agent(data: dict, llm: object) -> AgentAI:
    """
    The function creates an agent with the dataframes extracted from the files.

    Args:
        data: a dictionary with dataframes extracted from the sent data.
        llm: llm langchain object

    Returns: object AgentAI
    """

    agent = AgentAI(
        data=list(data.values()),
        llm=llm,
        max_attempts=MAX_ATTEMPTS,
        whitelist=WHITELIST_ENV,
        verbose=VERBOSE,
    )

    st.session_state["agent_var"] = agent

    return agent


def extract_dataframes(raw_files: list) -> dict:
    """
    This function extracts data from the loaded files and converts it into a dictionary.

    Args:
        raw_files: upload_File object.

    Returns:
        dfs: dictionary with dataframes.
    """

    dfs = {}

    for raw_file in raw_files:
        if raw_file.name.split(".")[1] == "csv":
            var = st.session_state["count_var"]
            csv_name = f"DF_{var}___{raw_file.name.split('.')[0]}"
            df = pd.read_csv(raw_file, encoding="utf-8", encoding_errors="replace")

            dfs[csv_name] = df
            st.session_state["count_var"] += 1

        elif (raw_file.name.split(".")[1] == "xlsx") or (
            raw_file.name.split(".")[1] == "xls"
        ):
            var = st.session_state["count_var"]
            # Read the Excel file
            xls = pd.ExcelFile(raw_file)

            # Iterate through each sheet in the Excel file and store them into dataframes
            for index, sheet_name in enumerate(xls.sheet_names):
                var = st.session_state["count_var"]
                dfs[f"DF_{var}___Sheet-{index}__{sheet_name}"] = pd.read_excel(
                    raw_file, sheet_name=sheet_name
                )
                st.session_state["count_var"] += 1

    return dfs


@st.cache_resource
def _warmup_rag_resources() -> None:
    """Prime embedding model + Chroma so the first user RAG query is faster."""
    warmup_rag_embedder()


def main() -> None:
    """
    Main function as entry point for the script.
    """

    process_styles()

    # Header title
    st.markdown("####")
    header = st.container(border=False)
    header.markdown("####")
    header.header("Chatbot bất động sản", divider="violet")
    header.markdown("####")
    header.markdown("##### Trợ lý AI hỗ trợ phân tích dữ liệu bất động sản")
    st.markdown("##")

    # Always use the single hardcoded dataset (no file upload).
    data_df = load_main_dataset()
    data = {"DF_1___vietnam_housing_dataset_cleaned": data_df}

    _warmup_rag_resources()

    with st.sidebar:
        st.session_state["sample_var"] = N_SAMPLES

        # Button to delete chat history
        side_container_2 = st.sidebar.container(border=True)
        side_container_2.button(
            "🗑️ &nbsp;&nbsp;Xóa hội thoại", on_click=clear_chat_history
        )

        # Temperature only (hardcoded dataset + Ollama model).
        side_container_3 = st.sidebar.container(border=True)
        llm_temp = side_container_3.slider(
            "🌡️ &nbsp;Temperature", 0.0, 1.0, 0.0, key="temperature"
        )

    # Always use Ollama local model with hardcoded model name.
    llm = _get_or_create_chat_ollama(OLLAMA_MODEL, OLLAMA_BASE_URL, llm_temp)
    # Reuse one AgentAI across Streamlit reruns; only refresh LLM + data (e.g. temperature slider).
    agent = st.session_state.get("agent_var")
    if agent is None:
        agent = get_llm_agent(data, llm)
    else:
        agent.llm = llm
        agent.data = list(data.values())
    process_chat(agent, llm, data)


if __name__ == "__main__":
    main()
