# App
import os
from pathlib import Path
import json
import re
import traceback
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from colorama import Fore
from agent import AgentAI
from prompt import process_prompt
from styles import process_styles

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

    # Show messages from chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "question" in message:
                st.markdown(message["question"])
            elif "response" in message:
                if isinstance(message["response"], str):
                    st.write(message["response"])
                elif isinstance(message["response"], Figure):
                    st.pyplot(message["response"])
                elif (
                    isinstance(message["response"], dict)
                    and "figure" in message["response"]
                ):
                    # Dict từ plot agent: {"figure": Figure, "analysis": str}
                    fig_obj = message["response"]["figure"]
                    if isinstance(fig_obj, Figure):
                        st.pyplot(fig_obj)
                    analysis_text = message["response"].get("analysis", "")
                    if analysis_text:
                        st.markdown(analysis_text)
                elif isinstance(message["response"], list) or isinstance(
                    message["response"], tuple
                ):
                    for i in range(len(message["response"])):
                        if isinstance(message["response"][i], Figure):
                            st.pyplot(message["response"][i])
                        else:
                            st.write(message["response"][i])
                elif isinstance(message["response"], dict):
                    for key_, value_ in message["response"].items():
                        if isinstance(message["response"][key_], Figure):
                            st.pyplot(message["response"][key_])
                        else:
                            st.write(message["response"][key_])
                else:
                    st.write(message["response"])
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

            # ── BƯỚC 2: Build prompt + Agent chat ────────────────────────────
            print(f"\n{Fore.CYAN}[STEP 2] BUILD PROMPT & AGENT CHAT{Fore.RESET}")
            with st.spinner("Đang phân tích..."):
                try:
                    prompt = process_prompt(
                        st.session_state.messages, effective_user_question, data, llm
                    )
                    print(
                        f"{Fore.WHITE}  Prompt built ({len(prompt)} chars){Fore.RESET}"
                    )
                    print(
                        f"{Fore.CYAN}[STEP 3] LLM INVOKE + CODE EXECUTION{Fore.RESET}"
                    )
                    response = llm_agent.chat(prompt)
                    print(
                        f"{Fore.GREEN}  ✓ Agent response type: {type(response).__name__}{Fore.RESET}"
                    )
                except Exception as e:
                    exception_name = type(e).__name__
                    track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
                    response = f"EXCEPTION ERROR: {exception_name}: {track_line}"
                    print(
                        f"{Fore.RED}  ✗ Agent FAILED: {exception_name} {track_line}{Fore.RESET}"
                    )
                    # raise sys.exc_info()[0]

                st.session_state["last_code"] = llm_agent.get_last_code()
                # Code in context
                st.session_state["context_code_var"] = True

                # Convert output with numbers (int and float) to string
                response = (
                    str(response)
                    if isinstance(response, int) or isinstance(response, float)
                    else response
                )
                # Convert nd.array output to string
                response = (
                    str(response.item())
                    if isinstance(response, np.ndarray)
                    else response
                )

                # ── BƯỚC 4: Xử lý kiểu kết quả trả về ───────────────────────
                print(f"\n{Fore.CYAN}[STEP 4] XỬ LÝ KẾT QUẢ AGENT{Fore.RESET}")
                print(
                    f"{Fore.LIGHTYELLOW_EX}CODE RESPONSE:{Fore.RESET}", type(response)
                )
                if isinstance(response, dict):
                    print(
                        f"{Fore.WHITE}  response keys: {list(response.keys())}{Fore.RESET}"
                    )
                elif isinstance(response, str):
                    preview = response[:80].replace("\n", " ")
                    print(f"{Fore.WHITE}  response preview: {preview!r}{Fore.RESET}")

                # ── Tool-Calling Insight ─────────────────────────────────────
                # Sau khi có figure/response, gọi LLM với tool-calling để tính
                # số liệu thực từ DataFrame rồi sinh insight câu hỏi người dùng.
                fig_for_insight: Figure | None = None
                agent_analysis: str | None = None

                if ENABLE_TOOL_INSIGHT:
                    if isinstance(response, dict):
                        cand = response.get("figure")
                        if isinstance(cand, Figure):
                            fig_for_insight = cand
                        agent_analysis = response.get("analysis")
                    elif isinstance(response, Figure):
                        fig_for_insight = response

                tool_insight_text: str | None = None
                # Single tool-insight pass per request (after agent) — avoids duplicate Ollama rounds.
                if ENABLE_TOOL_INSIGHT and (
                    fig_for_insight is not None or isinstance(response, str)
                ):
                    print(f"\n{Fore.CYAN}[STEP 5] TOOL-CALLING INSIGHT{Fore.RESET}")
                    print(
                        f"{Fore.WHITE}  has_figure={fig_for_insight is not None} | is_str={isinstance(response, str)}{Fore.RESET}"
                    )
                    main_df = _get_main_df(data)
                    with st.spinner("🔧 Đang gọi tool phân tích số liệu..."):
                        tool_insight_text = run_tool_insight(
                            user_question=effective_user_question,
                            df=main_df,
                            ollama_base_url=OLLAMA_BASE_URL,
                            model=OLLAMA_MODEL,
                            verbose=VERBOSE,
                        )
                    if tool_insight_text:
                        preview = tool_insight_text[:100].replace("\n", " ")
                        print(
                            f"{Fore.GREEN}  ✓ Tool insight OK ({len(tool_insight_text)} chars): {preview!r}{Fore.RESET}"
                        )
                    else:
                        print(f"{Fore.YELLOW}  ⚠ Tool insight trống{Fore.RESET}")
                else:
                    print(
                        f"\n{Fore.CYAN}[STEP 5] TOOL-CALLING INSIGHT — bỏ qua{Fore.RESET}"
                    )
                    print(
                        f"{Fore.YELLOW}  ENABLE={ENABLE_TOOL_INSIGHT} | figure={fig_for_insight is not None} | str={isinstance(response, str)}{Fore.RESET}"
                    )
                # ─────────────────────────────────────────────────────────────

                # ── BƯỚC 6: Lưu session ───────────────────────────────────────
                print(f"\n{Fore.CYAN}[STEP 6] LƯU SESSION & RERUN{Fore.RESET}")
                try:
                    if isinstance(response, str):
                        # Handles exceptions
                        if response.split()[0] == "EXCEPTION":
                            print(
                                f"{Fore.RED}  ✗ EXCEPTION trong response → hiển thị thông báo lỗi{Fore.RESET}"
                            )
                            st.session_state["response_error_var"] = response
                            if tool_insight_text:
                                response = (
                                    "Mình gặp lỗi khi dựng biểu đồ tự động, nhưng vẫn rút được kết luận từ tools:\n\n"
                                    f"**🔧 Phân tích từ Tool Insight:**\n\n{tool_insight_text}"
                                )
                            else:
                                response = (
                                    "Xin lỗi, mình không thể đáp ứng yêu cầu của bạn. "
                                    "Bạn hãy xóa lịch sử hội thoại và thử lại nhé."
                                )
                            # Hide last code
                            st.session_state["last_code"] = None
                            _stream_text(response)
                        else:
                            if tool_insight_text:
                                # Stream base response first, then stream tool insight
                                _stream_text(response)
                                st.markdown("\n\n---")
                                st.markdown("**🔧 Phân tích từ Tool Insight:**\n")
                                _stream_text(tool_insight_text)
                                response = (
                                    f"{response}\n\n---\n"
                                    f"**🔧 Phân tích từ Tool Insight:**\n\n"
                                    f"{tool_insight_text}"
                                )
                            else:
                                _stream_text(response)
                            print(f"{Fore.GREEN}  ✓ Lưu response dạng str{Fore.RESET}")
                        st.session_state.messages.append(
                            {"role": "assistant", "response": response}
                        )

                    elif isinstance(response, dict) and "figure" in response:
                        # Chuẩn hoá response dict: giữ figure + ghép insight
                        combined_insight = ""
                        if agent_analysis:
                            combined_insight += agent_analysis
                        if tool_insight_text:
                            separator = "\n\n---\n" if combined_insight else ""
                            combined_insight += (
                                f"{separator}**🔧 Phân tích từ Tool Insight:**\n\n"
                                f"{tool_insight_text}"
                            )
                        if combined_insight:
                            response["analysis"] = combined_insight
                        fig_out = response["figure"]
                        if isinstance(fig_out, Figure):
                            st.pyplot(fig_out)
                        if combined_insight:
                            _stream_text(combined_insight)
                        print(
                            f"{Fore.GREEN}  ✓ Lưu response dạng dict+figure | has_analysis={bool(combined_insight)}{Fore.RESET}"
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "response": response}
                        )
                    elif isinstance(response, Figure):
                        payload = {"figure": response}
                        if tool_insight_text:
                            payload["analysis"] = (
                                "**🔧 Phân tích từ Tool Insight:**\n\n"
                                f"{tool_insight_text}"
                            )
                        st.pyplot(response)
                        if tool_insight_text:
                            st.markdown("**🔧 Phân tích từ Tool Insight:**\n")
                            _stream_text(tool_insight_text)
                        print(
                            f"{Fore.GREEN}  ✓ Lưu response dạng figure | has_analysis={bool(tool_insight_text)}{Fore.RESET}"
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "response": payload}
                        )

                    else:
                        print(
                            f"{Fore.GREEN}  ✓ Lưu response dạng {type(response).__name__}{Fore.RESET}"
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "response": response}
                        )
                except Exception as e:
                    exception_name = type(e).__name__
                    track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
                    message_ = "Lỗi khi hiển thị kết quả:"
                    print(
                        f"{Fore.RED}  ✗ Lỗi lưu session: {exception_name} {track_line}{Fore.RESET}"
                    )
                    st.error(f"{message_}  <{exception_name}: {track_line}>")
                    # raise sys.exc_info()[0]

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
