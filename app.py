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

# ── Load .env (nếu có) ───────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv chưa cài — bỏ qua, dùng env system

# ── LiteLLM wrapper ──────────────────────────────────────────────────────────
from litellm import completion as litellm_completion

from colorama import Fore
from agent import AgentAI
from prompt import process_prompt
from styles import process_styles

from tool_executor import run_tool_insight
from rag_docs_manager import build_rag_context, warmup_rag_embedder
from conversation_logger import list_conversations, load_conversation, save_conversation

# ── Parameters ───────────────────────────────────────────────────────────────
WHITELIST_ENV = ["json", "statsmodels", "scipy", "datetime"]
N_SAMPLES = 5        # Number of samples sent to the prompt in non-private mode
MAX_ATTEMPTS = 10    # Number of calls to LLM in case of execution error
VERBOSE = False      # Show final prompt and errors

DATASET_PATH = (
    Path(__file__).resolve().parent / "datasets" / "vietnam_housing_dataset_cleaned.csv"
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Tool-calling insight
ENABLE_TOOL_INSIGHT = True

# ── Provider configuration ────────────────────────────────────────────────────
PROVIDER_OPTIONS = ["Ollama (local)", "Gemini", "OpenAI"]

# Default model suggestions per provider
DEFAULT_MODELS = {
    "Ollama (local)": "qwen2.5:7b",
    "Gemini": "gemini/gemini-2.0-flash",
    "OpenAI": "gpt-4o-mini",
}

# ── LiteLLM wrapper class (tương thích interface cũ dùng .invoke / .stream) ──

class LiteLLMWrapper:
    """
    Wrapper bọc litellm.completion() thành interface tương thích
    với ChatOllama (có .invoke(), .stream(), .temperature).
    """

    def __init__(self, model: str, temperature: float = 0.0, api_base: str | None = None):
        self.model = model
        self.temperature = temperature
        self.api_base = api_base  # dùng cho Ollama

    def _build_kwargs(self) -> dict:
        kwargs: dict = {
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        return kwargs

    def invoke(self, prompt: str) -> "LiteLLMResponse":
        kwargs = self._build_kwargs()
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        response = litellm_completion(**kwargs)
        return LiteLLMResponse(response.choices[0].message.content or "")

    def stream(self, prompt: str):
        kwargs = self._build_kwargs()
        kwargs["messages"] = [{"role": "user", "content": prompt}]
        kwargs["stream"] = True
        for chunk in litellm_completion(**kwargs):
            delta = chunk.choices[0].delta
            token = delta.content or ""
            if token:
                yield LiteLLMResponse(token)


class LiteLLMResponse:
    """Giả lập AIMessage của LangChain để code cũ dùng .content không cần sửa."""
    def __init__(self, content: str):
        self.content = content

    def __str__(self) -> str:
        return self.content


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _check_ollama_connection(base_url: str) -> bool:
    """Kiểm tra Ollama có đang chạy không."""
    import urllib.request
    try:
        req = urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        return req.status == 200
    except Exception:
        return False


def _list_ollama_models(base_url: str) -> list[str]:
    """Trả về danh sách model đang có trong Ollama local."""
    import urllib.request
    import json as _json
    try:
        req = urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        data = _json.loads(req.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ── LLM factory ──────────────────────────────────────────────────────────────

def _build_llm(provider: str, model: str, temperature: float) -> LiteLLMWrapper:
    """Tạo LiteLLMWrapper phù hợp với provider được chọn."""
    if provider == "Ollama (local)":
        # litellm dùng prefix "ollama/" cho Ollama
        litellm_model = model if model.startswith("ollama/") else f"ollama/{model}"
        return LiteLLMWrapper(
            model=litellm_model,
            temperature=temperature,
            api_base=OLLAMA_BASE_URL,
        )
    elif provider == "Gemini":
        # litellm dùng prefix "gemini/" cho Google Gemini
        litellm_model = model if model.startswith("gemini/") else f"gemini/{model}"
        return LiteLLMWrapper(model=litellm_model, temperature=temperature)
    elif provider == "OpenAI":
        return LiteLLMWrapper(model=model, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _get_or_create_llm(provider: str, model: str, temperature: float) -> LiteLLMWrapper:
    """
    Reuse wrapper nếu provider/model không đổi; chỉ cập nhật temperature.
    """
    key = "_cached_litellm_wrapper"
    llm: LiteLLMWrapper | None = st.session_state.get(key)
    if llm is not None:
        same = (
            getattr(llm, "_provider", None) == provider
            and getattr(llm, "_model_raw", None) == model
        )
        if same:
            llm.temperature = temperature
            return llm

    llm = _build_llm(provider, model, temperature)
    llm._provider = provider   # type: ignore[attr-defined]
    llm._model_raw = model     # type: ignore[attr-defined]
    st.session_state[key] = llm
    return llm


def _get_classifier_llm(main_llm: LiteLLMWrapper) -> LiteLLMWrapper:
    """
    Router model riêng (nếu OLLAMA_ROUTER_MODEL được set) cho classify_intent_llm.
    Nếu không, dùng main_llm.
    """
    router_model = os.getenv("OLLAMA_ROUTER_MODEL", "").strip()
    if not router_model:
        return main_llm

    provider = getattr(main_llm, "_provider", "Ollama (local)")
    if provider != "Ollama (local)":
        return main_llm  # router model chỉ áp dụng khi dùng Ollama

    key = "_cached_ollama_router_llm"
    llm = st.session_state.get(key)
    if llm is not None and getattr(llm, "_model_raw", None) == router_model:
        llm.temperature = 0.0
        return llm

    llm = LiteLLMWrapper(
        model=f"ollama/{router_model}",
        temperature=0.0,
        api_base=OLLAMA_BASE_URL,
    )
    llm._provider = "Ollama (local)"  # type: ignore[attr-defined]
    llm._model_raw = router_model     # type: ignore[attr-defined]
    st.session_state[key] = llm
    return llm


# ── Dataset ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_main_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH, encoding="utf-8", encoding_errors="replace")


def _get_main_df(data: dict) -> pd.DataFrame:
    return next(iter(data.values()))


# ── Intent helpers ────────────────────────────────────────────────────────────

def _is_greeting(text: str) -> bool:
    t = text.strip().lower()
    return bool(
        re.search(
            r"(^(xin chào|chào|hello|hi|hey)\b)|(\b(xin chào|chào bạn|hello|hi|hey)\b)",
            t,
        )
    )


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


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
        "cột ", "cot ", "column", "price", "giá", "area", "diện tích",
        "bedroom", "bathroom", "province", "district",
    ]
    return any(marker in q for marker in followup_markers) or len(q.split()) <= 3


def _rewrite_followup_question_for_analysis(
    user_question: str, messages: list[dict] | None
) -> str:
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
    q = user_question.strip()

    if _is_dataset_followup_for_analysis(q, messages):
        return (False, "")

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
        "dataset", "tập dữ liệu", "các cột", "có những cột", "dữ liệu gồm",
        "cột nào", "features", "danh sách cột", "liệt kê cột", "tên các cột",
        "schema", "dtypes", "kiểu dữ liệu", "bao nhiêu dòng", "số dòng",
        "số cột", "kích thước", "shape của",
    ]
    context_keywords = [
        "có ý nghĩa gì", "ý nghĩa", "nghĩa là gì", "là gì", "giải thích",
        "câu chuyện", "phía sau", "nguồn gốc", "thuộc tính",
        "trường ", "field", "column", "describe", "what is", "what does", "explain",
    ]
    help_keywords = [
        "bạn làm gì", "bạn có thể", "giúp", "hướng dẫn", "cách dùng", "tính năng",
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

    if _contains_any(q, context_keywords):
        return (True, "__RAG_CONTEXT__")

    visualization_keywords = [
        "vẽ", "biểu đồ", "đồ thị", "plot", "chart", "scatter", "histogram",
        "hist", "heatmap", "map", "bản đồ", "bar", "line", "pie", "box",
        "violin", "treemap", "sunburst", "ohlc", "candlestick", "surface",
        "nhiều nhất", "ít nhất", "cao nhất", "thấp nhất", "top ", "xếp hạng",
        "so sánh", "nhiều hơn", "ít hơn", "tỉnh nào", "thành nào", "cái nào",
        "loại nào", "phân phối", "thống kê", "ranking", "ảnh hưởng", "tác động",
        "nhà to hơn", "đồ họa", "thống kê mô tả", "phân tích dữ liệu",
        "xu hướng giá", "theo tỉnh", "theo thành phố", "theo loại nhà",
        "thống kê mô tả", "bảng thống kê", "tạo bảng", "describe",
        "tổng quan dữ liệu", "summary", "tóm tắt dữ liệu",
    ]
    if _contains_any(q, visualization_keywords):
        return (False, "")

    if _contains_any(q, dataset_keywords):
        return (True, "__DATASET_OVERLAY__")

    analysis_stats_keywords = [
        "trung bình", "trung vi", "trung bình giá", "tbc ", "độ lệch chuẩn",
        "phương sai", "variance", "tương quan", "covariance", "hồi quy",
        "regression", "ngoại lệ", "outlier", "quantile", "phân vị", "tứ phân vị",
        "min ", " max ", "nhỏ nhất", "lớn nhất", "tổng số", "đếm số",
        "bao nhiêu căn", "bao nhiêu nhà", "chênh lệch", "khác biệt", "tỉ lệ",
        "phần trăm", "phân trăm", "so sánh giá", "tb giá", "giá tb",
        "giá trung bình", "giá trung vi", "số căn", "số bản ghi",
        "số lượng nhà", "đếm theo", "tần suất", "frequency",
    ]
    if _contains_any(q, analysis_stats_keywords):
        return (False, "")

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
    Dùng LLM phân loại intent.
    Trả về một trong: greeting | dataset_info | context_query |
                      visualization_request | direct_chat
    """
    intent_labels = [
        "greeting", "dataset_info", "context_query",
        "visualization_request", "direct_chat",
    ]

    classifier_prompt = f"""Intent classifier — Vietnamese real-estate data chatbot.
Return ONE JSON: {{"intent": "<label>"}}. No extra text.

greeting           : pure hello, no data task.
dataset_info       : wants column names / schema / shape — not column meanings.
context_query      : asks meaning of a SPECIFIC column (e.g. "cột X là gì?"). NOT for analytical questions.
visualization_request : chart, table, stats, OR analytical hypothesis about data
                     (e.g. "Liệu nhà to thì giá cao?", "Tỉnh nào đắt nhất?").
                     Prefer this over context_query when in doubt.
direct_chat        : everything else.

Message: {user_question}""".strip()

    prev_temp = getattr(llm, "temperature", None)
    try:
        if hasattr(llm, "temperature"):
            llm.temperature = 0.0
        resp_obj = llm.invoke(classifier_prompt)
        content = resp_obj.content if hasattr(resp_obj, "content") else str(resp_obj)

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


# ── Streaming helpers ─────────────────────────────────────────────────────────

def _stream_llm_response(llm: object, prompt: str) -> str:
    full_text = ""
    try:
        placeholder = st.empty()
        for chunk in llm.stream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_text += token
            placeholder.markdown(full_text + "▌")
        placeholder.markdown(full_text)
    except (AttributeError, NotImplementedError):
        resp_obj = llm.invoke(prompt)
        full_text = resp_obj.content if hasattr(resp_obj, "content") else str(resp_obj)
        st.markdown(full_text)
    return full_text


def _stream_text(text: str) -> None:
    st.markdown(text)


# ── Session state ─────────────────────────────────────────────────────────────

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

# ── Approval workflow state ───────────────────────────────────────────────────
# pending_approval: dict with keys "code", "prompt", "question"  — or None
if "pending_approval" not in st.session_state:
    st.session_state["pending_approval"] = None
# approval_decision: "accepted" | "rejected" | None
if "approval_decision" not in st.session_state:
    st.session_state["approval_decision"] = None
# ace_editor_version: bumped each time code is updated via edit-instruction,
# used as part of the st_ace key so the editor remounts with fresh content.
if "ace_editor_version" not in st.session_state:
    st.session_state["ace_editor_version"] = 0


def clear_chat_history() -> None:
    messages = st.session_state.get("messages", [])
    latest_code = st.session_state.get("last_code") or None
    if messages:
        saved_path = save_conversation(messages, latest_code)
        if saved_path:
            print(f"{Fore.LIGHTGREEN_EX}[Archive] Đã lưu hội thoại → {saved_path}{Fore.RESET}")
        else:
            print(f"{Fore.YELLOW}[Archive] Không lưu được hội thoại{Fore.RESET}")
    st.session_state["last_code"] = ""
    st.session_state["context_code_var"] = False
    st.session_state["response_error_var"] = ""
    st.session_state["messages"] = []
    st.session_state["pending_approval"] = None
    st.session_state["approval_decision"] = None
    agent = st.session_state.get("agent_var")
    if agent is not None:
        agent.chat_stop()
    st.session_state["agent_var"] = None


# ── Archive rendering ─────────────────────────────────────────────────────────

def _render_archived_message(message: dict) -> None:
    role = message.get("role", "assistant")
    msg_type = message.get("type", "text")
    content = message.get("content", "")
    code = message.get("code")

    with st.chat_message(role):
        if role == "user":
            st.markdown(content)
            return
        if msg_type == "chart_image":
            b64 = message.get("chart_image_b64")
            if b64:
                import base64
                png_bytes = base64.b64decode(b64)
                st.image(png_bytes)
            if content:
                st.markdown(content)
        else:
            if content:
                st.markdown(content)
        if code:
            with st.expander("Python code"):
                st.code(code, language="python")


def _render_archive_view(filepath: str) -> None:
    try:
        log = load_conversation(filepath)
    except Exception as e:
        st.error(f"Không thể đọc log: {e}")
        return
    saved_at = log.get("saved_at", "")
    st.caption(f"📁 Lưu lúc: {saved_at}  •  Chỉ xem, không thể chat")
    st.divider()
    for message in log.get("messages", []):
        _render_archived_message(message)


# ── Agent factory ─────────────────────────────────────────────────────────────

def get_llm_agent(data: dict, llm: object) -> AgentAI:
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
    dfs = {}
    for raw_file in raw_files:
        ext = raw_file.name.split(".")[1]
        if ext == "csv":
            var = st.session_state["count_var"]
            csv_name = f"DF_{var}___{raw_file.name.split('.')[0]}"
            df = pd.read_csv(raw_file, encoding="utf-8", encoding_errors="replace")
            dfs[csv_name] = df
            st.session_state["count_var"] += 1
        elif ext in ("xlsx", "xls"):
            xls = pd.ExcelFile(raw_file)
            for index, sheet_name in enumerate(xls.sheet_names):
                var = st.session_state["count_var"]
                dfs[f"DF_{var}___Sheet-{index}__{sheet_name}"] = pd.read_excel(
                    raw_file, sheet_name=sheet_name
                )
                st.session_state["count_var"] += 1
    return dfs


@st.cache_resource
def _warmup_rag_resources() -> None:
    warmup_rag_embedder()


# ── Chat processing ───────────────────────────────────────────────────────────

def _render_message(message: dict) -> None:
    """Render a single assistant/user message stored in session_state.messages."""
    with st.chat_message(message["role"]):
        if "question" in message:
            st.markdown(message["question"])
        elif "response" in message:
            resp = message["response"]
            if isinstance(resp, str):
                st.write(resp)
            elif isinstance(resp, Figure):
                st.pyplot(resp)
            elif isinstance(resp, dict) and "figure" in resp:
                fig_obj = resp["figure"]
                if isinstance(fig_obj, Figure):
                    st.pyplot(fig_obj)
                analysis_text = resp.get("analysis", "")
                if analysis_text:
                    st.markdown(analysis_text)
            elif isinstance(resp, (list, tuple)):
                for item in resp:
                    if isinstance(item, Figure):
                        st.pyplot(item)
                    else:
                        st.write(item)
            elif isinstance(resp, dict):
                for key_, value_ in resp.items():
                    if isinstance(value_, Figure):
                        st.pyplot(value_)
                    else:
                        st.write(value_)
            else:
                st.write(resp)
        elif "error" in message:
            st.text(message["error"])


def _build_edit_prompt(original_code: str, user_instruction: str) -> str:
    """
    Build a focused prompt that asks the LLM to apply the human's edit instruction
    to the existing code block, returning only the revised code.
    """
    return (
        "You are a Python code editor. The user wants to modify the code below.\n"
        "Apply ONLY the requested change. Keep everything else identical.\n"
        "Return ONE fenced python block and nothing else.\n\n"
        f"<current_code>\n```python\n{original_code}\n```\n</current_code>\n\n"
        f"<edit_instruction>{user_instruction}</edit_instruction>"
    )


def _render_approval_panel(llm_agent: AgentAI, llm: object, data: dict) -> None:
    """
    Display the pending-approval panel.

    Shows an editable code area with three action buttons:
    - Accept  → execute the (possibly edited) code → store result → clear pending → rerun
    - Reject  → clear pending → append rejection notice → rerun
    - Tạo lại code → dedicated chat input where the human types an edit instruction;
                     LLM applies it surgically to the current code → update pending → rerun
    """
    pending = st.session_state["pending_approval"]
    if pending is None:
        return

    st.divider()
    st.markdown("### 🔍 Duyệt code trước khi thực thi")
    st.caption(
        "AI đã sinh ra đoạn code bên dưới. Bạn có thể chỉnh sửa trực tiếp trong ô code, "
        "hoặc dùng ô **'Yêu cầu chỉnh sửa'** để ra lệnh cho AI sửa theo ý muốn."
    )

    # ── Editable code editor with Python syntax highlighting ──────────────────
    ace_version = st.session_state.get("ace_editor_version", 0)
    ace_key = f"approval_code_editor_v{ace_version}"
    try:
        from streamlit_ace import st_ace
        edited_code = st_ace(
            value=pending["code"],
            language="python",
            theme="monokai",
            font_size=13,
            tab_size=4,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            auto_update=True,
            height=350,
            key=ace_key,
        )
        if edited_code is None:
            edited_code = pending["code"]
    except ImportError:
        st.warning(
            "💡 Cài `streamlit-ace` để có syntax highlighting: "
            "`pip install streamlit-ace`"
        )
        edited_code = st.text_area(
            "Python code (có thể chỉnh sửa trực tiếp)",
            value=pending["code"],
            height=350,
            key=ace_key,
        )

    # ── Action buttons ────────────────────────────────────────────────────────
    col_accept, col_reject = st.columns([3, 1])
    with col_accept:
        accept_clicked = st.button(
            "✅ Chấp nhận & Thực thi",
            type="primary",
            use_container_width=True,
            key="btn_approve",
        )
    with col_reject:
        reject_clicked = st.button(
            "❌ Từ chối",
            type="secondary",
            use_container_width=True,
            key="btn_reject",
        )

    # ── Dedicated edit-instruction chat input ─────────────────────────────────
    st.markdown("---")
    st.markdown(
        "##### 🔄 Tạo lại code theo yêu cầu",
    )
    st.caption(
        "Nhap huong dan chinh sua — AI se ap dung dung thay doi do vao code hien tai.  \n"
        'Vi du: "doi nguong outlier tu 3 xuong 2 standard deviations", '
        '"dung mau xanh la cho cot gia", "them duong trung binh vao bieu do"'
    )

    edit_col, send_col = st.columns([5, 1])
    with edit_col:
        edit_instruction = st.text_input(
            label="Yêu cầu chỉnh sửa code",
            placeholder="Ví dụ: đổi outlier threshold từ 3 xuống 2 standard deviations...",
            label_visibility="collapsed",
            key="edit_instruction_input",
        )
    with send_col:
        send_edit = st.button(
            "Gửi ↵",
            type="secondary",
            use_container_width=True,
            key="btn_send_edit",
        )

    # ── Handlers ──────────────────────────────────────────────────────────────

    if reject_clicked:
        print(f"{Fore.YELLOW}[APPROVAL] Rejected by user{Fore.RESET}")
        st.session_state["pending_approval"] = None
        st.session_state["approval_decision"] = "rejected"
        st.session_state.messages.append(
            {"role": "assistant", "response": "⛔ Yêu cầu đã bị từ chối. Code sẽ không được thực thi."}
        )
        st.rerun()

    if send_edit and edit_instruction.strip():
        instruction = edit_instruction.strip()
        base_code = edited_code
        print(f"{Fore.CYAN}[APPROVAL] Edit instruction: {instruction!r}{Fore.RESET}")
        with st.spinner("✏️ Đang áp dụng chỉnh sửa..."):
            try:
                edit_prompt = _build_edit_prompt(base_code, instruction)
                raw = llm_agent._invoke_llm_blocking(edit_prompt)
                import re as _re
                match = _re.search(r"```python(.*?)```", raw, _re.DOTALL)
                new_code = match.group(1) if match else base_code
                print(f"{Fore.GREEN}  ✓ Edit applied ({len(new_code.splitlines())} lines){Fore.RESET}")
            except Exception as e:
                exception_name = type(e).__name__
                track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
                st.error(f"Lỗi khi áp dụng chỉnh sửa: {exception_name} {track_line}")
                st.rerun()
        st.session_state["pending_approval"] = {
            "code": new_code,
            "prompt": pending["prompt"],
            "question": pending["question"],
        }
        st.session_state["last_code"] = new_code
        st.session_state["ace_editor_version"] = (
            st.session_state.get("ace_editor_version", 0) + 1
        )
        st.rerun()

    if accept_clicked:
        print(f"{Fore.GREEN}[APPROVAL] Accepted by user — executing approved code...{Fore.RESET}")
        st.session_state["pending_approval"] = None
        st.session_state["approval_decision"] = "accepted"
        llm_agent.last_code = edited_code

        effective_user_question = pending["question"]
        with st.spinner("⚙️ Đang thực thi code đã được duyệt..."):
            try:
                response = llm_agent.chat_execute(edited_code)
                print(f"{Fore.GREEN}  ✓ Execution OK — type: {type(response).__name__}{Fore.RESET}")
            except Exception as e:
                exception_name = type(e).__name__
                track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
                response = f"EXCEPTION ERROR: {exception_name}: {track_line}"
                print(f"{Fore.RED}  ✗ Execution FAILED: {exception_name} {track_line}{Fore.RESET}")

        st.session_state["last_code"] = llm_agent.get_last_code()
        st.session_state["context_code_var"] = True

        response = str(response) if isinstance(response, (int, float)) else response
        response = str(response.item()) if isinstance(response, np.ndarray) else response

        # ── Tool-Calling Insight ──────────────────────────────────────────────
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
        if ENABLE_TOOL_INSIGHT and (fig_for_insight is not None or isinstance(response, str)):
            print(f"{Fore.CYAN}[APPROVAL] Running tool insight...{Fore.RESET}")
            main_df = _get_main_df(data)
            _llm: LiteLLMWrapper = llm  # type: ignore[assignment]

            # ── PATCHED: dùng litellm_model + api_base thay vì ollama_base_url + model ──
            _litellm_model: str = getattr(_llm, "model", "ollama/qwen2.5:7b")
            _api_base: str | None = getattr(_llm, "api_base", None)

            with st.spinner("🔧 Đang gọi tool phân tích số liệu..."):
                tool_insight_text = run_tool_insight(
                    user_question=effective_user_question,
                    df=main_df,
                    litellm_model=_litellm_model,
                    api_base=_api_base,
                    temperature=getattr(_llm, "temperature", 0.1),
                    verbose=VERBOSE,
                )

        # ── Persist result ────────────────────────────────────────────────────
        try:
            if isinstance(response, str):
                if response.split() and response.split()[0] == "EXCEPTION":
                    st.session_state["response_error_var"] = response
                    if tool_insight_text:
                        response = (
                            "Mình gặp lỗi khi dựng biểu đồ tự động, "
                            "nhưng vẫn rút được kết luận từ tools:\n\n"
                            f"**🔧 Phân tích:**\n\n{tool_insight_text}"
                        )
                    else:
                        response = (
                            "Xin lỗi, mình không thể đáp ứng yêu cầu của bạn. "
                            "Bạn hãy xóa lịch sử hội thoại và thử lại nhé."
                        )
                    st.session_state["last_code"] = None
                else:
                    if tool_insight_text:
                        response = (
                            f"{response}\n\n---\n**🔧 Phân tích:**\n\n{tool_insight_text}"
                        )
                st.session_state.messages.append({"role": "assistant", "response": response})

            elif isinstance(response, dict) and "figure" in response:
                combined_insight = ""
                if agent_analysis:
                    combined_insight += agent_analysis
                if tool_insight_text:
                    sep = "\n\n---\n" if combined_insight else ""
                    combined_insight += f"{sep}**🔧 Phân tích:**\n\n{tool_insight_text}"
                if combined_insight:
                    response["analysis"] = combined_insight
                st.session_state.messages.append({"role": "assistant", "response": response})

            elif isinstance(response, Figure):
                payload: dict = {"figure": response}
                if tool_insight_text:
                    payload["analysis"] = "**🔧 Phân tích:**\n\n" + tool_insight_text
                st.session_state.messages.append({"role": "assistant", "response": payload})

            else:
                st.session_state.messages.append({"role": "assistant", "response": response})

        except Exception as e:
            exception_name = type(e).__name__
            track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
            st.error(f"Lỗi khi lưu kết quả: <{exception_name}: {track_line}>")

        st.rerun()


def process_chat(llm_agent: AgentAI, llm: object, data: dict) -> None:
    with st.chat_message("assistant"):
        st.write("Chào bạn! Mình sẵn sàng giúp bạn khám phá dữ liệu. Bắt đầu thôi?")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Render conversation history ───────────────────────────────────────────
    for message in st.session_state.messages:
        _render_message(message)

    # ── Code / error expander ─────────────────────────────────────────────────
    _, col1, _ = st.columns([1, 8, 1])
    if st.session_state["last_code"]:
        with col1:
            with st.expander("Python code"):
                st.code(st.session_state["last_code"], language="python")
    elif st.session_state["response_error_var"] != "":
        with col1:
            with st.expander("Lỗi phản hồi"):
                st.error(st.session_state["response_error_var"])

    # ── Approval panel (shown when code is pending) ───────────────────────────
    if st.session_state["pending_approval"] is not None:
        _render_approval_panel(llm_agent, llm, data)
        st.info("⏳ Đang chờ bạn xét duyệt code ở trên trước khi tiếp tục.")
        return

    # ── Chat input ────────────────────────────────────────────────────────────
    if user_question := st.chat_input("Nhập câu hỏi để mình phân tích dữ liệu cho bạn..."):
        st.session_state["response_error_var"] = ""
        user_question = user_question.strip()
        st.session_state.messages.append({"role": "user", "question": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        effective_user_question = _rewrite_followup_question_for_analysis(
            user_question, st.session_state.messages
        )

        with st.chat_message("assistant"):
            print(f"\n{Fore.CYAN}{'='*60}{Fore.RESET}")
            print(f"{Fore.CYAN}[STEP 1] PHÂN LOẠI INTENT{Fore.RESET}")
            print(f"{Fore.WHITE}  Câu hỏi: {user_question!r}{Fore.RESET}")
            if effective_user_question != user_question:
                print(f"{Fore.WHITE}  rewritten_question={effective_user_question!r}{Fore.RESET}")

            should_short_circuit, hint_response = should_answer_normally(
                effective_user_question,
                _get_classifier_llm(llm),
                st.session_state.messages,
            )

            print(
                f"{Fore.YELLOW}  should_short_circuit={should_short_circuit} | "
                f"hint={hint_response[:60] if hint_response else ''!r}{Fore.RESET}"
            )

            if should_short_circuit:
                response = hint_response

                if hint_response == "__DATASET_OVERLAY__":
                    print(f"{Fore.YELLOW}  → Trả lời nhanh: DATASET_OVERLAY{Fore.RESET}")
                    main_df = _get_main_df(data)
                    response = answer_dataset_overview(main_df)
                    _stream_text(response)

                elif hint_response == "__RAG_CONTEXT__":
                    print(f"{Fore.YELLOW}  → Trả lời nhanh: RAG_CONTEXT{Fore.RESET}")
                    try:
                        rag_context = build_rag_context(effective_user_question)
                        if rag_context:
                            rag_prompt = (
                                "Dùng thông tin sau để trả lời câu hỏi bằng tiếng Việt.\n\n"
                                f"Thông tin:\n{rag_context}\n\n"
                                f"Câu hỏi: {effective_user_question}"
                            )
                            response = _stream_llm_response(llm, rag_prompt)
                        else:
                            fallback_prompt = (
                                "Data assistant. Vietnamese only. No code.\n\n"
                                f"{effective_user_question}"
                            )
                            response = _stream_llm_response(llm, fallback_prompt)
                            print(f"{Fore.YELLOW}  ⚠ RAG không tìm thấy doc — LLM tự trả lời{Fore.RESET}")
                    except Exception as e:
                        response = "Mình chưa thể tìm thấy thông tin ngữ cảnh. Bạn thử hỏi lại nhé."
                        st.markdown(response)
                        print(f"{Fore.RED}  ✗ RAG context FAILED: {e}{Fore.RESET}")

                elif hint_response == "__DIRECT_LLM_CHAT__":
                    print(f"{Fore.YELLOW}  → Trả lời nhanh: DIRECT_LLM_CHAT{Fore.RESET}")
                    chat_prompt = (
                        "Data assistant for Vietnamese users. No code, no plot requests. "
                        f"Vietnamese only.\n\n{effective_user_question}"
                    )
                    try:
                        response = _stream_llm_response(llm, chat_prompt)
                        print(f"{Fore.GREEN}  ✓ LLM direct chat OK ({len(response)} chars){Fore.RESET}")
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
                st.session_state.messages.append({"role": "assistant", "response": response})
                print(f"{Fore.CYAN}{'='*60}{Fore.RESET}\n")
                st.rerun()

            # ── STEP 2: Build prompt ──────────────────────────────────────────
            print(f"\n{Fore.CYAN}[STEP 2] BUILD PROMPT{Fore.RESET}")
            with st.spinner("Đang phân tích..."):
                try:
                    prompt = process_prompt(
                        st.session_state.messages, effective_user_question, data, llm
                    )
                    print(f"{Fore.WHITE}  Prompt built ({len(prompt)} chars){Fore.RESET}")
                except Exception as e:
                    exception_name = type(e).__name__
                    track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
                    st.error(f"Lỗi xây dựng prompt: {exception_name} {track_line}")
                    st.rerun()

            # ── STEP 3: LLM → code (NO execution yet) ────────────────────────
            print(f"\n{Fore.CYAN}[STEP 3] LLM GENERATE CODE (pending approval){Fore.RESET}")
            try:
                generated_code = llm_agent.chat_generate(prompt)
                print(f"{Fore.GREEN}  ✓ Code generated ({len(generated_code.splitlines())} lines){Fore.RESET}")
            except Exception as e:
                exception_name = type(e).__name__
                track_line = f" L-{traceback.extract_tb(e.__traceback__)[0].lineno}"
                error_msg = f"Mình gặp lỗi khi sinh code: {exception_name} {track_line}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "response": error_msg})
                st.rerun()

            # ── STEP 4: Store pending approval — do NOT execute ───────────────
            print(f"\n{Fore.CYAN}[STEP 4] STORE PENDING APPROVAL{Fore.RESET}")
            st.session_state["pending_approval"] = {
                "code": generated_code,
                "prompt": prompt,
                "question": effective_user_question,
            }
            st.session_state["last_code"] = generated_code
            st.session_state["context_code_var"] = True
            st.session_state["ace_editor_version"] = 0  # reset for the new approval cycle
            print(f"{Fore.CYAN}{'='*60}{Fore.RESET}\n")
            st.rerun()


# ── Sidebar: model selector + Ollama status ───────────────────────────────────

def _render_model_selector() -> tuple[str, str]:
    """
    Vẽ phần chọn provider + model trong sidebar.
    Trả về (provider, model_name).
    """
    with st.sidebar:

        provider = st.selectbox(
            "Provider",
            PROVIDER_OPTIONS,
            key="llm_provider",
        )

        # ── Ollama: trạng thái kết nối + chọn model từ danh sách local ──
        if provider == "Ollama (local)":
            is_connected = _check_ollama_connection(OLLAMA_BASE_URL)
            if is_connected:
                st.success(f"🟢 Ollama đang chạy  \n`{OLLAMA_BASE_URL}`")
                local_models = _list_ollama_models(OLLAMA_BASE_URL)
                if local_models:
                    default_idx = 0
                    saved_model = st.session_state.get("_ollama_model_select")
                    if saved_model and saved_model in local_models:
                        default_idx = local_models.index(saved_model)
                    model = st.selectbox(
                        "Model (local)",
                        local_models,
                        index=default_idx,
                        key="_ollama_model_select",
                    )
                else:
                    st.warning("Chưa có model nào. Chạy `ollama pull <model>` để tải.")
                    model = st.text_input(
                        "Nhập tên model",
                        value=DEFAULT_MODELS["Ollama (local)"],
                        key="_ollama_model_text",
                    )
            else:
                st.error(f"🔴 Không kết nối được Ollama  \n`{OLLAMA_BASE_URL}`")
                st.caption("Chạy `ollama serve` để khởi động.")
                model = st.text_input(
                    "Tên model (sẽ dùng khi kết nối lại)",
                    value=DEFAULT_MODELS["Ollama (local)"],
                    key="_ollama_model_text",
                )

        # ── Gemini ────────────────────────────────────────────────────────
        elif provider == "Gemini":
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            if not gemini_key:
                st.warning("⚠️ Chưa có GEMINI_API_KEY trong `.env`")
            else:
                st.success("🟢 GEMINI_API_KEY đã được đặt")
            model = st.text_input(
                "Model",
                value=st.session_state.get("_gemini_model", DEFAULT_MODELS["Gemini"]),
                key="_gemini_model",
                help="Ví dụ: gemini/gemini-2.0-flash, gemini/gemini-1.5-pro",
            )

        # ── OpenAI ────────────────────────────────────────────────────────
        else:
            openai_key = os.getenv("OPENAI_API_KEY", "")
            if not openai_key:
                st.warning("⚠️ Chưa có OPENAI_API_KEY trong `.env`")
            else:
                st.success("🟢 OPENAI_API_KEY đã được đặt")
            model = st.text_input(
                "Model",
                value=st.session_state.get("_openai_model", DEFAULT_MODELS["OpenAI"]),
                key="_openai_model",
                help="Ví dụ: gpt-4o-mini, gpt-4o, gpt-4-turbo",
            )

    return provider, model.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    process_styles()

    st.markdown("####")
    header = st.container(border=False)
    header.markdown("####")
    header.header("Chatbot bất động sản", divider="violet")
    header.markdown("####")
    _viewing_archive = st.session_state.get("archive_selector", "💬 Chat hiện tại") != "💬 Chat hiện tại"
    if _viewing_archive:
        header.markdown("##### 📁 Đang xem lịch sử hội thoại")
    else:
        header.markdown("##### Trợ lý AI hỗ trợ phân tích dữ liệu bất động sản")
    st.markdown("##")

    data_df = load_main_dataset()
    data = {"DF_1___vietnam_housing_dataset_cleaned": data_df}

    _warmup_rag_resources()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.session_state["sample_var"] = N_SAMPLES

        side_container_2 = st.sidebar.container(border=True)
        side_container_2.button("🗑️ &nbsp;&nbsp;Xóa hội thoại", on_click=clear_chat_history)

        _CURRENT_LABEL = "💬 Chat hiện tại"
        log_files = list_conversations()
        archive_options = [_CURRENT_LABEL]
        archive_label_to_path: dict[str, str] = {}
        for fp in log_files:
            basename = os.path.basename(fp)
            label = basename.replace("conversation_", "").replace(".json", "")
            label = label.replace("_", " ", 1).replace("-", ":", 2)
            archive_options.append(label)
            archive_label_to_path[label] = fp

        side_archive = st.sidebar.container(border=True)
        selected_archive_label = side_archive.selectbox(
            "🗂️ Lịch sử hội thoại",
            options=archive_options,
            index=0,
            key="archive_selector",
        )
        selected_log = (
            None
            if selected_archive_label == _CURRENT_LABEL
            else archive_label_to_path.get(selected_archive_label)
        )

        side_container_3 = st.sidebar.container(border=True)
        llm_temp = side_container_3.slider(
            "🌡️ &nbsp;Temperature", 0.0, 1.0, 0.0, key="temperature"
        )

    # ── LLM provider + model selector ────────────────────────────────────────
    provider, model = _render_model_selector()
    llm = _get_or_create_llm(provider, model, llm_temp)

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = st.session_state.get("agent_var")
    if agent is None:
        agent = get_llm_agent(data, llm)
    else:
        agent.llm = llm
        agent.data = list(data.values())

    if selected_log is not None:
        _render_archive_view(selected_log)
    else:
        process_chat(agent, llm, data)


if __name__ == "__main__":
    main()