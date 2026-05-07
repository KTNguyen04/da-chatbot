# chart_insight.py
"""
Phân tích biểu đồ Plotly bằng Vision LLM (Moondream) qua Ollama.

Luồng hoạt động:
  1. Render Plotly figure → ảnh PNG (bytes) bằng kaleido.
  2. Encode PNG → Base64 string.
  3. Gửi đến Ollama /api/chat với field "images" (multimodal format).
  4. Trả về chuỗi insight.

Yêu cầu:
  - Ollama đang chạy và đã pull model Moondream:
      ollama pull moondream
  - Thư viện kaleido (dùng để render Plotly → PNG):
      pip install kaleido
"""

import base64
import io
import json
import requests
import traceback
import plotly.graph_objs as go
from colorama import Fore

# ── Hằng số mặc định ──────────────────────────────────────────────────────────
VISION_MODEL_DEFAULT = "moondream:1.8b"
OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
INSIGHT_TIMEOUT = 120  # giây – Moondream nhẹ nhưng inference lần đầu chậm hơn
IMG_WIDTH = 1200  # px – độ rộng ảnh render
IMG_HEIGHT = 700  # px – độ cao ảnh render
IMG_SCALE = 2  # hệ số scale (retina-like, rõ hơn)


# ── Bước 1: Render figure → PNG bytes ─────────────────────────────────────────
def _fig_to_png_bytes(fig: go.Figure) -> bytes:
    """
    Chuyển Plotly figure thành ảnh PNG (bytes) bằng kaleido.
    Raises ImportError nếu kaleido chưa được cài.
    Raises RuntimeError nếu render thất bại.
    """
    try:
        png_bytes: bytes = fig.to_image(
            format="png",
            width=IMG_WIDTH,
            height=IMG_HEIGHT,
            scale=IMG_SCALE,
        )
        if not png_bytes:
            raise RuntimeError("fig.to_image() trả về bytes rỗng.")
        return png_bytes
    except ImportError:
        raise ImportError("Kaleido chưa được cài. Chạy: pip install kaleido")
    except Exception as e:
        raise RuntimeError(f"Không thể render figure thành PNG: {e}")


# ── Bước 2: Encode PNG bytes → Base64 string ──────────────────────────────────
def _png_bytes_to_base64(png_bytes: bytes) -> str:
    """Encode PNG bytes thành chuỗi Base64 (không có prefix data URI)."""
    return base64.b64encode(png_bytes).decode("utf-8")


# ── Bước 3: Build prompt gửi đến vision model ─────────────────────────────────
def _build_vision_prompt(user_question: str, lang: str = "vi") -> str:
    """
    Tạo câu hỏi ngắn gọn để gửi kèm ảnh biểu đồ.
    Moondream hoạt động tốt nhất với prompt ngắn, cụ thể.
    """
    if lang == "vi":
        return (
            f'Đây là một biểu đồ dữ liệu. Câu hỏi gốc của người dùng là: "{user_question}"\n\n'
            "Hãy phân tích biểu đồ này và trả lời bằng tiếng Việt, súc tích (tối đa 150 từ):\n"
            "1. Giá trị nổi bật (cao nhất, thấp nhất, xu hướng chính).\n"
            "2. Nhận xét liên quan đến câu hỏi.\n"
            "3. Gợi ý hoặc kết luận ngắn."
        )
    else:
        return (
            f'This is a data chart. The user\'s original question was: "{user_question}"\n\n'
            "Analyze this chart and respond concisely (max 120 words):\n"
            "1. Key values (highest, lowest, main trend).\n"
            "2. Insight relevant to the question.\n"
            "3. Brief recommendation or conclusion."
        )


# ── Bước 4: Gọi Ollama multimodal API ─────────────────────────────────────────
def _call_ollama_vision(
    base64_image: str,
    prompt_text: str,
    chat_url: str,
    model: str,
    timeout: int = INSIGHT_TIMEOUT,
    verbose: bool = False,
) -> str:
    """
    Gửi ảnh Base64 + prompt đến Ollama /api/chat (multimodal).

    Ollama multimodal format:
      {
        "model": "moondream",
        "messages": [
          {
            "role": "user",
            "content": "<prompt text>",
            "images": ["<base64_string>"]   ← field images chứa list base64
          }
        ],
        "stream": false
      }
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
                "images": [base64_image],  # Ollama nhận list base64 strings
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        },
    }

    if verbose:
        print(
            f"\n{Fore.LIGHTCYAN_EX}[VisionInsight] POST → {chat_url} "
            f"| model={model} | image_b64_len={len(base64_image)}{Fore.RESET}"
        )

    resp = requests.post(chat_url, json=payload, timeout=timeout)
    resp.raise_for_status()

    data = resp.json()
    content = data.get("message", {}).get("content", "").strip()
    return content if content else "⚠️ Model không trả về nội dung."


# ── Public API ────────────────────────────────────────────────────────────────
def generate_chart_insight(
    fig: go.Figure,
    user_question: str,
    ollama_base_url: str = OLLAMA_BASE_URL_DEFAULT,
    model: str = VISION_MODEL_DEFAULT,
    lang: str = "vi",
    verbose: bool = False,
) -> str:
    """
    Render biểu đồ Plotly thành ảnh PNG, encode Base64 và gửi đến
    Moondream (hoặc bất kỳ vision model nào trên Ollama) để sinh insight.

    Args:
        fig:              Plotly Figure cần phân tích.
        user_question:    Câu hỏi gốc của người dùng.
        ollama_base_url:  URL gốc của Ollama server.
        model:            Tên model vision trên Ollama (mặc định: "moondream").
        lang:             Ngôn ngữ phản hồi ("vi" hoặc "en").
        verbose:          In log debug nếu True.

    Returns:
        Chuỗi insight từ vision model, hoặc thông báo lỗi.
    """
    chat_url = f"{ollama_base_url.rstrip('/')}/api/chat"

    # ── Bước 1: Render → PNG ───────────────────────────────────────────────────
    try:
        png_bytes = _fig_to_png_bytes(fig)
        if verbose:
            print(
                f"\n{Fore.LIGHTCYAN_EX}[VisionInsight] PNG rendered: "
                f"{len(png_bytes):,} bytes{Fore.RESET}"
            )
    except ImportError as e:
        return f"⚠️ {e}"
    except RuntimeError as e:
        return f"⚠️ {e}"
    except Exception as e:
        return f"⚠️ Lỗi không xác định khi render ảnh: {e}"

    # ── Bước 2: Encode Base64 ─────────────────────────────────────────────────
    try:
        b64_image = _png_bytes_to_base64(png_bytes)
    except Exception as e:
        return f"⚠️ Lỗi encode Base64: {e}"

    # ── Bước 3: Build prompt ──────────────────────────────────────────────────
    prompt_text = _build_vision_prompt(user_question, lang)

    # ── Bước 4: Gọi Ollama vision API ────────────────────────────────────────
    try:
        insight = _call_ollama_vision(
            base64_image=b64_image,
            prompt_text=prompt_text,
            chat_url=chat_url,
            model=model,
            timeout=INSIGHT_TIMEOUT,
            verbose=verbose,
        )
        if verbose:
            print(
                f"\n{Fore.LIGHTGREEN_EX}[VisionInsight] Insight received "
                f"({len(insight)} chars){Fore.RESET}"
            )
        return insight

    except requests.exceptions.ConnectionError:
        return (
            f"⚠️ Không thể kết nối Ollama tại `{chat_url}`. "
            "Hãy chắc chắn Ollama đang chạy và model đã được pull:\n"
            f"    ollama pull {model}"
        )
    except requests.exceptions.Timeout:
        return (
            f"⚠️ Timeout khi chờ phản hồi từ model `{model}` "
            f"(>{INSIGHT_TIMEOUT}s). Thử tăng INSIGHT_TIMEOUT."
        )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        body = ""
        if e.response is not None:
            try:
                body = e.response.json().get("error", e.response.text[:200])
            except Exception:
                body = e.response.text[:200]
        return (
            f"⚠️ Ollama trả về lỗi HTTP {status}: {body}\n"
            f"Hãy chắc chắn model `{model}` đã được pull:\n"
            f"    ollama pull {model}"
        )
    except Exception as e:
        tb = traceback.format_exc()
        if verbose:
            print(
                f"{Fore.LIGHTRED_EX}[VisionInsight] Unexpected error:\n{tb}{Fore.RESET}"
            )
        return f"⚠️ Lỗi không xác định: {e}"
