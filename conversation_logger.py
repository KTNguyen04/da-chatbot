import base64
import io
import json
import os
from datetime import datetime
import re

import numpy as np
from matplotlib.figure import Figure

LOG_DIR = "./conversation_logs"


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars, arrays, and NaN/Inf values."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _extract_message_payload(message: dict) -> tuple[str, object]:
    role = message.get("role", "assistant")
    if role == "user":
        return str(message.get("question", message.get("content", ""))), None
    return str(message.get("response", message.get("content", ""))), message.get("response")


def _is_code_message(content: str) -> bool:
    return "```" in content


def _extract_code_block(content: str) -> str | None:
    if not content:
        return None
    match = re.search(r"```(?:python)?\s*\n(.*?)```", content, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _figure_to_b64_png(figure: Figure) -> str | None:
    """Render matplotlib Figure to base64-encoded PNG string."""
    try:
        buf = io.BytesIO()
        figure.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception:
        return None


def _extract_figure(payload: object) -> Figure | None:
    """Extract matplotlib Figure from a raw response or a {'figure': Fig} dict."""
    if isinstance(payload, Figure):
        return payload
    if isinstance(payload, dict):
        figure = payload.get("figure")
        if isinstance(figure, Figure):
            return figure
    return None


def _serialize_message(message: dict) -> dict:
    role = message.get("role", "assistant")
    content, payload = _extract_message_payload(message)
    serialized = {
        "role": role,
        "content": content,
        "type": "text",
        "chart_image_b64": None,
        "code": None,
    }

    figure = _extract_figure(payload)
    if role == "assistant" and figure is not None:
        serialized["type"] = "chart_image"
        serialized["chart_image_b64"] = _figure_to_b64_png(figure)
        analysis = ""
        if isinstance(payload, dict):
            analysis = str(payload.get("analysis", "")).strip()
        serialized["content"] = analysis

    if role == "assistant":
        explicit_code = str(message.get("code", "")).strip()
        if explicit_code:
            serialized["code"] = explicit_code
        elif _is_code_message(content):
            serialized["code"] = _extract_code_block(content)

    if role == "assistant" and serialized["code"] and serialized["type"] == "text":
        serialized["type"] = "code"

    return serialized


def _attach_latest_code_to_last_assistant(
    messages: list[dict], latest_code: str | None
) -> list[dict]:
    if not latest_code:
        return messages
    cloned = [dict(message) for message in messages]
    for idx in range(len(cloned) - 1, -1, -1):
        message = cloned[idx]
        if message.get("role") == "assistant":
            message.setdefault("code", latest_code)
            cloned[idx] = message
            break
    return cloned


def save_conversation(messages: list[dict], latest_code: str | None = None) -> str | None:
    if not messages:
        return None

    _ensure_log_dir()
    now = datetime.now()
    filename = f"conversation_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    filepath = os.path.join(LOG_DIR, filename)
    prepared_messages = _attach_latest_code_to_last_assistant(messages, latest_code)
    payload = {
        "saved_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [_serialize_message(message) for message in prepared_messages],
    }
    with open(filepath, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    return filepath


def list_conversations() -> list[str]:
    _ensure_log_dir()
    files = [
        os.path.join(LOG_DIR, file_name)
        for file_name in os.listdir(LOG_DIR)
        if file_name.endswith(".json")
    ]
    return sorted(files, reverse=True)


def load_conversation(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)