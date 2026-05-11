import base64
import json
import os
from datetime import datetime

import plotly.graph_objs as go
import plotly.io as pio

LOG_DIR = "./conversation_logs"


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _extract_message_payload(message: dict) -> tuple[str, object]:
    role = message.get("role", "assistant")
    if role == "user":
        return str(message.get("question", message.get("content", ""))), None
    return str(message.get("response", message.get("content", ""))), message.get("response")


def _is_code_message(content: str) -> bool:
    return "```" in content


def _figure_to_b64_png(figure: go.Figure) -> str | None:
    try:
        png_bytes = pio.to_image(figure, format="png")
    except Exception:
        return None
    return base64.b64encode(png_bytes).decode("utf-8")


def _extract_figure(payload: object) -> go.Figure | None:
    if isinstance(payload, go.Figure):
        return payload
    if isinstance(payload, dict):
        figure = payload.get("figure")
        if isinstance(figure, go.Figure):
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
    }

    figure = _extract_figure(payload)
    if role == "assistant" and figure is not None:
        serialized["type"] = "chart_image"
        serialized["chart_image_b64"] = _figure_to_b64_png(figure)
        analysis = ""
        if isinstance(payload, dict):
            analysis = str(payload.get("analysis", "")).strip()
        serialized["content"] = analysis
    elif role == "assistant" and _is_code_message(content):
        serialized["type"] = "code"

    return serialized


def save_conversation(messages: list[dict]) -> str | None:
    if not messages:
        return None

    _ensure_log_dir()
    now = datetime.now()
    filename = f"conversation_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    filepath = os.path.join(LOG_DIR, filename)
    payload = {
        "saved_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [_serialize_message(message) for message in messages],
    }
    with open(filepath, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
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
