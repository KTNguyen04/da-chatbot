from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import plotly.graph_objs as go


ARCHIVE_DIRNAME = "conversation_logs"


@dataclass(frozen=True)
class ArchiveRef:
    filename: str
    path: Path


def _archive_dir(base_dir: Path | None = None) -> Path:
    root = base_dir or Path(__file__).resolve().parent
    return root / ARCHIVE_DIRNAME


def _ensure_archive_dir(base_dir: Path | None = None) -> Path:
    d = _archive_dir(base_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _timestamp_id(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%Y-%m-%d_%H-%M-%S")


def list_archives(base_dir: Path | None = None) -> list[ArchiveRef]:
    d = _ensure_archive_dir(base_dir)
    files = sorted(d.glob("*.json"), key=lambda p: p.name, reverse=True)
    return [ArchiveRef(filename=p.name, path=p) for p in files]


def load_conversation(filename: str, base_dir: Path | None = None) -> dict[str, Any]:
    p = _ensure_archive_dir(base_dir) / filename
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_conversation(
    messages: list[dict[str, Any]] | None,
    base_dir: Path | None = None,
    dt: datetime | None = None,
) -> ArchiveRef | None:
    if not messages:
        return None

    archive_id = _timestamp_id(dt)
    payload = {
        "id": archive_id,
        "savedAt": (dt or datetime.now()).isoformat(),
        "messages": _serialize_session_messages(messages),
    }

    d = _ensure_archive_dir(base_dir)
    filename = f"{archive_id}.json"
    path = d / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return ArchiveRef(filename=filename, path=path)


def _serialize_session_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "assistant")
        if role == "user":
            content = str(m.get("question", ""))
            out.append(_mk_text_item(role="user", content=content))
            continue

        if "response" in m:
            out.extend(_serialize_assistant_response(m["response"]))
            continue

        if "error" in m:
            out.append(_mk_text_item(role="assistant", content=str(m["error"])))
            continue

    return out


def _serialize_assistant_response(response: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if isinstance(response, str):
        items.append(_mk_text_item(role="assistant", content=response))
        return items

    if isinstance(response, go.Figure):
        items.append(_mk_chart_item(role="assistant", figure=response))
        return items

    if isinstance(response, dict) and "figure" in response:
        fig = response.get("figure")
        if isinstance(fig, go.Figure):
            items.append(_mk_chart_item(role="assistant", figure=fig))
        analysis = response.get("analysis", "")
        if analysis:
            items.append(_mk_text_item(role="assistant", content=str(analysis)))
        return items

    if isinstance(response, (list, tuple)):
        for part in response:
            items.extend(_serialize_assistant_response(part))
        return items

    if isinstance(response, dict):
        # Generic dict responses: flatten in a stable key order.
        for k in sorted(response.keys(), key=lambda x: str(x)):
            v = response[k]
            if isinstance(v, go.Figure):
                items.append(_mk_chart_item(role="assistant", figure=v))
            else:
                items.append(_mk_text_item(role="assistant", content=str(v)))
        return items

    items.append(_mk_text_item(role="assistant", content=str(response)))
    return items


def _mk_text_item(role: str, content: str) -> dict[str, Any]:
    t = "code" if "```" in content else "text"
    return {"role": role, "content": content, "type": t}


def _mk_chart_item(role: str, figure: go.Figure) -> dict[str, Any]:
    png_b64 = _try_plotly_png_base64(figure)
    if png_b64:
        return {
            "role": role,
            "type": "chart",
            # Keep schema simple: store as a string.
            "content": f"data:image/png;base64,{png_b64}",
        }

    # Fallback: store Plotly JSON (replay uses st.plotly_chart).
    return {
        "role": role,
        "type": "chart",
        "content": json.dumps(figure.to_plotly_json(), ensure_ascii=False),
    }


def _try_plotly_png_base64(fig: go.Figure) -> str | None:
    try:
        png_bytes = fig.to_image(format="png")  # requires Kaleido at runtime
        return base64.b64encode(png_bytes).decode("ascii")
    except Exception:
        return None


def iter_archive_messages(archive: dict[str, Any]) -> Iterable[dict[str, Any]]:
    msgs = archive.get("messages") or []
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                yield m

