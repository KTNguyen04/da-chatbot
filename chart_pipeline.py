"""
Chart pipeline aligned with chatbot.py intent flow (JSON → deferred execution).

- Single LLM round for intent + code (no Agent retry loop).
- Plotly execution only on explicit user action (or cached replay).
- Post-process figures for large-n scatter performance.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import secrets
from typing import Any

import pandas as pd
import plotly.graph_objects as go


MAX_SCATTER_POINTS = 15_000
PLOTLY_RENDER_CONFIG: dict[str, Any] = {"displaylogo": False}


def new_chart_message_id() -> str:
    return secrets.token_hex(8)


def st_session() -> dict:
    import streamlit as st

    return st.session_state


def _stable_code_fingerprint(code: str, df: pd.DataFrame) -> str:
    h = hashlib.sha256(code.strip().encode("utf-8")).hexdigest()[:32]
    sig = f"{df.shape}:{','.join(map(str, df.columns))}"
    return f"{h}:{hash(sig) & 0xFFFFFFFF:x}"


def get_session_figure_cache() -> dict[str, dict[str, Any]]:
    """Maps execution fingerprint → {"figure": go.Figure, "analysis": str|None}."""
    return st_session().setdefault("_chart_figure_cache", {})


def take_from_exec_cache(fingerprint: str) -> dict[str, Any] | None:
    return get_session_figure_cache().get(fingerprint)


def store_in_exec_cache(
    fingerprint: str, fig: go.Figure, analysis: str | None
) -> None:
    cache = get_session_figure_cache()
    cache[fingerprint] = {"figure": fig, "analysis": analysis}
    if len(cache) > 24:
        for k in list(cache.keys())[:-24]:
            cache.pop(k, None)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """First top-level JSON object in model output (same idea as chatbot.py regex)."""
    if not text or not text.strip():
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def optimize_plotly_figure(fig: go.Figure) -> go.Figure:
    """
    Reduce client-side work for very large scatter-like traces.
    Does not change aggregated charts (bar, box, etc.) meaningfully.
    """
    if fig is None or not isinstance(fig, go.Figure):
        return fig
    rng = random.Random(42)
    for i, tr in enumerate(list(fig.data)):
        tname = type(tr).__name__
        if "Scatter" not in tname:
            continue
        xs = getattr(tr, "x", None)
        if xs is None:
            continue
        try:
            n = len(xs)
        except TypeError:
            continue
        if n <= MAX_SCATTER_POINTS:
            continue
        idx = sorted(rng.sample(range(n), k=MAX_SCATTER_POINTS))
        sx = pd.Series(xs)
        new_x = sx.iloc[idx].tolist()
        ys = getattr(tr, "y", None)
        new_y = None
        if ys is not None:
            new_y = pd.Series(ys).iloc[idx].tolist()
        fig.data[i].update(x=new_x, y=new_y)
    try:
        fig.update_layout(uirevision="constant")
    except Exception:
        pass
    return fig


def normalize_intent_payload(raw: dict[str, Any]) -> dict[str, Any]:
    intent = str(raw.get("intent", "TEXT")).upper().strip()
    if intent not in ("CHART", "TABLE", "TEXT"):
        intent = "TEXT"
    return {
        "intent": intent,
        "idea": str(raw.get("idea", "") or ""),
        "explanation": str(raw.get("explanation", "") or ""),
        "code": str(raw.get("code", "") or "").strip(),
        "table_query": str(raw.get("table_query", "") or "").strip(),
    }


def eval_table_query(table_query: str, df_by_name: dict[str, pd.DataFrame]) -> Any:
    """Evaluate a pandas expression like chatbot's table path (trusted LLM output)."""
    import pandas as pd

    safe_builtins = {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
    }
    env = {"pd": pd, **df_by_name}
    return eval(table_query, {"__builtins__": safe_builtins}, env)
