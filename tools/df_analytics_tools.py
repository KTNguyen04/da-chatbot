# tools/df_analytics_tools.py
"""
Generic tabular analysis tools (dataset-agnostic).
Use alongside domain-specific tools (e.g. housing_tools).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None  # type: ignore

_df: Optional[pd.DataFrame] = None

def set_dataframe(df: pd.DataFrame) -> None:
    global _df
    _df = df


def _require_df() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("DataFrame not initialized; call set_dataframe() first.")
    return _df


def _resolve_column(df: pd.DataFrame, name: str) -> Optional[str]:
    if name in df.columns:
        return name
    lower = name.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == lower:
            return str(c)
    return None


def _mask_from_conditions(
    df: pd.DataFrame, conditions: list[dict[str, Any]]
) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    for cond in conditions:
        col = cond.get("column")
        op = str(cond.get("op", "eq")).lower()
        val = cond.get("value")
        if not col:
            continue
        rc = _resolve_column(df, col)
        if rc is None:
            return pd.Series(False, index=df.index)
        s = df[rc]

        if op == "isna":
            mask &= s.isna()
        elif op == "notna":
            mask &= s.notna()
        elif op == "eq":
            mask &= s == val
        elif op == "ne":
            mask &= s != val
        elif op in ("gt", "gte", "lt", "lte", "between"):
            sn = pd.to_numeric(s, errors="coerce")
            if op == "gt":
                mask &= sn > float(val)  # type: ignore[arg-type]
            elif op == "gte":
                mask &= sn >= float(val)  # type: ignore[arg-type]
            elif op == "lt":
                mask &= sn < float(val)  # type: ignore[arg-type]
            elif op == "lte":
                mask &= sn <= float(val)  # type: ignore[arg-type]
            elif op == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
                lo, hi = float(val[0]), float(val[1])
                mask &= sn.between(lo, hi, inclusive="both")
        elif op == "in_list" and isinstance(val, list):
            mask &= s.isin(val)
        elif op == "contains" and val is not None:
            mask &= s.astype(str).str.contains(str(val), case=False, na=False, regex=False)
        else:
            mask &= pd.Series(False, index=df.index)
    return mask


# ── Tool: data profile ────────────────────────────────────────────────────────


def get_data_profile(
    max_columns: int = 80,
    sample_rows: int = 3,
) -> dict[str, Any]:
    df = _require_df()
    n, m = len(df), len(df.columns)
    cols = list(df.columns)[: max(1, min(max_columns, 200))]
    details = []
    for c in cols:
        s = df[c]
        null_pct = round(float(s.isna().mean() * 100), 2) if n else 0.0
        nun = int(s.nunique(dropna=True))
        id_like = (n > 1 and nun == n) or (nun > 50 and nun / max(n, 1) > 0.95)
        details.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "null_pct": null_pct,
                "nunique": nun,
                "flag_id_like": bool(id_like),
            }
        )

    dup_pct = round(float(df.duplicated().mean() * 100), 2) if n else 0.0
    sample = df[cols].head(min(sample_rows, 5)).to_dict(orient="records")

    datetime_hints = [c for c in cols if "date" in c.lower() or "time" in c.lower()]

    return {
        "tool": "get_data_profile",
        "row_count": int(n),
        "column_count": int(m),
        "columns_shown": len(cols),
        "duplicate_row_pct": dup_pct,
        "column_summaries": details,
        "datetime_hint_columns": datetime_hints[:15],
        "sample_rows": sample,
        "insight": (
            f"Bảng có {n:,} dòng và {m} cột. "
            f"{dup_pct}% dòng trùng lặp. "
            f"Đã mô tả {len(details)} cột (dtype, thiếu, số giá trị khác nhau)."
        ),
    }


# ── Tool: column profile ────────────────────────────────────────────────────────


def profile_column(column: str, top_n: int = 15) -> dict[str, Any]:
    df = _require_df()
    rc = _resolve_column(df, column)
    if rc is None:
        return {"error": f"Cột '{column}' không tồn tại.", "columns": list(df.columns)}

    s = df[rc]
    top_n = max(3, min(int(top_n), 50))
    nulls = int(s.isna().sum())

    if pd.api.types.is_numeric_dtype(s) or pd.to_numeric(s, errors="coerce").notna().sum() > 0.8 * len(s):
        num = pd.to_numeric(s, errors="coerce").dropna()
        if num.empty:
            return {"error": f"Cột '{rc}' không có giá trị số hợp lệ."}
        out: dict[str, Any] = {
            "tool": "profile_column",
            "column": rc,
            "kind": "numeric",
            "count": int(num.shape[0]),
            "missing": nulls,
            "mean": round(float(num.mean()), 4),
            "std": round(float(num.std()), 4) if len(num) > 1 else 0.0,
            "min": round(float(num.min()), 4),
            "max": round(float(num.max()), 4),
            "q25": round(float(num.quantile(0.25)), 4),
            "median": round(float(num.median()), 4),
            "q75": round(float(num.quantile(0.75)), 4),
        }
        out["insight"] = (
            f"'{rc}' kiểu số: trung bình {out['mean']}, trung vị {out['median']}, "
            f"từ {out['min']} đến {out['max']}, {nulls} giá trị thiếu."
        )
        return out

    vc = s.astype("string").fillna("<NA>").value_counts().head(top_n)
    records = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
    top = records[0] if records else {}
    return {
        "tool": "profile_column",
        "column": rc,
        "kind": "categorical",
        "count_non_na": int(s.notna().sum()),
        "missing": nulls,
        "nunique": int(s.nunique(dropna=True)),
        "top_categories": records,
        "insight": (
            f"'{rc}' phân loại: {int(s.nunique(dropna=True))} giá trị khác nhau. "
            f"Nhiều nhất: '{top.get('value')}' ({top.get('count', 0):,})."
        ),
    }


# ── Tool: filter + optional aggregate ───────────────────────────────────────────


def filter_rows(
    conditions: list[dict[str, Any]],
    value_column: Optional[str] = None,
) -> dict[str, Any]:
    """
    conditions: list of {column, op, value}. Combined with AND.
    op: eq, ne, gt, gte, lt, lte, between, in_list, contains, isna, notna
    """
    df = _require_df()
    if not conditions:
        return {"error": "conditions không được rỗng."}

    mask = _mask_from_conditions(df, conditions)
    sub = df.loc[mask]
    n = int(len(sub))
    out: dict[str, Any] = {
        "tool": "filter_rows",
        "matched_rows": n,
        "conditions": conditions,
    }
    if value_column:
        rvc = _resolve_column(df, value_column)
        if rvc is None:
            out["error"] = f"Cột giá trị '{value_column}' không tồn tại."
            return out
        num = pd.to_numeric(sub[rvc], errors="coerce").dropna()
        if len(num) == 0:
            out["value_stats"] = {"error": "Không có số liệu hợp lệ sau lọc."}
        else:
            out["value_stats"] = {
                "column": rvc,
                "mean": round(float(num.mean()), 4),
                "median": round(float(num.median()), 4),
                "std": round(float(num.std()), 4) if len(num) > 1 else 0.0,
                "min": round(float(num.min()), 4),
                "max": round(float(num.max()), 4),
            }
    out["insight"] = f"Sau khi lọc (AND), còn {n:,} dòng."
    if "value_stats" in out and isinstance(out["value_stats"], dict) and "mean" in out["value_stats"]:
        vs = out["value_stats"]
        out["insight"] += f" Trung bình '{vs['column']}': {vs['mean']}."
    return out


# ── Tool: pivot / BI summary ────────────────────────────────────────────────────


def pivot_summary(
    index_column: str,
    value_column: str,
    aggfunc: str = "mean",
    columns_column: Optional[str] = None,
    top_n_index: int = 20,
) -> dict[str, Any]:
    df = _require_df()
    idx = _resolve_column(df, index_column)
    val = _resolve_column(df, value_column)
    if not idx or not val:
        return {"error": "index_column hoặc value_column không hợp lệ."}

    top_n_index = max(5, min(int(top_n_index), 80))
    work = df[[idx, val]].copy()
    if columns_column:
        cc2 = _resolve_column(df, columns_column)
        if not cc2:
            return {"error": f"columns_column '{columns_column}' không tồn tại."}
        work[cc2] = df[cc2]

    work[val] = pd.to_numeric(work[val], errors="coerce")
    work = work.dropna(subset=[val])

    allowed = {"mean", "sum", "median", "count", "min", "max"}
    af = aggfunc if aggfunc in allowed else "mean"

    if columns_column:
        cc = _resolve_column(df, columns_column)
        assert cc is not None
        top_idx = work[idx].astype(str).value_counts().head(top_n_index).index
        work = work[work[idx].astype(str).isin(top_idx)]
        pt = pd.pivot_table(
            work,
            values=val,
            index=idx,
            columns=cc,
            aggfunc=af,
            fill_value=np.nan,
        )
        mat = pt.round(4)
        out_tbl = mat.to_dict()
        insight = f"Bảng pivot {af} của '{val}' theo '{idx}' và '{cc}' (giới hạn {top_n_index} nhóm hàng)."
    else:
        top_idx = work[idx].astype(str).value_counts().head(top_n_index).index
        g = work[work[idx].astype(str).isin(top_idx)].groupby(idx, observed=True)[val]
        agg_map: dict[str, Callable[..., Any]] = {
            "mean": g.mean,
            "median": g.median,
            "sum": g.sum,
            "count": g.count,
            "min": g.min,
            "max": g.max,
        }
        ser = agg_map[af]().sort_values(ascending=False)
        records = [{"group": str(k), str(af): round(float(v), 4)} for k, v in ser.items()]
        out_tbl = {"records": records}
        best = records[0] if records else {}
        insight = f"{af.capitalize()} '{val}' theo '{idx}': cao nhất '{best.get('group')}' = {best.get(af, 'n/a')}."

    return {
        "tool": "pivot_summary",
        "aggfunc": af,
        "table": out_tbl,
        "insight": insight,
    }


# ── Tool: correlations ────────────────────────────────────────────────────────


def numeric_correlation_pairs(top_k: int = 20, min_abs_corr: float = 0.05) -> dict[str, Any]:
    df = _require_df()
    num = df.select_dtypes(include=["number"]).copy()
    for c in num.columns:
        num[c] = pd.to_numeric(num[c], errors="coerce")
    num = num.dropna(axis=1, how="all")
    if num.shape[1] < 2:
        return {"error": "Không đủ cột số để tính tương quan."}

    cmat = num.corr(numeric_only=True)
    pairs = []
    cols = list(cmat.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = cmat.loc[a, b]
            if pd.isna(v):
                continue
            pairs.append((abs(float(v)), float(v), a, b))
    pairs.sort(reverse=True, key=lambda t: t[0])
    top_k = max(5, min(int(top_k), 100))
    min_abs_corr = float(min_abs_corr)
    filtered = [p for p in pairs if p[0] >= min_abs_corr][:top_k]
    records = [{"col_a": p[2], "col_b": p[3], "correlation": round(p[1], 4)} for p in filtered]
    top = records[0] if records else {}
    return {
        "tool": "numeric_correlation_pairs",
        "numeric_columns_used": cols,
        "sample_size": int(num.dropna().shape[0]),
        "pairs": records,
        "caveats": ["Tương quan không suy ra nhân quả."],
        "insight": (
            f"Cặp tuyến tính mạnh nhất: {top.get('col_a')} vs {top.get('col_b')} "
            f"(r={top.get('correlation')})." if top else "Không có cặp tương quan đáng kể."
        ),
    }


# ── Tool: two-sample test ─────────────────────────────────────────────────────


def compare_two_groups_stat_test(
    value_column: str,
    group_column: str,
    group_value_a: Any,
    group_value_b: Any,
    test: str = "mannwhitney",
) -> dict[str, Any]:
    if scipy_stats is None:
        return {"error": "scipy chưa được cài đặt."}

    df = _require_df()
    vc = _resolve_column(df, value_column)
    gc = _resolve_column(df, group_column)
    if not vc or not gc:
        return {"error": "Tên cột không hợp lệ."}

    work = df[[vc, gc]].copy()
    work[vc] = pd.to_numeric(work[vc], errors="coerce")
    work = work.dropna(subset=[vc])
    a = work.loc[work[gc] == group_value_a, vc]
    b = work.loc[work[gc] == group_value_b, vc]
    if len(a) < 3 or len(b) < 3:
        return {"error": "Một trong hai nhóm quá ít dữ liệu sau khi lọc (cần ≥3 điểm mỗi nhóm)."}

    test = test.lower()
    if test == "welch_t":
        res = scipy_stats.ttest_ind(a.values, b.values, equal_var=False)
        stat, pvalue = float(res.statistic), float(res.pvalue)
        test_name = "Welch t-test"
    else:
        res = scipy_stats.mannwhitneyu(a.values, b.values, alternative="two-sided")
        stat, pvalue = float(res.statistic), float(res.pvalue)
        test_name = "Mann–Whitney U"

    med_a, med_b = float(np.median(a)), float(np.median(b))
    p_ok = pvalue == pvalue and not np.isnan(pvalue)
    p_str = f"{float(pvalue):.4g}" if p_ok else "n/a"
    return {
        "tool": "compare_two_groups_stat_test",
        "test": test_name,
        "value_column": vc,
        "group_column": gc,
        "group_a": str(group_value_a),
        "group_b": str(group_value_b),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "median_a": round(med_a, 4),
        "median_b": round(med_b, 4),
        "statistic": float(stat) if not np.isnan(stat) else None,
        "p_value_two_sided": float(pvalue) if p_ok else None,
        "alpha": 0.05,
        "caveats": [
            "Giả định độc lập giữa các quan sát.",
            "Kết quả phụ thuộc chất lượng dữ liệu và ngoại lệ.",
        ],
        "insight": (
            f"{test_name}: p={p_str}. "
            f"Trung vị nhóm A={med_a:.4g}, nhóm B={med_b:.4g}."
        ),
    }


# ── Tool: chi-square ────────────────────────────────────────────────────────────


def chi_square_categorical_association(column_a: str, column_b: str) -> dict[str, Any]:
    if scipy_stats is None:
        return {"error": "scipy chưa được cài đặt."}

    df = _require_df()
    ca = _resolve_column(df, column_a)
    cb = _resolve_column(df, column_b)
    if not ca or not cb:
        return {"error": "Tên cột không hợp lệ."}

    ct = pd.crosstab(df[ca].astype("string").fillna("<NA>"), df[cb].astype("string").fillna("<NA>"))
    if ct.size == 0:
        return {"error": "Không đủ dữ liệu cho bảng chéo."}

    chi2, p, dof, expected = scipy_stats.chi2_contingency(ct.values)
    return {
        "tool": "chi_square_categorical_association",
        "column_a": ca,
        "column_b": cb,
        "chi2_statistic": round(float(chi2), 4),
        "degrees_of_freedom": int(dof),
        "p_value": float(p),
        "table_shape": {"rows": int(ct.shape[0]), "cols": int(ct.shape[1])},
        "caveats": [
            "Ô hiếm làm lệch chi-square; cần đủ tần suất kỳ vọng.",
        ],
        "insight": (
            f"Kiểm định độc lập giữa '{ca}' và '{cb}': chi²={float(chi2):.3f}, p={float(p):.4g}."
        ),
    }


# ── Tool: visualization hints ───────────────────────────────────────────────────


def recommend_visualizations(user_intent_keywords: Optional[str] = None) -> dict[str, Any]:
    df = _require_df()
    nums = list(df.select_dtypes(include=["number"]).columns)
    cats = [c for c in df.columns if c not in nums][:40]
    hint = (user_intent_keywords or "").lower()

    rec: list[str] = []
    if len(nums) >= 2 and any(k in hint for k in ("scatter", "correlation", "relate", "liên hệ", "tương quan")):
        rec.append("scatter_2d_plot: hai cột số quan trọng nhất (alpha thấp nếu đông điểm).")
    if nums and cats:
        rec.append("box_plot: phân phối số theo một cột phân loại.")
        rec.append("bar_plot: trung bình / tổng theo nhóm (sắp xếp giảm dần).")
    if cats:
        rec.append("histogram_plot: phân phối một biến số chính.")
    if len(cats) >= 1:
        vc = df[cats[0]].astype(str)
        if vc.nunique() <= 12:
            rec.append("pie_plot: chỉ khi ít nhóm (≤12) để dễ đọc.")
    if "time" in "".join(df.columns).lower() or "date" in hint:
        rec.append("line_plot: nếu có cột thời gian sau khi parse datetime.")

    if not rec:
        rec.append("bar_plot hoặc histogram_plot cho cột số/categorical phổ biến.")

    return {
        "tool": "recommend_visualizations",
        "numeric_columns": nums[:30],
        "sample_categorical_columns": cats[:15],
        "suggestions": rec,
        "insight": "Gợi ý chart dựa trên dtype và từ khóa ý định (có thể tinh chỉnh theo câu hỏi).",
    }


TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_data_profile",
            "description": (
                "Summarize the loaded dataframe: row/column counts, per-column dtype/null rate/nunique, "
                "duplicate rate, sample rows. Call first when exploring an unfamiliar dataset or when the user "
                "asks what columns exist or data quality overview."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_columns": {
                        "type": "integer",
                        "description": "Max columns to describe in detail (default 80)",
                        "default": 80,
                    },
                    "sample_rows": {
                        "type": "integer",
                        "description": "Number of example rows to include (default 3)",
                        "default": 3,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "profile_column",
            "description": (
                "Profile one column: if numeric, return summary stats; if categorical, return value counts. "
                "Case-insensitive column name match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Column name"},
                    "top_n": {
                        "type": "integer",
                        "description": "For categorical, max categories to list (default 15)",
                        "default": 15,
                    },
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_rows",
            "description": (
                "Apply AND-ed filter conditions, return matched row count and optional numeric summary after filters. "
                "Each condition: {column, op, value}. ops: eq, ne, gt, gte, lt, lte, between ([low,high]), "
                "in_list, contains (substring), isna, notna."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of {column, op, value}",
                    },
                    "value_column": {
                        "type": "string",
                        "description": "Optional numeric column to summarize on the filtered subset",
                    },
                },
                "required": ["conditions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pivot_summary",
            "description": (
                "Pivot-style aggregation: aggregate a numeric column by one row dimension, "
                "optionally also split by a second categorical columns_column. "
                "aggfunc: mean, median, sum, count, min, max."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index_column": {"type": "string"},
                    "value_column": {"type": "string"},
                    "aggfunc": {
                        "type": "string",
                        "enum": ["mean", "median", "sum", "count", "min", "max"],
                        "default": "mean",
                    },
                    "columns_column": {
                        "type": "string",
                        "description": "Optional second categorical dimension",
                    },
                    "top_n_index": {
                        "type": "integer",
                        "description": "Limit to top N index categories by frequency",
                        "default": 20,
                    },
                },
                "required": ["index_column", "value_column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "numeric_correlation_pairs",
            "description": (
                "List strongest Pearson correlations between numeric columns (pairs, sorted by absolute value)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "default": 20},
                    "min_abs_corr": {
                        "type": "number",
                        "description": "Minimum |r| to include (default 0.05)",
                        "default": 0.05,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_two_groups_stat_test",
            "description": (
                "Compare a numeric column between two levels of a categorical group_column "
                "using Mann–Whitney U (nonparametric) or Welch t-test. Use for A/B style questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value_column": {"type": "string"},
                    "group_column": {"type": "string"},
                    "group_value_a": {"description": "Label of first group"},
                    "group_value_b": {"description": "Label of second group"},
                    "test": {
                        "type": "string",
                        "enum": ["mannwhitney", "welch_t"],
                        "default": "mannwhitney",
                    },
                },
                "required": ["value_column", "group_column", "group_value_a", "group_value_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chi_square_categorical_association",
            "description": (
                "Chi-square test of independence on two categorical columns (contingency table). "
                "Use for association between two nominal fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column_a": {"type": "string"},
                    "column_b": {"type": "string"},
                },
                "required": ["column_a", "column_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_visualizations",
            "description": (
                "Suggest matplotlib/seaborn chart types (by internal graph_type names used in this app) "
                "based on dataframe dtypes and optional user intent keywords."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_intent_keywords": {
                        "type": "string",
                        "description": "Short phrase or keywords from the user's question",
                    }
                },
                "required": [],
            },
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_data_profile": get_data_profile,
    "profile_column": profile_column,
    "filter_rows": filter_rows,
    "pivot_summary": pivot_summary,
    "numeric_correlation_pairs": numeric_correlation_pairs,
    "compare_two_groups_stat_test": compare_two_groups_stat_test,
    "chi_square_categorical_association": chi_square_categorical_association,
    "recommend_visualizations": recommend_visualizations,
}
