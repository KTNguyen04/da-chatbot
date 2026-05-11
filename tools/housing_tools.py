# tools/housing_tools.py
"""
Analysis tools for the Vietnam housing dataset.

Each tool exposes:
  - TOOL_DEFINITIONS: English schema text for the LLM (when to call, how arguments work)
  - TOOL_FUNCTIONS: Python implementations (numeric summaries; user-facing prose still Vietnamese via the chat model)

Dataset columns include:
  Address, Area, Frontage, Access Road, House direction, Balcony direction,
  Floors, Bedrooms, Bathrooms, Legal status, Furniture state, Price, Project,
  Ward_Street, District, Province, Price_per_m2, Is_Project, Area_group, Has_certificate
"""

import pandas as pd
import numpy as np
from typing import Any, Optional, List

# ── DataFrame reference (injected by tool_executor) ─────────────────────────
_df: Optional[pd.DataFrame] = None


def set_dataframe(df: pd.DataFrame) -> None:
    """Called by tool_executor before any tool runs."""
    global _df
    _df = df


def _require_df() -> pd.DataFrame:
    if _df is None:
        raise RuntimeError("DataFrame not initialized; call set_dataframe() first.")
    return _df


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS (OpenAI / Ollama function-calling; descriptions in English)
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS: list[dict] = [

    # ── 1. Province / city ranking by listing count ────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_province_ranking",
            "description": (
                "Rank provinces or cities by how many listings appear in the dataset. "
                "Use when the user asks (possibly in Vietnamese) which province appears most/least, "
                "distribution by province, top provinces, counts per province, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "How many provinces/cities to return (default 15; use a large value for all)",
                        "default": 15
                    },
                    "ascending": {
                        "type": "boolean",
                        "description": "True = smallest counts first, False = largest first (default False)",
                        "default": False
                    }
                },
                "required": []
            }
        }
    },

    # ── 2. Descriptive statistics for a numeric column ─────────────────────────
    {
        "type": "function",
        "function": {
            "name": "describe_numeric_column",
            "description": (
                "Compute descriptive statistics (min, max, mean, median, std, percentiles) "
                "for one numeric column. Use when the user asks for summary stats, averages, spread, "
                "or distribution of a numeric field (questions may be phrased in Vietnamese)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Numeric column name (e.g. 'Price', 'Area', 'Bedrooms')"
                    }
                },
                "required": ["column"]
            }
        }
    },

    # ── 3. Aggregate a numeric column by a categorical group ───────────────────
    {
        "type": "function",
        "function": {
            "name": "compare_mean_by_group",
            "description": (
                "Aggregate a numeric column by a categorical column (mean/median/sum/count/min/max). "
                "Examples: mean price by province, mean area by bedroom count, price per m² by district. "
                "User wording may be Vietnamese (e.g. comparing price across groups)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value_column": {
                        "type": "string",
                        "description": "Numeric column to aggregate (e.g. 'Price', 'Area', 'Price_per_m2')"
                    },
                    "group_column": {
                        "type": "string",
                        "description": "Categorical column to group by (e.g. 'Province', 'District', 'Legal status')"
                    },
                    "agg_func": {
                        "type": "string",
                        "enum": ["mean", "median", "sum", "count", "min", "max"],
                        "description": "Aggregation function (default: mean)",
                        "default": "mean"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Return only the top N groups by aggregated value (default 15)",
                        "default": 15
                    }
                },
                "required": ["value_column", "group_column"]
            }
        }
    },

    # ── 4. Counts and shares by category ───────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "count_by_category",
            "description": (
                "Count rows and percentage share for each value of a categorical column. "
                "Use for composition / mix questions (legal status mix, furniture mix, direction mix, etc.); "
                "user questions may be Vietnamese."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Categorical column to count (e.g. 'Legal status', 'Furniture state', 'House direction')"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Keep only the top N most frequent groups (0 = all groups; default 20)",
                        "default": 20
                    }
                },
                "required": ["column"]
            }
        }
    },

    # ── 5. Filter rows then summarize a numeric column ──────────────────────────
    {
        "type": "function",
        "function": {
            "name": "filter_and_summarize",
            "description": (
                "Filter rows by one equality condition, then compute summary stats on a numeric column. "
                "Use for questions like average price for 3-bedroom homes in a given province, "
                "or stats after filtering by certificate / legal status (Vietnamese phrasing is common)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_column": {
                        "type": "string",
                        "description": "Column used to filter (e.g. 'Province', 'Bedrooms', 'Has_certificate')"
                    },
                    "filter_value": {
                        "description": "Filter value (string or number)",
                        "oneOf": [{"type": "string"}, {"type": "number"}]
                    },
                    "stat_column": {
                        "type": "string",
                        "description": "Numeric column to summarize after filtering (e.g. 'Price', 'Area')"
                    }
                },
                "required": ["filter_column", "filter_value", "stat_column"]
            }
        }
    },

    # ── 6. Outlier detection (IQR) ─────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "detect_outliers",
            "description": (
                "Detect outliers in a numeric column using the IQR rule. "
                "Use when the user asks about unusual prices, extreme values, or skewed tails "
                "(often expressed in Vietnamese)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Numeric column to scan (e.g. 'Price', 'Area', 'Price_per_m2')"
                    }
                },
                "required": ["column"]
            }
        }
    },
    # ── 7. Price drivers (correlation-style signals) ────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "analyze_price_drivers",
            "description": (
                "Summarize how strongly price relates to structural numeric features "
                "(area, floors, bedrooms, bathrooms, frontage). "
                "Use for broad 'what drives price' or 'bigger house more expensive' style questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "description": "Target price column (default: Price)",
                        "default": "Price",
                    },
                    "feature_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Feature columns to analyze "
                            "(default: Area, Floors, Bedrooms, Bathrooms, Frontage)"
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    # ── 8. Price vs size by quantile buckets ───────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "price_vs_size_summary",
            "description": (
                "Bin area into quantile groups and report mean/median price per bin. "
                "Use to answer whether larger homes are always more expensive (overlap and monotonicity)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area_column": {
                        "type": "string",
                        "description": "Area column name (default: Area)",
                        "default": "Area",
                    },
                    "price_column": {
                        "type": "string",
                        "description": "Price column name (default: Price)",
                        "default": "Price",
                    },
                    "bins": {
                        "type": "integer",
                        "description": "Number of quantile buckets (default 5)",
                        "default": 5,
                    },
                },
                "required": [],
            },
        },
    },
    # ── 9. Price by structural groups ──────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "structure_group_price_compare",
            "description": (
                "Compare price across structural groups (floors, bedrooms, bathrooms, frontage buckets). "
                "Returns group-level medians/means to support structure-vs-price narratives."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Structural columns to compare "
                            "(default: Floors, Bedrooms, Bathrooms)"
                        ),
                    },
                    "price_column": {
                        "type": "string",
                        "description": "Price column name (default: Price)",
                        "default": "Price",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Max groups to return per column",
                        "default": 10,
                    },
                },
                "required": [],
            },
        },
    },
    # ── 10. Intent bundle: physical structure vs price ─────────────────────────
    {
        "type": "function",
        "function": {
            "name": "analyze_property_structure_price_impact",
            "description": (
                "Bundle analysis for how size and structure (Area, Floors, Bedrooms, Bathrooms, Frontage) "
                "associate with sale price. Use for the 'physical structure vs price' analytical intent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "price_column": {
                        "type": "string",
                        "description": "Price column name (default: Price)",
                        "default": "Price",
                    }
                },
                "required": [],
            },
        },
    },
    # ── 11. Intent bundle: bigger home = higher price? ─────────────────────────
    {
        "type": "function",
        "function": {
            "name": "analyze_bigger_house_premium",
            "description": (
                "Test the 'larger homes are always more expensive' assumption using area quantile bins, "
                "median price by bin, and simple exception rates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area_column": {
                        "type": "string",
                        "description": "Area column name (default: Area)",
                        "default": "Area",
                    },
                    "price_column": {
                        "type": "string",
                        "description": "Price column name (default: Price)",
                        "default": "Price",
                    },
                    "bins": {
                        "type": "integer",
                        "description": "Number of area quantile bins (default 5)",
                        "default": 5,
                    },
                },
                "required": [],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_province_ranking(top_n: int = 15, ascending: bool = False) -> dict[str, Any]:
    """
    Xếp hạng tỉnh/thành theo số lượng bất động sản.
    Tự động detect tên cột Province (hỗ trợ cả 'Province' và 'province').
    """
    df = _require_df()

    # Tìm cột Province (case-insensitive)
    province_col = None
    for col in df.columns:
        if col.lower() == "province":
            province_col = col
            break

    if province_col is None:
        return {
            "error": "Không tìm thấy cột 'Province' trong dataset.",
            "available_columns": list(df.columns)
        }

    counts = df[province_col].value_counts(ascending=ascending)
    total = len(df)

    if top_n and top_n > 0:
        counts = counts.head(top_n)

    records = [
        {
            "province": str(k),
            "count": int(v),
            "percentage": round(v / total * 100, 2)
        }
        for k, v in counts.items()
    ]

    top1 = records[0] if records else {}
    direction_label = "ít nhất" if ascending else "nhiều nhất"

    return {
        "tool": "get_province_ranking",
        "total_records": total,
        "total_provinces": int(df[province_col].nunique()),
        "top_n": top_n,
        "ascending": ascending,
        "records": records,
        "insight": (
            f"Dataset có {total:,} bất động sản tại {df[province_col].nunique()} tỉnh/thành. "
            f"Tỉnh/thành xuất hiện {direction_label}: **{top1.get('province')}** "
            f"với {top1.get('count', 0):,} bất động sản ({top1.get('percentage', 0)}%)."
        )
    }


def describe_numeric_column(column: str) -> dict[str, Any]:
    df = _require_df()
    if column not in df.columns:
        return {"error": f"Cột '{column}' không tồn tại. Các cột số: {list(df.select_dtypes(include='number').columns)}"}

    col = pd.to_numeric(df[column], errors="coerce").dropna()
    if col.empty:
        return {"error": f"Cột '{column}' không chứa dữ liệu số hợp lệ."}

    stats = {
        "tool": "describe_numeric_column",
        "column": column,
        "count": int(col.count()),
        "mean": round(float(col.mean()), 2),
        "median": round(float(col.median()), 2),
        "std": round(float(col.std()), 2),
        "min": round(float(col.min()), 2),
        "max": round(float(col.max()), 2),
        "q25": round(float(col.quantile(0.25)), 2),
        "q75": round(float(col.quantile(0.75)), 2),
        "missing": int(df[column].isna().sum()),
    }
    stats["insight"] = (
        f"Cột '{column}': trung bình {stats['mean']:,}, "
        f"trung vị {stats['median']:,}, "
        f"dao động từ {stats['min']:,} đến {stats['max']:,}."
    )
    return stats


def compare_mean_by_group(
    value_column: str,
    group_column: str,
    agg_func: str = "mean",
    top_n: int = 15
) -> dict[str, Any]:
    df = _require_df()
    for col in [value_column, group_column]:
        if col not in df.columns:
            return {"error": f"Cột '{col}' không tồn tại. Các cột hiện có: {list(df.columns)}"}

    numeric_col = pd.to_numeric(df[value_column], errors="coerce")
    grouped = numeric_col.groupby(df[group_column])

    agg_map = {
        "mean": grouped.mean,
        "median": grouped.median,
        "sum": grouped.sum,
        "count": grouped.count,
        "min": grouped.min,
        "max": grouped.max,
    }
    result_series = agg_map.get(agg_func, grouped.mean)().sort_values(ascending=False)
    top_series = result_series.head(top_n)

    records = [
        {"group": str(k), "value": round(float(v), 2)}
        for k, v in top_series.items()
    ]
    best = records[0]
    return {
        "tool": "compare_mean_by_group",
        "value_column": value_column,
        "group_column": group_column,
        "agg_func": agg_func,
        "records": records,
        "insight": (
            f"{agg_func.capitalize()} của '{value_column}' theo '{group_column}': "
            f"cao nhất là '{best['group']}' với {best['value']:,}."
        )
    }


def count_by_category(column: str, top_n: int = 20) -> dict[str, Any]:
    df = _require_df()
    if column not in df.columns:
        return {"error": f"Cột '{column}' không tồn tại. Các cột hiện có: {list(df.columns)}"}

    counts = df[column].value_counts()
    total = len(df)
    if top_n > 0:
        counts = counts.head(top_n)

    records = [
        {"category": str(k), "count": int(v), "percentage": round(v / total * 100, 2)}
        for k, v in counts.items()
    ]
    top = records[0]
    return {
        "tool": "count_by_category",
        "column": column,
        "total_records": total,
        "records": records,
        "insight": (
            f"'{column}' có {len(records)} nhóm. Nhiều nhất: '{top['category']}' "
            f"({top['count']:,} = {top['percentage']}%)."
        )
    }


def filter_and_summarize(
    filter_column: str,
    filter_value: Any,
    stat_column: str
) -> dict[str, Any]:
    df = _require_df()
    for col in [filter_column, stat_column]:
        if col not in df.columns:
            return {"error": f"Cột '{col}' không tồn tại. Các cột hiện có: {list(df.columns)}"}

    try:
        mask = df[filter_column] == filter_value
        if mask.sum() == 0:
            numeric_val = float(filter_value)  # type: ignore
            mask = pd.to_numeric(df[filter_column], errors="coerce") == numeric_val
    except (ValueError, TypeError):
        pass

    filtered = df[mask]
    if filtered.empty:
        return {
            "error": f"Không tìm thấy dữ liệu với {filter_column} = '{filter_value}'."
        }

    col_num = pd.to_numeric(filtered[stat_column], errors="coerce").dropna()
    stats = {
        "tool": "filter_and_summarize",
        "filter": f"{filter_column} = '{filter_value}'",
        "matched_records": int(mask.sum()),
        "stat_column": stat_column,
        "mean": round(float(col_num.mean()), 2),
        "median": round(float(col_num.median()), 2),
        "min": round(float(col_num.min()), 2),
        "max": round(float(col_num.max()), 2),
        "std": round(float(col_num.std()), 2),
    }
    stats["insight"] = (
        f"Với {filter_column}='{filter_value}' ({stats['matched_records']:,} bản ghi): "
        f"'{stat_column}' trung bình = {stats['mean']:,}, dao động {stats['min']:,} – {stats['max']:,}."
    )
    return stats


def detect_outliers(column: str) -> dict[str, Any]:
    df = _require_df()
    if column not in df.columns:
        return {"error": f"Cột '{column}' không tồn tại."}

    col = pd.to_numeric(df[column], errors="coerce").dropna()
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    outliers = col[(col < lower) | (col > upper)]
    pct = round(len(outliers) / len(col) * 100, 2)

    return {
        "tool": "detect_outliers",
        "column": column,
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(iqr), 2),
        "lower_bound": round(float(lower), 2),
        "upper_bound": round(float(upper), 2),
        "outlier_count": int(len(outliers)),
        "outlier_percentage": pct,
        "insight": (
            f"Cột '{column}': phát hiện {len(outliers):,} outlier ({pct}%) "
            f"nằm ngoài khoảng [{lower:.2f}, {upper:.2f}]."
        )
    }


def analyze_price_drivers(
    target_column: str = "Price", feature_columns: Optional[list[str]] = None
) -> dict[str, Any]:
    df = _require_df()
    default_features = ["Area", "Floors", "Bedrooms", "Bathrooms", "Frontage"]
    feature_columns = feature_columns or default_features

    if target_column not in df.columns:
        return {"error": f"Cột target '{target_column}' không tồn tại."}

    valid_features = [c for c in feature_columns if c in df.columns]
    if not valid_features:
        return {
            "error": "Không có feature hợp lệ để phân tích.",
            "available_columns": list(df.columns),
        }

    num_df = df[[target_column] + valid_features].copy()
    for col in num_df.columns:
        num_df[col] = pd.to_numeric(num_df[col], errors="coerce")
    num_df = num_df.dropna()
    if len(num_df) < 10:
        return {"error": "Không đủ dữ liệu số hợp lệ để phân tích price drivers."}

    correlations = {}
    slopes = {}
    for col in valid_features:
        cor = num_df[target_column].corr(num_df[col])
        correlations[col] = round(float(cor), 4) if pd.notna(cor) else None

        x = num_df[col].values
        y = num_df[target_column].values
        if np.std(x) == 0:
            slopes[col] = None
        else:
            slope = float(np.polyfit(x, y, 1)[0])
            slopes[col] = round(slope, 6)

    ranked = sorted(
        [
            {"feature": k, "correlation": v, "slope": slopes.get(k)}
            for k, v in correlations.items()
            if v is not None
        ],
        key=lambda it: abs(it["correlation"]),
        reverse=True,
    )

    top_feature = ranked[0]["feature"] if ranked else "N/A"
    top_corr = ranked[0]["correlation"] if ranked else 0

    return {
        "tool": "analyze_price_drivers",
        "target_column": target_column,
        "features_analyzed": valid_features,
        "sample_size": int(len(num_df)),
        "drivers_ranked": ranked,
        "key_metrics": {
            "top_driver": top_feature,
            "top_driver_correlation": top_corr,
        },
        "suggested_charts": ["scatter_2d_plot", "box_plot", "bar_plot"],
        "caveats": [
            "Tương quan không đồng nghĩa quan hệ nhân quả.",
            "Kết quả phụ thuộc chất lượng dữ liệu và biến bị thiếu.",
        ],
        "insight": (
            f"Trong {len(num_df):,} mẫu hợp lệ, biến liên quan mạnh nhất với '{target_column}' "
            f"là **{top_feature}** (corr={top_corr}). "
            "Nên kết hợp biểu đồ scatter và so sánh theo nhóm để kết luận chắc hơn."
        ),
    }


def price_vs_size_summary(
    area_column: str = "Area", price_column: str = "Price", bins: int = 5
) -> dict[str, Any]:
    df = _require_df()
    for col in [area_column, price_column]:
        if col not in df.columns:
            return {"error": f"Cột '{col}' không tồn tại."}

    work = df[[area_column, price_column]].copy()
    work[area_column] = pd.to_numeric(work[area_column], errors="coerce")
    work[price_column] = pd.to_numeric(work[price_column], errors="coerce")
    work = work.dropna()
    if len(work) < 10:
        return {"error": "Không đủ dữ liệu để phân tích quan hệ diện tích và giá."}

    bins = max(3, min(int(bins), 10))
    work["size_bin"] = pd.qcut(work[area_column], q=bins, duplicates="drop")
    grouped = (
        work.groupby("size_bin")[price_column]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    grouped["size_bin"] = grouped["size_bin"].astype(str)

    records = []
    for _, row in grouped.iterrows():
        records.append(
            {
                "size_bin": row["size_bin"],
                "count": int(row["count"]),
                "mean_price": round(float(row["mean"]), 2),
                "median_price": round(float(row["median"]), 2),
                "min_price": round(float(row["min"]), 2),
                "max_price": round(float(row["max"]), 2),
            }
        )

    medians = [r["median_price"] for r in records]
    monotonic_non_decreasing = all(
        medians[i] <= medians[i + 1] for i in range(len(medians) - 1)
    )

    return {
        "tool": "price_vs_size_summary",
        "area_column": area_column,
        "price_column": price_column,
        "bins_used": len(records),
        "records": records,
        "key_metrics": {
            "monotonic_non_decreasing_median_price": monotonic_non_decreasing,
            "lowest_bin_median_price": medians[0] if medians else None,
            "highest_bin_median_price": medians[-1] if medians else None,
        },
        "suggested_charts": ["bar_plot", "box_plot", "scatter_2d_plot"],
        "caveats": [
            "Giá theo nhóm diện tích có thể bị nhiễu bởi vị trí/pháp lý.",
            "Có thể xuất hiện ngoại lệ khiến xu hướng không đơn điệu.",
        ],
        "insight": (
            "Giá trung vị theo nhóm diện tích "
            + ("có xu hướng tăng đều." if monotonic_non_decreasing else "không tăng đều ở mọi nhóm.")
            + " Điều này cho thấy 'nhà to hơn' thường đắt hơn, nhưng không phải luôn luôn."
        ),
    }


def structure_group_price_compare(
    group_columns: Optional[list[str]] = None,
    price_column: str = "Price",
    top_n: int = 10,
) -> dict[str, Any]:
    df = _require_df()
    default_groups = ["Floors", "Bedrooms", "Bathrooms"]
    group_columns = group_columns or default_groups

    if price_column not in df.columns:
        return {"error": f"Cột giá '{price_column}' không tồn tại."}

    valid_groups = [c for c in group_columns if c in df.columns]
    if not valid_groups:
        return {"error": "Không có cột nhóm hợp lệ để so sánh."}

    price_series = pd.to_numeric(df[price_column], errors="coerce")
    top_n = max(3, min(int(top_n), 30))

    comparison = {}
    for col in valid_groups:
        gdf = pd.DataFrame({"group": df[col], "price": price_series}).dropna()
        grouped = (
            gdf.groupby("group")["price"]
            .agg(["count", "mean", "median"])
            .sort_values("median", ascending=False)
            .head(top_n)
            .reset_index()
        )
        rows = []
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "group": str(row["group"]),
                    "count": int(row["count"]),
                    "mean_price": round(float(row["mean"]), 2),
                    "median_price": round(float(row["median"]), 2),
                }
            )
        comparison[col] = rows

    strongest_col = max(
        comparison.keys(),
        key=lambda c: (comparison[c][0]["median_price"] - comparison[c][-1]["median_price"])
        if len(comparison[c]) >= 2
        else 0,
    )

    return {
        "tool": "structure_group_price_compare",
        "price_column": price_column,
        "group_columns": valid_groups,
        "comparison": comparison,
        "key_metrics": {
            "strongest_grouping_signal": strongest_col,
        },
        "suggested_charts": ["bar_plot", "box_plot"],
        "caveats": [
            "So sánh theo nhóm chưa kiểm soát đồng thời các biến khác.",
            "Nên kết hợp với tương quan đa biến để kết luận chắc chắn.",
        ],
        "insight": (
            f"So sánh theo nhóm cho thấy biến cấu trúc nổi bật nhất là **{strongest_col}** "
            "khi xét chênh lệch giá trung vị giữa các nhóm."
        ),
    }


def analyze_property_structure_price_impact(price_column: str = "Price") -> dict[str, Any]:
    """
    Intent 1: đo ảnh hưởng của kích thước/cấu trúc tới giá.
    """
    drivers = analyze_price_drivers(target_column=price_column)
    if "error" in drivers:
        return drivers

    structure = structure_group_price_compare(
        group_columns=["Floors", "Bedrooms", "Bathrooms", "Frontage"],
        price_column=price_column,
        top_n=10,
    )
    if "error" in structure:
        return structure

    return {
        "tool": "analyze_property_structure_price_impact",
        "price_column": price_column,
        "drivers_ranked": drivers.get("drivers_ranked", []),
        "group_comparison": structure.get("comparison", {}),
        "key_metrics": {
            "top_driver": drivers.get("key_metrics", {}).get("top_driver"),
            "top_driver_correlation": drivers.get("key_metrics", {}).get(
                "top_driver_correlation"
            ),
            "strongest_grouping_signal": structure.get("key_metrics", {}).get(
                "strongest_grouping_signal"
            ),
        },
        "suggested_charts": ["scatter_2d_plot", "box_plot", "bar_plot"],
        "chart_guidance": (
            "Ưu tiên scatter_2d_plot để xem quan hệ Area-Price; "
            "dùng box_plot/bar_plot để so sánh phân phối giá theo Floors/Bedrooms/Bathrooms/Frontage."
        ),
        "insight": (
            "Phân tích tổng hợp cho thấy giá chịu tác động bởi cả biến liên tục (diện tích) "
            "và biến cấu trúc theo nhóm (số tầng/phòng/mặt tiền)."
        ),
    }


def analyze_bigger_house_premium(
    area_column: str = "Area", price_column: str = "Price", bins: int = 5
) -> dict[str, Any]:
    """
    Intent 2: kiểm chứng 'nhà to hơn có luôn đắt hơn không'.
    """
    summary = price_vs_size_summary(
        area_column=area_column, price_column=price_column, bins=bins
    )
    if "error" in summary:
        return summary

    df = _require_df()
    work = df[[area_column, price_column]].copy()
    work[area_column] = pd.to_numeric(work[area_column], errors="coerce")
    work[price_column] = pd.to_numeric(work[price_column], errors="coerce")
    work = work.dropna().sort_values(area_column)
    if len(work) < 10:
        return {"error": "Không đủ dữ liệu để kiểm chứng giả định diện tích và giá."}

    # Tỷ lệ ngoại lệ cục bộ: điểm sau có diện tích lớn hơn nhưng giá thấp hơn điểm trước.
    area_vals = work[area_column].values
    price_vals = work[price_column].values
    violation_count = 0
    comparisons = 0
    for i in range(1, len(work)):
        if area_vals[i] > area_vals[i - 1]:
            comparisons += 1
            if price_vals[i] < price_vals[i - 1]:
                violation_count += 1
    violation_rate = round((violation_count / comparisons) * 100, 2) if comparisons else 0.0

    monotonic = summary.get("key_metrics", {}).get(
        "monotonic_non_decreasing_median_price", False
    )
    conclusion = (
        "Nhà lớn hơn thường có giá cao hơn, nhưng không phải lúc nào cũng đúng."
        if not monotonic or violation_rate > 0
        else "Trong dữ liệu hiện tại, giá trung vị tăng theo diện tích và ít thấy ngoại lệ."
    )

    return {
        "tool": "analyze_bigger_house_premium",
        "area_column": area_column,
        "price_column": price_column,
        "bins_used": summary.get("bins_used"),
        "size_price_summary": summary.get("records", []),
        "key_metrics": {
            "monotonic_non_decreasing_median_price": monotonic,
            "local_price_violation_rate_percent": violation_rate,
            "pairwise_comparisons": comparisons,
        },
        "suggested_charts": ["box_plot", "scatter_2d_plot", "bar_plot"],
        "chart_guidance": (
            "Dùng box_plot theo size_bin để kiểm tra chồng lấn phân phối giá; "
            "dùng scatter_2d_plot để nhìn ngoại lệ; bar_plot để tóm tắt median theo nhóm diện tích."
        ),
        "insight": conclusion,
    }


# ── Export cho registry ───────────────────────────────────────────────────────
TOOL_FUNCTIONS: dict[str, callable] = {
    "get_province_ranking": get_province_ranking,
    "describe_numeric_column": describe_numeric_column,
    "compare_mean_by_group": compare_mean_by_group,
    "count_by_category": count_by_category,
    "filter_and_summarize": filter_and_summarize,
    "detect_outliers": detect_outliers,
    "analyze_price_drivers": analyze_price_drivers,
    "price_vs_size_summary": price_vs_size_summary,
    "structure_group_price_compare": structure_group_price_compare,
    "analyze_property_structure_price_impact": analyze_property_structure_price_impact,
    "analyze_bigger_house_premium": analyze_bigger_house_premium,
}
