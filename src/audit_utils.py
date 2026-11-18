"""
Data and Modality Audit Utilities for Amazon Reviews 2023 (Musical Instruments).
Provides robust schema audits, semantic missing value quantification, and modality checks.
"""

import json
import re
from typing import Any
import pandas as pd
import numpy as np


def clean_price_to_float(val: Any) -> float:
    """Extract numeric price from strings like '$19.99', '19.99', or return NaN."""
    if val is None or pd.isna(val):
        return np.nan
    try:
        if isinstance(val, (int, float)):
            return float(val) if not np.isnan(val) else np.nan
        s = str(val).strip()
        if s in ["", "—", "-", "None", "nan", "null"]:
            return np.nan
        match = re.search(r"(\d+(\.\d+)?)", s.replace(",", ""))
        if match:
            return float(match.group(1))
    except Exception:
        return np.nan
    return np.nan


def safe_nunique(series: pd.Series) -> int:
    """Safely calculate unique values across unhashable or nested types."""
    try:
        return int(series.nunique())
    except TypeError:
        return int(series.apply(lambda x: str(x) if x is not None else "").nunique())


def is_semantically_missing(val: Any) -> bool:
    """
    Check if a value is effectively missing (NaN, None, empty string,
    whitespace-only, dash placeholder, or empty serialized collection).
    """
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str):
        s = val.strip()
        if s in ["", "—", "-", "None", "nan", "null", "[]", "{}"]:
            return True
    if isinstance(val, (list, dict, tuple)) and len(val) == 0:
        return True
    if isinstance(val, np.ndarray) and val.size == 0:
        return True
    return False


def audit_table_schema(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Compute rigorous missingness by checking both physical NaNs
    and semantic empty placeholders (empty strings, empty JSONs).
    """
    records = []
    total = len(df)
    for col in df.columns:
        # Detect missing semantics
        missing_mask = df[col].apply(is_semantically_missing)
        n_missing = int(missing_mask.sum())
        pct_missing = (n_missing / total) * 100.0 if total > 0 else 0.0
        n_unique = safe_nunique(df[col])
        records.append({
            "table": table_name,
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": n_missing,
            "missing_pct": round(pct_missing, 2),
            "unique_count": n_unique
        })
    return pd.DataFrame(records)


def audit_reviews_schema(df_rev: pd.DataFrame) -> pd.DataFrame:
    return audit_table_schema(df_rev, "reviews")


def audit_meta_schema(df_meta: pd.DataFrame) -> pd.DataFrame:
    return audit_table_schema(df_meta, "metadata")


def parse_json_safely(val: Any) -> Any:
    """Safely deserialize JSON string or return native empty list/dict."""
    if val is None or is_semantically_missing(val):
        return []
    if isinstance(val, (list, dict, tuple)):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def safe_collection_len(val: Any) -> int:
    """Safely determine element length regardless of serialization."""
    parsed = parse_json_safely(val)
    if isinstance(parsed, (list, dict, tuple)):
        return len(parsed)
    return 1 if parsed else 0


def audit_modality_coverage(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate modality availability across the joined dataset:
    - Text modality
    - Review images
    - Product catalog images
    - Structured Metadata (price, store, features, average_rating)
    """
    total = len(df_merged)

    # 1. Text coverage
    has_review_text = ~df_merged["text"].apply(is_semantically_missing)
    has_review_title = ~df_merged["title_rev"].apply(is_semantically_missing)
    has_any_text = has_review_text | has_review_title

    # 2. Image coverage
    has_user_images = df_merged["images_rev"].apply(lambda x: len(parse_json_safely(x)) > 0)
    has_product_images = df_merged["images_meta"].apply(lambda x: len(parse_json_safely(x)) > 0)

    # 3. Structured / Metadata coverage
    has_numeric_price = ~df_merged["price_numeric"].isna()
    has_store = ~df_merged["store"].apply(is_semantically_missing)
    has_features = df_merged["features"].apply(lambda x: len(parse_json_safely(x)) > 0)
    has_product_rating = ~df_merged["average_rating"].isna()

    # Combinations
    text_and_price = has_any_text & has_numeric_price
    text_and_meta = has_any_text & has_numeric_price & has_store
    text_and_product_img = has_any_text & has_product_images
    full_multimodal = has_any_text & has_product_images & has_numeric_price

    metrics = [
        ("Review Text (Non-empty)", has_review_text.sum()),
        ("Review Title (Non-empty)", has_review_title.sum()),
        ("Any Text Available", has_any_text.sum()),
        ("User Uploaded Images", has_user_images.sum()),
        ("Product Catalog Images", has_product_images.sum()),
        ("Valid Numeric Price", has_numeric_price.sum()),
        ("Brand / Store Name", has_store.sum()),
        ("Product Feature List", has_features.sum()),
        ("Catalog Rating History (CRITICAL LEAKAGE RISKS)", has_product_rating.sum()),
        ("Modality: Text + Price", text_and_price.sum()),
        ("Modality: Text + Price + Store", text_and_meta.sum()),
        ("Modality: Text + Product Image", text_and_product_img.sum()),
        ("Full Multimodal (Text + Img + Price)", full_multimodal.sum()),
    ]

    records = []
    for label, count in metrics:
        pct = round((count / total) * 100.0, 2) if total > 0 else 0.0
        records.append({
            "modality_aspect": label,
            "available_count": int(count),
            "coverage_percentage": pct
        })

    return pd.DataFrame(records)