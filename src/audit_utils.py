"""
Data and Modality Audit Utilities for Amazon Reviews 2023 (Musical Instruments).
Provides robust checks for missingness, duplication, schema integrity, and modality availability.
Handles numpy arrays, lists, and unhashable types safely.
"""

import json
import re
from typing import Any
import pandas as pd
import numpy as np


def clean_price_to_float(val: Any) -> float:
    """Extract numeric price from strings like '$19.99', '19.99', or handle missing/dash."""
    if val is None:
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
    """Safely calculate unique values even if series contains lists, dicts, or ndarrays."""
    try:
        return int(series.nunique())
    except TypeError:
        return int(series.apply(lambda x: str(x) if x is not None else "").nunique())


def audit_reviews_schema(df_rev: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values, unique counts, and types for reviews."""
    records = []
    total = len(df_rev)
    for col in df_rev.columns:
        n_missing = int(df_rev[col].isna().sum())
        pct_missing = (n_missing / total) * 100.0 if total > 0 else 0.0
        n_unique = safe_nunique(df_rev[col])
        records.append({
            "table": "reviews",
            "column": col,
            "dtype": str(df_rev[col].dtype),
            "missing_count": n_missing,
            "missing_pct": round(pct_missing, 2),
            "unique_count": n_unique
        })
    return pd.DataFrame(records)


def audit_meta_schema(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values, unique counts, and types for metadata."""
    records = []
    total = len(df_meta)
    for col in df_meta.columns:
        n_missing = int(df_meta[col].isna().sum())
        pct_missing = (n_missing / total) * 100.0 if total > 0 else 0.0
        n_unique = safe_nunique(df_meta[col])
        records.append({
            "table": "metadata",
            "column": col,
            "dtype": str(df_meta[col].dtype),
            "missing_count": n_missing,
            "missing_pct": round(pct_missing, 2),
            "unique_count": n_unique
        })
    return pd.DataFrame(records)


def safe_collection_len(val: Any) -> int:
    """
    Safely determine the length of an element whether it's a list, numpy array,
    dictionary, or a JSON-serialized string.
    """
    if val is None:
        return 0
    if isinstance(val, (list, tuple, dict)):
        return len(val)
    if isinstance(val, np.ndarray):
        return int(val.size)
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str in ["[]", "{}", '""']:
            return 0
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, (list, dict, tuple)):
                return len(parsed)
            return 1 if parsed else 0
        except Exception:
            return 1 if len(val_str) > 0 else 0
    return 1


def audit_modality_coverage(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate modality availability across the joined dataset:
    - Text modality (review body & title)
    - Review images (user-submitted)
    - Product images (catalog)
    - Tabular/Metadata (price, store, features, average_rating)
    """
    total = len(df_merged)

    # 1. Text coverage
    has_review_text = df_merged["text"].fillna("").astype(str).str.strip().str.len() > 0
    has_review_title = df_merged["title_rev"].fillna("").astype(str).str.strip().str.len() > 0
    has_any_text = has_review_text | has_review_title

    # 2. Image coverage (Robust check)
    has_user_images = df_merged["images_rev"].apply(lambda x: safe_collection_len(x) > 0)
    has_product_images = df_merged["images_meta"].apply(lambda x: safe_collection_len(x) > 0)

    # 3. Structured / Metadata coverage
    has_numeric_price = ~df_merged["price_numeric"].isna()
    has_store = df_merged["store"].fillna("").astype(str).str.strip().str.len() > 0
    has_features = df_merged["features"].apply(lambda x: safe_collection_len(x) > 0)
    has_product_rating = ~df_merged["average_rating"].isna()

    # Multimodal combinations
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
        ("Catalog Rating History", has_product_rating.sum()),
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