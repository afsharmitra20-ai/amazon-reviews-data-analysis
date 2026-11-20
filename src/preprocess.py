"""
Phase 2 - Step 2: Principled Preprocessing Pipeline & Custom Transformers.
Compatible with Scikit-Learn. Enforces zero data leakage by learning
statistics (medians, scalers, vocabularies) strictly on the Training set.
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def extract_numeric_price(val):
    """Safely extracts numeric price float from raw strings or returns NaN."""
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


class MusicalInstrumentsPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end reproducible preprocessing transformer for Amazon Musical Instruments:
    1. Quarantines forbidden target/temporal leakage columns.
    2. Imputes missing prices using Training median + creates binary missingness indicator.
    3. Transforms skewed price via log1p and robust z-score standardization.
    4. Unifies text fields (title + body) and cleans whitespace.
    5. Categorizes high-cardinality stores into Top-K + 'Other'.
    """

    def __init__(self, top_k_stores: int = 50):
        self.top_k_stores = top_k_stores
        # Learned statistics on Train
        self.median_price_ = None
        self.price_mean_log_ = None
        self.price_std_log_ = None
        self.top_stores_ = None

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        
        # 1. Learn Price Imputation Median strictly on Training set
        numeric_prices = df["price_raw"].apply(extract_numeric_price)
        valid_prices = numeric_prices.dropna()
        self.median_price_ = float(valid_prices.median()) if len(valid_prices) > 0 else 19.99

        # 2. Learn Mean and Std of log-transformed prices for scaling
        imputed_prices = numeric_prices.fillna(self.median_price_)
        log_prices = np.log1p(imputed_prices)
        self.price_mean_log_ = float(log_prices.mean())
        self.price_std_log_ = float(log_prices.std()) if log_prices.std() > 0 else 1.0

        # 3. Learn Top-K Store Brands
        clean_stores = df["store"].fillna("Unknown_Store").astype(str).str.strip()
        store_counts = clean_stores.value_counts()
        self.top_stores_ = set(store_counts.head(self.top_k_stores).index)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # --- A. Sanitize and Exclude Forbidden Leakage Columns ---
        leakage_cols = ["average_rating", "rating_number", "helpful_vote"]
        for col in leakage_cols:
            if col in df.columns:
                df = df.drop(columns=[col])

        # --- B. Principled Missing Price Handling ---
        raw_numeric_prices = df["price_raw"].apply(extract_numeric_price)
        df["price_is_missing"] = raw_numeric_prices.isna().astype(int)
        
        # Impute with fitted median
        df["price_imputed"] = raw_numeric_prices.fillna(self.median_price_)
        
        # Log1p transformation & Standardization
        df["price_log"] = np.log1p(df["price_imputed"])
        df["price_scaled"] = (df["price_log"] - self.price_mean_log_) / self.price_std_log_

        # --- C. Text Concatenation & Hygiene ---
        title_clean = df["title_rev"].fillna("").astype(str).str.strip()
        body_clean = df["text"].fillna("").astype(str).str.strip()
        
        # Combine title and body into unified text
        df["clean_text"] = (title_clean + ". " + body_clean).str.strip()
        df["clean_text"] = df["clean_text"].apply(lambda s: re.sub(r"\s+", " ", s))

        # Basic surface text features
        df["char_count"] = df["clean_text"].str.len()
        df["word_count"] = df["clean_text"].apply(lambda s: len(s.split()))

        # --- D. Store Brand Normalization ---
        store_series = df["store"].fillna("Unknown_Store").astype(str).str.strip()
        df["store_normalized"] = store_series.apply(
            lambda s: s if s in self.top_stores_ else "Other"
        )

        # --- E. Verified Purchase Binary ---
        df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(int)

        return df