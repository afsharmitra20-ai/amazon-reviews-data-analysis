"""
Phase 2 - Step 3: Domain-Specific Feature Engineering for Musical Instruments.
Extracts stylometric, musical lexicon, catalog metadata, and multimodal signals
compatible with Scikit-Learn transformers (fit on Train, transform on all splits).
Robust against NumPy array truth-value evaluations.
"""

import re
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Domain lexicons for Musical Instruments
AUDIO_KEYWORDS = re.compile(
    r"\b(sound|tone|noise|buzz|hum|pitch|tune|tuning|sustain|acoustic|volume|distortion|frequency)\b",
    re.IGNORECASE
)
BUILD_KEYWORDS = re.compile(
    r"\b(sturdy|cheap|plastic|wood|metal|broke|broken|defect|flimsy|solid|durability|craftsmanship)\b",
    re.IGNORECASE
)
VALUE_KEYWORDS = re.compile(
    r"\b(price|worth|expensive|cost|bargain|value|dollar|penny|money|overpriced)\b",
    re.IGNORECASE
)


def safe_json_count(val):
    """
    Counts items in serialized list, numpy array, or collection
    safely without ambiguous truth-value errors on arrays.
    """
    if val is None:
        return 0
    # Handle array-like or collection types directly first
    if isinstance(val, (list, tuple, dict)):
        return len(val)
    if isinstance(val, np.ndarray):
        return int(val.size)
    if isinstance(val, float) and np.isnan(val):
        return 0
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str in ["[]", "{}", '""', "None", "nan", "null"]:
            return 0
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, (list, tuple, dict)):
                return len(parsed)
            if isinstance(parsed, np.ndarray):
                return int(parsed.size)
            return 1 if parsed else 0
        except Exception:
            return 1 if len(val_str) > 0 else 0
    return 0


class MusicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Constructs domain-specific features without target leakage.
    Fitted statistics (like maximum anchor timestamp) are learned exclusively on Train.
    """

    def __init__(self):
        self.anchor_timestamp_ = None

    def fit(self, X: pd.DataFrame, y=None):
        # Anchor timestamp: max timestamp in the training split
        valid_ts = X["timestamp"].dropna()
        self.anchor_timestamp_ = int(valid_ts.max()) if len(valid_ts) > 0 else 1680000000000
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        texts = df["clean_text"].fillna("").astype(str)
        words_list = texts.apply(lambda s: s.split())
        word_counts = words_list.apply(len).clip(lower=1)  # avoid division by zero
        char_counts = texts.str.len().clip(lower=1)

        # --- A. Stylometric & Affective Features ---
        df["caps_ratio"] = texts.apply(lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1))
        df["exclamation_count"] = texts.apply(lambda s: s.count("!"))
        df["exclamation_density"] = df["exclamation_count"] / word_counts
        df["question_count"] = texts.apply(lambda s: s.count("?"))
        df["question_density"] = df["question_count"] / word_counts
        df["avg_word_length"] = char_counts / word_counts
        
        # Lexical Diversity (Type-Token Ratio)
        df["lexical_diversity"] = words_list.apply(lambda ws: len(set(ws)) / max(len(ws), 1))

        # --- B. Musical Domain Lexicon Features ---
        df["audio_quality_mentions"] = texts.apply(lambda s: len(AUDIO_KEYWORDS.findall(s)))
        df["build_durability_mentions"] = texts.apply(lambda s: len(BUILD_KEYWORDS.findall(s)))
        df["value_price_mentions"] = texts.apply(lambda s: len(VALUE_KEYWORDS.findall(s)))

        # --- C. Catalog & Multimodal Metadata Signals ---
        df["feature_bullets_count"] = df["features"].apply(safe_json_count)
        df["has_user_images"] = df["images_rev"].apply(lambda x: int(safe_json_count(x) > 0))
        df["has_product_images"] = df["images_meta"].apply(lambda x: int(safe_json_count(x) > 0))
        
        # --- D. Temporal Features ---
        # Review age in days relative to the training set anchor
        df["review_age_days"] = ((self.anchor_timestamp_ - df["timestamp"]) / (1000 * 60 * 60 * 24)).clip(lower=0)

        # --- E. Cross-Modality Interaction ---
        df["price_to_word_ratio"] = df["price_log"] / word_counts

        return df