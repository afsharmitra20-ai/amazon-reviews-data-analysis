"""
Phase 2 - Step 4: Text and Multimodal Representation Builder.
Builds:
1. Classical Sparse Lexical Representation (TF-IDF with n-grams (1, 2))
2. Dense Pre-trained Semantic Embeddings (Sentence-Transformers MiniLM-L6-v2)
3. Normalized Tabular Domain-Feature Matrix
Strictly enforces Zero Data Leakage: TF-IDF vocab is fitted exclusively on Train.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer


class MultimodalRepresentationBuilder:
    """
    Constructs both sparse lexical (TF-IDF) and dense semantic (MiniLM)
    representations along with standardized tabular feature matrices.
    """

    def __init__(self, max_tfidf_features: int = 5000, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.max_tfidf_features = max_tfidf_features
        self.embedding_model_name = embedding_model_name
        
        # Models
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            min_df=3
        )
        self.embedding_model = None
        self.tabular_scaler = StandardScaler()
        self.tabular_feature_cols = []

    def fit_tfidf(self, train_texts: pd.Series):
        """Fit TF-IDF vocabulary strictly on Training set."""
        print(f"[*] Fitting TF-IDF Vectorizer (max_features={self.max_tfidf_features}) strictly on Train...")
        clean_texts = train_texts.fillna("").astype(str)
        self.tfidf_vectorizer.fit(clean_texts)
        print(f"    -> Vocabulary size learned: {len(self.tfidf_vectorizer.vocabulary_):,} tokens.")
        return self

    def transform_tfidf(self, texts: pd.Series) -> sparse.csr_matrix:
        """Transforms text series into sparse TF-IDF CSR matrix."""
        clean_texts = texts.fillna("").astype(str)
        return self.tfidf_vectorizer.transform(clean_texts)

    def extract_dense_embeddings(self, texts: pd.Series, batch_size: int = 64) -> np.ndarray:
        """
        Extracts 384-dimensional dense semantic embeddings using MiniLM.
        Independent per instance (no leakage vector).
        """
        if self.embedding_model is None:
            print(f"[*] Loading Pre-trained SentenceTransformer: '{self.embedding_model_name}'...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

        clean_texts = texts.fillna("").astype(str).tolist()
        print(f"[*] Encoding {len(clean_texts):,} texts into 384-d dense vectors (batch_size={batch_size})...")
        embeddings = self.embedding_model.encode(
            clean_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return np.asarray(embeddings, dtype=np.float32)

    def fit_transform_tabular(self, df_train: pd.DataFrame, 
                             df_val: pd.DataFrame, 
                             df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts and standardizes numeric engineered features using Train statistics.
        """
        self.tabular_feature_cols = [
            "price_imputed", "price_is_missing", "price_scaled",
            "caps_ratio", "exclamation_density", "question_density",
            "avg_word_length", "lexical_diversity", "audio_quality_mentions",
            "build_durability_mentions", "value_price_mentions",
            "feature_bullets_count", "has_user_images", "has_product_images",
            "review_age_days", "price_to_word_ratio", "verified_purchase"
        ]
        
        print(f"[*] Standardizing {len(self.tabular_feature_cols)} tabular features using Train parameters...")
        X_train_num = df_train[self.tabular_feature_cols].fillna(0).values
        X_val_num = df_val[self.tabular_feature_cols].fillna(0).values
        X_test_num = df_test[self.tabular_feature_cols].fillna(0).values

        # Fit strictly on Train
        self.tabular_scaler.fit(X_train_num)
        
        return (
            self.tabular_scaler.transform(X_train_num),
            self.tabular_scaler.transform(X_val_num),
            self.tabular_scaler.transform(X_test_num)
        )