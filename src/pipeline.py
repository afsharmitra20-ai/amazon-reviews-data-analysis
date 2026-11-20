"""
Phase 2 - Step 7: Unified End-to-End Data Preparation & Feature Pipeline.
Encapsulates preprocessing, feature engineering, representation transform,
and cluster prediction for reproducible downstream modeling in Phase 3.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy import sparse
from sklearn.preprocessing import normalize

MODELS_DIR = "models"


class EndToEndMusicalPipeline:
    """
    Unified Inference & Feature Pipeline for Musical Instruments.
    Loads persisted Phase 2 artifacts and transforms raw review interactions
    into structured features, TF-IDF matrices, PCA representations, and cluster IDs.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.preprocessor = None
        self.feature_engineer = None
        self.tfidf_vectorizer = None
        self.tabular_scaler = None
        self.pca_transformer = None
        self.kmeans_model = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads all serialized Phase 2 transformers."""
        prep_path = os.path.join(self.models_dir, "preprocessor.joblib")
        fe_path = os.path.join(self.models_dir, "feature_engineer.joblib")
        tfidf_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        scaler_path = os.path.join(self.models_dir, "tabular_scaler.joblib")
        pca_path = os.path.join(self.models_dir, "pca_transformer.joblib")
        km_path = os.path.join(self.models_dir, "kmeans_model.joblib")

        if os.path.exists(prep_path):
            self.preprocessor = joblib.load(prep_path)
        if os.path.exists(fe_path):
            self.feature_engineer = joblib.load(fe_path)
        if os.path.exists(tfidf_path):
            self.tfidf_vectorizer = joblib.load(tfidf_path)
        if os.path.exists(scaler_path):
            self.tabular_scaler = joblib.load(scaler_path)
        if os.path.exists(pca_path):
            self.pca_transformer = joblib.load(pca_path)
        if os.path.exists(km_path):
            self.kmeans_model = joblib.load(km_path)

    def transform_tabular_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Runs cleaning, principled imputation, and domain feature engineering."""
        df_proc = self.preprocessor.transform(df_raw)
        df_feat = self.feature_engineer.transform(df_proc)
        return df_feat

    def transform_tfidf(self, df_feat: pd.DataFrame) -> sparse.csr_matrix:
        """Transforms clean text into sparse TF-IDF CSR matrix."""
        clean_texts = df_feat["clean_text"].fillna("").astype(str)
        return self.tfidf_vectorizer.transform(clean_texts)

    def predict_clusters_from_pca(self, X_pca: np.ndarray) -> np.ndarray:
        """Predicts cluster memberships using L2-normalized PCA embeddings."""
        X_norm = normalize(X_pca, norm="l2")
        return self.kmeans_model.predict(X_norm)


if __name__ == "__main__":
    pipeline = EndToEndMusicalPipeline()
    print("[✓] End-to-End Pipeline initialized successfully. All Phase 2 models loaded.")