"""
Phase 2 - Step 6: Unsupervised Pattern Discovery, Parameter Selection & Stability Analysis.
Implements:
1. L2-Normalized K-Means (Cosine-equivalent semantic clustering) strictly fitted on Train.
2. Multi-metric hyperparameter search across k in [2..10] (Silhouette, Inertia, Calinski-Harabasz).
3. Rigorous 3-axis stability verification (Random Seeds, 80% Subsampling, Scaling shift) via ARI & NMI.
4. Differential TF-IDF keyword extraction & Multimodal cluster profiling.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


class UnsupervisedClusterAnalyzer:
    """
    End-to-end unsupervised clustering, hyperparameter tuning, stability testing,
    and multimodal domain profiling.
    """

    def __init__(self, k_range: Tuple[int, int] = (2, 10), random_state: int = 42):
        self.k_min, self.k_max = k_range
        self.random_state = random_state
        self.optimal_k_ = None
        self.kmeans_model_ = None
        self.k_metrics_df_ = None

    def prepare_features(self, X_pca: np.ndarray) -> np.ndarray:
        """L2-normalizes PCA embeddings so Euclidean K-Means optimizes cosine similarity."""
        return normalize(X_pca, norm="l2")

    def search_optimal_k(self, X_train_pca: np.ndarray, min_domain_k: int = 4, sil_sample_size: int = 6000) -> pd.DataFrame:
        """
        Evaluates k across [k_min, k_max] using Inertia, Silhouette, and Calinski-Harabasz.
        Selects optimal k maximizing Silhouette within meaningful domain granularity (k >= min_domain_k).
        """
        X_norm = self.prepare_features(X_train_pca)
        records = []

        print(f"[*] Evaluating K-Means across k = {self.k_min} to {self.k_max}...")
        for k in range(self.k_min, self.k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X_norm)

            sil = silhouette_score(
                X_norm, labels,
                sample_size=min(sil_sample_size, len(X_norm)),
                random_state=self.random_state
            )
            ch = calinski_harabasz_score(X_norm, labels)
            min_cluster_pct = (pd.Series(labels).value_counts(normalize=True).min()) * 100

            records.append({
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(sil),
                "calinski_harabasz": float(ch),
                "smallest_cluster_pct": round(float(min_cluster_pct), 2)
            })
            print(f"    -> k={k:2d} | Silhouette: {sil:.4f} | Inertia: {km.inertia_:8.1f} | CH: {ch:7.1f} | Min Cluster: {min_cluster_pct:.1f}%")

        self.k_metrics_df_ = pd.DataFrame(records)

        # Select optimal k in non-trivial regime (k >= min_domain_k) with healthy cluster sizes (>= 3%)
        candidates = self.k_metrics_df_[
            (self.k_metrics_df_["k"] >= min_domain_k) &
            (self.k_metrics_df_["smallest_cluster_pct"] >= 3.0)
        ]
        if candidates.empty:
            candidates = self.k_metrics_df_[self.k_metrics_df_["k"] >= min_domain_k]

        best_row = candidates.loc[candidates["silhouette"].idxmax()]
        self.optimal_k_ = int(best_row["k"])
        print(f"[+] Selected Optimal k* = {self.optimal_k_} (Silhouette = {best_row['silhouette']:.4f})")
        return self.k_metrics_df_

    def fit_final_model(self, X_train_pca: np.ndarray) -> np.ndarray:
        """Fits final K-Means model with optimal_k_ strictly on Training set."""
        if self.optimal_k_ is None:
            raise ValueError("Call search_optimal_k before fit_final_model.")
        X_norm = self.prepare_features(X_train_pca)
        self.kmeans_model_ = KMeans(
            n_clusters=self.optimal_k_,
            n_init=20,
            random_state=self.random_state
        )
        return self.kmeans_model_.fit_predict(X_norm)

    def predict(self, X_pca: np.ndarray) -> np.ndarray:
        """Assigns unseen Validation/Test instances to nearest learned cluster centroids."""
        X_norm = self.prepare_features(X_pca)
        return self.kmeans_model_.predict(X_norm)

    def evaluate_stability(self, X_train_pca: np.ndarray, base_labels: np.ndarray) -> pd.DataFrame:
        """
        Tests pattern stability across 3 dimensions required by guidelines:
        1. Random Seed Perturbations (5 seeds)
        2. 80% Subsampling Bootstrap (5 subsamples)
        3. Feature Scaling Shift (Raw PCA & StandardScaled PCA vs L2-Normalized PCA)
        """
        X_norm = self.prepare_features(X_train_pca)
        k = self.optimal_k_
        results = []

        # 1. Seed Stability
        test_seeds = [7, 100, 2023, 777, 999]
        for s in test_seeds:
            km_seed = KMeans(n_clusters=k, n_init=10, random_state=s)
            pred_seed = km_seed.fit_predict(X_norm)
            results.append({
                "perturbation_type": "Seed Variation",
                "setting": f"seed={s}",
                "ari": float(adjusted_rand_score(base_labels, pred_seed)),
                "nmi": float(normalized_mutual_info_score(base_labels, pred_seed))
            })

        # 2. Subsample (80%) Stability
        rng = np.random.default_rng(self.random_state)
        n_total = len(X_norm)
        sub_size = int(0.80 * n_total)
        for i in range(1, 6):
            sub_idx = rng.choice(n_total, size=sub_size, replace=False)
            km_sub = KMeans(n_clusters=k, n_init=10, random_state=self.random_state + i)
            km_sub.fit(X_norm[sub_idx])
            # Predict full train labels from subsample-learned centroids
            pred_full_from_sub = km_sub.predict(X_norm)
            results.append({
                "perturbation_type": "80% Subsample",
                "setting": f"subsample_run_{i}",
                "ari": float(adjusted_rand_score(base_labels, pred_full_from_sub)),
                "nmi": float(normalized_mutual_info_score(base_labels, pred_full_from_sub))
            })

        # 3. Scaling Variations
        # 3a. Raw unnormalized PCA
        km_raw = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        pred_raw = km_raw.fit_predict(X_train_pca)
        results.append({
            "perturbation_type": "Scaling Shift",
            "setting": "Unnormalized_PCA",
            "ari": float(adjusted_rand_score(base_labels, pred_raw)),
            "nmi": float(normalized_mutual_info_score(base_labels, pred_raw))
        })

        # 3b. Z-Score Standardized PCA (equal variance weighting per PC)
        X_zscore = StandardScaler().fit_transform(X_train_pca)
        km_z = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
        pred_z = km_z.fit_predict(X_zscore)
        results.append({
            "perturbation_type": "Scaling Shift",
            "setting": "ZScore_EqualVar_PCA",
            "ari": float(adjusted_rand_score(base_labels, pred_z)),
            "nmi": float(normalized_mutual_info_score(base_labels, pred_z))
        })

        return pd.DataFrame(results)

    @staticmethod
    def build_cluster_profiles(
        df_train: pd.DataFrame,
        labels: np.ndarray,
        X_train_tfidf: sparse.csr_matrix,
        vocab: np.ndarray,
        top_n_words: int = 8
    ) -> pd.DataFrame:
        """
        Extracts Differential TF-IDF signature terms and multimodal behavioral metrics per cluster.
        """
        df_work = df_train.copy()
        df_work["cluster"] = labels

        # Ignore HTML artifact tokens like 'br'
        ignore_tokens = {"br", "quot", "amp_html"}
        global_tfidf_mean = np.asarray(X_train_tfidf.mean(axis=0)).flatten()

        profiles = []
        total_n = len(df_work)

        for c in sorted(np.unique(labels)):
            mask = (labels == c)
            sub_df = df_work[mask]
            c_size = int(mask.sum())

            # Differential TF-IDF: words disproportionately enriched in cluster c
            cluster_tfidf_mean = np.asarray(X_train_tfidf[mask].mean(axis=0)).flatten()
            diff_scores = cluster_tfidf_mean - global_tfidf_mean
            sorted_idx = diff_scores.argsort()[::-1]

            top_terms: List[str] = []
            for idx in sorted_idx:
                tok = str(vocab[idx])
                if tok not in ignore_tokens and not tok.startswith("br "):
                    top_terms.append(tok)
                if len(top_terms) >= top_n_words:
                    break

            profiles.append({
                "cluster_id": int(c),
                "count": c_size,
                "share_pct": round((c_size / total_n) * 100, 2),
                "mean_rating": round(float(sub_df["rating"].mean()), 3),
                "critical_rate_pct": round(float((sub_df["rating"] <= 2).mean() * 100), 2),
                "five_star_pct": round(float((sub_df["rating"] == 5).mean() * 100), 2),
                "median_price_usd": round(float(sub_df["price_imputed"].median()), 2),
                "mean_price_usd": round(float(sub_df["price_imputed"].mean()), 2),
                "missing_price_pct": round(float(sub_df["price_is_missing"].mean() * 100), 2),
                "mean_word_count": round(float(sub_df["word_count"].mean()), 1),
                "audio_mentions": round(float(sub_df["audio_quality_mentions"].mean()), 2),
                "build_mentions": round(float(sub_df["build_durability_mentions"].mean()), 2),
                "verified_pct": round(float(sub_df["verified_purchase"].mean() * 100), 1),
                "top_distinctive_terms": ", ".join(top_terms)
            })

        return pd.DataFrame(profiles)