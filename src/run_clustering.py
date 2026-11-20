"""
Phase 2 - Step 6: Execution Pipeline for Clustering, Hyperparameter Selection,
Stability Verification, and Multimodal Cluster Interpretation.
Generates:
- figures/clustering_k_selection.png
- figures/clustering_manifold_2d.png
- figures/clustering_stability_ari.png
- figures/clustering_profiles_summary.png
- reports/phase2/clustering_metrics_by_k.csv
- reports/phase2/cluster_profiles.csv
- reports/phase2/clustering_audit.txt
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.manifold import TSNE
from clustering import UnsupervisedClusterAnalyzer

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
FIGURES_DIR = "figures"
REPORTS_DIR = os.path.join("reports", "phase2")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12.5,
    "figure.titlesize": 14
})


def plot_k_selection(k_df: pd.DataFrame, optimal_k: int, out_path: str):
    """Dual panel plot: Elbow (Inertia) and Silhouette Score across k."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Inertia Elbow
    ax1.plot(k_df["k"], k_df["inertia"], marker="o", lw=2, color="#1D3557")
    ax1.axvline(optimal_k, color="#E63946", linestyle="--", lw=1.8, label=f"Selected k* = {optimal_k}")
    ax1.set_title("Elbow Method (Within-Cluster Sum of Squares)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_xticks(k_df["k"])
    ax1.legend()

    # Right: Silhouette Score
    ax2.plot(k_df["k"], k_df["silhouette"], marker="s", lw=2, color="#2A9D8F")
    ax2.axvline(optimal_k, color="#E63946", linestyle="--", lw=1.8, label=f"Selected k* = {optimal_k}")
    opt_sil = k_df.loc[k_df["k"] == optimal_k, "silhouette"].values[0]
    ax2.scatter([optimal_k], [opt_sil], color="#E63946", s=90, zorder=5)
    ax2.set_title("Silhouette Score Across Candidate k")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Mean Silhouette Coefficient")
    ax2.set_xticks(k_df["k"])
    ax2.legend()

    plt.suptitle("Unsupervised Hyperparameter Selection (L2-Normalized 50-D Semantic Space)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved k-selection plot to: {out_path}")


def plot_cluster_manifold(X_train_pca: np.ndarray, labels: np.ndarray, profiles_df: pd.DataFrame, out_path: str):
    """Projects 4,000 samples via t-SNE and colors by discovered cluster ID + centroids."""
    sample_n = min(4000, len(X_train_pca))
    print(f"[*] Computing 2D t-SNE projection for cluster visualization (N={sample_n})...")
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate="auto",
        init="random",
        random_state=42,
        max_iter=1000
    )
    coords = tsne.fit_transform(X_train_pca[:sample_n, :30])
    sub_labels = labels[:sample_n]

    # Map cluster ID to short label with top 2 words
    label_map = {}
    for _, row in profiles_df.iterrows():
        cid = int(row["cluster_id"])
        top2 = ", ".join(row["top_distinctive_terms"].split(", ")[:2])
        label_map[cid] = f"C{cid}: {top2} ({row['mean_rating']:.2f}★)"

    df_plot = pd.DataFrame({
        "Dim 1": coords[:, 0],
        "Dim 2": coords[:, 1],
        "Cluster": [label_map[c] for c in sub_labels],
        "cid": sub_labels
    })

    plt.figure(figsize=(10, 6.8))
    palette = sns.color_palette("tab10", n_colors=len(profiles_df))
    sns.scatterplot(
        data=df_plot, x="Dim 1", y="Dim 2",
        hue="Cluster", hue_order=[label_map[c] for c in sorted(label_map.keys())],
        palette=palette, alpha=0.72, s=30, edgecolor="none"
    )

    # Annotate cluster centers in 2D space
    for cid in sorted(label_map.keys()):
        c_pts = df_plot[df_plot["cid"] == cid]
        cx, cy = c_pts["Dim 1"].mean(), c_pts["Dim 2"].mean()
        plt.text(
            cx, cy, f"C{cid}", fontsize=11, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.85)
        )

    plt.title("Discovered Unsupervised Clusters in 2D Semantic Manifold")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Cluster Signature & Mean Rating", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved 2D cluster manifold plot to: {out_path}")


def plot_stability_results(stab_df: pd.DataFrame, out_path: str):
    """Visualizes ARI and NMI stability across Seed, Subsample, and Scaling tests."""
    plt.figure(figsize=(9.5, 5))
    df_melt = stab_df.melt(
        id_vars=["perturbation_type", "setting"],
        value_vars=["ari", "nmi"],
        var_name="Metric",
        value_name="Score"
    )
    df_melt["Metric"] = df_melt["Metric"].str.upper()

    sns.barplot(
        data=df_melt, x="setting", y="Score", hue="Metric",
        palette=["#2B6CB0", "#48BB78"]
    )
    plt.axhline(0.75, color="darkred", linestyle="--", alpha=0.6, label="High Stability Threshold (0.75)")
    plt.ylim(0, 1.08)
    plt.xticks(rotation=32, ha="right")
    plt.title("Cluster Stability Analysis Across Seeds, 80% Subsamples & Scaling Shifts")
    plt.xlabel("Experimental Perturbation Condition")
    plt.ylabel("Agreement Score with Baseline Clustering (ARI / NMI)")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Saved stability analysis plot to: {out_path}")


def plot_cluster_profiles(profiles_df: pd.DataFrame, out_path: str):
    """Multi-panel comparison linking clusters to Star Rating, Critical %, Price, and Length."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    c_labels = [f"C{int(r['cluster_id'])}" for _, r in profiles_df.iterrows()]

    # Panel 1: Mean Rating & Critical Rate
    sns.barplot(
        x=c_labels, y=profiles_df["mean_rating"], hue=c_labels,
        palette="Blues_d", legend=False, ax=axes[0]
    )
    axes[0].set_ylim(1, 5.2)
    axes[0].set_title("Mean Star Rating by Cluster")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Mean Rating (1 - 5)")
    for i, v in enumerate(profiles_df["mean_rating"]):
        axes[0].text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=9.5, fontweight="bold")

    # Panel 2: Critical Complaint Rate (<= 2 stars)
    sns.barplot(
        x=c_labels, y=profiles_df["critical_rate_pct"], hue=c_labels,
        palette="Reds_d", legend=False, ax=axes[1]
    )
    axes[1].set_ylim(0, max(profiles_df["critical_rate_pct"]) * 1.25 + 2)
    axes[1].set_title("Dissatisfaction Rate (Rating ≤ 2 Stars)")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Critical Reviews (%)")
    for i, v in enumerate(profiles_df["critical_rate_pct"]):
        axes[1].text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9.5, fontweight="bold")

    # Panel 3: Median Product Price (USD)
    sns.barplot(
        x=c_labels, y=profiles_df["median_price_usd"], hue=c_labels,
        palette="Greens_d", legend=False, ax=axes[2]
    )
    axes[2].set_ylim(0, max(profiles_df["median_price_usd"]) * 1.25 + 2)
    axes[2].set_title("Median Product Price by Cluster ($)")
    axes[2].set_xlabel("Cluster")
    axes[2].set_ylabel("Median Price (USD)")
    for i, v in enumerate(profiles_df["median_price_usd"]):
        axes[2].text(i, v + 0.6, f"${v:.1f}", ha="center", fontsize=9.5, fontweight="bold")

    plt.suptitle("Multimodal Behavioral & Economic Profiles Across Discovered Clusters", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved multimodal cluster profile summary plot to: {out_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("[*] Loading 50-D PCA embeddings, TF-IDF matrices, and feature tables...")
    X_train_pca = np.load(os.path.join(PROCESSED_DIR, "train_pca50.npy"))
    X_val_pca = np.load(os.path.join(PROCESSED_DIR, "val_pca50.npy"))
    X_test_pca = np.load(os.path.join(PROCESSED_DIR, "test_pca50.npy"))

    X_train_tfidf = sparse.load_npz(os.path.join(PROCESSED_DIR, "train_tfidf.npz"))
    tfidf_vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    vocab = np.array(tfidf_vec.get_feature_names_out())

    df_train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_features.parquet"))

    # 1. Hyperparameter Selection
    analyzer = UnsupervisedClusterAnalyzer(k_range=(2, 10), random_state=42)
    k_metrics_df = analyzer.search_optimal_k(X_train_pca, min_domain_k=4)
    k_metrics_df.to_csv(os.path.join(REPORTS_DIR, "clustering_metrics_by_k.csv"), index=False)

    # 2. Fit Final Model on Train & Predict All Splits
    print(f"\n[*] Fitting final K-Means (k*={analyzer.optimal_k_}) strictly on Train...")
    train_labels = analyzer.fit_final_model(X_train_pca)
    val_labels = analyzer.predict(X_val_pca)
    test_labels = analyzer.predict(X_test_pca)

    np.save(os.path.join(PROCESSED_DIR, "train_clusters.npy"), train_labels)
    np.save(os.path.join(PROCESSED_DIR, "val_clusters.npy"), val_labels)
    np.save(os.path.join(PROCESSED_DIR, "test_clusters.npy"), test_labels)
    joblib.dump(analyzer.kmeans_model_, os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    print("[✓] Persisted cluster assignments and K-Means model.")

    # 3. Stability Analysis (Seed, Subsample, Scaling)
    print("\n[*] Running 3-Axis Stability Analysis (Seeds, 80% Subsamples, Scaling)...")
    stab_df = analyzer.evaluate_stability(X_train_pca, train_labels)
    stab_df.to_csv(os.path.join(REPORTS_DIR, "clustering_stability.csv"), index=False)

    # 4. Multimodal Cluster Profiling & Interpretation
    print("[*] Extracting differential TF-IDF keywords and multimodal cluster profiles...")
    profiles_df = analyzer.build_cluster_profiles(df_train, train_labels, X_train_tfidf, vocab)
    profiles_df.to_csv(os.path.join(REPORTS_DIR, "cluster_profiles.csv"), index=False)

    # 5. Generate Publication-Ready Visualizations
    plot_k_selection(k_metrics_df, analyzer.optimal_k_, os.path.join(FIGURES_DIR, "clustering_k_selection.png"))
    plot_cluster_manifold(X_train_pca, train_labels, profiles_df, os.path.join(FIGURES_DIR, "clustering_manifold_2d.png"))
    plot_stability_results(stab_df, os.path.join(FIGURES_DIR, "clustering_stability_ari.png"))
    plot_cluster_profiles(profiles_df, os.path.join(FIGURES_DIR, "clustering_profiles_summary.png"))

    # 6. Write Comprehensive Audit Report
    seed_ari_mean = stab_df[stab_df["perturbation_type"] == "Seed Variation"]["ari"].mean()
    sub_ari_mean = stab_df[stab_df["perturbation_type"] == "80% Subsample"]["ari"].mean()

    audit_path = os.path.join(REPORTS_DIR, "clustering_audit.txt")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("=== PHASE 2 - STEP 6: PATTERN DISCOVERY & CLUSTERING AUDIT REPORT ===\n\n")
        f.write("1. HYPERPARAMETER SELECTION JUSTIFICATION:\n")
        f.write(f"   - Evaluated Range: k in [2 .. 10] on L2-Normalized 50-D Semantic Embeddings.\n")
        f.write(f"   - Selected Optimal k*: {analyzer.optimal_k_}\n")
        f.write("   - Rationale: While k=2,3 capture coarse polarity/verbosity splits, selecting the Silhouette peak\n")
        f.write("     within k >= 4 resolves distinct musical product categories AND sentiment failure modes.\n\n")
        f.write(k_metrics_df.to_string(index=False) + "\n\n")

        f.write("2. PATTERN STABILITY VERIFICATION (ADJUSTED RAND INDEX & NMI):\n")
        f.write(f"   - Mean Seed Stability ARI (5 random seeds):       {seed_ari_mean:.4f}\n")
        f.write(f"   - Mean 80% Subsample Stability ARI (5 runs):      {sub_ari_mean:.4f}\n")
        f.write(stab_df.to_string(index=False) + "\n\n")

        f.write("3. MULTIMODAL CLUSTER PROFILES & LINK TO RESEARCH QUESTION:\n")
        for _, r in profiles_df.iterrows():
            f.write(f"   • Cluster C{int(r['cluster_id'])} ({int(r['count']):,} reviews | {r['share_pct']}% of Train):\n")
            f.write(f"     - Distinctive Terms: {r['top_distinctive_terms']}\n")
            f.write(f"     - Rating Profile:    Mean = {r['mean_rating']:.3f} ★ | Critical (<=2★) = {r['critical_rate_pct']}% | 5★ = {r['five_star_pct']}%\n")
            f.write(f"     - Price & Length:    Median Price = ${r['median_price_usd']:.2f} (Missing: {r['missing_price_pct']}%) | Mean Words = {r['mean_word_count']}\n")
            f.write(f"     - Domain Signals:    Audio Mentions = {r['audio_mentions']} | Build Mentions = {r['build_mentions']}\n\n")

    print(f"\n[✓] Full clustering audit report saved to: {audit_path}")
    print("\n" + open(audit_path, "r", encoding="utf-8").read())


if __name__ == "__main__":
    main()