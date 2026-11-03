"""
Step 3: Exploratory Data Analysis (EDA) for Amazon Reviews 2023 (Musical Instruments).
Generates publication-quality charts for univariate and multimodal multivariate distributions.
Robust handling of numpy arrays and JSON data.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual style setup
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})

RAW_REVIEWS_PATH = os.path.join("data", "raw", "reviews_Musical_Instruments_sampled.parquet")
RAW_META_PATH = os.path.join("data", "raw", "meta_Musical_Instruments_sampled.parquet")
FIGURES_DIR = "figures"
REPORTS_DIR = os.path.join("reports", "phase1")


def clean_price(val):
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


def safe_json_len(val):
    """Safely calculate collection length without boolean evaluation on arrays."""
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


def load_and_enrich_data():
    print("[*] Loading datasets for EDA...")
    df_rev = pd.read_parquet(RAW_REVIEWS_PATH)
    df_meta = pd.read_parquet(RAW_META_PATH)

    df_meta["price_numeric"] = df_meta["price_raw"].apply(clean_price)

    # Merge
    df = pd.merge(df_rev, df_meta, on="parent_asin", how="inner", suffixes=("_rev", "_meta"))

    # Feature enrichment for EDA
    df["review_char_len"] = df["text"].fillna("").astype(str).str.len()
    df["review_word_count"] = df["text"].fillna("").astype(str).apply(lambda s: len(s.split()))
    df["has_user_image"] = df["images_rev"].apply(lambda x: safe_json_len(x) > 0)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df["year"] = df["datetime"].dt.year
    df["is_helpful"] = df["helpful_vote"] > 0

    return df


def plot_rating_distribution(df, out_path):
    plt.figure(figsize=(7, 4.5))
    rating_counts = df["rating"].value_counts().sort_index()
    total = len(df)
    
    ax = sns.barplot(x=rating_counts.index.astype(int), y=rating_counts.values, palette="Blues_d")
    plt.title("Distribution of Star Ratings (Imbalance Audit)")
    plt.xlabel("Star Rating (1 - 5)")
    plt.ylabel("Number of Reviews")
    plt.ylim(0, max(rating_counts.values) * 1.15)

    for p in ax.patches:
        h = p.get_height()
        pct = (h / total) * 100
        ax.annotate(f"{int(h):,}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., h + total * 0.01),
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Saved: {out_path}")


def plot_review_length_distribution(df, out_path):
    plt.figure(figsize=(8, 4.5))
    q99 = df["review_word_count"].quantile(0.99)
    subset = df[df["review_word_count"] <= q99]

    sns.histplot(subset["review_word_count"], bins=40, kde=True, color="teal")
    plt.title(f"Review Word Count Distribution (Up to 99th Percentile: {int(q99)} words)")
    plt.xlabel("Word Count per Review")
    plt.ylabel("Frequency")
    median_val = df["review_word_count"].median()
    mean_val = df["review_word_count"].mean()

    plt.axvline(median_val, color="red", linestyle="--", label=f"Median: {median_val:.0f} words")
    plt.axvline(mean_val, color="darkorange", linestyle=":", label=f"Mean: {mean_val:.1f} words")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Saved: {out_path}")


def plot_length_vs_rating(df, out_path):
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x="rating",
        y="review_word_count",
        data=df,
        showfliers=False,
        palette="Spectral"
    )
    plt.title("Review Length (Word Count) vs. Rating")
    plt.xlabel("Star Rating")
    plt.ylabel("Word Count (Outliers suppressed for clarity)")

    means = df.groupby("rating")["review_word_count"].mean()
    plt.plot(range(len(means)), means.values, color="black", marker="o", linestyle="--", label="Mean Word Count")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Saved: {out_path}")


def plot_temporal_trend(df, out_path):
    plt.figure(figsize=(9, 4.5))
    yearly_counts = df.dropna(subset=["year"]).groupby("year").size()
    yearly_counts = yearly_counts[yearly_counts.index >= 2010]

    ax = sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker="o", color="navy", lw=2)
    plt.title("Temporal Evolution of Reviews (2010 - 2023)")
    plt.xlabel("Year")
    plt.ylabel("Annual Review Count")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Saved: {out_path}")


def plot_helpful_multimodal_interaction(df, out_path):
    plt.figure(figsize=(8, 5))
    df["has_img_label"] = df["has_user_image"].map({True: "With User Photos", False: "Text Only"})

    rate_df = df.groupby("has_img_label")["is_helpful"].mean().reset_index()
    rate_df["is_helpful"] = rate_df["is_helpful"] * 100

    ax = sns.barplot(x="has_img_label", y="is_helpful", data=rate_df, palette="Pastel1")
    plt.title("Impact of Modality on Review Helpfulness (% Helpful Votes > 0)")
    plt.xlabel("Review Modality")
    plt.ylabel("Percentage Receiving Helpful Votes (%)")
    plt.ylim(0, max(rate_df["is_helpful"]) * 1.3)

    for p in ax.patches:
        val = p.get_height()
        ax.annotate(f"{val:.1f}%", (p.get_x() + p.get_width() / 2., val + 1),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Saved: {out_path}")


def generate_eda_summary(df, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== STATISTICAL SUMMARY OF EXPLORATORY DATA ANALYSIS ===\n\n")

        # Rating stats
        f.write("1. RATING DISTRIBUTION & CLASS IMBALANCE:\n")
        val_counts = df["rating"].value_counts(normalize=True).sort_index() * 100
        for r, pct in val_counts.items():
            f.write(f"   - Rating {int(r)} Stars: {pct:.2f}%\n")
        f.write(f"   -> Positive Bias (Rating >= 4): {(df['rating'] >= 4).mean()*100:.2f}%\n\n")

        # Text length stats
        f.write("2. TEXT LENGTH CHARACTERISTICS (WORDS):\n")
        f.write(f"   - Mean Word Count: {df['review_word_count'].mean():.2f}\n")
        f.write(f"   - Median Word Count: {df['review_word_count'].median():.2f}\n")
        f.write(f"   - Standard Deviation: {df['review_word_count'].std():.2f}\n")
        f.write(f"   - Short Reviews (<= 5 words): {(df['review_word_count'] <= 5).sum()} ({(df['review_word_count'] <= 5).mean()*100:.2f}%)\n")
        f.write(f"   - Long Detailed Reviews (>= 100 words): {(df['review_word_count'] >= 100).sum()} ({(df['review_word_count'] >= 100).mean()*100:.2f}%)\n\n")

        # Price stats
        f.write("3. PRODUCT PRICING (USD):\n")
        valid_prices = df["price_numeric"].dropna()
        f.write(f"   - Available price records: {len(valid_prices):,}\n")
        f.write(f"   - Median Price: ${valid_prices.median():.2f}\n")
        f.write(f"   - Mean Price: ${valid_prices.mean():.2f}\n")
        f.write(f"   - 25th - 75th Percentile: ${valid_prices.quantile(0.25):.2f} - ${valid_prices.quantile(0.75):.2f}\n")
        f.write(f"   - Max Price: ${valid_prices.max():.2f}\n\n")

        # Helpfulness stats
        f.write("4. HELPFULNESS & MULTIMODAL CORRELATIONS:\n")
        f.write(f"   - Overall Helpfulness Ratio: {(df['helpful_vote'] > 0).mean()*100:.2f}%\n")
        with_img_helpful = df[df['has_user_image']]['is_helpful'].mean() * 100
        no_img_helpful = df[~df['has_user_image']]['is_helpful'].mean() * 100
        f.write(f"   - Helpfulness for Reviews WITH user images: {with_img_helpful:.2f}%\n")
        f.write(f"   - Helpfulness for Reviews WITHOUT images: {no_img_helpful:.2f}%\n")
        
        corr = df[["rating", "review_word_count", "price_numeric", "helpful_vote"]].corr()
        f.write(f"\nPearson Correlation Matrix:\n{corr.to_string()}\n")

    print(f"[✓] EDA Statistical Summary written to: {out_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = load_and_enrich_data()

    print("[*] Generating Publication-Ready Figures...")
    plot_rating_distribution(df, os.path.join(FIGURES_DIR, "eda_rating_distribution.png"))
    plot_review_length_distribution(df, os.path.join(FIGURES_DIR, "eda_word_count_distribution.png"))
    plot_length_vs_rating(df, os.path.join(FIGURES_DIR, "eda_length_vs_rating.png"))
    plot_temporal_trend(df, os.path.join(FIGURES_DIR, "eda_temporal_trend.png"))
    plot_helpful_multimodal_interaction(df, os.path.join(FIGURES_DIR, "eda_modality_helpfulness.png"))

    generate_eda_summary(df, os.path.join(REPORTS_DIR, "eda_summary.txt"))


if __name__ == "__main__":
    main()