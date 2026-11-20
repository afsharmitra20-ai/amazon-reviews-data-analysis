"""
Step 1 of Phase 2: Leakage-Free Item-Disjoint Stratified Partitioning.
Splits 20,000 interactions into 70% Train, 15% Validation, 15% Test
grouped strictly on 'parent_asin' to prevent item/cold-start leakage.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_REVIEWS_PATH = os.path.join("data", "raw", "reviews_Musical_Instruments_sampled.parquet")
RAW_META_PATH = os.path.join("data", "raw", "meta_Musical_Instruments_sampled.parquet")
PROCESSED_DIR = os.path.join("data", "processed")
REPORTS_DIR = os.path.join("reports", "phase2")

# Critical Leakage Columns to be isolated/quarantined from features
LEAKAGE_COLUMNS = ["average_rating", "rating_number", "helpful_vote"]


def load_and_merge_raw():
    print("[*] Loading raw datasets...")
    df_rev = pd.read_parquet(RAW_REVIEWS_PATH)
    df_meta = pd.read_parquet(RAW_META_PATH)
    
    # Inner join on parent_asin (100% verified match from Phase 1)
    df = pd.merge(df_rev, df_meta, on="parent_asin", how="inner", suffixes=("_rev", "_meta"))
    print(f"[+] Total merged observations: {len(df):,} across {df['parent_asin'].nunique():,} unique products.")
    return df


def perform_item_disjoint_split(df: pd.DataFrame, seed: int = 42, 
                                train_size: float = 0.70, 
                                val_size: float = 0.15, 
                                test_size: float = 0.15):
    """
    Performs stratified item-level partitioning.
    Guarantees: set(train_items) ∩ set(val_items) ∩ set(test_items) == empty.
    """
    assert np.isclose(train_size + val_size + test_size, 1.0), "Splits must sum to 1.0"

    print(f"[*] Aggregating item-level profiles for stratified item partitioning (Seed={seed})...")
    # Compute representative rating per product for stratification
    item_stats = df.groupby("parent_asin").agg(
        review_count=("rating", "count"),
        mean_rating=("rating", "mean")
    ).reset_index()

    # Discretize item rating into 4 bins to ensure balanced distribution across sets
    item_stats["strat_bin"] = pd.qcut(item_stats["mean_rating"], q=4, labels=False, duplicates="drop")

    # Step 1: Split into Train and Temp (Val + Test)
    temp_size = val_size + test_size
    train_items_df, temp_items_df = train_test_split(
        item_stats,
        test_size=temp_size,
        stratify=item_stats["strat_bin"],
        random_state=seed
    )

    # Step 2: Split Temp into Validation and Test (equal halves: 0.15 / 0.30 = 0.5)
    val_ratio_within_temp = val_size / temp_size
    val_items_df, test_items_df = train_test_split(
        temp_items_df,
        test_size=(1.0 - val_ratio_within_temp),
        stratify=temp_items_df["strat_bin"],
        random_state=seed
    )

    train_items = set(train_items_df["parent_asin"])
    val_items = set(val_items_df["parent_asin"])
    test_items = set(test_items_df["parent_asin"])

    # Strict Zero-Leakage Invariance Checks
    assert len(train_items.intersection(val_items)) == 0, "FATAL: Train-Val item overlap detected!"
    assert len(train_items.intersection(test_items)) == 0, "FATAL: Train-Test item overlap detected!"
    assert len(val_items.intersection(test_items)) == 0, "FATAL: Val-Test item overlap detected!"

    # Map back to observation-level rows
    df_train = df[df["parent_asin"].isin(train_items)].copy().reset_index(drop=True)
    df_val = df[df["parent_asin"].isin(val_items)].copy().reset_index(drop=True)
    df_test = df[df["parent_asin"].isin(test_items)].copy().reset_index(drop=True)

    df_train["split_set"] = "train"
    df_val["split_set"] = "val"
    df_test["split_set"] = "test"

    return df_train, df_val, df_test


def audit_splits(df_train, df_val, df_test, out_report_path):
    """
    Computes rigorous metrics verifying split integrity, item isolation,
    and target class balance.
    """
    total_rows = len(df_train) + len(df_val) + len(df_test)
    
    train_items = set(df_train["parent_asin"])
    val_items = set(df_val["parent_asin"])
    test_items = set(df_test["parent_asin"])

    train_users = set(df_train["user_id"])
    val_users = set(df_val["user_id"])
    test_users = set(df_test["user_id"])

    # Rating distribution comparisons
    def get_dist(series):
        return series.value_counts(normalize=True).sort_index().round(4) * 100

    train_dist = get_dist(df_train["rating"])
    val_dist = get_dist(df_val["rating"])
    test_dist = get_dist(df_test["rating"])

    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write("=== PHASE 2 - STEP 1: ITEM-DISJOINT SPLIT AUDIT REPORT ===\n\n")
        f.write(f"Total Dataset Size: {total_rows:,} rows\n")
        f.write(f"• Train Set: {len(df_train):,} rows ({len(df_train)/total_rows*100:.2f}%) | {len(train_items):,} unique products\n")
        f.write(f"• Val Set:   {len(df_val):,} rows ({len(df_val)/total_rows*100:.2f}%) | {len(val_items):,} unique products\n")
        f.write(f"• Test Set:  {len(df_test):,} rows ({len(df_test)/total_rows*100:.2f}%) | {len(test_items):,} unique products\n\n")

        f.write("--- LEAKAGE SANITY VERIFICATION ---\n")
        f.write(f"• Item Overlap (Train ∩ Val):  {len(train_items.intersection(val_items))} items (MUST BE 0)\n")
        f.write(f"• Item Overlap (Train ∩ Test): {len(train_items.intersection(test_items))} items (MUST BE 0)\n")
        f.write(f"• Item Overlap (Val ∩ Test):   {len(val_items.intersection(test_items))} items (MUST BE 0)\n")
        f.write(f"• Zero-Leakage Status: PASSED (Strictly Item-Disjoint Cold-Start Partition)\n\n")

        f.write("--- USER OVERLAP INCIDENCE (OBSERVATIONAL AUDIT) ---\n")
        f.write(f"• User Overlap (Train ∩ Val):  {len(train_users.intersection(val_users))} users\n")
        f.write(f"• User Overlap (Train ∩ Test): {len(train_users.intersection(test_users))} users\n")
        f.write(f"• User Overlap (Val ∩ Test):   {len(val_users.intersection(test_users))} users\n\n")

        f.write("--- TARGET DISTRIBUTION BALANCE (PERCENTAGES) ---\n")
        f.write(f"{'Rating':<10}{'Train (%)':<15}{'Val (%)':<15}{'Test (%)':<15}\n")
        f.write("-" * 55 + "\n")
        for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
            t_pct = train_dist.get(r, 0.0)
            v_pct = val_dist.get(r, 0.0)
            te_pct = test_dist.get(r, 0.0)
            f.write(f"{int(r):<10}{t_pct:<15.2f}{v_pct:<15.2f}{te_pct:<15.2f}\n")

        f.write("\n--- QUARANTINED LEAKAGE COLUMNS (NEVER TO BE USED AS FEATURES) ---\n")
        for col in LEAKAGE_COLUMNS:
            f.write(f"• [FORBIDDEN FEATURE] {col}: Target / Temporal feedback leakage risk.\n")

    print(f"[✓] Full split audit saved to: {out_report_path}")
    print("\n" + open(out_report_path, "r", encoding="utf-8").read())


def main():
    parser = argparse.ArgumentParser(description="Leakage-Free Item-Disjoint Split Generator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df_merged = load_and_merge_raw()

    df_train, df_val, df_test = perform_item_disjoint_split(
        df_merged,
        seed=args.seed,
        train_size=args.train_ratio,
        val_size=args.val_ratio,
        test_size=args.test_ratio
    )

    # Save disjoint partitions as compressed Parquets
    train_path = os.path.join(PROCESSED_DIR, "train.parquet")
    val_path = os.path.join(PROCESSED_DIR, "val.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test.parquet")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    print(f"[✓] Partitions saved to {PROCESSED_DIR}:\n    - Train: {train_path}\n    - Val:   {val_path}\n    - Test:  {test_path}")

    # Generate and display audit report
    audit_report_path = os.path.join(REPORTS_DIR, "split_audit.txt")
    audit_splits(df_train, df_val, df_test, audit_report_path)


if __name__ == "__main__":
    main()