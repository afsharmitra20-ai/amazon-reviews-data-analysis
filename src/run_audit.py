"""
Execution pipeline for Step 2: Comprehensive Data & Modality Audit.
Generates structured audit logs, checks duplicate interactions,
and verifies join integrity between reviews and product metadata.
"""

import os
import pandas as pd
from audit_utils import (
    clean_price_to_float,
    audit_reviews_schema,
    audit_meta_schema,
    audit_modality_coverage
)

REVIEWS_FILE = os.path.join("data", "raw", "reviews_Musical_Instruments_sampled.parquet")
META_FILE = os.path.join("data", "raw", "meta_Musical_Instruments_sampled.parquet")
OUTPUT_DIR = os.path.join("reports", "phase1")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[*] Starting Data and Modality Audit...")

    # 1. Load Parquet datasets
    print(f"[*] Reading reviews: {REVIEWS_FILE}")
    df_rev = pd.read_parquet(REVIEWS_FILE)
    print(f"[*] Reading metadata: {META_FILE}")
    df_meta = pd.read_parquet(META_FILE)

    print(f"[+] Loaded {len(df_rev)} reviews and {len(df_meta)} metadata records.")

    # 2. Schema and Missingness Audit
    audit_rev_df = audit_reviews_schema(df_rev)
    audit_meta_df = audit_meta_schema(df_meta)

    audit_rev_df.to_csv(os.path.join(OUTPUT_DIR, "audit_reviews_schema.csv"), index=False)
    audit_meta_df.to_csv(os.path.join(OUTPUT_DIR, "audit_meta_schema.csv"), index=False)
    print("[✓] Saved schema missingness audits.")

    # 3. Clean numeric price
    df_meta["price_numeric"] = df_meta["price_raw"].apply(clean_price_to_float)

    # 4. Key Relationships and Join Integrity
    rev_unique_parents = set(df_rev["parent_asin"].dropna().unique())
    meta_unique_parents = set(df_meta["parent_asin"].dropna().unique())
    matched_parents = rev_unique_parents.intersection(meta_unique_parents)
    match_rate = (len(matched_parents) / len(rev_unique_parents)) * 100.0

    print(f"[*] Parent ASIN matching: {len(matched_parents)} / {len(rev_unique_parents)} ({match_rate:.2f}%)")

    # 5. Join Datasets
    df_merged = pd.merge(
        df_rev,
        df_meta,
        on="parent_asin",
        how="inner",
        suffixes=("_rev", "_meta")
    )
    print(f"[+] Inner-joined dataset size: {len(df_merged)} rows.")

    # 6. Duplication Check
    dup_users_items = df_merged.duplicated(subset=["user_id", "parent_asin"]).sum()
    exact_text_dups = df_merged.duplicated(subset=["text"]).sum()

    # 7. Modality Availability Audit
    modality_df = audit_modality_coverage(df_merged)
    modality_df.to_csv(os.path.join(OUTPUT_DIR, "audit_modality_availability.csv"), index=False)
    print("[✓] Saved modality availability audit.")

    # 8. Generate Summary Text Report
    summary_report_path = os.path.join(OUTPUT_DIR, "data_audit_summary.txt")
    with open(summary_report_path, "w", encoding="utf-8") as f:
        f.write("=== AMAZON REVIEWS 2023 (MUSICAL INSTRUMENTS) AUDIT REPORT ===\n\n")
        f.write(f"Total Sampled Reviews: {len(df_rev):,}\n")
        f.write(f"Total Matched Metadata Items: {len(df_meta):,}\n")
        f.write(f"Merged Dataset Observations: {len(df_merged):,}\n")
        f.write(f"Unique Reviewers: {df_rev['user_id'].nunique():,}\n")
        f.write(f"Unique Parent Products: {len(rev_unique_parents):,}\n")
        f.write(f"Metadata Linkage Coverage: {match_rate:.2f}%\n\n")
        f.write("--- INTEGRITY & HYGIENE CHECKS ---\n")
        f.write(f"Duplicate (user_id, parent_asin) pairs: {dup_users_items}\n")
        f.write(f"Exact Duplicate Review Texts: {exact_text_dups}\n")
        f.write(f"Verified Purchases: {df_merged['verified_purchase'].sum()} ({(df_merged['verified_purchase'].mean()*100):.2f}%)\n")
        f.write(f"Reviews with helpful_vote > 0: {(df_merged['helpful_vote'] > 0).sum()}\n\n")
        f.write("--- MODALITY AVAILABILITY OVERVIEW ---\n")
        for _, row in modality_df.iterrows():
            f.write(f"• {row['modality_aspect']}: {row['available_count']:,} ({row['coverage_percentage']}%)\n")

    print(f"[✓] Full audit summary written to: {summary_report_path}")
    print("\n" + open(summary_report_path, "r", encoding="utf-8").read())


if __name__ == "__main__":
    main()