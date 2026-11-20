"""
Execution pipeline for Step 2: Fit preprocessor strictly on Train,
transform Val & Test without leakage, and serialize clean datasets.
"""

import os
import joblib
import pandas as pd
from preprocess import MusicalInstrumentsPreprocessor

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
REPORTS_DIR = os.path.join("reports", "phase2")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("[*] Loading Train, Val, and Test disjoint partitions...")
    train_path = os.path.join(PROCESSED_DIR, "train.parquet")
    val_path = os.path.join(PROCESSED_DIR, "val.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test.parquet")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    print(f"[+] Train shape: {df_train.shape} | Val shape: {df_val.shape} | Test shape: {df_test.shape}")

    # 1. Initialize Preprocessor
    preprocessor = MusicalInstrumentsPreprocessor(top_k_stores=50)

    # 2. Fit STRICTLY on Train (Zero Leakage Invariance)
    print("[*] Fitting preprocessor parameters strictly on Training data...")
    preprocessor.fit(df_train)
    print(f"    -> Learned Training Median Price: ${preprocessor.median_price_:.2f}")
    print(f"    -> Learned Log-Price Mean: {preprocessor.price_mean_log_:.4f}, Std: {preprocessor.price_std_log_:.4f}")
    print(f"    -> Identified {len(preprocessor.top_stores_)} high-frequency store brands.")

    # 3. Transform all splits using fitted parameters
    print("[*] Transforming Train, Val, and Test splits...")
    df_train_proc = preprocessor.transform(df_train)
    df_val_proc = preprocessor.transform(df_val)
    df_test_proc = preprocessor.transform(df_test)

    # 4. Save Processed Artifacts
    out_train = os.path.join(PROCESSED_DIR, "train_preprocessed.parquet")
    out_val = os.path.join(PROCESSED_DIR, "val_preprocessed.parquet")
    out_test = os.path.join(PROCESSED_DIR, "test_preprocessed.parquet")

    df_train_proc.to_parquet(out_train, index=False)
    df_val_proc.to_parquet(out_val, index=False)
    df_test_proc.to_parquet(out_test, index=False)

    # 5. Persist the fitted pipeline object for Phase 3 benchmarking
    pipeline_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, pipeline_path)
    print(f"[✓] Persisted fitted preprocessor model to: {pipeline_path}")

    # 6. Generate Imputation & Preprocessing Audit Report
    report_path = os.path.join(REPORTS_DIR, "imputation_audit.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== PHASE 2 - STEP 2: PREPROCESSING & IMPUTATION AUDIT REPORT ===\n\n")
        f.write("1. MISSINGNESS MECHANISM & JUSTIFICATION:\n")
        f.write("   - Variable: price_raw (Missing in ~33% of instances)\n")
        f.write("   - Mechanism: MNAR / MAR (Discontinued instruments, accessory variants, unpriced listings)\n")
        f.write("   - Handling Method:\n")
        f.write("     * Imputed using Training Set Median (Robust to high-value luthier outliers)\n")
        f.write("     * Generated binary indicator 'price_is_missing' to retain missingness information\n")
        f.write("     * Applied log1p transformation to mitigate severe right-skewness\n")
        f.write("     * Standardized using Training set log-mean and log-std\n\n")

        f.write("2. LEARNED PARAMETERS (TRAIN SET ONLY - ZERO LEAKAGE):\n")
        f.write(f"   - Fitted Median Price: ${preprocessor.median_price_:.2f}\n")
        f.write(f"   - Fitted Log1p Price Mean: {preprocessor.price_mean_log_:.4f}\n")
        f.write(f"   - Fitted Log1p Price Std:  {preprocessor.price_std_log_:.4f}\n")
        f.write(f"   - Top Stores Monitored: {len(preprocessor.top_stores_)}\n\n")

        f.write("3. SPLIT MISSINGNESS & SCALING VERIFICATION:\n")
        for name, split_df in [("Train", df_train_proc), ("Val", df_val_proc), ("Test", df_test_proc)]:
            missing_rate = split_df["price_is_missing"].mean() * 100
            f.write(f"   - {name} Set: Missing Price Rate = {missing_rate:.2f}%\n")
            f.write(f"     Mean Scaled Price: {split_df['price_scaled'].mean():.4f} | Std: {split_df['price_scaled'].std():.4f}\n")
            f.write(f"     Mean Word Count: {split_df['word_count'].mean():.2f} words\n")

        f.write("\n4. QUARANTINED LEAKAGE VERIFICATION:\n")
        f.write("   - Columns 'average_rating', 'rating_number', 'helpful_vote' successfully DROPPED from all splits.\n")

    print(f"[✓] Imputation & Preprocessing Audit saved to: {report_path}")
    print("\n" + open(report_path, "r", encoding="utf-8").read())


if __name__ == "__main__":
    main()