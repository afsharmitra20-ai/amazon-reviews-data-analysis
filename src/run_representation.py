"""
Phase 2 - Step 4: Execution Pipeline for Representation Extraction.
Extracts and persists:
- TF-IDF Sparse Matrices (Train, Val, Test)
- Dense Semantic Embeddings (Train, Val, Test)
- Standardized Tabular Matrices (Train, Val, Test)
- Fitted TF-IDF Vectorizer & Tabular Scaler objects
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from representation import MultimodalRepresentationBuilder

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
REPORTS_DIR = os.path.join("reports", "phase2")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("[*] Loading engineered feature datasets...")
    train_path = os.path.join(PROCESSED_DIR, "train_features.parquet")
    val_path = os.path.join(PROCESSED_DIR, "val_features.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test_features.parquet")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    builder = MultimodalRepresentationBuilder(max_tfidf_features=5000)

    # -------------------------------------------------------------
    # 1. Classical Sparse Representation: TF-IDF
    # -------------------------------------------------------------
    builder.fit_tfidf(df_train["clean_text"])

    print("[*] Transforming texts to TF-IDF CSR matrices...")
    X_train_tfidf = builder.transform_tfidf(df_train["clean_text"])
    X_val_tfidf = builder.transform_tfidf(df_val["clean_text"])
    X_test_tfidf = builder.transform_tfidf(df_test["clean_text"])

    sparse.save_npz(os.path.join(PROCESSED_DIR, "train_tfidf.npz"), X_train_tfidf)
    sparse.save_npz(os.path.join(PROCESSED_DIR, "val_tfidf.npz"), X_val_tfidf)
    sparse.save_npz(os.path.join(PROCESSED_DIR, "test_tfidf.npz"), X_test_tfidf)
    print(f"[✓] TF-IDF matrices saved: Train {X_train_tfidf.shape}, Val {X_val_tfidf.shape}, Test {X_test_tfidf.shape}")

    # -------------------------------------------------------------
    # 2. Modern Dense Representation: MiniLM Embeddings
    # -------------------------------------------------------------
    print("\n[*] Extracting dense semantic embeddings (MiniLM-L6-v2)...")
    X_train_emb = builder.extract_dense_embeddings(df_train["clean_text"])
    X_val_emb = builder.extract_dense_embeddings(df_val["clean_text"])
    X_test_emb = builder.extract_dense_embeddings(df_test["clean_text"])

    np.save(os.path.join(PROCESSED_DIR, "train_embeddings.npy"), X_train_emb)
    np.save(os.path.join(PROCESSED_DIR, "val_embeddings.npy"), X_val_emb)
    np.save(os.path.join(PROCESSED_DIR, "test_embeddings.npy"), X_test_emb)
    print(f"[✓] Dense embeddings saved: Train {X_train_emb.shape}, Val {X_val_emb.shape}, Test {X_test_emb.shape}")

    # -------------------------------------------------------------
    # 3. Standardized Tabular Feature Matrices
    # -------------------------------------------------------------
    print("\n[*] Standardizing tabular domain features...")
    X_train_tab, X_val_tab, X_test_tab = builder.fit_transform_tabular(df_train, df_val, df_test)

    np.save(os.path.join(PROCESSED_DIR, "train_tabular.npy"), X_train_tab)
    np.save(os.path.join(PROCESSED_DIR, "val_tabular.npy"), X_val_tab)
    np.save(os.path.join(PROCESSED_DIR, "test_tabular.npy"), X_test_tab)
    print(f"[✓] Tabular feature matrices saved: Train {X_train_tab.shape}")

    # -------------------------------------------------------------
    # 4. Persist Fitted Artifacts
    # -------------------------------------------------------------
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    scaler_path = os.path.join(MODELS_DIR, "tabular_scaler.joblib")
    joblib.dump(builder.tfidf_vectorizer, vectorizer_path)
    joblib.dump(builder.tabular_scaler, scaler_path)
    print(f"[✓] Persisted models to {MODELS_DIR}/")

    # -------------------------------------------------------------
    # 5. Compile Representation Audit Report
    # -------------------------------------------------------------
    # Top 20 terms by mean TF-IDF in Train
    mean_tfidf = np.asarray(X_train_tfidf.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[::-1][:20]
    feature_names = np.array(builder.tfidf_vectorizer.get_feature_names_out())
    top_tokens = [(feature_names[i], round(mean_tfidf[i], 4)) for i in top_indices]

    audit_path = os.path.join(REPORTS_DIR, "representation_audit.txt")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("=== PHASE 2 - STEP 4: REPRESENTATION AUDIT REPORT ===\n\n")
        f.write("1. REPRESENTATION COMPARISON & SPECIFICATIONS:\n")
        f.write("   • Classical Representation: Sparse TF-IDF (Unigrams & Bigrams)\n")
        f.write(f"     - Train Shape: {X_train_tfidf.shape} (Sparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]))*100:.2f}%)\n")
        f.write(f"     - Val Shape:   {X_val_tfidf.shape}\n")
        f.write(f"     - Test Shape:  {X_test_tfidf.shape}\n\n")
        
        f.write("   • Modern Semantic Representation: Pre-trained Sentence-Transformers (all-MiniLM-L6-v2)\n")
        f.write(f"     - Embedding Dimension: {X_train_emb.shape[1]} (Dense L2-normalized vectors)\n")
        f.write(f"     - Train Shape: {X_train_emb.shape}\n")
        f.write(f"     - Val Shape:   {X_val_emb.shape}\n")
        f.write(f"     - Test Shape:  {X_test_emb.shape}\n\n")

        f.write("   • Tabular Structured Modality:\n")
        f.write(f"     - Features Count: {X_train_tab.shape[1]}\n")
        f.write(f"     - Feature List: {', '.join(builder.tabular_feature_cols)}\n\n")

        f.write("2. TOP 20 HIGH-SALIENCE TOKENS IN TRAINING CORPUS (MEAN TF-IDF):\n")
        for rank, (tok, score) in enumerate(top_tokens, 1):
            f.write(f"   {rank:2d}. {tok:<18} (Mean TF-IDF: {score})\n")

    print(f"\n[✓] Representation audit written to: {audit_path}")
    print("\n" + open(audit_path, "r", encoding="utf-8").read())


if __name__ == "__main__":
    main()