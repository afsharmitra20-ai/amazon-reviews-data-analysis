"""
Robust Data Ingestion Module for Amazon Reviews 2023 (Musical Instruments)
Direct UCSD streaming + schema sanitation for robust Parquet serialization.
"""

import os
import gzip
import json
import time
import argparse
from typing import Dict, Any, List, Set
import requests
import pandas as pd
from tqdm import tqdm

DEFAULT_SAMPLE_SIZE = 20000
CATEGORY = "Musical_Instruments"

BASE_UCSD_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
REVIEWS_URL = f"{BASE_UCSD_URL}/review_categories/{CATEGORY}.jsonl.gz"
META_URL = f"{BASE_UCSD_URL}/meta_categories/meta_{CATEGORY}.jsonl.gz"


def stream_jsonl_gz_with_retry(url: str, max_retries: int = 5):
    """
    Robust stream generator with retry logic for network drops.
    """
    for attempt in range(1, max_retries + 1):
        try:
            session = requests.Session()
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with gzip.GzipFile(fileobj=response.raw) as gz_stream:
                for line in gz_stream:
                    if line.strip():
                        yield json.loads(line.decode("utf-8"))
            return
        except (requests.exceptions.RequestException, EOFError, OSError) as e:
            print(f"\n[!] Network glitch encountered ({e}). Retry {attempt}/{max_retries} in 5 seconds...")
            time.sleep(5)

    print("[!] Max retries reached.")


def fetch_sampled_reviews(sample_size: int, step_skip: int = 5) -> pd.DataFrame:
    print(f"[*] Connecting to UCSD Reviews stream:\n    {REVIEWS_URL}")
    records: List[Dict[str, Any]] = []

    count = 0
    pbar = tqdm(total=sample_size, desc="Streaming Reviews")

    for row in stream_jsonl_gz_with_retry(REVIEWS_URL):
        count += 1
        if count % step_skip != 0:
            continue

        records.append({
            "rating": float(row.get("rating", 0.0)),
            "title": str(row.get("title", "") or ""),
            "text": str(row.get("text", "") or ""),
            "images": json.dumps(row.get("images", [])),  # Safely serialize nested list
            "asin": str(row.get("asin", "")),
            "parent_asin": str(row.get("parent_asin", "")),
            "user_id": str(row.get("user_id", "")),
            "timestamp": int(row.get("timestamp", 0)),
            "verified_purchase": bool(row.get("verified_purchase", False)),
            "helpful_vote": int(row.get("helpful_vote", 0))
        })
        pbar.update(1)

        if len(records) >= sample_size:
            break

    pbar.close()
    return pd.DataFrame(records)


def fetch_matching_metadata(target_parents: Set[str]) -> pd.DataFrame:
    print(f"[*] Connecting to UCSD Metadata stream:\n    {META_URL}")
    print(f"[*] Querying metadata for {len(target_parents)} unique parent items...")

    records: List[Dict[str, Any]] = []
    found_asins: Set[str] = set()

    pbar = tqdm(total=len(target_parents), desc="Matching Metadata")

    for row in stream_jsonl_gz_with_retry(META_URL):
        p_asin = str(row.get("parent_asin", ""))
        if p_asin in target_parents and p_asin not in found_asins:
            # Handle price cleanly: keep raw string or empty
            raw_price = row.get("price")
            price_str = str(raw_price) if raw_price is not None else ""

            records.append({
                "parent_asin": p_asin,
                "main_category": str(row.get("main_category", "") or ""),
                "title": str(row.get("title", "") or ""),
                "average_rating": float(row.get("average_rating")) if row.get("average_rating") is not None else None,
                "rating_number": int(row.get("rating_number")) if row.get("rating_number") is not None else None,
                "price_raw": price_str,
                "store": str(row.get("store", "") or ""),
                "features": json.dumps(row.get("features", [])),
                "description": json.dumps(row.get("description", [])),
                "images": json.dumps(row.get("images", [])),
                "details": json.dumps(row.get("details", {}))
            })
            found_asins.add(p_asin)
            pbar.update(1)

            if len(found_asins) == len(target_parents):
                break

    pbar.close()
    print(f"[+] Successfully extracted {len(records)} metadata records.")
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Fault-tolerant streaming ingestion")
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    reviews_file = os.path.join(args.out_dir, f"reviews_{CATEGORY}_sampled.parquet")
    meta_file = os.path.join(args.out_dir, f"meta_{CATEGORY}_sampled.parquet")

    # 1. Load or fetch reviews
    if os.path.exists(reviews_file):
        print(f"[✓] Existing reviews sample detected: {reviews_file}. Skipping re-download.")
        df_reviews = pd.read_parquet(reviews_file)
    else:
        df_reviews = fetch_sampled_reviews(sample_size=args.size)
        df_reviews.to_parquet(reviews_file, index=False)
        print(f"[✓] Reviews saved successfully to: {reviews_file}")

    # 2. Fetch matching metadata
    unique_parents = set(df_reviews["parent_asin"].dropna().unique())
    df_meta = fetch_matching_metadata(unique_parents)
    df_meta.to_parquet(meta_file, index=False)
    print(f"[✓] Metadata saved successfully ({len(df_meta)} rows) to: {meta_file}")


if __name__ == "__main__":
    main()