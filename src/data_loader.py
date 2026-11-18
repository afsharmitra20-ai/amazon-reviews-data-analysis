"""
Robust Data Ingestion Module for Amazon Reviews 2023 (Musical Instruments)
Streams directly from UCSD mirror with systematic stride sampling (step_skip=5)
and preserves genuine None/null representations for missing values.
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
from urllib3.exceptions import HTTPError as Urllib3HTTPError

DEFAULT_SAMPLE_SIZE = 20000
DEFAULT_STEP_SKIP = 5
CATEGORY = "Musical_Instruments"

BASE_UCSD_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
REVIEWS_URL = f"{BASE_UCSD_URL}/review_categories/{CATEGORY}.jsonl.gz"
META_URL = f"{BASE_UCSD_URL}/meta_categories/meta_{CATEGORY}.jsonl.gz"


def stream_jsonl_gz_with_retry(url: str, max_retries: int = 3):
    session = requests.Session()
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, stream=True, timeout=40)
            response.raise_for_status()

            with gzip.GzipFile(fileobj=response.raw) as gz_stream:
                for line in gz_stream:
                    if line.strip():
                        yield json.loads(line.decode("utf-8"))
            return
        except (requests.exceptions.RequestException, EOFError, OSError, Urllib3HTTPError, Exception) as e:
            print(f"\n[!] Network stream interrupted ({e}).")
            if attempt < max_retries:
                print(f"[*] Retry {attempt}/{max_retries} in 3 seconds...")
                time.sleep(3)
            else:
                print("[*] Stream terminated.")
                return


def fetch_sampled_reviews(sample_size: int, step_skip: int = DEFAULT_STEP_SKIP) -> pd.DataFrame:
    print(f"[*] Connecting to UCSD Reviews stream:\n    {REVIEWS_URL}")
    records: List[Dict[str, Any]] = []

    count = 0
    pbar = tqdm(total=sample_size, desc="Streaming Reviews")

    for row in stream_jsonl_gz_with_retry(REVIEWS_URL):
        count += 1
        if count % step_skip != 0:
            continue

        # Retain actual None values instead of casting everything to empty strings
        text_val = row.get("text")
        title_val = row.get("title")

        records.append({
            "rating": float(row.get("rating", 0.0)),
            "title": str(title_val).strip() if title_val else None,
            "text": str(text_val).strip() if text_val else None,
            "images": json.dumps(row.get("images", [])),
            "asin": row.get("asin"),
            "parent_asin": row.get("parent_asin"),
            "user_id": row.get("user_id"),
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

    try:
        for row in stream_jsonl_gz_with_retry(META_URL):
            p_asin = str(row.get("parent_asin", ""))
            if p_asin in target_parents and p_asin not in found_asins:
                raw_price = row.get("price")
                store_val = row.get("store")
                cat_val = row.get("main_category")

                records.append({
                    "parent_asin": p_asin,
                    "main_category": str(cat_val).strip() if cat_val else None,
                    "title": row.get("title"),
                    "average_rating": float(row.get("average_rating")) if row.get("average_rating") is not None else None,
                    "rating_number": int(row.get("rating_number")) if row.get("rating_number") is not None else None,
                    "price_raw": str(raw_price).strip() if raw_price is not None else None,
                    "store": str(store_val).strip() if store_val else None,
                    "features": json.dumps(row.get("features", [])),
                    "description": json.dumps(row.get("description", [])),
                    "images": json.dumps(row.get("images", [])),
                    "details": json.dumps(row.get("details", {}))
                })
                found_asins.add(p_asin)
                pbar.update(1)

                if len(found_asins) == len(target_parents):
                    break
    except Exception as e:
        print(f"\n[!] Handled exception during streaming: {e}")

    pbar.close()
    print(f"[+] Successfully extracted {len(records)} metadata records.")
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Streaming ingestion for Amazon Reviews 2023 (UCSD)")
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample size for reviews")
    parser.add_argument("--out_dir", type=str, default="data/raw", help="Target output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    reviews_file = os.path.join(args.out_dir, f"reviews_{CATEGORY}_sampled.parquet")
    meta_file = os.path.join(args.out_dir, f"meta_{CATEGORY}_sampled.parquet")

    # 1. Reviews
    if os.path.exists(reviews_file):
        print(f"[✓] Existing reviews sample detected: {reviews_file}. Skipping re-download.")
        df_reviews = pd.read_parquet(reviews_file)
    else:
        df_reviews = fetch_sampled_reviews(sample_size=args.size, step_skip=DEFAULT_STEP_SKIP)
        df_reviews.to_parquet(reviews_file, index=False)
        print(f"[✓] Reviews saved successfully to: {reviews_file}")

    # 2. Metadata
    if os.path.exists(meta_file):
        print(f"[✓] Existing metadata detected: {meta_file}. Skipping re-download.")
    else:
        unique_parents = set(df_reviews["parent_asin"].dropna().unique())
        df_meta = fetch_matching_metadata(unique_parents)
        df_meta.to_parquet(meta_file, index=False)
        print(f"[✓] Metadata saved successfully ({len(df_meta)} rows) to: {meta_file}")


if __name__ == "__main__":
    main()