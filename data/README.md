# Data Ingestion Documentation

## Dataset Source
- **Platform:** Hugging Face Datasets (`McAuley-Lab/Amazon-Reviews-2023`)
- **Subset:** `Musical_Instruments`
- **Modality:** Reviews (Text, Ratings, User Images) + Metadata (Item features, Price, Descriptions, Catalog Images)

## Sampling Methodology
Due to the massive footprint of the full 2023 corpus, streaming ingestion (`streaming=True`) is employed:
1. `raw_review_Musical_Instruments` is pseudo-randomly streamed with fixed seed `42` to collect a representative sample of 20,000 interactions.
2. Distinct `parent_asin` identifiers are mapped directly to `raw_meta_Musical_Instruments` via streaming filter.
3. Artifacts are serialized in Apache Parquet format inside `data/raw/` (ignored by Git tracking).

## Replication Command
```bash
python src/data_loader.py --size 20000 --seed 42 --out_dir data/raw