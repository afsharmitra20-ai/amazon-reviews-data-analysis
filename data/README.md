# Data Ingestion & Sampling Documentation

## Dataset Source
- **Official Repository Endpoint:** UCSD McAuley Lab (`https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw`)
- **Subset Category:** `Musical_Instruments`
- **Modality Coverage:** Reviews (`Musical_Instruments.jsonl.gz`) and Matching Metadata (`meta_Musical_Instruments.jsonl.gz`).

## Ingestion Architecture & Systematic Stride Sampling
Because the complete uncompressed corpus spans several gigabytes, streaming ingestion is conducted line-by-line directly over HTTP:
1. **Systematic Stride Sampling:** A fixed stride sampling interval (`step_skip = 5`) is enforced across the stream to prevent sampling bias localized to the beginning of the file, capturing an evenly distributed set of 20,000 interactions.
2. **Metadata Linkage:** Unique product identifiers (`parent_asin`) collected from the reviews stream are matched against the product metadata stream.
3. **Missing Value Preservation:** True missing elements (`null`/`None`) are preserved during ingestion to prevent schema masking.
4. **Local Artifacts:** Data artifacts are serialized in Apache Parquet format inside `data/raw/` (excluded from Git tracking via `.gitignore`).

## Reproduction Command
```powershell
python src/data_loader.py --size 20000