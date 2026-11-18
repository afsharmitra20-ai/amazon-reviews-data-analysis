# Amazon Reviews 2023 - Multimodal Analysis (Musical Instruments)
**Data Analysis Course Project - Fall 1405**  
**Track B: Amazon Reviews'23 | Phase 1 Completed**

Github repo: https://afsharmitra20-ai:@github.com/afsharmitra20-ai/amazon-reviews-data-analysis.git

## Overview
This repository contains the end-to-end data analysis pipeline for the **Musical Instruments** category from the Amazon Reviews 2023 corpus. The project explores the interaction between review text and product metadata (pricing, store/brand, catalog images) to understand customer satisfaction and predict review ratings without information leakage.

## Directory Structure
```text
├── README.md                 # Project root overview
├── PROJECT_CARD.md           # Formal project specification (14 mandatory points)
├── AI_USAGE.md               # Transparent AI tool usage disclosure
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # Ingested Parquet subsets (ignored by git)
│   └── README.md             # Ingestion and replication guide
├── src/
│   ├── data_loader.py        # Fault-tolerant streaming ingestion from UCSD
│   ├── audit_utils.py        # Data hygiene and modality audit utilities
│   ├── run_audit.py          # Execution script for data & modality audit
│   ├── run_eda.py            # Univariate, temporal, and multimodal EDA
│   └── generate_phase1_pdf.py# Automated compilation of Phase 1 PDF report
├── notebooks/
│   ├── 01_data_audit.ipynb   # Interactive data audit notebook
│   └── 02_eda_multimodal.ipynb # Interactive EDA and visual distributions
├── figures/                  # Publication-ready visualizations
└── reports/
    └── phase1.pdf            # Formal Phase 1 PDF deliverable