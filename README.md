# Retail Customer Intelligence Platform

Portfolio project for retail customer analytics, progressing from data auditing and customer features to customer segmentation, repeat-purchase prediction, and customer lifetime value estimation.

## Current status

- The official UCI Online Retail workbook is retained under `data/raw/`; raw and processed transaction files are excluded from Git by `.gitignore`.
- `src/data/` provides ingestion, validation, candidate partitioning, reconciliation, and SQLite loading.
- `src/features/customer_features.py` creates a reconciled customer-level RFM and activity table from the approved candidate cohort.
- Segmentation, prediction, CLV estimation, and dashboard functionality are not implemented yet.

## Environment setup

Python 3.10 or newer is recommended. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `.xlsx` reader dependency is `openpyxl`.

## Retail data ingestion and quality audit

The source was obtained from the [official UCI Online Retail dataset page](https://archive.ics.uci.edu/dataset/352/online+retail). The downloaded archive and extracted source workbook are retained unchanged in `data/raw/`. The UCI dataset is licensed CC BY 4.0; cite Chen (2015), *Online Retail*, UCI Machine Learning Repository, DOI [10.24432/C5BW33](https://doi.org/10.24432/C5BW33).

Activate the environment, then run:

```bash
source .venv/bin/activate
python -m src.data.pipeline
```

Optional paths can be supplied with `--input`, `--processed-dir`, and `--report-dir`.

The pipeline retains every source row in `data/processed/transactions_audited.csv`, adds quality flags, and writes separate cancellation/return-like, customer-sales-candidate, and candidate-exclusion files. It also loads corresponding tables into `data/processed/online_retail.sqlite`. Row counts, quality flag counts, source checksum, run time, and configured assumptions are recorded under `reports/generated/`.

The candidate cohort is a starting rule, not a silent deletion policy: it requires positive quantity and unit price, a valid invoice/date/customer identifier, and no cancellation invoice marker. Negative quantities are called return-like for review; duplicates and other anomalies are retained and flagged. Review the quality report before using the candidate cohort for modeling.

## Customer-level feature table

After the ingestion pipeline has created `data/processed/customer_sales_candidates.csv`, build the initial customer-level feature table with:

```bash
source .venv/bin/activate
python -m src.features.customer_features
```

The command writes `data/processed/customer_features.csv`, replaces the `customer_features` table in `data/processed/online_retail.sqlite`, and writes `reports/generated/customer_feature_reconciliation.json`. It reports line, invoice, customer, and gross-sales tie-outs.

Features include recency, distinct invoice frequency, gross sales in GBP, first/last purchase dates, customer tenure, total units, product breadth, transaction-line count, and average order value. The snapshot date is one day after the latest invoice date in the candidate cohort. Monetary value is gross revenue from positive candidate rows; it is not profit or customer lifetime value.

## Project layout

- `data/raw/`: untouched source archive and workbook (Git-ignored)
- `data/processed/`: local audited extracts and SQLite database (Git-ignored)
- `src/data/`: ingestion and quality audit modules
- `src/features/`: customer-level feature aggregation
- `src/segmentation/`, `src/models/`, `src/evaluation/`: reserved source packages
- `notebooks/`: analysis notebooks
- `reports/generated/`: run-level quality and reconciliation reports
- `app/`: reserved for a future dashboard
- `tests/`: reserved for automated checks
