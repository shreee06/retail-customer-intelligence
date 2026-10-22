"""Run reproducible Online Retail ingestion, audit, and reconciliation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ingestion import load_transactions
from ..paths import project_relative
from .validation import audit_transactions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORTS = PROJECT_ROOT / "reports" / "generated"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(
    source_path: Path = DEFAULT_SOURCE,
    processed_dir: Path = DEFAULT_PROCESSED,
    report_dir: Path = DEFAULT_REPORTS,
) -> dict:
    """Create full-row audit, candidate partitions, reports, and a SQLite DB."""
    source_path = Path(source_path).resolve()
    processed_dir = Path(processed_dir).resolve()
    report_dir = Path(report_dir).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source workbook not found: {source_path}")

    raw = load_transactions(source_path)
    audited, quality = audit_transactions(raw)

    # This partition is intentionally permissive: all cancellation invoice
    # rows and all negative-quantity rows are retained for reconciliation.
    cancellation_returns = audited.loc[audited["is_cancellation_or_return"]].copy()
    candidates = audited.loc[audited["is_customer_sales_candidate"]].copy()
    exclusions = audited.loc[~audited["is_customer_sales_candidate"]].copy()
    net_spend_cancellations = audited.loc[audited["is_net_spend_cancellation"]].copy()

    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    row_counts = [
        {"stage": "source_rows_loaded", "rows": int(len(raw)), "meaning": "Rows read from the source workbook."},
        {"stage": "audit_rows_retained", "rows": int(len(audited)), "meaning": "All source rows retained with quality flags."},
        {"stage": "cancellations_and_returns", "rows": int(len(cancellation_returns)), "meaning": "Rows with a cancellation invoice marker or negative quantity; may overlap."},
        {"stage": "customer_sales_candidates", "rows": int(len(candidates)), "meaning": "Positive quantity and unit price; valid invoice/date/customer identifiers; not cancellation-marked; not a non-product line; not an extra copy of an exact duplicate."},
        {"stage": "excluded_from_customer_sales_candidates", "rows": int(len(exclusions)), "meaning": "Rows not meeting the candidate cohort rules; retained in audit and exclusions output."},
        {"stage": "net_spend_cancellation_lines", "rows": int(len(net_spend_cancellations)), "meaning": "Valid cancelled/returned product lines subtracted from customer net spend."},
        {"stage": "rows_requiring_review", "rows": int(quality["rows_requiring_review"]), "meaning": "Rows with one or more non-exclusion review flags; flags can overlap."},
    ]

    audit_path = processed_dir / "transactions_audited.csv"
    candidates_path = processed_dir / "customer_sales_candidates.csv"
    returns_path = processed_dir / "cancellations_returns.csv"
    exclusions_path = processed_dir / "candidate_exclusions.csv"
    net_cancellations_path = processed_dir / "net_spend_cancellations.csv"
    database_path = processed_dir / "online_retail.sqlite"

    audited.to_csv(audit_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    cancellation_returns.to_csv(returns_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)
    net_spend_cancellations.to_csv(net_cancellations_path, index=False)

    # SQLite is the initial SQL workflow; data remain local and ignored by Git.
    with sqlite3.connect(database_path) as connection:
        audited.to_sql("transactions_audited", connection, if_exists="replace", index=False, chunksize=10000)
        candidates.to_sql("customer_sales_candidates", connection, if_exists="replace", index=False, chunksize=10000)
        cancellation_returns.to_sql("cancellations_returns", connection, if_exists="replace", index=False, chunksize=10000)
        exclusions.to_sql("candidate_exclusions", connection, if_exists="replace", index=False, chunksize=10000)
        net_spend_cancellations.to_sql("net_spend_cancellations", connection, if_exists="replace", index=False, chunksize=10000)

    reconciliation = pd.DataFrame(row_counts)
    reconciliation.to_csv(report_dir / "reconciliation.csv", index=False)

    metadata = {
        "source_file": source_path.name,
        "source_url": SOURCE_URL,
        "source_sha256": sha256_file(source_path),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pandas_version": pd.__version__,
        "row_counts": row_counts,
        "quality": quality,
        "assumptions": {
            "cancellation_invoice": "InvoiceNo starts with C, case-insensitive, per UCI documentation.",
            "return_like": "Negative Quantity is separately flagged as return-like; this is a review signal, not a definitive UCI-coded return label.",
            "customer_sales_candidate": "Positive Quantity and UnitPrice, valid six-digit invoice (optional C is excluded), parseable InvoiceDate, and valid five-digit CustomerID; all other rows remain in audit outputs.",
            "exact_duplicates": "All copies stay flagged (row_is_exact_duplicate); only copies after the first are excluded from the cohort (exact_duplicate_extra_copy).",
            "non_product_lines": "Postage, manual, carriage, bank charges, discounts, samples and fees are excluded from revenue/RFM (non_product_line).",
            "description_anomalies": "Flagged for review and never automatically removed.",
        },
        "outputs": {
            "full_audit_csv": project_relative(audit_path),
            "customer_sales_candidates_csv": project_relative(candidates_path),
            "cancellations_returns_csv": project_relative(returns_path),
            "candidate_exclusions_csv": project_relative(exclusions_path),
            "net_spend_cancellations_csv": project_relative(net_cancellations_path),
            "sqlite_database": project_relative(database_path),
            "reconciliation_csv": project_relative(report_dir / "reconciliation.csv"),
        },
    }
    summary_path = report_dir / "data_quality.json"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE, help="Source .xlsx path")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.input, args.processed_dir, args.report_dir)
    print(f"Source rows loaded: {result['quality']['row_count']:,}")
    print(f"Customer sales candidates: {result['row_counts'][3]['rows']:,}")
    print(f"Cancellation/return-like rows: {result['row_counts'][2]['rows']:,}")
    print(f"Quality report: {args.report_dir.resolve() / 'data_quality.json'}")
    print(f"Reconciliation: {args.report_dir.resolve() / 'reconciliation.csv'}")


if __name__ == "__main__":
    main()
