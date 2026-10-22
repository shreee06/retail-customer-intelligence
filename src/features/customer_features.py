"""Aggregate audited customer-sales candidates into a customer-level RFM table."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..paths import project_relative

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "customer_sales_candidates.csv"
DEFAULT_CANCELLATIONS = PROJECT_ROOT / "data" / "processed" / "net_spend_cancellations.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "generated" / "customer_feature_reconciliation.json"
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "processed" / "online_retail.sqlite"
REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "is_customer_sales_candidate",
}


def load_candidate_sales(input_path: Path) -> pd.DataFrame:
    """Load only the candidate fields needed to build customer features."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Candidate sales file not found: {input_path}")

    header = pd.read_csv(input_path, nrows=0).columns
    missing = sorted(REQUIRED_COLUMNS - set(header))
    if missing:
        raise ValueError("Candidate file is missing required columns: " + ", ".join(missing))

    return pd.read_csv(
        input_path,
        usecols=sorted(REQUIRED_COLUMNS),
        dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string"},
    )


def load_cancellation_lines(input_path: Path) -> pd.DataFrame:
    """Load cancelled/returned product lines used to compute net spend."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Net-spend cancellation file not found: {input_path}")
    return pd.read_csv(
        input_path,
        usecols=["CustomerID", "Quantity", "UnitPrice"],
        dtype={"CustomerID": "string"},
    )


def build_customer_features(
    candidate_sales: pd.DataFrame, cancellation_lines: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict]:
    """Build RFM and activity features, returning feature data and tie-out data.

    `gross_spend_gbp` is purchases only. `cancelled_spend_gbp` is the value of
    the customer's cancelled/returned lines (a positive amount) and
    `monetary_gbp` is net spend: gross minus cancelled.
    """
    sales = candidate_sales.copy()
    is_candidate = sales["is_customer_sales_candidate"].astype("string").str.lower().eq("true")
    if not is_candidate.all():
        raise ValueError("Input contains rows that are not marked as customer sales candidates")

    customer_id = sales["CustomerID"].astype("string").str.strip()
    invoice_id = sales["InvoiceNo"].astype("string").str.strip()
    product_id = sales["StockCode"].astype("string").str.strip()
    quantity = pd.to_numeric(sales["Quantity"], errors="coerce")
    unit_price = pd.to_numeric(sales["UnitPrice"], errors="coerce")
    invoice_date = pd.to_datetime(sales["InvoiceDate"], errors="coerce")

    invalid = (
        customer_id.isna()
        | customer_id.eq("")
        | invoice_id.isna()
        | invoice_id.eq("")
        | invoice_id.str.upper().str.startswith("C", na=False)
        | quantity.isna()
        | quantity.le(0)
        | unit_price.isna()
        | unit_price.le(0)
        | invoice_date.isna()
    )
    if invalid.any():
        raise ValueError(f"Candidate input contains {int(invalid.sum())} rows outside the approved cohort rules")

    sales["CustomerID"] = customer_id
    sales["InvoiceNo"] = invoice_id
    sales["StockCode"] = product_id
    sales["_quantity"] = quantity
    sales["_unit_price"] = unit_price
    sales["_invoice_date"] = invoice_date
    sales["_line_revenue_gbp"] = (quantity * unit_price).round(2)

    if cancellation_lines is None or cancellation_lines.empty:
        cancelled = pd.Series(dtype="float64")
    else:
        cancel_value = (
            pd.to_numeric(cancellation_lines["Quantity"]) * pd.to_numeric(cancellation_lines["UnitPrice"])
        ).abs()
        cancelled = cancel_value.groupby(
            cancellation_lines["CustomerID"].astype("string").str.strip()
        ).sum()

    snapshot_date = sales["_invoice_date"].max().normalize() + pd.Timedelta(days=1)
    features = (
        sales.groupby("CustomerID", as_index=False, sort=True)
        .agg(
            first_purchase_date=("_invoice_date", "min"),
            last_purchase_date=("_invoice_date", "max"),
            frequency_orders=("InvoiceNo", "nunique"),
            gross_spend_gbp=("_line_revenue_gbp", "sum"),
            total_units=("_quantity", "sum"),
            unique_products=("StockCode", "nunique"),
            transaction_lines=("InvoiceNo", "size"),
        )
    )
    features["cancelled_spend_gbp"] = (
        features["CustomerID"].map(cancelled).fillna(0.0).round(2)
    )
    features["gross_spend_gbp"] = features["gross_spend_gbp"].round(2)
    features["monetary_gbp"] = (
        features["gross_spend_gbp"] - features["cancelled_spend_gbp"]
    ).round(2)
    features["net_spend_nonpositive"] = features["monetary_gbp"] <= 0
    features["recency_days"] = (
        snapshot_date - features["last_purchase_date"].dt.normalize()
    ).dt.days.astype("int64")
    features["customer_tenure_days"] = (
        features["last_purchase_date"].dt.normalize()
        - features["first_purchase_date"].dt.normalize()
    ).dt.days.astype("int64")
    features["average_order_value_gbp"] = (
        features["monetary_gbp"] / features["frequency_orders"]
    ).round(2)
    features["snapshot_date"] = snapshot_date.date().isoformat()
    features["total_units"] = features["total_units"].round(2)

    source_lines = int(len(sales))
    source_invoices = int(sales["InvoiceNo"].nunique())
    source_revenue = float(sales["_line_revenue_gbp"].sum())
    feature_lines = int(features["transaction_lines"].sum())
    feature_invoices = int(features["frequency_orders"].sum())
    feature_revenue = float(features["gross_spend_gbp"].sum())
    reconciliation = {
        "source_candidate_lines": source_lines,
        "feature_transaction_lines": feature_lines,
        "source_distinct_invoices": source_invoices,
        "summed_customer_invoice_frequency": feature_invoices,
        "source_gross_sales_gbp": round(source_revenue, 2),
        "summed_customer_gross_spend_gbp": round(feature_revenue, 2),
        "summed_customer_cancelled_spend_gbp": round(float(features["cancelled_spend_gbp"].sum()), 2),
        "summed_customer_net_spend_gbp": round(float(features["monetary_gbp"].sum()), 2),
        "customers_with_nonpositive_net_spend": int(features["net_spend_nonpositive"].sum()),
        "source_customers": int(sales["CustomerID"].nunique()),
        "feature_rows": int(len(features)),
        "snapshot_date": snapshot_date.date().isoformat(),
        "checks": {
            "line_count_matches": source_lines == feature_lines,
            "invoice_frequency_matches": source_invoices == feature_invoices,
            "gross_sales_matches": math.isclose(source_revenue, feature_revenue, abs_tol=0.01),
            "customer_count_matches": int(sales["CustomerID"].nunique()) == int(len(features)),
        },
    }
    return features, reconciliation


def run_feature_pipeline(
    input_path: Path = DEFAULT_INPUT,
    cancellations_path: Path = DEFAULT_CANCELLATIONS,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
    database_path: Path = DEFAULT_DATABASE,
) -> dict:
    input_path = Path(input_path).resolve()
    cancellations_path = Path(cancellations_path).resolve()
    output_path = Path(output_path).resolve()
    report_path = Path(report_path).resolve()
    database_path = Path(database_path).resolve()

    candidates = load_candidate_sales(input_path)
    cancellations = load_cancellation_lines(cancellations_path)
    features, reconciliation = build_customer_features(candidates, cancellations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(output_path, index=False)
    with sqlite3.connect(database_path) as connection:
        features.to_sql(
            "customer_features", connection, if_exists="replace", index=False, chunksize=1000
        )

    report = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": project_relative(input_path),
        "output_file": project_relative(output_path),
        "sqlite_database": project_relative(database_path),
        "sqlite_table": "customer_features",
        "feature_scope": "RFM and customer activity summary from customer-sales candidates only.",
        "revenue_interpretation": "Net spend in GBP (purchases minus cancellations/returns, Quantity x UnitPrice); gross spend is a separate column; not profit and not CLV.",
        "definitions": {
            "recency_days": "Days from last purchase date to one day after the latest candidate invoice date.",
            "frequency_orders": "Distinct non-cancellation InvoiceNo values per customer.",
            "gross_spend_gbp": "Sum of candidate purchase lines (Quantity x UnitPrice).",
            "cancelled_spend_gbp": "Value of the customer's valid cancelled/returned product lines.",
            "monetary_gbp": "Net spend: gross_spend_gbp minus cancelled_spend_gbp.",
            "net_spend_nonpositive": "True when net spend is zero or negative; review before using in RFM.",
            "customer_tenure_days": "Calendar days between the customer's first and last candidate purchase dates.",
        },
        "reconciliation": reconciliation,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cancellations", type=Path, default=DEFAULT_CANCELLATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_feature_pipeline(
        args.input, args.cancellations, args.output, args.report, args.database
    )
    result = report["reconciliation"]
    print(f"Customers: {result['feature_rows']:,}")
    print(f"Candidate lines reconciled: {result['feature_transaction_lines']:,}")
    print(f"Gross spend reconciled: £{result['summed_customer_gross_spend_gbp']:,.2f}")
    print(f"Net spend: £{result['summed_customer_net_spend_gbp']:,.2f}")
    print(f"Snapshot date: {result['snapshot_date']}")
    print(f"Feature table: {args.output.resolve()}")
    print(f"Reconciliation report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
