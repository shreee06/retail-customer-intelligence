"""Leakage-aware 90-day repeat purchase and expected revenue models."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/customer_sales_candidates.csv"
PREDICTIONS = ROOT / "data/processed/repeat_purchase_predictions.csv"
REPORT = ROOT / "reports/generated/model_evaluation.json"
DATABASE = ROOT / "data/processed/online_retail.sqlite"
HORIZON_DAYS = 90
FEATURES = ["recency_days", "frequency_orders", "monetary_gbp", "total_units",
            "unique_products", "transaction_lines", "customer_tenure_days"]


def make_temporal_snapshots(sales: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> tuple[pd.DataFrame, list[str]]:
    """Build one customer cohort per monthly cutoff using only strictly prior sales."""
    needed = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    if needed - set(sales.columns):
        raise ValueError(f"Input missing fields: {sorted(needed - set(sales.columns))}")
    x = sales.copy()
    x["CustomerID"] = x.CustomerID.astype("string").str.strip()
    x["InvoiceNo"] = x.InvoiceNo.astype("string").str.strip()
    x["InvoiceDate"] = pd.to_datetime(x.InvoiceDate, errors="coerce")
    x["Quantity"] = pd.to_numeric(x.Quantity, errors="coerce")
    x["UnitPrice"] = pd.to_numeric(x.UnitPrice, errors="coerce")
    x = x.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"])
    x = x[(x.CustomerID != "") & (x.InvoiceNo != "") & ~x.InvoiceNo.str.upper().str.startswith("C")
          & (x.Quantity > 0) & (x.UnitPrice > 0)]
    x["_revenue"] = x.Quantity * x.UnitPrice
    first, last = x.InvoiceDate.min().normalize(), x.InvoiceDate.max().normalize()
    earliest = (first + pd.Timedelta(days=max(120, horizon_days + 30))).to_period("M").to_timestamp()
    latest = last - pd.Timedelta(days=horizon_days)
    cutoffs = pd.date_range(earliest, latest, freq="MS")
    if len(cutoffs) < 3:
        raise ValueError("At least three monthly temporal snapshots are required for training and holdout.")
    rows = []
    for cutoff in cutoffs:
        prior = x[x.InvoiceDate < cutoff]
        future = x[(x.InvoiceDate >= cutoff) & (x.InvoiceDate < cutoff + pd.Timedelta(days=horizon_days))]
        if prior.empty:
            continue
        snap = prior.groupby("CustomerID", sort=False).agg(
            first_purchase_date=("InvoiceDate", "min"), last_purchase_date=("InvoiceDate", "max"),
            frequency_orders=("InvoiceNo", "nunique"), monetary_gbp=("_revenue", "sum"),
            total_units=("Quantity", "sum"), unique_products=("StockCode", "nunique"),
            transaction_lines=("InvoiceNo", "size")).reset_index()
        snapshot_day = cutoff.normalize()
        snap["recency_days"] = (snapshot_day - snap.last_purchase_date.dt.normalize()).dt.days
        snap["customer_tenure_days"] = (snap.last_purchase_date.dt.normalize() - snap.first_purchase_date.dt.normalize()).dt.days
        outcomes = future.groupby("CustomerID").agg(
            _future_orders=("InvoiceNo", "nunique"), _future_revenue=("_revenue", "sum"))
        snap = snap.merge(outcomes, on="CustomerID", how="left")
        snap[["_future_orders", "_future_revenue"]] = snap[["_future_orders", "_future_revenue"]].fillna(0)
        snap["repeat_purchase_90d"] = snap._future_orders.gt(0).astype("int8")
        snap["revenue_90d_gbp"] = snap._future_revenue.round(2)
        snap["snapshot_date"] = cutoff.date().isoformat()
        rows.append(snap.drop(columns=["first_purchase_date", "last_purchase_date", "_future_orders", "_future_revenue"]))
    return pd.concat(rows, ignore_index=True), [d.date().isoformat() for d in cutoffs]


def fit_and_evaluate(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit on earlier cutoffs and evaluate once on the final temporal cutoff."""
    dates = sorted(snapshots.snapshot_date.unique())
    if len(dates) < 3:
        raise ValueError("Need at least three usable snapshots.")
    holdout_date = dates[-1]
    train = snapshots[snapshots.snapshot_date < holdout_date].copy()
    test = snapshots[snapshots.snapshot_date == holdout_date].copy()
    if train.repeat_purchase_90d.nunique() < 2 or test.repeat_purchase_90d.nunique() < 2:
        raise ValueError("Temporal training and holdout cohorts must contain both outcome classes.")
    classifier = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
    classifier.fit(train[FEATURES], train.repeat_purchase_90d)
    probability = classifier.predict_proba(test[FEATURES])[:, 1]
    predicted = (probability >= .5).astype(int)
    y = test.repeat_purchase_90d.to_numpy()
    class_metrics = {"accuracy": float(accuracy_score(y, predicted)),
                     "precision": float(precision_score(y, predicted, zero_division=0)),
                     "recall": float(recall_score(y, predicted, zero_division=0)),
                     "f1": float(f1_score(y, predicted, zero_division=0)),
                     "roc_auc": float(roc_auc_score(y, probability)),
                     "average_precision": float(average_precision_score(y, probability)),
                     "confusion_matrix_labels_0_1": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}
    revenue_model = HistGradientBoostingRegressor(max_iter=100, learning_rate=.08, max_leaf_nodes=15,
                                                   l2_regularization=1.0, random_state=42)
    revenue_model.fit(train[FEATURES], np.log1p(train.revenue_90d_gbp.clip(lower=0)))
    estimate = np.expm1(revenue_model.predict(test[FEATURES])).clip(min=0)
    actual = test.revenue_90d_gbp.to_numpy()
    revenue_metrics = {"mae_gbp": float(mean_absolute_error(actual, estimate)),
                       "rmse_gbp": float(np.sqrt(mean_squared_error(actual, estimate))),
                       "r2": float(r2_score(actual, estimate)) if len(actual) > 1 else None,
                       "zero_baseline_mae_gbp": float(mean_absolute_error(actual, np.zeros_like(actual)))}
    # Expected value combines repeat probability and conditional positive spend forecast.
    positive_train = train[train.revenue_90d_gbp > 0]
    conditional = HistGradientBoostingRegressor(max_iter=100, learning_rate=.08, max_leaf_nodes=15,
                                                 l2_regularization=1.0, random_state=43)
    conditional.fit(positive_train[FEATURES], np.log1p(positive_train.revenue_90d_gbp))
    conditional_spend = np.expm1(conditional.predict(test[FEATURES])).clip(min=0)
    expected = probability * conditional_spend
    test["repeat_purchase_probability"] = probability
    test["predicted_repeat_purchase_90d"] = predicted
    test["predicted_90d_revenue_gbp"] = estimate.round(2)
    test["expected_90d_value_gbp"] = expected.round(2)
    report = {"horizon_days": HORIZON_DAYS, "target": "one or more distinct invoices in the next 90 days",
              "feature_policy": "All predictors use invoices strictly before each snapshot cutoff; CustomerID is not a predictor.",
              "training_snapshot_dates": dates[:-1], "holdout_snapshot_date": holdout_date,
              "training_rows": int(len(train)), "holdout_rows": int(len(test)),
              "training_customers": int(train.CustomerID.nunique()), "holdout_customers": int(test.CustomerID.nunique()),
              "repeat_purchase_metrics": class_metrics, "revenue_metrics": revenue_metrics,
              "value_definition": "Expected 90-day gross sales proxy = predicted repeat probability × predicted positive 90-day gross sales; not margin-based lifetime profit.",
              "expected_value_metrics": {"mae_gbp": float(mean_absolute_error(actual, expected)),
                                         "rmse_gbp": float(np.sqrt(mean_squared_error(actual, expected)))} }
    return test, report


def run(input_path: Path = INPUT, output_path: Path = PREDICTIONS,
        report_path: Path = REPORT, database_path: Path = DATABASE) -> dict:
    cols = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
    sales = pd.read_csv(input_path, usecols=cols, dtype={"InvoiceNo": "string", "StockCode": "string", "CustomerID": "string"})
    snapshots, cutoffs = make_temporal_snapshots(sales)
    predictions, report = fit_and_evaluate(snapshots)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    report["all_snapshot_dates"] = cutoffs
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with sqlite3.connect(database_path) as db:
        predictions.to_sql("repeat_purchase_holdout", db, if_exists="replace", index=False)
        snapshots.to_sql("repeat_purchase_snapshots", db, if_exists="replace", index=False, chunksize=1000)
    return report


if __name__ == "__main__":
    result = run()
    print(f"Holdout {result['holdout_snapshot_date']}: {result['holdout_rows']:,} customers")
    print("Repeat purchase metrics:", json.dumps(result["repeat_purchase_metrics"], indent=2))
    print("90-day gross revenue MAE: £{:,.2f}".format(result["revenue_metrics"]["mae_gbp"]))
