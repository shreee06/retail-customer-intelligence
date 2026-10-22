"""Transparent, rule-based RFM segmentation for the Online Retail cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data/processed/customer_features.csv"
OUTPUT = ROOT / "data/processed/customer_segments.csv"
PROFILE = ROOT / "reports/generated/segmentation_profile.json"


def assign_rfm_segments(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Assign interpretable segments using cohort quartiles (no fitted model)."""
    required = {"CustomerID", "recency_days", "frequency_orders", "monetary_gbp"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    out = features.copy()
    if out.empty:
        raise ValueError("Cannot segment an empty customer table")
    recency_q25, recency_q75 = (float(out.recency_days.quantile(q)) for q in (.25, .75))
    frequency_q75 = float(out.frequency_orders.quantile(.75))
    monetary_q75 = float(out.monetary_gbp.quantile(.75))
    r = out.recency_days
    f = out.frequency_orders
    m = out.monetary_gbp
    out["segment"] = "Developing"
    out.loc[r.le(recency_q25), "segment"] = "Recent"
    out.loc[f.ge(frequency_q75), "segment"] = "Loyal"
    out.loc[r.ge(recency_q75) & (f.ge(frequency_q75) | m.ge(monetary_q75)), "segment"] = "At Risk"
    out.loc[r.le(recency_q25) & f.ge(frequency_q75) & m.ge(monetary_q75), "segment"] = "Champions"
    thresholds = {"recency_days_q25": recency_q25, "recency_days_q75": recency_q75,
                  "frequency_orders_q75": frequency_q75, "monetary_gbp_q75": monetary_q75}
    return out, thresholds


def run(features_path: Path = FEATURES, output_path: Path = OUTPUT, profile_path: Path = PROFILE) -> dict:
    features = pd.read_csv(features_path)
    segments, thresholds = assign_rfm_segments(features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(output_path, index=False)
    summary = segments.groupby("segment", as_index=False).agg(
        customers=("CustomerID", "nunique"), mean_recency_days=("recency_days", "mean"),
        median_orders=("frequency_orders", "median"), mean_gross_sales_gbp=("monetary_gbp", "mean"))
    result = {"method": "cohort quartile rules", "thresholds": thresholds,
              "customer_count": int(len(segments)), "segment_profile": summary.round(2).to_dict("records"),
              "definitions": {"Champions": "recency <= Q25 and frequency and monetary >= Q75",
                              "At Risk": "recency >= Q75 and either frequency or monetary >= Q75",
                              "Loyal": "frequency >= Q75 after higher-priority rules",
                              "Recent": "recency <= Q25 after higher-priority rules",
                              "Developing": "remaining customers"}}
    profile_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(f"Segmented {report['customer_count']:,} customers -> {OUTPUT}")
    for row in report["segment_profile"]:
        print(f"{row['segment']}: {row['customers']:,}")
