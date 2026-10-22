"""Local Streamlit dashboard for customer segments and model metrics."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = ROOT / "data/processed/customer_segments.csv"
PREDICTIONS = ROOT / "data/processed/repeat_purchase_predictions.csv"
EVALUATION = ROOT / "reports/generated/model_evaluation.json"
st.set_page_config(page_title="Retail Customer Intelligence", page_icon="🛍️", layout="wide")
st.title("Retail Customer Intelligence Platform")
st.caption("Customer segments, repeat-purchase propensity, and a 90-day gross-sales value proxy")

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

if not SEGMENTS.exists():
    st.info("Build analysis outputs first: run the segmentation and repeat-purchase commands in the README.")
    st.stop()
segments = load_csv(SEGMENTS)
required = {"segment", "CustomerID", "recency_days", "frequency_orders", "monetary_gbp"}
if required - set(segments.columns):
    st.error(f"Segment output is missing columns: {sorted(required - set(segments.columns))}")
    st.stop()
profiles = segments.groupby("segment", as_index=False).agg(
    customers=("CustomerID", "nunique"), mean_recency_days=("recency_days", "mean"),
    median_orders=("frequency_orders", "median"), mean_gross_sales_gbp=("monetary_gbp", "mean"))
a, b, c = st.columns(3)
a.metric("Customers", f"{segments.CustomerID.nunique():,}")
a.metric("Gross sales in cohort", f"£{segments.monetary_gbp.sum():,.0f}")
b.metric("Segments", str(segments.segment.nunique()))
c.metric("Snapshot", str(segments.snapshot_date.iloc[0]) if "snapshot_date" in segments else "Full-period cohort")
st.subheader("Segment profile")
st.bar_chart(profiles.set_index("segment")["customers"])
st.dataframe(profiles.round(2), use_container_width=True, hide_index=True)
chosen = st.selectbox("Explore segment", sorted(segments.segment.unique()))
subset = segments[segments.segment == chosen]
st.subheader(f"{chosen}: behavior summary")
summary = {"customers": int(subset.CustomerID.nunique()),
           "mean_recency_days": round(float(subset.recency_days.mean()), 1),
           "median_orders": float(subset.frequency_orders.median()),
           "mean_gross_sales_gbp": round(float(subset.monetary_gbp.mean()), 2)}
x, y, z = st.columns(3)
x.metric("Mean recency", f"{summary['mean_recency_days']:.0f} days")
y.metric("Median orders", f"{summary['median_orders']:.0f}")
z.metric("Mean gross sales", f"£{summary['mean_gross_sales_gbp']:,.2f}")
if PREDICTIONS.exists() and EVALUATION.exists():
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    st.subheader("Repeat-purchase and 90-day value model")
    st.caption(f"Temporal holdout: {evaluation['holdout_snapshot_date']} · {evaluation['horizon_days']}-day horizon · {evaluation['holdout_rows']:,} customers")
    metrics = evaluation["repeat_purchase_metrics"]
    cards = st.columns(4)
    cards[0].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    cards[1].metric("Precision", f"{metrics['precision']:.3f}")
    cards[2].metric("Recall", f"{metrics['recall']:.3f}")
    cards[3].metric("F1", f"{metrics['f1']:.3f}")
    st.write(f"Holdout gross-revenue MAE: £{evaluation['revenue_metrics']['mae_gbp']:,.2f}. Expected-value MAE: £{evaluation['expected_value_metrics']['mae_gbp']:,.2f}.")
    st.caption("This is a 90-day gross-sales proxy, not profit or full lifetime value. Holdout actuals are shown for evaluation.")
else:
    st.info("Run the repeat-purchase module to add temporal holdout metrics and value estimates.")
st.caption("For exploration only; validate campaigns and model thresholds before operational use.")
