"""Local Streamlit dashboard. Reads only generated pipeline outputs; shows no invented numbers.

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(os.environ.get("RETAIL_DASHBOARD_ROOT", Path(__file__).resolve().parents[1]))
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "generated"

# name -> (file, command that creates it)
SOURCES = {
    "data_quality": (REPORTS / "data_quality.json", "python -m src.data.pipeline"),
    "segments": (PROCESSED / "customer_segments.csv",
                 "python -m src.data.pipeline && python -m src.features.customer_features && python -m src.segmentation.rfm"),
    "segment_profile": (REPORTS / "segmentation_profile.json", "python -m src.segmentation.rfm"),
    "repeat": (REPORTS / "model_evaluation.json", "python -m src.models.repeat_purchase"),
    "clv_forecast": (PROCESSED / "clv_forecast.csv", "python -m src.models.clv"),
    "clv_eval": (REPORTS / "clv_evaluation.json", "python -m src.models.clv"),
}
PAGES = [
    "Business overview",
    "Segments",
    "Customer behaviour",
    "Repeat-purchase model",
    "90-day predicted customer value",
    "Recommendations",
    "Methodology and limitations",
]
SEGMENT_METHODS = {
    "Band rules (default)": ("segment", "band_rules"),
    "Quartile rules (old)": ("segment_quartile", "quartile_rules"),
    "KMeans (comparison)": ("segment_kmeans", "kmeans"),
}
FLAGGED = "Flagged: zero/negative net spend"
COLOR = {"model": "#2a6fdb", "baseline": "#9aa5b1", "actual": "#e07b00"}


# --- loading ---------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _read(path: str, modified: float):
    if path.endswith(".json"):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.read_csv(path, dtype={"CustomerID": "string"})


def load(name: str):
    """Return the parsed file, or None if it is missing or unreadable."""
    path, _ = SOURCES[name]
    if not path.is_file():
        return None
    try:
        return _read(str(path), path.stat().st_mtime)
    except Exception as exc:  # corrupt file: tell the user instead of crashing
        st.error(f"Could not read {path.name}: {exc}")
        return None


def require(*names: str) -> dict | None:
    """Load all named sources; if any are missing say which step to run."""
    data = {name: load(name) for name in names}
    missing = [name for name, value in data.items() if value is None]
    if missing:
        st.warning("Some outputs this section needs are missing. Run the step below, then reload the page.")
        for name in missing:
            path, command = SOURCES[name]
            st.markdown(f"- `{path.relative_to(ROOT)}` is missing. Run: `{command}`")
        return None
    return data


def gbp(value, digits: int = 0) -> str:
    return f"£{value:,.{digits}f}"


def pct(value, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


def stage_rows(quality: dict) -> dict:
    return {row["stage"]: row["rows"] for row in quality["row_counts"]}


# --- pages -------------------------------------------------------------------


def page_overview():
    st.header("Business overview")
    data = require("data_quality", "segments", "segment_profile")
    if not data:
        return
    quality, segments, profile = data["data_quality"], data["segments"], data["segment_profile"]
    stages, flags = stage_rows(quality), quality["quality"]["flag_counts"]
    gross, cancelled = segments.gross_spend_gbp.sum(), segments.cancelled_spend_gbp.sum()

    c = st.columns(4)
    c[0].metric("Source rows loaded", f"{stages['source_rows_loaded']:,}")
    c[1].metric("Candidate sales rows", f"{stages['customer_sales_candidates']:,}")
    c[2].metric("Customers", f"{profile['customers_total']:,}")
    c[3].metric("Net revenue", gbp(profile["total_net_spend_gbp"]))
    c = st.columns(4)
    c[0].metric("Gross spend (purchases)", gbp(gross))
    c[1].metric("Cancelled / returned", gbp(cancelled))
    c[2].metric("Rows in exact-duplicate groups", f"{flags['exact_duplicate_row']:,}")
    c[3].metric("Extra duplicate copies excluded", f"{flags['exact_duplicate_extra_copy']:,}")
    st.caption(
        f"Net revenue = gross purchases minus cancellations and returns. Customers flagged for zero or negative "
        f"net spend: {profile['customers_flagged']}. Non-product lines excluded (postage, manual, fees): "
        f"{flags['non_product_line']:,} rows. Reference date for recency: {profile['reference_date']}."
    )

    st.subheader("Why rows were left out of the candidate sales")
    reasons = pd.Series(quality["quality"]["candidate_exclusion_reason_counts"])
    reasons = reasons[reasons > 0].sort_values()
    fig = px.bar(x=reasons.values, y=reasons.index, orientation="h",
                 labels={"x": "Rows (a row can have several reasons)", "y": ""})
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Row counts by stage")
    table = pd.DataFrame(quality["row_counts"]).rename(columns={"stage": "Stage", "rows": "Rows", "meaning": "Meaning"})
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(f"Source file: {quality['source_file']} (SHA-256 {quality['source_sha256'][:16]}…). "
               f"Pipeline run: {quality['run_timestamp_utc']}.")


def page_segments():
    st.header("Segments")
    data = require("segments", "segment_profile")
    if not data:
        return
    segments, profile = data["segments"], data["segment_profile"]
    method = st.radio("Segmentation method", list(SEGMENT_METHODS), horizontal=True)
    column, key = SEGMENT_METHODS[method]
    rows = pd.DataFrame(profile["profiles"][key])
    names = sorted(rows.segment)
    chosen = st.multiselect("Show segments", names, default=names)
    shown = rows[rows.segment.isin(chosen)]
    if shown.empty:
        st.info("Pick at least one segment.")
        return

    c = st.columns(2)
    fig = px.bar(shown.sort_values("customers", ascending=False), x="segment", y="share_of_customers_pct",
                 labels={"share_of_customers_pct": "% of customers", "segment": ""}, title="Share of customers")
    c[0].plotly_chart(fig, use_container_width=True)
    fig = px.bar(shown.sort_values("total_net_spend_gbp", ascending=False), x="segment", y="share_of_net_revenue_pct",
                 labels={"share_of_net_revenue_pct": "% of net revenue", "segment": ""}, title="Share of net revenue")
    c[1].plotly_chart(fig, use_container_width=True)

    st.subheader("Profile")
    table = shown.rename(columns={
        "segment": "Segment", "customers": "Customers", "share_of_customers_pct": "% customers",
        "mean_recency_days": "Avg recency (days)", "mean_frequency_orders": "Avg orders",
        "mean_net_spend_gbp": "Avg net spend (£)", "total_net_spend_gbp": "Total net spend (£)",
        "share_of_net_revenue_pct": "% net revenue"})
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("Rules")
    if key == "band_rules":
        st.write(f"Recency: {profile['band_rules']['recency']}. Frequency: {profile['band_rules']['frequency']}.")
        st.table(pd.DataFrame(profile["band_rules"]["segments"].items(), columns=["Segment", "Rule"]))
    elif key == "quartile_rules":
        cut = profile["quartile_cutoffs"]
        st.write(f"Cut-offs from the data: recency Q25 {cut['recency_days_q25']:.0f} days, Q75 {cut['recency_days_q75']:.0f} days; "
                 f"frequency Q75 {cut['frequency_orders_q75']:.0f} orders; net spend Q75 {gbp(cut['monetary_gbp_q75'], 2)}.")
        st.table(pd.DataFrame(profile["quartile_rule_definitions"].items(), columns=["Segment", "Rule"]))
    else:
        info = profile["kmeans"]
        st.write(f"KMeans on log-scaled, standardised recency, frequency and net spend. k = {info['chosen_k']} "
                 f"(best silhouette {info['silhouette_by_k'][str(info['chosen_k'])]:.3f}). Clusters are numbered by average net spend.")
    st.caption(f"Customers in the flagged group (zero or negative net spend, not ranked): {profile['customers_flagged']}.")

    st.subheader("Customers in the chosen segments")
    subset = segments[segments[column].isin(chosen)].sort_values("monetary_gbp", ascending=False)
    st.write(f"{len(subset):,} customers. Top 200 by net spend:")
    cols = ["CustomerID", column, "recency_days", "frequency_orders", "gross_spend_gbp", "cancelled_spend_gbp", "monetary_gbp"]
    st.dataframe(subset[cols].head(200).rename(columns={
        column: "Segment", "recency_days": "Recency (days)", "frequency_orders": "Orders",
        "gross_spend_gbp": "Gross spend (£)", "cancelled_spend_gbp": "Cancelled (£)", "monetary_gbp": "Net spend (£)"}),
        hide_index=True, use_container_width=True)


def page_behaviour():
    st.header("Customer behaviour")
    data = require("segments")
    if not data:
        return
    segments = data["segments"]
    ranked = segments[segments.segment != FLAGGED]
    names = sorted(ranked.segment.unique())
    c = st.columns(3)
    chosen = c[0].multiselect("Segments", names, default=names)
    recency = c[1].slider("Recency (days since last purchase)", 0, int(ranked.recency_days.max()),
                          (0, int(ranked.recency_days.max())))
    min_orders = c[2].slider("Minimum orders", 1, int(min(ranked.frequency_orders.max(), 50)), 1)
    view = ranked[ranked.segment.isin(chosen) & ranked.recency_days.between(*recency)
                  & (ranked.frequency_orders >= min_orders)]
    st.write(f"{len(view):,} of {len(ranked):,} ranked customers match. The {len(segments) - len(ranked)} flagged "
             f"customers (zero or negative net spend) are not shown.")
    if view.empty:
        st.info("No customers match these filters.")
        return

    c = st.columns(2)
    c[0].plotly_chart(px.histogram(view, x="recency_days", nbins=40, color="segment",
                                   labels={"recency_days": "Recency (days)"}, title="Recency"), use_container_width=True)
    c[1].plotly_chart(px.histogram(view, x="frequency_orders", nbins=40, color="segment", log_y=True,
                                   labels={"frequency_orders": "Orders"}, title="Frequency (log count)"),
                      use_container_width=True)
    c = st.columns(2)
    c[0].plotly_chart(px.histogram(view, x="monetary_gbp", nbins=50, log_x=True, color="segment",
                                   labels={"monetary_gbp": "Net spend (£, log scale)"}, title="Net spend"),
                      use_container_width=True)
    c[1].plotly_chart(px.box(view, x="segment", y="monetary_gbp", log_y=True,
                             labels={"monetary_gbp": "Net spend (£, log scale)", "segment": ""},
                             title="Net spend by segment"), use_container_width=True)
    sample = view.sample(min(len(view), 2500), random_state=0)
    st.plotly_chart(px.scatter(sample, x="recency_days", y="monetary_gbp", color="segment", log_y=True,
                               size_max=6, opacity=.6, hover_data=["CustomerID", "frequency_orders"],
                               labels={"recency_days": "Recency (days)", "monetary_gbp": "Net spend (£, log scale)"},
                               title=f"Recency vs net spend ({len(sample):,} customers shown)"),
                    use_container_width=True)
    st.subheader("Summary of the filtered customers")
    summary = view[["recency_days", "frequency_orders", "gross_spend_gbp", "cancelled_spend_gbp", "monetary_gbp"]].describe().T
    summary.index = ["Recency (days)", "Orders", "Gross spend (£)", "Cancelled (£)", "Net spend (£)"]
    st.dataframe(summary.round(2), use_container_width=True)


def page_repeat():
    st.header("Repeat-purchase model")
    data = require("repeat")
    if not data:
        return
    r = data["repeat"]
    st.write("**Question:** will a customer buy at least once in the 90 days after a cutoff date? "
             "The table compares two simple baselines with logistic regression, all on the same unseen test period.")
    labels = {"baseline_always_yes_metrics": "Baseline: everyone repeats",
              "baseline_recent_purchase_rule_metrics": "Baseline: bought in the last 90 days",
              "repeat_purchase_metrics": "Logistic regression"}
    rows = []
    for key, label in labels.items():
        m = r[key]
        rows.append({"Model": label, "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1": m["f1"], "Accuracy": m["accuracy"]})
    table = pd.DataFrame(rows)
    st.dataframe(table.style.format({c: "{:.3f}" for c in table.columns[1:]}), hide_index=True, use_container_width=True)
    long = table.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig = px.bar(long, x="Metric", y="Value", color="Model", barmode="group",
                 color_discrete_sequence=[COLOR["baseline"], "#c4cdd6", COLOR["model"]])
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Test cohort: {r['holdout_rows']:,} customers; {r['holdout_repeat_rate'] * 100:.1f}% of them did buy again. "
               "F1 favours 'everyone repeats' because over half of customers repeat; compare the AUC columns for ranking quality.")
    tn, fp = r["repeat_purchase_metrics"]["confusion_matrix_labels_0_1"][0]
    fn, tp = r["repeat_purchase_metrics"]["confusion_matrix_labels_0_1"][1]
    st.write(f"Logistic regression at a 0.5 threshold: {tp:,} correct repeat calls, {fp:,} false alarms, "
             f"{fn:,} missed repeaters, {tn:,} correct non-repeat calls.")

    st.subheader("How the split avoids leakage")
    st.markdown(
        "- **Cutoff dates are monthly.** At each cutoff the model only sees purchases *before* it. The target is whether "
        "the customer buys in the next 90 days.\n"
        "- **The test is the latest cutoff.** Training uses only cutoffs whose whole 90-day outcome window ends *before* "
        "the test starts, so no training answer overlaps the test period.")
    split = pd.DataFrame({"Role": ["Train"] * len(r["training_snapshot_dates"]) + ["Test"],
                          "Cutoff date": r["training_snapshot_dates"] + [r["holdout_snapshot_date"]],
                          "Outcome window ends": r["training_outcome_windows_end"] + [r["holdout_outcome_window_end"]]})
    st.dataframe(split, hide_index=True, use_container_width=True)
    st.caption(f"Training rows: {r['training_rows']:,}. Test rows: {r['holdout_rows']:,}. "
               "The same customers can appear in both; that is fine because their features come only from the past.")


def page_value():
    st.header("90-day predicted customer value")
    data = require("clv_eval", "clv_forecast")
    if not data:
        return
    ev, forecast = data["clv_eval"], data["clv_forecast"]
    net_all = ev["all_ranked_customers"]["net"]
    ratio = net_all["model"]["predicted_over_actual"]
    st.error(
        f"**Ranking tool only, not a validated forecast of totals.** On the holdout the model predicted "
        f"{gbp(net_all['model']['total_predicted_gbp'])} of net spend against {gbp(net_all['model']['total_actual_gbp'])} actual: "
        f"it under-predicts the total by about {(1 - ratio) * 100:.0f}%. Use it to order customers, not to budget pounds.")
    st.write(
        "**What it is:** a *prediction* of each customer's net spend in the 90 days after the cutoff "
        f"({ev['holdout_window']}), made only from purchases before {ev['cutoff']}. It is not historical spend. "
        "Net spend = purchases minus cancellations and returns.")
    st.caption(ev["definition"])

    st.subheader("Holdout comparison against the baseline")
    scope = st.radio("Customers", ["All ranked customers", "Only customers with 2+ purchase days"], horizontal=True)
    block = ev["all_ranked_customers" if scope.startswith("All") else "multi_purchase_day_only"]
    rows = []
    for kind in ("net", "gross"):
        for who, label in (("model", "Predicted value model"), ("baseline", "Baseline (past spend rate)")):
            m = block[kind][who]
            rows.append({"Measure": kind.capitalize(), "Method": label, "Customers": m["customers"],
                         "Predicted total (£)": m["total_predicted_gbp"], "Actual total (£)": m["total_actual_gbp"],
                         "Predicted / actual": m["predicted_over_actual"], "MAE (£)": m["mae_gbp"],
                         "Rank correlation (Spearman)": m["spearman"]})
    table = pd.DataFrame(rows)
    st.dataframe(table.style.format({"Predicted total (£)": "{:,.0f}", "Actual total (£)": "{:,.0f}",
                                     "Predicted / actual": "{:.2f}", "MAE (£)": "{:,.2f}",
                                     "Rank correlation (Spearman)": "{:.3f}"}),
                 hide_index=True, use_container_width=True)
    st.caption("Baseline = the customer's average daily spend before the cutoff, times 90. The baseline's total is close, "
               "but the model ranks customers slightly better (higher rank correlation, lower MAE).")

    st.subheader("Do higher predictions mean higher actual spend?")
    kind = st.radio("Measure", ["Net", "Gross"], horizontal=True)
    deciles = pd.DataFrame(ev["deciles_net_all_ranked" if kind == "Net" else "deciles_gross_all_ranked"])
    fig = go.Figure()
    fig.add_bar(x=deciles.decile, y=deciles.mean_predicted_gbp, name="Average predicted", marker_color=COLOR["model"])
    fig.add_bar(x=deciles.decile, y=deciles.mean_actual_gbp, name="Average actual", marker_color=COLOR["actual"])
    fig.update_layout(barmode="group", height=400, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=f"{kind} spend per customer (£)",
                      xaxis=dict(title="Decile of predicted value (10 = highest)", dtick=1))
    st.plotly_chart(fig, use_container_width=True)
    top = deciles.iloc[-1]
    st.caption(f"Actual spend rises with predicted decile, so the ordering works. The top decile ({int(top.customers)} customers) "
               f"produced {pct(top.total_actual_share_pct)} of actual {kind.lower()} spend. Every decile's actual is above "
               "its prediction, which is the under-prediction shown above.")

    st.subheader("Customers the model cannot see")
    ft = ev["first_time_customers_in_window"]
    st.write(f"{ft['first_time_customers']:,} customers made their first purchase inside the 90-day window. They are not "
             f"forecast, yet they produced {gbp(ft['their_actual_net_gbp'])} of net spend, {pct(ft['net_share_pct'])} "
             f"of the window's {gbp(ft['all_actual_net_gbp'])}.")

    st.subheader("Highest predicted customers")
    ranked = forecast[~forecast.excluded_nonpositive_calibration_net.astype(bool)].sort_values("predicted_net", ascending=False)
    top20 = ranked.head(20)[["CustomerID", "purchase_days", "net_spend", "predicted_net", "actual_net"]].rename(columns={
        "purchase_days": "Purchase days (before cutoff)", "net_spend": "Past net spend (£)",
        "predicted_net": "Predicted 90-day net (£)", "actual_net": "Actual 90-day net (£, holdout)"})
    st.dataframe(top20.round(2), hide_index=True, use_container_width=True)
    cov = ev["coverage"]
    st.caption(f"Covered: {cov['ranked_customers']:,} customers with purchases before the cutoff. Excluded: "
               f"{cov['excluded_nonpositive_calibration_net']} with zero or negative net spend before the cutoff. "
               f"Single-purchase-day customers ({cov['single_purchase_day_customers_in_ranked']:,}) get the average order value. "
               "The 'actual' column is holdout data, shown for evaluation only.")


def _finding(label: str, title: str, evidence: str):
    with st.container(border=True):
        st.markdown(f"**{label}** — {title}")
        st.write(evidence)


def page_recommendations():
    st.header("Recommendations")
    data = require("segment_profile", "repeat", "clv_eval", "segments")
    if not data:
        return
    prof = {r["segment"]: r for r in data["segment_profile"]["profiles"]["band_rules"]}
    repeat, ev, segments = data["repeat"], data["clv_eval"], data["segments"]
    st.caption("A *finding* is a number in this dataset. A *hypothesis* is an idea the data suggests but has not tested; "
               "it needs a controlled test (a treated group and a held-out comparison group) before acting at scale.")

    st.subheader("Data-supported findings")
    if {"Champions", "At Risk", "New"} <= set(prof):
        ch, ar = prof["Champions"], prof["At Risk"]
        _finding("Data-supported finding", "Revenue is concentrated in Champions",
                 f"Champions are {pct(ch['share_of_customers_pct'])} of customers ({int(ch['customers']):,}) but "
                 f"{pct(ch['share_of_net_revenue_pct'])} of net revenue.")
        _finding("Data-supported finding", "A frequent-but-quiet group exists",
                 f"At Risk customers ({int(ar['customers']):,}, {pct(ar['share_of_customers_pct'])} of customers) ordered on average "
                 f"{ar['mean_frequency_orders']:.1f} times, average net spend {gbp(ar['mean_net_spend_gbp'])}, and have been silent "
                 f"for {ar['mean_recency_days']:.0f} days on average. They hold {pct(ar['share_of_net_revenue_pct'])} of net revenue.")
    m, b = repeat["repeat_purchase_metrics"], repeat["baseline_recent_purchase_rule_metrics"]
    _finding("Data-supported finding", "Repeat purchase is predictable, but only moderately better than a simple rule",
             f"On unseen data logistic regression reached ROC-AUC {m['roc_auc']:.3f} vs {b['roc_auc']:.3f} for the "
             f"'bought in the last 90 days' rule (PR-AUC {m['pr_auc']:.3f} vs {b['pr_auc']:.3f}).")
    top = ev["deciles_net_all_ranked"][-1]
    _finding("Data-supported finding", "Predicted value orders customers usefully",
             f"The top predicted decile produced {pct(top['total_actual_share_pct'])} of actual net spend in the holdout "
             f"(average {gbp(top['mean_actual_gbp'])} per customer). Totals are under-predicted, so use it for ranking only.")
    ft = ev["first_time_customers_in_window"]
    _finding("Data-supported finding", "New customers are a large share of near-term spend",
             f"Customers who first bought inside the 90-day window made up {pct(ft['net_share_pct'])} of its net spend, "
             "and none of them can be forecast by the value model.")
    gross, cancelled = segments.gross_spend_gbp.sum(), segments.cancelled_spend_gbp.sum()
    _finding("Data-supported finding", "Cancellations are material but modest",
             f"Cancelled and returned value is {gbp(cancelled)}, {pct(cancelled / gross)} of gross spend.")

    st.subheader("Hypotheses to test")
    n = lambda seg: f"{int(prof[seg]['customers']):,}" if seg in prof else "the"
    _finding("Hypothesis to test", "A win-back offer to At Risk customers recovers spend",
             f"Offer the {n('At Risk')} At Risk customers a reminder or incentive; keep a random half untreated. "
             "Measure orders and net spend in the following 90 days in both halves. The data shows the group is valuable "
             "and quiet; it does not show that an offer changes behaviour.")
    _finding("Hypothesis to test", "A second-purchase nudge converts New customers into repeat customers",
             f"The {n('New')} New customers (one order, recently active) could get a follow-up message. Compare the share "
             "who place a second order within 90 days against an untreated group.")
    _finding("Hypothesis to test", "Targeting the highest predicted-value decile beats random targeting",
             "Run retention outreach on the top one or two deciles of the 90-day predicted customer value versus a random "
             "sample of equal size, and compare incremental net spend.")
    _finding("Hypothesis to test", "Reducing contact with Lost customers saves cost without losing revenue",
             f"The {n('Lost')} Lost customers (silent for 181+ days) could be suppressed from routine campaigns for one "
             "cycle. Check that their spend does not fall against a group still contacted.")


def page_methodology():
    st.header("Methodology and limitations")
    quality = load("data_quality")
    profile = load("segment_profile")
    repeat = load("repeat")
    ev = load("clv_eval")
    st.subheader("Data and cleaning")
    if quality:
        st.write(f"Source: {quality['source_file']} (UCI Online Retail). Every source row is kept in the audit file with flags; "
                 "nothing is deleted.")
    else:
        st.info(f"Run `{SOURCES['data_quality'][1]}` to show data details.")
    st.markdown(
        "- **Candidate sales:** valid invoice, date and customer; positive quantity and price; not a cancellation.\n"
        "- **Exact duplicates:** every copy is flagged; only copies after the first are excluded (`exact_duplicate_extra_copy`).\n"
        "- **Non-product lines** (postage, manual entries, carriage, bank charges, discounts, samples, fees) are excluded from "
        "revenue and segmentation (`non_product_line`).\n"
        "- **Gross spend** (`gross_spend_gbp`) = purchases only. **Net spend** (`monetary_gbp` in the files) = purchases minus "
        "valid cancelled/returned product lines. Customers with zero or negative net spend are flagged and kept out of the "
        "ranked segments.")
    st.subheader("Segments")
    if profile:
        st.write(f"Reference date {profile['reference_date']} ({profile['reference_date_meaning']}) "
                 f"Band rules: recency {profile['band_rules']['recency']}; frequency {profile['band_rules']['frequency']}.")
    st.subheader("Repeat-purchase model")
    if repeat:
        st.write(f"{repeat['target']}. {repeat['feature_policy']} {repeat['split_policy']}")
    st.subheader("90-day predicted customer value")
    if ev:
        rules = ev["rules"]
        st.write(f"{ev['definition']} Calibration period {ev['calibration_period']}; holdout window {ev['holdout_window']}.")
        st.markdown(f"- **Purchase unit:** {rules['purchase_unit']}.\n- **Cancellation rate:** {rules['cancellation_rate']}.\n"
                    f"- **Baseline:** {rules['baseline']}.")
        fit = ev["fit_parameters"]["bgnbd_fit"]
        st.caption(f"The purchase-count model's dropout parameters sit at their bounds (a at lower bound: {fit['a_at_lower_bound']}, "
                   f"b at upper bound: {fit['b_at_upper_bound']}): the data shows almost no permanent dropout, so the model "
                   "behaves like a purchase-rate model.")
    st.subheader("Limitations")
    missing_id = (f"{quality['quality']['flag_counts']['missing_customer_id']:,} source rows"
                  if quality else "rows with no CustomerID")
    st.markdown(
        "- **About one year of data.** There is no full seasonal cycle, and metrics come from one test period, so they could "
        "change in another.\n"
        "- **The value forecast ignores seasonality** and under-predicts the autumn surge; it is a ranking tool, not validated "
        "for pound totals.\n"
        "- **First-time customers are invisible to the value model.**\n"
        "- **Segment cut-offs are judgement-based** (60 and 181 days, 4+ orders), not fitted to the data.\n"
        f"- **Customers without an ID** ({missing_id}) cannot be analysed per customer.\n"
        "- **Cancellations** are matched to customers by ID only, not to the original sale, and rates are a rough scaling.\n"
        "- **Correlation, not cause.** Findings describe this dataset; recommendations marked as hypotheses need a controlled test.")


# --- app ---------------------------------------------------------------------

st.set_page_config(page_title="Retail Customer Intelligence", page_icon="🛍️", layout="wide")
st.title("Retail Customer Intelligence Platform")
st.caption("UCI Online Retail data. Every figure comes from the generated pipeline outputs.")
page = st.sidebar.radio("Section", PAGES)
st.sidebar.caption("Net spend = purchases minus cancellations and returns. Gross spend = purchases only.")

{
    PAGES[0]: page_overview,
    PAGES[1]: page_segments,
    PAGES[2]: page_behaviour,
    PAGES[3]: page_repeat,
    PAGES[4]: page_value,
    PAGES[5]: page_recommendations,
    PAGES[6]: page_methodology,
}[page]()
