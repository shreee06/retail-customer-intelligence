"""Transparent RFM segmentation (band rules = default, quartile rules) and a KMeans comparison.

Monetary is NET spend (`monetary_gbp` = purchases minus cancellations). Customers
with zero or negative net spend are not ranked; they form their own flagged group.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data/processed/customer_features.csv"
OUTPUT = ROOT / "data/processed/customer_segments.csv"
PROFILE = ROOT / "reports/generated/segmentation_profile.json"

FLAGGED_GROUP = "Flagged: zero/negative net spend"
REQUIRED = {"CustomerID", "recency_days", "frequency_orders", "monetary_gbp", "net_spend_nonpositive"}

# Band rules (days since last purchase, counted from the reference date; orders).
ACTIVE_MAX_DAYS = 60       # bought within ~2 months
DORMANT_MIN_DAYS = 181     # silent for more than ~6 months
FREQUENT_MIN_ORDERS = 4    # 4+ distinct orders in the year
KMEANS_K_RANGE = range(3, 9)


def split_flagged(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ranked customers, flagged customers with zero/negative net spend)."""
    missing = REQUIRED - set(features.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")
    if features.empty:
        raise ValueError("Cannot segment an empty customer table")
    flagged = features["net_spend_nonpositive"].astype(str).str.lower().eq("true")
    return features[~flagged].copy(), features[flagged].copy()


def quartile_cutoffs(ranked: pd.DataFrame) -> dict:
    """Cut-offs come from the ranked customers only (flagged ones never move them)."""
    return {
        "recency_days_q25": float(ranked.recency_days.quantile(.25)),
        "recency_days_q75": float(ranked.recency_days.quantile(.75)),
        "frequency_orders_q75": float(ranked.frequency_orders.quantile(.75)),
        "monetary_gbp_q75": float(ranked.monetary_gbp.quantile(.75)),
    }


def quartile_segments(ranked: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Original rules: later lines override earlier ones; 'Developing' is the leftover."""
    cut = quartile_cutoffs(ranked)
    r, f, m = ranked.recency_days, ranked.frequency_orders, ranked.monetary_gbp
    seg = pd.Series("Developing", index=ranked.index)
    seg[r.le(cut["recency_days_q25"])] = "Recent"
    seg[f.ge(cut["frequency_orders_q75"])] = "Loyal"
    seg[r.ge(cut["recency_days_q75"]) & (f.ge(cut["frequency_orders_q75"]) | m.ge(cut["monetary_gbp_q75"]))] = "At Risk"
    seg[r.le(cut["recency_days_q25"]) & f.ge(cut["frequency_orders_q75"]) & m.ge(cut["monetary_gbp_q75"])] = "Champions"
    return seg, cut


def band_segments(
    ranked: pd.DataFrame,
    active_max_days: int = ACTIVE_MAX_DAYS,
    dormant_min_days: int = DORMANT_MIN_DAYS,
    frequent_min_orders: int = FREQUENT_MIN_ORDERS,
) -> pd.Series:
    """Recency band x frequency band grid; every customer lands in exactly one cell."""
    r, f = ranked.recency_days, ranked.frequency_orders
    active = r.le(active_max_days)
    dormant = r.ge(dormant_min_days)
    cooling = ~active & ~dormant
    frequent = f.ge(frequent_min_orders)
    repeat = f.between(2, frequent_min_orders - 1)
    seg = pd.Series("Lost", index=ranked.index)  # one/occasional buyers silent 181+ days
    seg[cooling & ~frequent] = "Cooling Down"
    seg[active & repeat] = "Promising"
    seg[active & f.eq(1)] = "New"
    seg[frequent & ~active] = "At Risk"
    seg[frequent & active] = "Champions"
    return seg


def kmeans_segments(ranked: pd.DataFrame, k: int | None = None, seed: int = 42) -> tuple[pd.Series, dict]:
    """KMeans on log-scaled, standardised R, F, M. k = best silhouette unless given.

    Clusters are numbered 1..k by descending mean net spend so labels are stable.
    """
    matrix = np.log1p(ranked[["recency_days", "frequency_orders", "monetary_gbp"]].astype(float))
    scaled = StandardScaler().fit_transform(matrix)
    silhouettes = {}
    for candidate in KMEANS_K_RANGE:
        labels = KMeans(n_clusters=candidate, n_init=10, random_state=seed).fit_predict(scaled)
        silhouettes[candidate] = float(silhouette_score(scaled, labels))
    chosen = k or max(silhouettes, key=silhouettes.get)
    labels = pd.Series(KMeans(n_clusters=chosen, n_init=10, random_state=seed).fit_predict(scaled), index=ranked.index)
    order = ranked.groupby(labels).monetary_gbp.mean().sort_values(ascending=False).index
    rename = {old: f"Cluster {rank}" for rank, old in enumerate(order, start=1)}
    return labels.map(rename), {"chosen_k": int(chosen), "silhouette_by_k": silhouettes}


def profile(table: pd.DataFrame, column: str, total_net_spend: float) -> list[dict]:
    """Size, averages and share of total net revenue for each group in `column`."""
    grouped = table.groupby(column).agg(
        customers=("CustomerID", "nunique"),
        mean_recency_days=("recency_days", "mean"),
        mean_frequency_orders=("frequency_orders", "mean"),
        mean_net_spend_gbp=("monetary_gbp", "mean"),
        total_net_spend_gbp=("monetary_gbp", "sum"),
    ).reset_index().rename(columns={column: "segment"})
    grouped["share_of_customers_pct"] = grouped.customers / len(table) * 100
    grouped["share_of_net_revenue_pct"] = grouped.total_net_spend_gbp / total_net_spend * 100
    return grouped.sort_values("total_net_spend_gbp", ascending=False).round(2).to_dict("records")


def assign_segments(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Segment every customer; return the full table and a report dictionary."""
    ranked, flagged = split_flagged(features)
    if ranked.empty:
        raise ValueError("No customers with positive net spend to segment")
    out = features.copy()
    quartile, cutoffs = quartile_segments(ranked)
    bands = band_segments(ranked)
    clusters, cluster_info = kmeans_segments(ranked)
    out["segment"] = bands.reindex(out.index).fillna(FLAGGED_GROUP)  # default: band rules
    out["segment_quartile"] = quartile.reindex(out.index).fillna(FLAGGED_GROUP)
    out["segment_kmeans"] = clusters.reindex(out.index).fillna(FLAGGED_GROUP)
    total = float(out.monetary_gbp.sum())
    report = {
        "reference_date": str(out["snapshot_date"].iloc[0]) if "snapshot_date" in out else None,
        "reference_date_meaning": "One day after the latest invoice; recency_days = days from a customer's last purchase to this date.",
        "customers_total": int(len(out)),
        "customers_ranked": int(len(ranked)),
        "customers_flagged": int(len(flagged)),
        "flagged_customer_ids": sorted(flagged.CustomerID.astype(str)),
        "total_net_spend_gbp": round(total, 2),
        "quartile_cutoffs": cutoffs,
        "band_rules": {
            "recency": f"Active <= {ACTIVE_MAX_DAYS} days; Cooling {ACTIVE_MAX_DAYS + 1}-{DORMANT_MIN_DAYS - 1}; Dormant >= {DORMANT_MIN_DAYS}",
            "frequency": f"One order; Repeat 2-{FREQUENT_MIN_ORDERS - 1}; Frequent >= {FREQUENT_MIN_ORDERS}",
            "segments": {
                "Champions": "Frequent and Active",
                "At Risk": "Frequent and Cooling/Dormant",
                "Promising": "Repeat (2-3 orders) and Active",
                "New": "One order and Active",
                "Cooling Down": "One/Repeat and Cooling",
                "Lost": "One/Repeat and Dormant",
            },
        },
        "quartile_rule_definitions": {
            "Champions": "recency <= Q25 and frequency >= Q75 and net spend >= Q75",
            "At Risk": "recency >= Q75 and (frequency >= Q75 or net spend >= Q75)",
            "Loyal": "frequency >= Q75 (after the two rules above)",
            "Recent": "recency <= Q25 (after the rules above)",
            "Developing": "everyone else",
        },
        "kmeans": cluster_info,
        "agreement_adjusted_rand": {
            "quartile_vs_bands": float(adjusted_rand_score(quartile, bands)),
            "quartile_vs_kmeans": float(adjusted_rand_score(quartile, clusters)),
            "bands_vs_kmeans": float(adjusted_rand_score(bands, clusters)),
        },
        "profiles": {
            "band_rules": profile(out, "segment", total),
            "quartile_rules": profile(out, "segment_quartile", total),
            "kmeans": profile(out, "segment_kmeans", total),
        },
    }
    return out, report


def run(features_path: Path = FEATURES, output_path: Path = OUTPUT, profile_path: Path = PROFILE) -> dict:
    features = pd.read_csv(features_path, dtype={"CustomerID": "string"})
    segments, report = assign_segments(features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(output_path, index=False)
    profile_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(f"Ranked {result['customers_ranked']:,} customers; flagged {result['customers_flagged']}")
    print("Reference date:", result["reference_date"])
    print("Quartile cut-offs:", result["quartile_cutoffs"])
    print("KMeans:", result["kmeans"])
    print("Agreement (ARI):", result["agreement_adjusted_rand"])
    for name, rows in result["profiles"].items():
        print(f"\n{name}")
        print(pd.DataFrame(rows).to_string(index=False))
