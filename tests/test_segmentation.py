"""Tests for the RFM segmentation logic in src/segmentation/rfm.py."""

import numpy as np
import pandas as pd
import pytest

from src.segmentation.rfm import (
    FLAGGED_GROUP, assign_segments, band_segments, kmeans_segments,
    quartile_cutoffs, quartile_segments, split_flagged,
)


def customers(rows):
    """rows: (id, recency_days, frequency_orders, net_spend)."""
    frame = pd.DataFrame(rows, columns=["CustomerID", "recency_days", "frequency_orders", "monetary_gbp"])
    frame["net_spend_nonpositive"] = frame.monetary_gbp <= 0
    frame["snapshot_date"] = "2011-12-10"
    return frame


def synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return customers([
        (str(10000 + i), int(rng.integers(1, 370)), int(rng.integers(1, 15)), float(rng.integers(20, 5000)))
        for i in range(n)
    ])


# --- flagged group ----------------------------------------------------------


def test_zero_and_negative_net_spend_customers_are_split_out():
    table = customers([("1", 10, 3, 500.0), ("2", 10, 1, 0.0), ("3", 10, 1, -40.0)])

    ranked, flagged = split_flagged(table)

    assert ranked.CustomerID.tolist() == ["1"]
    assert flagged.CustomerID.tolist() == ["2", "3"]


def test_flagged_customers_are_labelled_in_every_segment_column():
    table = synthetic()
    table.loc[0, ["monetary_gbp", "net_spend_nonpositive"]] = [-5.0, True]
    table.loc[1, ["monetary_gbp", "net_spend_nonpositive"]] = [0.0, True]

    out, report = assign_segments(table)

    for column in ("segment", "segment_quartile", "segment_kmeans"):
        assert out.loc[[0, 1], column].eq(FLAGGED_GROUP).all()
        assert out.loc[2:, column].ne(FLAGGED_GROUP).all()
    assert report["customers_flagged"] == 2 and report["customers_ranked"] == len(table) - 2


def test_flagged_customers_do_not_move_the_quartile_cutoffs():
    base = synthetic()
    extra = pd.concat([base, customers([("999999", 1, 99, -1.0)])], ignore_index=True)

    ranked_base, _ = split_flagged(base)
    ranked_extra, _ = split_flagged(extra)

    assert quartile_cutoffs(ranked_base) == quartile_cutoffs(ranked_extra)


def test_missing_columns_and_empty_input_are_rejected():
    with pytest.raises(ValueError):
        split_flagged(synthetic().drop(columns=["net_spend_nonpositive"]))
    with pytest.raises(ValueError):
        split_flagged(synthetic().iloc[0:0])


# --- band rules -------------------------------------------------------------


@pytest.mark.parametrize(
    "recency,orders,expected",
    [
        (1, 4, "Champions"), (60, 4, "Champions"), (61, 4, "At Risk"), (300, 9, "At Risk"),
        (30, 3, "Promising"), (60, 2, "Promising"),
        (30, 1, "New"), (60, 1, "New"),
        (61, 1, "Cooling Down"), (180, 3, "Cooling Down"),
        (181, 1, "Lost"), (365, 3, "Lost"),
    ],
)
def test_band_boundaries(recency, orders, expected):
    table = customers([("1", recency, orders, 100.0)])

    assert band_segments(table).iloc[0] == expected


def test_every_ranked_customer_gets_exactly_one_band_segment():
    ranked, _ = split_flagged(synthetic(500, seed=3))

    seg = band_segments(ranked)

    assert len(seg) == len(ranked) and seg.notna().all()
    assert set(seg) <= {"Champions", "At Risk", "Promising", "New", "Cooling Down", "Lost"}


# --- quartile rules ---------------------------------------------------------


def test_quartile_cutoffs_are_the_data_quartiles():
    ranked = customers([(str(i), r, f, m) for i, (r, f, m) in enumerate(
        [(10, 1, 100.0), (20, 2, 200.0), (30, 3, 300.0), (40, 4, 400.0), (50, 5, 500.0)])])

    cut = quartile_cutoffs(ranked)

    assert cut == {"recency_days_q25": 20.0, "recency_days_q75": 40.0,
                   "frequency_orders_q75": 4.0, "monetary_gbp_q75": 400.0}


def test_quartile_champions_need_recent_frequent_and_high_spend():
    rows = [("c", 5, 20, 9000.0)] + [(f"x{i}", 100 + i, 1, 100.0 + i) for i in range(30)]
    ranked, _ = split_flagged(customers(rows))

    seg, _ = quartile_segments(ranked)

    assert seg.loc[ranked.CustomerID == "c"].iloc[0] == "Champions"
    assert (seg == "Champions").sum() == 1


# --- k-means ----------------------------------------------------------------


def test_kmeans_is_deterministic_and_ordered_by_spend():
    ranked, _ = split_flagged(synthetic(300, seed=1))

    a, info_a = kmeans_segments(ranked, k=4)
    b, _ = kmeans_segments(ranked, k=4)

    assert a.equals(b) and a.nunique() == 4 and info_a["chosen_k"] == 4
    means = ranked.groupby(a).monetary_gbp.mean()
    assert means["Cluster 1"] == means.max()


# --- profiles ---------------------------------------------------------------


def test_profiles_add_up_to_all_customers_and_all_net_revenue():
    out, report = assign_segments(synthetic(300, seed=2))

    for rows in report["profiles"].values():
        assert sum(r["customers"] for r in rows) == len(out)
        assert sum(r["total_net_spend_gbp"] for r in rows) == pytest.approx(out.monetary_gbp.sum(), abs=0.5)
        assert sum(r["share_of_net_revenue_pct"] for r in rows) == pytest.approx(100, abs=0.1)


# --- default segment and band coverage --------------------------------------


def test_default_segment_column_is_the_band_rules():
    out, _ = assign_segments(synthetic(300, seed=4))
    ranked = out[out.segment != FLAGGED_GROUP]

    assert ranked.segment.equals(band_segments(ranked))
    assert set(ranked.segment) <= {"Champions", "At Risk", "Promising", "New", "Cooling Down", "Lost"}
    assert "Developing" not in set(out.segment)


def test_every_frequent_customer_is_exactly_champions_or_at_risk():
    ranked, _ = split_flagged(synthetic(1000, seed=5))
    seg = band_segments(ranked)
    frequent = ranked.frequency_orders >= 4

    assert frequent.any()
    assert seg[frequent].isin(["Champions", "At Risk"]).all()
    assert not seg[~frequent].isin(["Champions", "At Risk"]).any()
    assert seg.isin(["Champions", "At Risk"]).sum() == frequent.sum()
