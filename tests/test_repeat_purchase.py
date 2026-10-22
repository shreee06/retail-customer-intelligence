"""Leakage tests for the 90-day repeat-purchase split in src/models/repeat_purchase.py."""

import pandas as pd
import pytest

from src.models.repeat_purchase import (
    FEATURES, build_snapshot, clean_sales, fit_and_evaluate, make_temporal_snapshots,
    monthly_cutoffs, split_cutoffs,
)

HORIZON = pd.Timedelta(days=90)


def sale(invoice, customer, when, qty=1, price=10.0, stock="85123"):
    return {"InvoiceNo": invoice, "StockCode": stock, "Quantity": qty,
            "InvoiceDate": pd.Timestamp(when), "UnitPrice": price, "CustomerID": customer}


def frame(*rows):
    return clean_sales(pd.DataFrame(list(rows)))


def month_starts(first, last):
    return list(pd.date_range(first, last, freq="MS"))


# --- split rules ------------------------------------------------------------


def test_every_training_window_ends_before_the_test_period():
    cutoffs = month_starts("2011-03-01", "2011-09-01")

    train, test = split_cutoffs(cutoffs)

    assert test == pd.Timestamp("2011-09-01")
    assert train == month_starts("2011-03-01", "2011-06-01")
    for cutoff in train:
        assert cutoff + HORIZON <= test


def test_overlapping_cutoffs_are_dropped_from_training():
    train, test = split_cutoffs(month_starts("2011-03-01", "2011-09-01"))

    # July and August outcome windows reach into the test period.
    assert pd.Timestamp("2011-07-01") not in train
    assert pd.Timestamp("2011-08-01") not in train


def test_boundary_cutoff_exactly_one_horizon_before_test_is_allowed():
    train, _ = split_cutoffs([pd.Timestamp("2011-06-01"), pd.Timestamp("2011-08-30")])
    assert train == [pd.Timestamp("2011-06-01")]  # 2011-06-01 + 90d = 2011-08-30


def test_one_day_of_overlap_is_rejected():
    with pytest.raises(ValueError):
        split_cutoffs([pd.Timestamp("2011-06-02"), pd.Timestamp("2011-08-30")])


def test_split_fails_when_no_clean_training_cutoff_exists():
    with pytest.raises(ValueError):
        split_cutoffs([pd.Timestamp("2011-08-01"), pd.Timestamp("2011-09-01")])


# --- target definition ------------------------------------------------------


def test_target_counts_purchases_inside_the_90_day_window_only():
    cutoff = pd.Timestamp("2011-03-01")
    sales = frame(
        sale("1", "A", "2011-01-10"), sale("2", "B", "2011-01-10"),
        sale("3", "C", "2011-01-10"), sale("4", "D", "2011-01-10"),
        sale("5", "A", "2011-03-01 00:00"),  # on the cutoff -> inside window
        sale("6", "B", "2011-05-29 23:59"),  # last moment of window -> inside
        sale("7", "C", "2011-05-30 00:00"),  # exactly cutoff + 90d -> outside
        # D buys only before the cutoff -> no repeat
    )

    snap = build_snapshot(sales, cutoff).set_index("CustomerID")

    assert snap.loc["A", "repeat_purchase_90d"] == 1
    assert snap.loc["B", "repeat_purchase_90d"] == 1
    assert snap.loc["C", "repeat_purchase_90d"] == 0
    assert snap.loc["D", "repeat_purchase_90d"] == 0


def test_customers_first_seen_after_the_cutoff_are_not_scored():
    sales = frame(sale("1", "A", "2011-01-10"), sale("2", "NEW", "2011-04-01"))

    snap = build_snapshot(sales, pd.Timestamp("2011-03-01"))

    assert snap.CustomerID.tolist() == ["A"]


# --- features use only the past ---------------------------------------------


def test_features_ignore_purchases_on_or_after_the_cutoff():
    cutoff = pd.Timestamp("2011-03-01")
    past = [sale("1", "A", "2011-01-10", qty=2, price=5.0), sale("2", "A", "2011-02-20")]
    future = [sale("3", "A", "2011-03-01 00:00", qty=500, price=99.0, stock="22222"),
              sale("4", "A", "2011-04-15", qty=500, price=99.0, stock="33333")]

    with_future = build_snapshot(frame(*past, *future), cutoff)
    past_only = build_snapshot(frame(*past), cutoff)

    pd.testing.assert_frame_equal(
        with_future[FEATURES].reset_index(drop=True), past_only[FEATURES].reset_index(drop=True))
    assert with_future.loc[0, "frequency_orders"] == 2
    assert with_future.loc[0, "recency_days"] == 9  # 2011-02-20 -> 2011-03-01


def test_changing_future_data_never_changes_features():
    cutoff = pd.Timestamp("2011-03-01")
    base = [sale("1", "A", "2011-01-10"), sale("2", "B", "2011-02-01"), sale("3", "A", "2011-03-20")]
    altered = [*base[:2], sale("3", "A", "2011-03-20", qty=1000), sale("9", "B", "2011-04-04")]

    a = build_snapshot(frame(*base), cutoff).sort_values("CustomerID")
    b = build_snapshot(frame(*altered), cutoff).sort_values("CustomerID")

    pd.testing.assert_frame_equal(a[FEATURES].reset_index(drop=True), b[FEATURES].reset_index(drop=True))


# --- end to end -------------------------------------------------------------


def synthetic_sales():
    """Customers on a monthly rhythm over Dec 2010 - Dec 2011, some who stop buying."""
    rows, n = [], 0
    for customer in range(1, 41):
        stops = pd.Timestamp("2011-12-01") if customer % 3 else pd.Timestamp("2011-04-15")
        step = 20 + (customer % 5) * 10
        when = pd.Timestamp("2010-12-01") + pd.Timedelta(days=customer % 7)
        while when < stops and when < pd.Timestamp("2011-12-09"):
            n += 1
            rows.append(sale(str(100000 + n), str(10000 + customer), when,
                             qty=1 + customer % 4, price=2.5 + customer % 3))
            when += pd.Timedelta(days=step)
    return pd.DataFrame(rows)


def test_no_training_outcome_window_reaches_the_test_period_end_to_end():
    snapshots, _ = make_temporal_snapshots(synthetic_sales())

    test_cohort, report = fit_and_evaluate(snapshots)

    test_start = pd.Timestamp(report["holdout_snapshot_date"])
    assert set(test_cohort.snapshot_date) == {report["holdout_snapshot_date"]}
    assert report["training_snapshot_dates"]
    for date, window_end in zip(report["training_snapshot_dates"], report["training_outcome_windows_end"]):
        assert pd.Timestamp(date) < test_start
        assert pd.Timestamp(window_end) < test_start
    assert report["holdout_snapshot_date"] not in report["training_snapshot_dates"]


def test_training_data_is_identical_without_any_test_period_sales():
    sales = synthetic_sales()
    snapshots, _ = make_temporal_snapshots(sales)
    _, report = fit_and_evaluate(snapshots)
    test_start = pd.Timestamp(report["holdout_snapshot_date"])
    trimmed = clean_sales(sales)
    trimmed = trimmed[trimmed.InvoiceDate < test_start]

    for date in report["training_snapshot_dates"]:
        full = build_snapshot(clean_sales(sales), date)
        without_test_period = build_snapshot(trimmed, date)
        pd.testing.assert_frame_equal(
            full.reset_index(drop=True), without_test_period.reset_index(drop=True))


def test_report_contains_baselines_and_all_requested_metrics():
    snapshots, _ = make_temporal_snapshots(synthetic_sales())

    _, report = fit_and_evaluate(snapshots)

    for key in ("baseline_always_yes_metrics", "baseline_recent_purchase_rule_metrics", "repeat_purchase_metrics"):
        assert {"roc_auc", "pr_auc", "precision", "recall", "f1"} <= set(report[key])
    assert report["baseline_always_yes_metrics"]["roc_auc"] == 0.5
