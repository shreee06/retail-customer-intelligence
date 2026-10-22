"""Tests for the 90-day net-spend forecast in src/models/clv.py."""

import numpy as np
import pandas as pd
import pytest

from src.models.clv import (
    actual_window, baseline_forecast, calibration_summary, cancellation_rates, decile_table,
    evaluate, first_time_customers, prepare_lines,
)

CUTOFF = pd.Timestamp("2011-09-01")


def buy(customer, when, value=100.0, invoice="1"):
    return {"CustomerID": customer, "InvoiceNo": invoice, "InvoiceDate": pd.Timestamp(when),
            "Quantity": 1, "UnitPrice": value}


def cancel(customer, when, value):
    return {"CustomerID": customer, "InvoiceDate": pd.Timestamp(when), "Quantity": -1, "UnitPrice": value}


def lines(buys, cancels=()):
    cancel_frame = pd.DataFrame(list(cancels), columns=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"])
    return prepare_lines(pd.DataFrame(buys), cancel_frame)


def synthetic(n=150, seed=0):
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2010-12-01"), pd.Timestamp("2011-11-29")
    buys, cancels, counter = [], [], 0
    for i in range(n):
        cid = str(10000 + i)
        rate = max(rng.gamma(1.0, 0.02), 0.002)
        first = start + pd.Timedelta(days=int(rng.integers(0, 200)))
        day = first
        while day <= end:
            counter += 1
            buys.append(buy(cid, day + pd.Timedelta(hours=10), value=float(rng.gamma(2, 60)), invoice=str(counter)))
            day += pd.Timedelta(days=max(1, int(rng.exponential(1 / rate))))
        if rng.random() < .3:
            cancels.append(cancel(cid, first + pd.Timedelta(days=int(rng.integers(1, 100))), float(rng.integers(5, 80))))
    return pd.DataFrame(buys), pd.DataFrame(cancels)


# --- calibration inputs -----------------------------------------------------


def test_same_day_invoices_count_as_one_purchase_day():
    purchases, cancels = lines([buy("A", "2011-03-01 09:00", 50, "1"), buy("A", "2011-03-01 15:00", 70, "2")])

    summary = calibration_summary(purchases, cancels, CUTOFF)

    assert summary.loc["A", "purchase_days"] == 1 and summary.loc["A", "frequency"] == 0
    assert summary.loc["A", "gross_spend"] == 120


def test_frequency_recency_and_age_follow_the_standard_definitions():
    purchases, cancels = lines([buy("A", "2011-03-01"), buy("A", "2011-04-01", 200), buy("A", "2011-06-01", 400)])

    row = calibration_summary(purchases, cancels, CUTOFF).loc["A"]

    assert row.frequency == 2                              # repeat purchase days
    assert row.recency == (pd.Timestamp("2011-06-01") - pd.Timestamp("2011-03-01")).days
    assert row["T"] == (CUTOFF - pd.Timestamp("2011-03-01")).days
    assert row.monetary_value == 300                       # mean of repeat days only (first day excluded)


def test_calibration_ignores_the_cutoff_day_and_everything_after():
    base = [buy("A", "2011-03-01"), buy("A", "2011-05-01")]
    later = [buy("A", "2011-09-01 00:00", 9999), buy("A", "2011-10-10", 9999), buy("B", "2011-09-15")]
    cancels = [cancel("A", "2011-09-01", 500)]

    with_future = calibration_summary(*lines(base + later, cancels), CUTOFF)
    without = calibration_summary(*lines(base), CUTOFF)

    pd.testing.assert_frame_equal(with_future, without)
    assert "B" not in with_future.index


def test_calibration_and_holdout_windows_do_not_overlap():
    buys = [buy("A", "2011-08-31 23:59"), buy("A", "2011-09-01 00:00", 10), buy("A", "2011-11-29 23:59", 20),
            buy("A", "2011-11-30 00:00", 40)]
    purchases, cancels = lines(buys)

    summary = calibration_summary(purchases, cancels, CUTOFF)
    window = actual_window(purchases, cancels, CUTOFF)

    assert summary.last_day.max() < CUTOFF <= purchases.InvoiceDate[purchases.InvoiceDate >= CUTOFF].min()
    assert summary.loc["A", "gross_spend"] == 100
    assert window.loc["A", "actual_gross"] == 30           # 2011-09-01 and 2011-11-29 only


# --- cancellation rate ------------------------------------------------------


def test_cancellation_rate_is_capped_between_zero_and_one():
    purchases, cancels = lines(
        [buy(c, d, 100) for c in "ABC" for d in ("2011-01-01", "2011-02-01", "2011-03-01")],
        [cancel("A", "2011-04-01", 5000), cancel("B", "2011-04-01", 30)])

    rate, pooled = cancellation_rates(calibration_summary(purchases, cancels, CUTOFF))

    assert rate["A"] == 1.0          # cancelled more than bought -> capped at 1
    assert rate["B"] == pytest.approx(0.1)
    assert rate["C"] == 0.0
    assert ((rate >= 0) & (rate <= 1)).all() and 0 <= pooled <= 1


def test_low_history_customers_get_the_pooled_rate():
    purchases, cancels = lines(
        [buy("A", d, 100) for d in ("2011-01-01", "2011-02-01", "2011-03-01")] + [buy("B", "2011-01-01", 100)],
        [cancel("A", "2011-04-01", 60), cancel("B", "2011-04-01", 100)])

    rate, pooled = cancellation_rates(calibration_summary(purchases, cancels, CUTOFF))

    assert pooled == pytest.approx(160 / 400)
    assert rate["A"] == pytest.approx(0.2)         # 3 purchase days -> own rate
    assert rate["B"] == pytest.approx(pooled)      # 1 purchase day -> pooled, not its own 100%


# --- baseline ---------------------------------------------------------------


def test_baseline_is_spend_per_observed_day_times_90_with_a_30_day_floor():
    summary = pd.DataFrame({"T": [180, 10], "net_spend": [1800.0, 300.0]}, index=["A", "B"])

    result = baseline_forecast(summary, "net_spend")

    assert result["A"] == pytest.approx(1800 / 180 * 90)
    assert result["B"] == pytest.approx(300 / 30 * 90)   # T=10 is floored to 30


# --- exclusions -------------------------------------------------------------


def test_nonpositive_net_flag_uses_calibration_data_only():
    base_sales, base_cancels = synthetic(seed=7)
    buys = [buy("GOOD", "2011-03-01", 100), buy("GOOD", "2011-09-10", 100),
            buy("BAD", "2011-03-01", 100), buy("BAD", "2011-09-10", 100)]
    cancels = [cancel("GOOD", "2011-10-01", 5000),    # huge return AFTER the cutoff: must not exclude
               cancel("BAD", "2011-04-01", 150)]      # return before the cutoff exceeds purchases
    sales = pd.concat([base_sales, pd.DataFrame(buys)], ignore_index=True)

    table, report = evaluate(sales, pd.concat([base_cancels, pd.DataFrame(cancels)], ignore_index=True))

    assert not table.loc["GOOD", "excluded_nonpositive_calibration_net"]
    assert table.loc["BAD", "excluded_nonpositive_calibration_net"]
    assert report["coverage"]["excluded_nonpositive_calibration_net"] == int(table.excluded_nonpositive_calibration_net.sum())


def test_first_time_customers_are_counted_with_their_share_of_window_spend():
    purchases, cancels = lines(
        [buy("OLD", "2011-03-01"), buy("OLD", "2011-09-05", 300), buy("NEW", "2011-09-10", 100)],
        [cancel("NEW", "2011-09-20", 40)])

    result = first_time_customers(purchases, cancels, CUTOFF, calibration_ids=["OLD"])

    assert result["first_time_customers"] == 1
    assert result["their_actual_gross_gbp"] == 100 and result["all_actual_gross_gbp"] == 400
    assert result["gross_share_pct"] == 25.0
    assert result["their_actual_net_gbp"] == 60 and result["net_share_pct"] == pytest.approx(60 / 360 * 100, abs=0.01)


# --- end to end on synthetic data -------------------------------------------


def test_forecast_is_finite_nonnegative_and_net_is_gross_times_one_minus_rate():
    sales, cancels = synthetic()

    table, report = evaluate(sales, cancels)

    assert np.isfinite(table[["expected_orders", "expected_order_value", "predicted_gross", "predicted_net",
                              "baseline_net", "probability_alive"]]).all().all()
    assert (table.predicted_gross >= 0).all() and (table.predicted_net >= 0).all()
    np.testing.assert_allclose(table.predicted_net, table.predicted_gross * (1 - table.cancellation_rate))
    assert table.cancellation_rate.between(0, 1).all()
    assert report["fit_parameters"]["bgnbd_fit"]["converged"]


def test_changing_holdout_data_never_changes_the_forecast():
    sales, cancels = synthetic(seed=1)
    altered = sales.copy()
    future = pd.to_datetime(altered.InvoiceDate) >= CUTOFF
    altered.loc[future, "UnitPrice"] = altered.loc[future, "UnitPrice"] * 7 + 1000

    a, _ = evaluate(sales, cancels)
    b, _ = evaluate(altered, cancels)

    forecast_columns = ["expected_orders", "expected_order_value", "predicted_gross", "predicted_net",
                        "baseline_gross", "baseline_net", "excluded_nonpositive_calibration_net"]
    pd.testing.assert_frame_equal(a[forecast_columns], b[forecast_columns])
    assert not a.actual_gross.equals(b.actual_gross)


def test_results_are_deterministic():
    sales, cancels = synthetic(seed=2)

    a, report_a = evaluate(sales, cancels)
    b, report_b = evaluate(sales, cancels)

    pd.testing.assert_frame_equal(a, b)
    assert report_a == report_b


def test_report_has_gross_and_net_metrics_with_and_without_single_purchase_customers():
    _, report = evaluate(*synthetic(seed=3))

    for block in ("all_ranked_customers", "multi_purchase_day_only"):
        for kind in ("gross", "net"):
            for who in ("model", "baseline"):
                assert {"total_predicted_gbp", "total_actual_gbp", "mae_gbp", "spearman"} <= set(report[block][kind][who])
    assert report["coverage"]["multi_purchase_day_customers_in_ranked"] < report["coverage"]["ranked_customers"]


def test_decile_table_has_ten_equal_groups_that_sum_to_the_total():
    frame = pd.DataFrame({"p": np.arange(1, 101, dtype=float), "a": np.arange(100, 0, -1, dtype=float)})

    table = pd.DataFrame(decile_table(frame, "p", "a"))

    assert table.decile.tolist() == list(range(1, 11)) and (table.customers == 10).all()
    assert table.mean_predicted_gbp.is_monotonic_increasing
    assert table.total_actual_gbp.sum() == frame.a.sum()


def test_too_few_customers_is_refused_instead_of_returning_garbage():
    sales = pd.DataFrame([buy("A", "2011-03-01"), buy("B", "2011-04-01", 50), buy("A", "2011-09-10", 10)])

    with pytest.raises(ValueError, match="calibration customers"):
        evaluate(sales, pd.DataFrame(columns=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]))


def test_order_value_falls_back_to_the_population_mean_when_few_customers_repeat():
    rows = [buy(f"C{i}", "2011-03-01", 100 + i, str(i)) for i in range(60)]  # nobody repeats
    sales = pd.DataFrame(rows + [buy("C0", "2011-04-01", 120, "x")])

    table, report = evaluate(sales, pd.DataFrame(columns=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]))

    assert set(table.order_value_source) == {"population_mean"}
    assert report["fit_parameters"]["gamma_gamma"] is None
