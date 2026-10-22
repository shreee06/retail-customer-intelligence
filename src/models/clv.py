"""90-day predicted net spend per customer (BG/NBD x Gamma-Gamma), evaluated on a holdout.

This is a FORECAST of future spend, not historical spend. Everything is fitted on
data strictly before `cutoff`; the window [cutoff, cutoff + 90 days) is only used
to score the forecast.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SALES = ROOT / "data/processed/customer_sales_candidates.csv"
CANCELLATIONS = ROOT / "data/processed/net_spend_cancellations.csv"
FEATURES = ROOT / "data/processed/customer_features.csv"
FORECAST = ROOT / "data/processed/clv_forecast.csv"
REPORT = ROOT / "reports/generated/clv_evaluation.json"

CUTOFF = "2011-09-01"
HORIZON_DAYS = 90
MIN_PURCHASE_DAYS_FOR_OWN_CANCEL_RATE = 3   # fewer days: rate is too noisy, use the pooled rate
BASELINE_MIN_OBSERVED_DAYS = 30             # avoids dividing by ~0 for brand-new customers
GAMMA_GAMMA_PENALIZER = 0.001
# BG/NBD dropout parameters a, b are bounded: on this data the likelihood is almost flat
# and the unbounded optimum is a ~ 0 / b ~ 0 (infinite forecasts). Within these bounds the
# fit settles at "almost no permanent dropout". b >= 1.5 keeps the expected-purchases
# formula numerically defined for customers with no repeat purchases (it returns NaN at b = 1).
BGNBD_A_BOUNDS = (0.01, 100.0)
BGNBD_B_BOUNDS = (1.5, 100.0)
MIN_GAMMA_GAMMA_CUSTOMERS = 20
MIN_CALIBRATION_CUSTOMERS = 50   # fewer than this: BG/NBD parameters are not identifiable


# --- data prep ---------------------------------------------------------------


def prepare_lines(sales: pd.DataFrame, cancellations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Purchases (value > 0) and cancelled value (positive amounts) with typed columns."""
    purchases = sales[["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]].copy()
    purchases["CustomerID"] = purchases.CustomerID.astype("string").str.strip()
    purchases["InvoiceDate"] = pd.to_datetime(purchases.InvoiceDate)
    purchases["value"] = pd.to_numeric(purchases.Quantity) * pd.to_numeric(purchases.UnitPrice)
    cancels = cancellations[["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]].copy()
    cancels["CustomerID"] = cancels.CustomerID.astype("string").str.strip()
    cancels["InvoiceDate"] = pd.to_datetime(cancels.InvoiceDate)
    cancels["value"] = (pd.to_numeric(cancels.Quantity) * pd.to_numeric(cancels.UnitPrice)).abs()
    return purchases, cancels


def purchase_days(purchases: pd.DataFrame) -> pd.DataFrame:
    """One row per customer per calendar day; invoices on the same day are summed."""
    day = purchases.assign(day=purchases.InvoiceDate.dt.normalize())
    return day.groupby(["CustomerID", "day"], as_index=False).value.sum()


def calibration_summary(purchases: pd.DataFrame, cancels: pd.DataFrame, cutoff) -> pd.DataFrame:
    """BG/NBD inputs and spend totals from data strictly before `cutoff`."""
    cutoff = pd.Timestamp(cutoff)
    days = purchase_days(purchases[purchases.InvoiceDate < cutoff])
    ordered = days.sort_values(["CustomerID", "day"])
    grouped = ordered.groupby("CustomerID")
    summary = grouped.agg(first_day=("day", "min"), last_day=("day", "max"),
                          purchase_days=("day", "size"), gross_spend=("value", "sum"))
    summary["frequency"] = summary.purchase_days - 1                       # repeat purchase days
    summary["recency"] = (summary.last_day - summary.first_day).dt.days     # first -> last purchase
    summary["T"] = (cutoff.normalize() - summary.first_day).dt.days         # first purchase -> cutoff
    repeat = ordered[ordered.duplicated("CustomerID", keep="first")]        # drop each customer's first day
    summary["monetary_value"] = repeat.groupby("CustomerID").value.mean().reindex(summary.index)
    prior_cancels = cancels[cancels.InvoiceDate < cutoff].groupby("CustomerID").value.sum()
    summary["cancelled_spend"] = prior_cancels.reindex(summary.index).fillna(0.0)
    summary["net_spend"] = summary.gross_spend - summary.cancelled_spend
    return summary


def cancellation_rates(summary: pd.DataFrame, min_days: int = MIN_PURCHASE_DAYS_FOR_OWN_CANCEL_RATE) -> tuple[pd.Series, float]:
    """Share of gross spend that was cancelled, capped to [0, 1].

    Customers with fewer than `min_days` purchase days get the pooled rate.
    """
    pooled = float(np.clip(summary.cancelled_spend.sum() / summary.gross_spend.sum(), 0, 1))
    own = (summary.cancelled_spend / summary.gross_spend.where(summary.gross_spend > 0)).clip(0, 1)
    rate = own.where(summary.purchase_days >= min_days, pooled).fillna(pooled)
    return rate.clip(0, 1), pooled


def baseline_forecast(summary: pd.DataFrame, spend_column: str, horizon_days: int = HORIZON_DAYS,
                      min_days: int = BASELINE_MIN_OBSERVED_DAYS) -> pd.Series:
    """Average daily calibration spend x horizon. Observed days = max(T, min_days)."""
    observed = summary["T"].clip(lower=min_days)
    return summary[spend_column] / observed * horizon_days


def actual_window(purchases: pd.DataFrame, cancels: pd.DataFrame, cutoff, horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    """Gross and net spend per customer inside [cutoff, cutoff + horizon)."""
    start = pd.Timestamp(cutoff)
    end = start + pd.Timedelta(days=horizon_days)
    gross = purchases[(purchases.InvoiceDate >= start) & (purchases.InvoiceDate < end)].groupby("CustomerID").value.sum()
    cancelled = cancels[(cancels.InvoiceDate >= start) & (cancels.InvoiceDate < end)].groupby("CustomerID").value.sum()
    out = pd.concat([gross.rename("actual_gross"), cancelled.rename("actual_cancelled")], axis=1).fillna(0.0)
    out["actual_net"] = out.actual_gross - out.actual_cancelled
    return out


# --- model ---------------------------------------------------------------


def bgnbd_log_likelihood(params, frequency, recency, T) -> float:
    """BG/NBD log-likelihood (Fader, Hardie & Lee 2005) summed over customers."""
    r, alpha, a, b = params
    x, tx = np.asarray(frequency, float), np.asarray(recency, float)
    T = np.asarray(T, float)
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    a3 = -(r + x) * np.log(alpha + T)
    a4 = np.log(a) - np.log(b + np.maximum(x, 1) - 1) - (r + x) * np.log(alpha + tx)
    return float(np.sum(a1 + a2 + np.where(x > 0, np.logaddexp(a3, a4), a3)))


def fit_bgnbd(frequency, recency, T) -> tuple[BetaGeoFitter, dict]:
    """Fit BG/NBD with bounded parameters; return a lifetimes fitter (for its prediction
    formulas) with the fitted parameters set, plus fit diagnostics."""
    (a_lo, a_hi), (b_lo, b_hi) = BGNBD_A_BOUNDS, BGNBD_B_BOUNDS
    bounds = [(np.log(1e-3), np.log(50)), (np.log(1e-2), np.log(1e4)),
              (np.log(a_lo), np.log(a_hi)), (np.log(b_lo), np.log(b_hi))]
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in ([0.5, 50, 1, 1], [0.3, 20, 0.5, 2], [1, 100, 2, 5], [0.2, 10, 0.1, 0.1]):
            result = minimize(lambda lp: -bgnbd_log_likelihood(np.exp(lp), frequency, recency, T),
                              np.log(start), method="L-BFGS-B", bounds=bounds)
            if best is None or result.fun < best.fun:
                best = result
    params = np.exp(best.x)
    fitter = BetaGeoFitter()
    fitter.params_ = pd.Series(dict(zip(("r", "alpha", "a", "b"), params)))
    diagnostics = {"log_likelihood": float(-best.fun), "converged": bool(best.success),
                   "a_at_lower_bound": bool(params[2] <= a_lo * 1.001), "b_at_upper_bound": bool(params[3] >= b_hi * 0.999),
                   "dropout_probability_mean": float(params[2] / (params[2] + params[3]))}
    return fitter, diagnostics


def forecast(summary: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> tuple[pd.DataFrame, dict]:
    """Fit BG/NBD and Gamma-Gamma on the calibration summary and forecast `horizon_days`."""
    if len(summary) < MIN_CALIBRATION_CUSTOMERS:
        raise ValueError(f"Need at least {MIN_CALIBRATION_CUSTOMERS} calibration customers, got {len(summary)}")
    out = summary.copy()
    bgf, bgf_fit = fit_bgnbd(out.frequency, out.recency, out["T"])
    out["expected_orders"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon_days, out.frequency, out.recency, out["T"])
    out["probability_alive"] = bgf.conditional_probability_alive(out.frequency, out.recency, out["T"])

    repeaters = out[(out.frequency > 0) & (out.monetary_value > 0)]
    all_day_values = out.gross_spend.sum() / out.purchase_days.sum()   # mean value of a purchase day, all customers
    out["expected_order_value"] = all_day_values
    out["order_value_source"] = "population_mean"
    gamma_gamma_params = None
    if len(repeaters) >= MIN_GAMMA_GAMMA_CUSTOMERS:   # too few repeat buyers: keep the population mean
        ggf = GammaGammaFitter(penalizer_coef=GAMMA_GAMMA_PENALIZER)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ggf.fit(repeaters.frequency, repeaters.monetary_value)
        use_own = (out.frequency > 0) & (out.monetary_value > 0)
        out.loc[use_own, "expected_order_value"] = ggf.conditional_expected_average_profit(
            out.loc[use_own, "frequency"], out.loc[use_own, "monetary_value"])
        out.loc[use_own, "order_value_source"] = "gamma_gamma"
        gamma_gamma_params = {k: float(v) for k, v in ggf.params_.items()}

    rate, pooled = cancellation_rates(out)
    out["cancellation_rate"] = rate
    out["predicted_gross"] = out.expected_orders * out.expected_order_value
    out["predicted_net"] = out.predicted_gross * (1 - out.cancellation_rate)
    out["baseline_gross"] = baseline_forecast(out, "gross_spend", horizon_days)
    out["baseline_net"] = baseline_forecast(out, "net_spend", horizon_days)
    params = {
        "bgnbd": {k: float(v) for k, v in bgf.params_.items()},
        "bgnbd_fit": bgf_fit,
        "gamma_gamma": gamma_gamma_params,
        "pooled_cancellation_rate": pooled,
        "population_mean_order_value_gbp": float(all_day_values),
        "gamma_gamma_fit_customers": int(len(repeaters)),
        "bgnbd_a_bounds": list(BGNBD_A_BOUNDS),
        "bgnbd_b_bounds": list(BGNBD_B_BOUNDS),
        "gamma_gamma_penalizer": GAMMA_GAMMA_PENALIZER,
    }
    return out, params


# --- evaluation ----------------------------------------------------------


def metrics(frame: pd.DataFrame, predicted: str, actual: str) -> dict:
    rho = spearmanr(frame[predicted], frame[actual])[0] if len(frame) > 2 else float("nan")
    return {
        "customers": int(len(frame)),
        "total_predicted_gbp": round(float(frame[predicted].sum()), 2),
        "total_actual_gbp": round(float(frame[actual].sum()), 2),
        "predicted_over_actual": round(float(frame[predicted].sum() / frame[actual].sum()), 3) if frame[actual].sum() else None,
        "mae_gbp": round(float((frame[predicted] - frame[actual]).abs().mean()), 2),
        "spearman": round(float(rho), 3),
    }


def decile_table(frame: pd.DataFrame, predicted: str, actual: str, bins: int = 10) -> list[dict]:
    """Decile 10 = highest predicted value. Compares mean predicted with mean actual."""
    if len(frame) < bins:  # not enough customers for deciles
        return []
    ranks = frame[predicted].rank(method="first")
    decile = pd.qcut(ranks, bins, labels=range(1, bins + 1)).astype(int)
    table = frame.groupby(decile).agg(customers=(predicted, "size"), mean_predicted_gbp=(predicted, "mean"),
                                      mean_actual_gbp=(actual, "mean"), total_actual_gbp=(actual, "sum"))
    table["total_actual_share_pct"] = table.total_actual_gbp / table.total_actual_gbp.sum() * 100
    return table.round(2).reset_index(names="decile").to_dict("records")


def first_time_customers(purchases: pd.DataFrame, cancels: pd.DataFrame, cutoff, calibration_ids,
                         horizon_days: int = HORIZON_DAYS) -> dict:
    """Customers with no purchase before the cutoff who buy inside the window."""
    start = pd.Timestamp(cutoff)
    end = start + pd.Timedelta(days=horizon_days)
    window = purchases[(purchases.InvoiceDate >= start) & (purchases.InvoiceDate < end)]
    new_ids = set(window.CustomerID) - set(calibration_ids)
    window_cancels = cancels[(cancels.InvoiceDate >= start) & (cancels.InvoiceDate < end)]
    gross_new = float(window[window.CustomerID.isin(new_ids)].value.sum())
    net_new = gross_new - float(window_cancels[window_cancels.CustomerID.isin(new_ids)].value.sum())
    gross_all = float(window.value.sum())
    net_all = gross_all - float(window_cancels.value.sum())
    return {
        "first_time_customers": len(new_ids),
        "their_actual_gross_gbp": round(gross_new, 2), "all_actual_gross_gbp": round(gross_all, 2),
        "gross_share_pct": round(gross_new / gross_all * 100, 2) if gross_all else None,
        "their_actual_net_gbp": round(net_new, 2), "all_actual_net_gbp": round(net_all, 2),
        "net_share_pct": round(net_new / net_all * 100, 2) if net_all else None,
    }


def evaluate(sales: pd.DataFrame, cancellations: pd.DataFrame, official_flagged_ids=(),
             cutoff: str = CUTOFF, horizon_days: int = HORIZON_DAYS) -> tuple[pd.DataFrame, dict]:
    purchases, cancels = prepare_lines(sales, cancellations)
    summary = calibration_summary(purchases, cancels, cutoff)
    predicted, params = forecast(summary, horizon_days)
    actual = actual_window(purchases, cancels, cutoff, horizon_days)
    table = predicted.join(actual).fillna({"actual_gross": 0.0, "actual_cancelled": 0.0, "actual_net": 0.0})
    # Flag computed from calibration data only (the full-year flag would use future data).
    table["excluded_nonpositive_calibration_net"] = table.net_spend <= 0
    table["single_purchase_day"] = table.frequency == 0
    ranked = table[~table.excluded_nonpositive_calibration_net]
    multi = ranked[~ranked.single_purchase_day]
    official = set(map(str, official_flagged_ids))

    def block(frame):
        return {"gross": {"model": metrics(frame, "predicted_gross", "actual_gross"),
                          "baseline": metrics(frame, "baseline_gross", "actual_gross")},
                "net": {"model": metrics(frame, "predicted_net", "actual_net"),
                        "baseline": metrics(frame, "baseline_net", "actual_net")}}

    report = {
        "definition": "Predicted NET spend in the 90 days after the cutoff = BG/NBD expected purchase days x Gamma-Gamma expected order value x (1 - cancellation rate). A forecast, not historical spend.",
        "cutoff": cutoff,
        "calibration_period": f"{summary.first_day.min().date()} to {(pd.Timestamp(cutoff) - pd.Timedelta(days=1)).date()}",
        "holdout_window": f"{cutoff} to {(pd.Timestamp(cutoff) + pd.Timedelta(days=horizon_days - 1)).date()}",
        "fit_parameters": params,
        "rules": {
            "purchase_unit": "distinct purchase days (invoices on the same day count once)",
            "cancellation_rate": f"own rate (cancelled / gross, calibration period only, capped to [0,1]) when >= {MIN_PURCHASE_DAYS_FOR_OWN_CANCEL_RATE} purchase days, else the pooled rate",
            "baseline": f"calibration spend / max(days from first purchase to cutoff, {BASELINE_MIN_OBSERVED_DAYS}) x {horizon_days}",
        },
        "coverage": {
            "customers_with_purchases_before_cutoff": int(len(table)),
            "excluded_nonpositive_calibration_net": int(table.excluded_nonpositive_calibration_net.sum()),
            "ranked_customers": int(len(ranked)),
            "single_purchase_day_customers_in_ranked": int(ranked.single_purchase_day.sum()),
            "single_purchase_day_customers_all": int(table.single_purchase_day.sum()),
            "multi_purchase_day_customers_in_ranked": int(len(multi)),
            "gamma_gamma_customers": int((table.order_value_source == "gamma_gamma").sum()),
            "own_cancellation_rate_customers": int((table.purchase_days >= MIN_PURCHASE_DAYS_FOR_OWN_CANCEL_RATE).sum()),
            "pooled_cancellation_rate_customers": int((table.purchase_days < MIN_PURCHASE_DAYS_FOR_OWN_CANCEL_RATE).sum()),
            "official_full_year_flagged_in_calibration": int(table.index.astype(str).isin(official).sum()),
            "official_full_year_flagged_in_ranked": int(ranked.index.astype(str).isin(official).sum()),
        },
        "all_ranked_customers": block(ranked),
        "multi_purchase_day_only": block(multi),
        "deciles_net_all_ranked": decile_table(ranked, "predicted_net", "actual_net"),
        "deciles_gross_all_ranked": decile_table(ranked, "predicted_gross", "actual_gross"),
        "deciles_net_baseline_all_ranked": decile_table(ranked, "baseline_net", "actual_net"),
        "first_time_customers_in_window": first_time_customers(purchases, cancels, cutoff, table.index, horizon_days),
    }
    return table, report


def run(sales_path: Path = SALES, cancellations_path: Path = CANCELLATIONS, features_path: Path = FEATURES,
        output_path: Path = FORECAST, report_path: Path = REPORT) -> dict:
    sales = pd.read_csv(sales_path, usecols=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"],
                        dtype={"CustomerID": "string", "InvoiceNo": "string"})
    cancellations = pd.read_csv(cancellations_path, usecols=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"],
                                dtype={"CustomerID": "string"})
    features = pd.read_csv(features_path, usecols=["CustomerID", "net_spend_nonpositive"], dtype={"CustomerID": "string"})
    flagged = features.loc[features.net_spend_nonpositive.astype(str).str.lower().eq("true"), "CustomerID"]
    table, report = evaluate(sales, cancellations, flagged)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: result[k] for k in ("calibration_period", "holdout_window", "coverage", "fit_parameters")}, indent=2))
    for name in ("all_ranked_customers", "multi_purchase_day_only"):
        print(f"\n{name}")
        for kind in ("gross", "net"):
            for who in ("model", "baseline"):
                print(f"  {kind:5s} {who:9s}", result[name][kind][who])
    print("\nNet deciles (model):")
    print(pd.DataFrame(result["deciles_net_all_ranked"]).to_string(index=False))
    print("\nNet deciles (baseline):")
    print(pd.DataFrame(result["deciles_net_baseline_all_ranked"]).to_string(index=False))
    print("\nGross deciles (model):")
    print(pd.DataFrame(result["deciles_gross_all_ranked"]).to_string(index=False))
    print("\nFirst-time customers:", json.dumps(result["first_time_customers_in_window"], indent=2))
