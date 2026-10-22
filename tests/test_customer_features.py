"""Unit tests for net/gross spend in src/features/customer_features.py."""

import pandas as pd

from src.features.customer_features import build_customer_features


def purchases(*rows):
    frame = pd.DataFrame(
        rows, columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
    )
    frame["is_customer_sales_candidate"] = True
    return frame


def cancels(*rows):
    return pd.DataFrame(rows, columns=["CustomerID", "Quantity", "UnitPrice"])


def test_net_spend_is_gross_minus_cancellations_and_gross_is_kept():
    sales = purchases(
        ("536365", "85123", 10, "2011-01-01 10:00", 2.0, "17850"),
        ("536366", "85124", 5, "2011-01-02 10:00", 4.0, "17850"),
    )
    features, rec = build_customer_features(sales, cancels(("17850", -3, 2.0)))

    row = features.iloc[0]
    assert row["gross_spend_gbp"] == 40.0
    assert row["cancelled_spend_gbp"] == 6.0
    assert row["monetary_gbp"] == 34.0
    assert not row["net_spend_nonpositive"]
    assert rec["checks"]["gross_sales_matches"]


def test_customer_with_zero_net_spend_is_flagged():
    sales = purchases(("536365", "23166", 100, "2011-01-18 10:00", 1.04, "12346"))
    features, rec = build_customer_features(sales, cancels(("12346", -100, 1.04)))

    assert features.iloc[0]["monetary_gbp"] == 0
    assert features.iloc[0]["net_spend_nonpositive"]
    assert rec["customers_with_nonpositive_net_spend"] == 1


def test_customer_with_negative_net_spend_is_flagged():
    sales = purchases(("536365", "23166", 1, "2011-01-18 10:00", 5.0, "16446"))
    features, _ = build_customer_features(sales, cancels(("16446", -3, 5.0)))

    assert features.iloc[0]["monetary_gbp"] == -10.0
    assert features.iloc[0]["net_spend_nonpositive"]


def test_cancellations_of_other_customers_are_not_netted():
    sales = purchases(
        ("536365", "85123", 1, "2011-01-01 10:00", 10.0, "17850"),
        ("536366", "85123", 1, "2011-01-01 11:00", 10.0, "13047"),
    )
    features, _ = build_customer_features(sales, cancels(("13047", -1, 10.0)))

    by_customer = features.set_index("CustomerID")["monetary_gbp"]
    assert by_customer["17850"] == 10.0
    assert by_customer["13047"] == 0.0


def test_no_cancellations_means_net_equals_gross():
    sales = purchases(("536365", "85123", 2, "2011-01-01 10:00", 3.0, "17850"))
    features, _ = build_customer_features(sales)

    assert features.iloc[0]["monetary_gbp"] == features.iloc[0]["gross_spend_gbp"] == 6.0
