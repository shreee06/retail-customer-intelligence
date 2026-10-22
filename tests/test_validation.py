"""Unit tests for the row-level audit rules in src/data/validation.py."""

import pandas as pd
import pytest

from src.data.validation import audit_transactions


def make_row(**overrides) -> dict:
    """Return one clean, eligible sale line; override fields to break rules."""
    row = {
        "InvoiceNo": "536365",
        "StockCode": "85123",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "Quantity": 6,
        "InvoiceDate": pd.Timestamp("2010-12-01 08:26:00"),
        "UnitPrice": 2.55,
        "CustomerID": "17850",
        "Country": "United Kingdom",
    }
    row.update(overrides)
    return row


def make_frame(*rows: dict) -> pd.DataFrame:
    """Build a frame shaped like load_transactions output."""
    frame = pd.DataFrame(list(rows))
    for column in ("InvoiceNo", "StockCode", "Description", "CustomerID", "Country"):
        frame[column] = frame[column].astype("string")
    frame.insert(0, "source_row_number", range(2, len(frame) + 2))
    return frame


def test_exact_duplicates_are_detected_despite_unique_row_numbers():
    frame = make_frame(make_row(), make_row(), make_row(StockCode="71053"))

    audited, summary = audit_transactions(frame)

    assert audited["row_is_exact_duplicate"].tolist() == [True, True, False]
    assert summary["flag_counts"]["exact_duplicate_row"] == 2


def test_every_duplicate_copy_is_flagged_but_only_extra_copies_are_excluded():
    audited, _ = audit_transactions(make_frame(make_row(), make_row()))

    assert audited["requires_review"].all()
    assert audited["row_is_exact_duplicate"].all()
    assert audited["is_customer_sales_candidate"].tolist() == [True, False]


def test_clean_sale_is_a_candidate_with_no_exclusion_reason():
    audited, summary = audit_transactions(make_frame(make_row()))

    assert audited.loc[0, "is_customer_sales_candidate"]
    assert audited.loc[0, "candidate_exclusion_reasons"] == ""
    assert summary["flag_counts"]["customer_sales_candidate"] == 1


def test_cancellation_invoice_is_excluded_with_all_reasons():
    frame = make_frame(make_row(InvoiceNo="C536379", Quantity=-1))

    audited, _ = audit_transactions(frame)

    assert not audited.loc[0, "is_customer_sales_candidate"]
    assert audited.loc[0, "is_cancellation_or_return"]
    reasons = audited.loc[0, "candidate_exclusion_reasons"].split("|")
    assert reasons == [
        "invoice_is_cancellation",
        "negative_quantity_return_like",
        "nonpositive_quantity",
    ]


def test_lowercase_cancellation_prefix_is_recognised():
    audited, _ = audit_transactions(make_frame(make_row(InvoiceNo="c536379", Quantity=-1)))

    assert audited.loc[0, "invoice_is_cancellation"]


def test_negative_quantity_without_c_prefix_is_return_like():
    audited, _ = audit_transactions(make_frame(make_row(Quantity=-5)))

    assert not audited.loc[0, "invoice_is_cancellation"]
    assert audited.loc[0, "is_cancellation_or_return"]
    assert not audited.loc[0, "is_customer_sales_candidate"]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"CustomerID": pd.NA}, "missing_customer_id"),
        ({"CustomerID": "1785"}, "invalid_customer_id_format"),
        ({"InvoiceNo": "A563185"}, "invalid_invoice_id"),
        ({"UnitPrice": 0.0}, "nonpositive_unit_price"),
        ({"UnitPrice": -11062.06}, "nonpositive_unit_price"),
        ({"InvoiceDate": "not a date"}, "invalid_timestamp"),
    ],
)
def test_rule_violations_exclude_the_row_with_a_named_reason(overrides, reason):
    audited, summary = audit_transactions(make_frame(make_row(**overrides)))

    assert not audited.loc[0, "is_customer_sales_candidate"]
    assert reason in audited.loc[0, "candidate_exclusion_reasons"].split("|")
    assert summary["candidate_exclusion_reason_counts"][reason] == 1


def test_no_rows_are_removed_and_input_is_not_modified():
    frame = make_frame(
        make_row(),
        make_row(InvoiceNo="C536379", Quantity=-1),
        make_row(CustomerID=pd.NA),
    )
    original = frame.copy()

    audited, summary = audit_transactions(frame)

    assert len(audited) == len(frame) == summary["row_count"] == 3
    assert audited["source_row_number"].tolist() == [2, 3, 4]
    pd.testing.assert_frame_equal(frame, original)


def test_invoice_shared_by_two_customers_is_flagged():
    frame = make_frame(make_row(), make_row(StockCode="71053", CustomerID="13047"))

    audited, _ = audit_transactions(frame)

    assert audited["invoice_has_multiple_customers"].all()
    assert audited["requires_review"].all()


# --- Step B rules -----------------------------------------------------------


def test_only_first_copy_of_exact_duplicate_stays_a_candidate():
    frame = make_frame(make_row(), make_row(), make_row(), make_row(StockCode="71053"))

    audited, _ = audit_transactions(frame)

    assert audited["row_is_exact_duplicate"].tolist() == [True, True, True, False]
    assert audited["is_customer_sales_candidate"].tolist() == [True, False, False, True]
    assert audited.loc[1, "candidate_exclusion_reasons"] == "exact_duplicate_extra_copy"
    assert len(audited) == 4  # nothing removed from the audit


def test_same_line_on_different_invoices_is_not_a_duplicate():
    audited, _ = audit_transactions(make_frame(make_row(), make_row(InvoiceNo="536366")))

    assert audited["is_customer_sales_candidate"].all()


@pytest.mark.parametrize(
    "code", ["POST", "M", "DOT", "C2", "BANK CHARGES", "D", "S", "B", "AMAZONFEE", "CRUK", "m", "post"]
)
def test_non_product_lines_are_excluded_with_named_reason(code):
    audited, summary = audit_transactions(make_frame(make_row(StockCode=code)))

    assert not audited.loc[0, "is_customer_sales_candidate"]
    assert audited.loc[0, "candidate_exclusion_reasons"] == "non_product_line"
    assert summary["candidate_exclusion_reason_counts"]["non_product_line"] == 1


def test_product_codes_with_letter_suffix_are_not_non_products():
    audited, _ = audit_transactions(make_frame(make_row(StockCode="85123A"), make_row(StockCode="15056BL", InvoiceNo="536366")))

    assert audited["is_customer_sales_candidate"].all()


def test_valid_cancelled_product_line_is_a_net_spend_cancellation():
    frame = make_frame(
        make_row(),
        make_row(InvoiceNo="C536379", Quantity=-2),
        make_row(InvoiceNo="C536380", Quantity=-1, StockCode="POST"),
        make_row(InvoiceNo="C536381", Quantity=-1, CustomerID=None),
    )

    audited, _ = audit_transactions(frame)

    assert audited["is_net_spend_cancellation"].tolist() == [False, True, False, False]
