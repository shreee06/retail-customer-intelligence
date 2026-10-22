"""Schema-oriented data-quality flags and conservative sales eligibility."""

from __future__ import annotations

import pandas as pd

from .ingestion import EXPECTED_COLUMNS


INVOICE_PATTERN = r"(?i)C?\d{6}"
CUSTOMER_PATTERN = r"\d{5}"
STOCK_CODE_PATTERN = r"\d{5}"
# Service, fee and adjustment lines (postage, manual entries, carriage, bank
# charges, discounts, samples, ...). They are not products a customer bought.
NON_PRODUCT_STOCK_CODES = frozenset(
    {"POST", "DOT", "M", "C2", "D", "S", "B", "BANK CHARGES", "AMAZONFEE", "CRUK"}
)


def audit_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return every source row with flags plus a summary of quality findings.

    Suspicious rows are never removed here. `is_customer_sales_candidate` is a
    conservative starting cohort only; other review flags remain visible.
    """
    audited = frame.copy()

    invoice = audited["InvoiceNo"].astype("string").str.strip()
    customer = audited["CustomerID"].astype("string").str.strip()
    stock_code = audited["StockCode"].astype("string").str.strip()
    description = audited["Description"].astype("string").str.strip()
    quantity = pd.to_numeric(audited["Quantity"], errors="coerce")
    unit_price = pd.to_numeric(audited["UnitPrice"], errors="coerce")
    parsed_date = pd.to_datetime(audited["InvoiceDate"], errors="coerce")

    audited["invoice_is_missing"] = invoice.isna() | invoice.eq("")
    audited["invoice_is_cancellation"] = invoice.str.upper().str.startswith("C", na=False)
    invoice_format_valid = invoice.str.fullmatch(INVOICE_PATTERN, na=False)
    customer_format_valid = customer.str.fullmatch(CUSTOMER_PATTERN, na=False)
    audited["invoice_has_valid_format"] = invoice_format_valid
    audited["invoice_id_has_invalid_format"] = (
        ~audited["invoice_is_missing"] & ~invoice_format_valid
    )
    audited["customer_id_is_missing"] = customer.isna() | customer.eq("")
    audited["customer_id_has_valid_format"] = customer_format_valid
    audited["customer_id_has_invalid_format"] = (
        ~audited["customer_id_is_missing"] & ~customer_format_valid
    )
    audited["stock_code_has_expected_format"] = stock_code.str.fullmatch(
        STOCK_CODE_PATTERN, na=False
    )
    audited["description_is_missing"] = description.isna() | description.eq("")
    audited["quantity_numeric"] = quantity
    audited["unit_price_numeric"] = unit_price
    audited["invoice_datetime_parsed"] = parsed_date
    audited["quantity_is_invalid"] = quantity.isna()
    audited["quantity_is_zero"] = quantity.eq(0)
    audited["quantity_is_negative"] = quantity.lt(0)
    audited["unit_price_is_invalid"] = unit_price.isna()
    audited["unit_price_is_zero"] = unit_price.eq(0)
    audited["unit_price_is_negative"] = unit_price.lt(0)
    audited["invoice_date_is_invalid"] = parsed_date.isna()
    # Compare source columns only: the added source_row_number is unique per
    # row, so including it would hide every real duplicate.
    audited["row_is_exact_duplicate"] = frame.duplicated(
        subset=list(EXPECTED_COLUMNS), keep=False
    )

    # Every copy stays flagged above; only copies after the first are dropped
    # from the candidate cohort so a duplicated line is counted once.
    audited["row_is_exact_duplicate_extra_copy"] = frame.duplicated(
        subset=list(EXPECTED_COLUMNS), keep="first"
    )
    audited["stock_code_is_non_product"] = stock_code.str.upper().isin(
        NON_PRODUCT_STOCK_CODES
    )

    invoice_customer_counts = (
        pd.DataFrame({"invoice": invoice, "customer": customer})
        .loc[lambda table: table["invoice"].notna() & table["customer"].notna()]
        .groupby("invoice")["customer"]
        .nunique()
    )
    audited["invoice_has_multiple_customers"] = (
        invoice.map(invoice_customer_counts).fillna(0).gt(1)
    )

    valid_product_rows = pd.DataFrame(
        {"stock_code": stock_code, "description": description}
    ).loc[lambda table: table["stock_code"].notna() & table["description"].notna()]
    product_description_counts = valid_product_rows.groupby("stock_code")[
        "description"
    ].nunique()
    audited["stock_code_has_multiple_descriptions"] = (
        stock_code.map(product_description_counts).fillna(0).gt(1)
    )

    cancellation_or_return = audited["invoice_is_cancellation"] | audited[
        "quantity_is_negative"
    ]
    audited["is_cancellation_or_return"] = cancellation_or_return

    # This rule defines a modeling cohort, not a deletion rule. Every excluded
    # row remains in the full audit and gets one or more explicit reasons.
    eligibility_checks = {
        "invoice_is_cancellation": audited["invoice_is_cancellation"],
        "negative_quantity_return_like": audited["quantity_is_negative"],
        "missing_invoice_id": audited["invoice_is_missing"],
        "invalid_invoice_id": audited["invoice_id_has_invalid_format"],
        "invalid_quantity_value": audited["quantity_is_invalid"],
        "nonpositive_quantity": quantity.le(0),
        "invalid_unit_price_value": audited["unit_price_is_invalid"],
        "nonpositive_unit_price": unit_price.le(0),
        "invalid_timestamp": audited["invoice_date_is_invalid"],
        "missing_customer_id": audited["customer_id_is_missing"],
        "invalid_customer_id_format": audited["customer_id_has_invalid_format"],
        "exact_duplicate_extra_copy": audited["row_is_exact_duplicate_extra_copy"],
        "non_product_line": audited["stock_code_is_non_product"],
    }
    reasons = [[] for _ in range(len(audited))]
    for reason, mask in eligibility_checks.items():
        for row_index in mask.fillna(False).to_numpy().nonzero()[0]:
            reasons[row_index].append(reason)
    audited["candidate_exclusion_reasons"] = ["|".join(items) for items in reasons]
    audited["is_customer_sales_candidate"] = audited[
        "candidate_exclusion_reasons"
    ].eq("")

    # Cancelled/returned product lines that can be netted against a customer's
    # purchases. Same hygiene as purchases, but the quantity is negative.
    audited["is_net_spend_cancellation"] = (
        audited["is_cancellation_or_return"]
        & quantity.lt(0).fillna(False)
        & unit_price.gt(0).fillna(False)
        & ~audited["invoice_is_missing"]
        & ~audited["invoice_id_has_invalid_format"]
        & ~audited["invoice_date_is_invalid"]
        & ~audited["customer_id_is_missing"]
        & ~audited["customer_id_has_invalid_format"]
        & ~audited["row_is_exact_duplicate_extra_copy"]
        & ~audited["stock_code_is_non_product"]
    )

    review_checks = {
        "exact_duplicate_row": audited["row_is_exact_duplicate"],
        "missing_product_description": audited["description_is_missing"],
        "unexpected_stock_code_format": ~audited["stock_code_has_expected_format"],
        "invoice_maps_to_multiple_customers": audited["invoice_has_multiple_customers"],
        "stock_code_maps_to_multiple_descriptions": audited[
            "stock_code_has_multiple_descriptions"
        ],
        "zero_quantity": audited["quantity_is_zero"],
        "zero_unit_price": audited["unit_price_is_zero"],
        "negative_unit_price": audited["unit_price_is_negative"],
    }
    audited["requires_review"] = False
    for mask in review_checks.values():
        audited["requires_review"] |= mask.fillna(False)

    flag_checks = {
        **{
            name: mask
            for name, mask in eligibility_checks.items()
        },
        **review_checks,
        "cancellation_invoice": audited["invoice_is_cancellation"],
        "negative_quantity_return_like": audited["quantity_is_negative"],
        "invalid_invoice_date": audited["invoice_date_is_invalid"],
        "cancellation_or_return_row": cancellation_or_return,
        "customer_sales_candidate": audited["is_customer_sales_candidate"],
    }
    summary = {
        "row_count": int(len(audited)),
        "flag_counts": {
            name: int(mask.fillna(False).sum()) for name, mask in flag_checks.items()
        },
        "candidate_exclusion_reason_counts": {
            reason: int(mask.fillna(False).sum())
            for reason, mask in eligibility_checks.items()
        },
        "rows_requiring_review": int(audited["requires_review"].sum()),
    }
    return audited, summary
