"""Reports must carry project-relative paths, never machine-specific ones."""

import json
from pathlib import Path

import pandas as pd

from src.data.pipeline import run_pipeline
from src.features.customer_features import run_feature_pipeline
from src.paths import PROJECT_ROOT, project_relative

REPORT_DIR = PROJECT_ROOT / "reports" / "generated"


def string_values(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from string_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from string_values(value)
    elif isinstance(node, str):
        yield node


def path_values(report: dict, keys=("outputs",)):
    return [
        value
        for key in ("input_file", "output_file", "sqlite_database", "source_file")
        for value in [report.get(key)]
        if value
    ] + [value for key in keys for value in report.get(key, {}).values()]


def assert_relative(paths):
    assert paths
    for value in paths:
        assert not value.startswith("/"), value
        assert "/home/" not in value, value


def test_project_relative_inside_and_outside_project(tmp_path):
    assert project_relative(PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx") == "data/raw/Online Retail.xlsx"
    assert project_relative(tmp_path / "out.csv") == "out.csv"


def test_pipeline_reports_contain_no_absolute_paths(tmp_path):
    source = tmp_path / "retail.xlsx"
    pd.DataFrame(
        {
            "InvoiceNo": ["536365", "C536379"],
            "StockCode": ["85123", "85123"],
            "Description": ["HEART", "HEART"],
            "Quantity": [6, -2],
            "InvoiceDate": pd.to_datetime(["2010-12-01 08:26", "2010-12-01 09:41"]),
            "UnitPrice": [2.55, 2.55],
            "CustomerID": ["17850", "17850"],
            "Country": ["United Kingdom"] * 2,
        }
    ).to_excel(source, index=False)
    processed, reports = tmp_path / "processed", tmp_path / "reports"

    quality = run_pipeline(source, processed, reports)
    features = run_feature_pipeline(
        processed / "customer_sales_candidates.csv",
        processed / "net_spend_cancellations.csv",
        processed / "customer_features.csv",
        reports / "feature_report.json",
        processed / "online_retail.sqlite",
    )

    assert_relative(path_values(quality))
    assert_relative(path_values(features))


def test_committed_reports_contain_no_absolute_paths():
    reports = sorted(REPORT_DIR.glob("*.json"))
    assert reports
    for report_path in reports:
        for value in string_values(json.loads(report_path.read_text(encoding="utf-8"))):
            assert not value.startswith("/"), (report_path.name, value)
            assert "/home/" not in value, (report_path.name, value)
