"""Smoke tests: the dashboard renders every section without exceptions."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT = Path(__file__).resolve().parents[1]
APP = str(PROJECT / "app" / "dashboard.py")
COMMITTED_JSON = ["data_quality", "model_evaluation", "segmentation_profile", "clv_evaluation"]
PAGES = ["Business overview", "Segments", "Customer behaviour", "Repeat-purchase model",
         "90-day predicted customer value", "Recommendations", "Methodology and limitations"]


def have_committed_json():
    return all((PROJECT / "reports" / "generated" / f"{name}.json").is_file() for name in COMMITTED_JSON)


@pytest.fixture
def fixture_root(tmp_path, monkeypatch):
    """A small project root: real committed reports plus tiny synthetic customer tables."""
    if not have_committed_json():
        pytest.skip("generated reports not available")
    reports, processed = tmp_path / "reports" / "generated", tmp_path / "data" / "processed"
    reports.mkdir(parents=True)
    processed.mkdir(parents=True)
    for name in COMMITTED_JSON:
        shutil.copy(PROJECT / "reports" / "generated" / f"{name}.json", reports / f"{name}.json")
    rng = np.random.default_rng(0)
    n = 120
    names = ["Champions", "At Risk", "Promising", "New", "Cooling Down", "Lost"]
    gross = rng.gamma(2, 400, n).round(2)
    cancelled = (gross * rng.uniform(0, .2, n)).round(2)
    segments = pd.DataFrame({
        "CustomerID": [str(10000 + i) for i in range(n)],
        "recency_days": rng.integers(1, 370, n), "frequency_orders": rng.integers(1, 20, n),
        "gross_spend_gbp": gross, "cancelled_spend_gbp": cancelled, "monetary_gbp": gross - cancelled,
        "net_spend_nonpositive": False,
        "segment": rng.choice(names, n), "segment_quartile": rng.choice(["Loyal", "Recent"], n),
        "segment_kmeans": rng.choice(["Cluster 1", "Cluster 2", "Cluster 3"], n)})
    segments.loc[0, ["segment", "segment_quartile", "segment_kmeans"]] = "Flagged: zero/negative net spend"
    segments.to_csv(processed / "customer_segments.csv", index=False)
    pd.DataFrame({
        "CustomerID": segments.CustomerID, "purchase_days": rng.integers(1, 9, n), "net_spend": gross - cancelled,
        "predicted_net": rng.gamma(2, 150, n), "actual_net": rng.gamma(2, 200, n),
        "excluded_nonpositive_calibration_net": False}).to_csv(processed / "clv_forecast.csv", index=False)
    monkeypatch.setenv("RETAIL_DASHBOARD_ROOT", str(tmp_path))
    return tmp_path


def run_page(page=None):
    app = AppTest.from_file(APP, default_timeout=60).run()
    if page:
        app.sidebar.radio[0].set_value(page).run()
    return app


def all_text(app):
    parts = []
    for group in (app.markdown, app.caption, app.header, app.subheader, app.title, app.error, app.warning, app.info):
        parts += [element.value for element in group]
    parts += [metric.label for metric in app.metric]
    return "\n".join(parts)


@pytest.mark.parametrize("page", PAGES)
def test_every_section_renders_without_exceptions(fixture_root, page):
    app = run_page(page)

    assert not app.exception, [e.value for e in app.exception]


@pytest.mark.parametrize("page", PAGES)
def test_missing_outputs_show_which_step_to_run_instead_of_crashing(tmp_path, monkeypatch, page):
    monkeypatch.setenv("RETAIL_DASHBOARD_ROOT", str(tmp_path))

    app = run_page(page)

    assert not app.exception, [e.value for e in app.exception]
    text = all_text(app)
    if page != "Methodology and limitations":
        assert "python -m src." in text and "is missing" in text
    assert not app.metric  # no number is invented when files are absent


def test_segment_methods_and_filters_work(fixture_root):
    app = run_page("Segments")
    for method in ("Quartile rules (old)", "KMeans (comparison)", "Band rules (default)"):
        app.radio[0].set_value(method).run()
        assert not app.exception, (method, [e.value for e in app.exception])


def test_behaviour_filters_with_no_matches_show_a_message(fixture_root):
    app = run_page("Customer behaviour")
    app.multiselect[0].set_value([]).run()

    assert not app.exception
    assert any("No customers match" in info.value for info in app.info)


def test_value_page_warns_with_the_underprediction_computed_from_the_report(fixture_root):
    report = json.loads((fixture_root / "reports" / "generated" / "clv_evaluation.json").read_text())
    ratio = report["all_ranked_customers"]["net"]["model"]["predicted_over_actual"]
    expected = f"under-predicts the total by about {(1 - ratio) * 100:.0f}%"

    app = run_page("90-day predicted customer value")

    assert any(expected in error.value for error in app.error)
    assert any("Ranking tool only" in error.value for error in app.error)


def test_overview_numbers_come_from_the_reports(fixture_root):
    quality = json.loads((fixture_root / "reports" / "generated" / "data_quality.json").read_text())
    rows = {r["stage"]: r["rows"] for r in quality["row_counts"]}

    app = run_page("Business overview")
    metrics = {m.label: m.value for m in app.metric}

    assert metrics["Source rows loaded"] == f"{rows['source_rows_loaded']:,}"
    assert metrics["Candidate sales rows"] == f"{rows['customer_sales_candidates']:,}"


def test_recommendations_are_all_labelled_and_never_say_validated_clv(fixture_root):
    app = run_page("Recommendations")
    text = all_text(app)

    assert "Data-supported finding" in text and "Hypothesis to test" in text
    assert text.count("Data-supported finding") >= 4 and text.count("Hypothesis to test") >= 3
    assert "CLV" not in text and "validated CLV" not in text


@pytest.mark.parametrize("page", PAGES)
def test_the_term_clv_never_appears(fixture_root, page):
    assert "CLV" not in all_text(run_page(page))


def test_dashboard_runs_on_the_real_outputs_when_present():
    needed = [PROJECT / "data" / "processed" / "customer_segments.csv", PROJECT / "data" / "processed" / "clv_forecast.csv"]
    if not all(path.is_file() for path in needed) or not have_committed_json():
        pytest.skip("real pipeline outputs not generated on this machine")
    for page in PAGES:
        app = run_page(page)
        assert not app.exception, (page, [e.value for e in app.exception])


def test_cancellation_share_on_recommendations_is_a_real_percentage(fixture_root):
    segments = pd.read_csv(fixture_root / "data" / "processed" / "customer_segments.csv")
    share = segments.cancelled_spend_gbp.sum() / segments.gross_spend_gbp.sum() * 100

    text = all_text(run_page("Recommendations"))

    assert f"{share:.1f}% of gross spend" in text


def decode(array):
    """Plotly may ship numeric arrays as {'dtype': ..., 'bdata': base64}."""
    if isinstance(array, dict):
        import base64
        return np.frombuffer(base64.b64decode(array["bdata"]), dtype=np.dtype(array["dtype"])).astype(float)
    return np.asarray(array, dtype=float)


def test_net_spend_histogram_has_data_on_a_log_axis(fixture_root):
    app = run_page("Customer behaviour")

    assert not app.exception
    charts = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    net = [c for c in charts if c["layout"].get("title", {}).get("text") == "Net spend"]
    assert net, "net spend histogram missing"
    values = np.concatenate([decode(trace.get("x", [])) for trace in net[0]["data"]])
    assert len(values) > 0 and np.isfinite(values).all()
    assert max(values) < 10   # log10 of pounds, not raw pounds
