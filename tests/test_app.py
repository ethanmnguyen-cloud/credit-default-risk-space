"""The Gradio app builds, and its score function answers like the model."""

from pathlib import Path

import pandas as pd
import pytest

import app

CASES = pd.read_csv(Path(app.__file__).parent / "model" / "parity_cases.csv")


def test_app_builds_with_one_input_per_raw_field():
    demo = app.build()
    assert demo is not None
    assert len(app.SPEC["inputs"]["raw_columns"]) == 23


def test_score_reports_the_model_probability():
    row = CASES.iloc[0]
    values = [row[c] for c in app.SPEC["inputs"]["raw_columns"]]
    headline, factors = app.score(*values)
    assert f"{row['expected_probability']:.1%}" in headline
    assert "What moved this estimate most" in factors


def test_score_explains_bad_input_instead_of_crashing():
    values = [app.EXAMPLES["On-time client"][c] for c in app.SPEC["inputs"]["raw_columns"]]
    values[0] = 0  # LIMIT_BAL
    headline, factors = app.score(*values)
    assert "Cannot score" in headline and factors == ""


@pytest.mark.parametrize("name", list(app.EXAMPLES))
def test_examples_are_valid_and_ordered_by_risk(name):
    values = [app.EXAMPLES[name][c] for c in app.SPEC["inputs"]["raw_columns"]]
    assert 0 < app.MODEL.predict(dict(zip(app.SPEC["inputs"]["raw_columns"], values))).probability < 1


def test_behind_example_scores_higher_than_on_time_example():
    p = {name: app.MODEL.predict(app.EXAMPLES[name]).probability for name in app.EXAMPLES}
    assert p["Client two months behind"] > p["On-time client"]


def test_feature_labels_are_readable():
    assert app._label("nom__PAY_1_2") == "September status = 2"
    assert app._label("log__BILL_AMT1") == "September bill"
    assert app._label("num__UTILIZATION") == "share of limit used"
