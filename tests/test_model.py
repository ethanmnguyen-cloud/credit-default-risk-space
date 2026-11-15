"""The deployed model must give the same answers as the model that was evaluated.

model/parity_cases.csv holds the raw records of the 5,993 held-out test clients and
the probability the evaluated scikit-learn pipeline gave each one (written by
`model export`). If any step drifts (a rename, a recode, a feature formula, a
scaling constant), these tests fail before the change can reach the Space.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_model import CreditModel, InputError

HERE = Path(__file__).resolve().parents[1]
CASES = pd.read_csv(HERE / "model" / "parity_cases.csv")
MODEL = CreditModel.load()


def record(i: int = 0, **changes) -> dict:
    row = CASES.iloc[i][MODEL.raw_columns].to_dict()
    row.update(changes)
    return row


# ------------------------------------------------------------------ parity
def test_parity_with_evaluated_model_on_every_test_client():
    probabilities = MODEL.predict_proba(CASES)
    gap = np.max(np.abs(probabilities - CASES["expected_probability"].to_numpy()))
    assert len(CASES) == 5993
    assert gap < 1e-9, f"largest gap {gap:.2e}"


def test_flags_match_and_reproduce_the_reported_confusion_matrix():
    flags = (MODEL.predict_proba(CASES) > MODEL.threshold).astype(int)
    assert (flags == CASES["expected_flag"].to_numpy()).all()
    actual = CASES["actual_default"].to_numpy()
    confusion = MODEL.spec["evaluation"]["test"]["confusion"]
    assert int(((flags == 1) & (actual == 1)).sum()) == confusion["tp"]
    assert int(((flags == 1) & (actual == 0)).sum()) == confusion["fp"]
    assert int(((flags == 0) & (actual == 1)).sum()) == confusion["fn"]
    assert int(((flags == 0) & (actual == 0)).sum()) == confusion["tn"]


def test_single_record_path_matches_batch_path():
    for i in (0, 1, 2, 500, 5992):
        single = MODEL.predict(record(i)).probability
        assert single == pytest.approx(CASES["expected_probability"].iloc[i], abs=1e-12)


def test_contributions_add_up_to_the_log_odds():
    result = MODEL.predict(record(7))
    log_odds = sum(value for _, value in result.contributions) + MODEL.intercept
    assert 1 / (1 + np.exp(-log_odds)) == pytest.approx(result.probability, abs=1e-12)


# ------------------------------------------------------------------ protocol steps
def test_undocumented_codes_are_merged_like_training():
    merged = MODEL.predict(record(3, EDUCATION=4)).probability
    for code in (0, 5, 6):
        assert MODEL.predict(record(3, EDUCATION=code)).probability == pytest.approx(merged)
    assert MODEL.predict(record(3, MARRIAGE=0)).probability == pytest.approx(
        MODEL.predict(record(3, MARRIAGE=3)).probability)


def test_pay_ratio_falls_back_to_one_when_nothing_was_owed():
    prepared = MODEL.prepare(MODEL.validate(pd.DataFrame([record(0, BILL_AMT2=0)])))
    assert prepared["PAY_RATIO"].iloc[0] == 1.0


def test_months_late_counts_positive_delays():
    prepared = MODEL.prepare(MODEL.validate(pd.DataFrame([record(
        0, PAY_0=2, PAY_2=1, PAY_3=0, PAY_4=-1, PAY_5=-2, PAY_6=3)])))
    assert prepared["MONTHS_LATE"].iloc[0] == 3


def test_negative_bill_overpayment_is_allowed():
    assert 0 <= MODEL.predict(record(0, BILL_AMT1=-5000)).probability <= 1


# ------------------------------------------------------------------ bad input
@pytest.mark.parametrize("changes, message", [
    ({"LIMIT_BAL": 0}, "LIMIT_BAL"),
    ({"AGE": 12}, "AGE"),
    ({"SEX": 3}, "SEX"),
    ({"PAY_0": 1.5}, "PAY_0"),
    ({"PAY_AMT1": -10}, "cannot be negative"),
    ({"BILL_AMT3": "lots"}, "must be a number"),
])
def test_bad_inputs_are_rejected_with_a_clear_message(changes, message):
    with pytest.raises(InputError, match=message):
        MODEL.predict(record(0, **changes))


def test_missing_input_is_named():
    row = record(0)
    del row["AGE"]
    with pytest.raises(InputError, match="AGE"):
        MODEL.predict(row)


# ------------------------------------------------------------------ the file itself
def test_model_file_describes_the_evaluated_model():
    spec = MODEL.spec
    assert spec["model"]["settings"]["C"] == 0.1
    assert spec["model"]["settings"]["class_weight"] == "balanced"
    assert len(spec["model"]["coefficients"]) == 89
    assert spec["provenance"]["max_abs_diff_sklearn"] < 1e-9
