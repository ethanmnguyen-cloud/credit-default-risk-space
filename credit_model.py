"""Credit default risk from a raw UCI client record, using only NumPy and pandas.

The fitted logistic regression lives in model/model.json as plain numbers
(written by `model export` in the training project). This module repeats the
exact steps the model was trained after:

    raw record -> rename PAY_0 to PAY_1 -> merge undocumented codes
      -> add UTILIZATION, PAY_RATIO, MONTHS_LATE
      -> impute, signed log, scale, one-hot -> weighted sum -> probability

tests/test_model.py checks that this gives the same probability as the
evaluated scikit-learn model for all 5,993 held-out clients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "model" / "model.json"


class InputError(ValueError):
    """A record the model cannot score, with a message a user can act on."""


@dataclass(frozen=True)
class Prediction:
    probability: float
    flagged: bool
    contributions: list[tuple[str, float]]


class CreditModel:
    def __init__(self, spec: Mapping[str, Any]):
        if spec.get("format_version") != 1:
            raise ValueError(f"Unsupported model format {spec.get('format_version')!r}.")
        self.spec = spec
        self.raw_columns: list[str] = list(spec["inputs"]["raw_columns"])
        self.codes: dict[str, dict[str, str]] = spec["inputs"]["codes"]
        self.protocol = spec["protocol"]
        self.blocks = spec["preprocessing"]
        self.coef = np.asarray(spec["model"]["coefficients"], dtype=float)
        self.intercept = float(spec["model"]["intercept"])
        self.threshold = float(spec["model"]["threshold"])
        self.feature_names: list[str] = list(spec["model"]["feature_names"])
        if len(self.coef) != len(self.feature_names):
            raise ValueError("Coefficient count does not match feature names.")

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> "CreditModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ public
    def predict(self, record: Mapping[str, Any]) -> Prediction:
        """Score one client. Raises InputError for anything the model cannot use."""
        frame = self.validate(pd.DataFrame([dict(record)]))
        design = self.design_matrix(self.prepare(frame))
        z = design @ self.coef + self.intercept
        probability = float(_sigmoid(z)[0])
        parts = design[0] * self.coef
        ranked = sorted(zip(self.feature_names, parts), key=lambda kv: -abs(kv[1]))
        return Prediction(probability, probability > self.threshold,
                          [(name, float(value)) for name, value in ranked])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Score many raw records at once (used by the tests and batch checks)."""
        design = self.design_matrix(self.prepare(self.validate(frame)))
        return _sigmoid(design @ self.coef + self.intercept)

    # ------------------------------------------------------------------ steps
    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.raw_columns if c not in frame.columns]
        if missing:
            raise InputError(f"Missing inputs: {', '.join(missing)}.")
        out = frame[self.raw_columns].copy()
        for column in self.raw_columns:
            values = pd.to_numeric(out[column], errors="coerce")
            if values.isna().any():
                raise InputError(f"{column} must be a number.")
            out[column] = values
        if (out["LIMIT_BAL"] <= 0).any():
            raise InputError("LIMIT_BAL (credit limit) must be greater than zero.")
        if ((out["AGE"] < 18) | (out["AGE"] > 100)).any():
            raise InputError("AGE must be between 18 and 100.")
        for column, allowed in self.codes.items():
            bad = ~out[column].astype(int).astype(str).isin(set(allowed))
            if (out[column] % 1 != 0).any() or bad.any():
                raise InputError(f"{column} must be one of the codes {', '.join(allowed)}.")
        pay_amounts = [c for c in self.raw_columns if c.startswith("PAY_AMT")]
        if (out[pay_amounts] < 0).any().any():
            raise InputError("Payment amounts (PAY_AMT1 to PAY_AMT6) cannot be negative.")
        return out

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rename, recode, and add the three row-wise features, as in training."""
        out = frame.rename(columns=self.protocol["rename"])
        for column, rule in self.protocol["recode"].items():
            text = out[column].astype(int).astype(str)
            out[column] = text.where(~text.isin(rule["codes"]), rule["into"])
        for feature in self.protocol["features"]:
            out[feature["name"]] = _feature(out, feature)
        for column in self.protocol["nominal_columns"]:
            out[column] = out[column].astype(int).astype(str)
        return out

    def design_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        parts = []
        for block in self.blocks:
            if block["kind"] == "numeric":
                values = frame[block["columns"]].to_numpy(dtype=float)
                values = np.where(np.isnan(values), np.asarray(block["medians"]), values)
                if block["signed_log"]:
                    values = np.sign(values) * np.log1p(np.abs(values))
                parts.append((values - np.asarray(block["means"])) / np.asarray(block["scales"]))
            elif block["kind"] == "one_hot":
                for column, cats in zip(block["columns"], block["categories"]):
                    codes = frame[column].to_numpy(dtype=str)
                    parts.append((codes[:, None] == np.asarray(cats)[None, :]).astype(float))
            else:
                raise ValueError(f"Unknown block kind {block['kind']!r}.")
        return np.hstack(parts)


# ---------------------------------------------------------------------------------
def _feature(frame: pd.DataFrame, spec: Mapping[str, Any]) -> pd.Series:
    """Same rules as modeling/features.py in the training project."""
    if spec["kind"] == "ratio":
        numerator = frame[spec["numerator"]].astype(float)
        denominator = frame[spec["denominator"]].astype(float)
        not_positive = denominator <= 0
        values = numerator / denominator.where(~not_positive)
        if spec.get("when_not_positive") is not None:
            values = values.where(~not_positive, float(spec["when_not_positive"]))
        return values.replace([np.inf, -np.inf], np.nan)
    if spec["kind"] == "count_at_least":
        block = frame[spec["columns"]].astype(float)
        return (block >= float(spec.get("threshold", 1))).sum(axis=1).astype(int)
    raise ValueError(f"Unknown feature kind {spec['kind']!r}.")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))
