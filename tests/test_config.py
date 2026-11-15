"""Catch configuration drift between the Space and the CI environment.

The Space installs Gradio from `sdk_version` in README.md, while CI installs it from
requirements-dev.txt. If the two disagree, tests pass on one version and the Space
runs another.
"""

import re
import sys
from pathlib import Path

import gradio
import numpy
import pandas

ROOT = Path(__file__).resolve().parents[1]


def front_matter() -> dict[str, str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    block = text.split("---")[1]
    return {k.strip(): v.strip().strip('"') for k, v in
            (line.split(":", 1) for line in block.strip().splitlines())}


def pins(path: str) -> dict[str, str]:
    found = {}
    for line in (ROOT / path).read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line.strip())
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def test_space_gradio_version_matches_ci():
    meta = front_matter()
    assert meta["sdk"] == "gradio"
    assert meta["sdk_version"] == pins("requirements-dev.txt")["gradio"] == gradio.__version__


def test_space_python_matches_ci():
    assert front_matter()["python_version"] == f"{sys.version_info.major}.{sys.version_info.minor}"


def test_runtime_pins_match_what_was_tested():
    runtime = pins("requirements.txt")
    assert runtime["numpy"] == numpy.__version__
    assert runtime["pandas"] == pandas.__version__


def test_app_file_exists():
    assert (ROOT / front_matter()["app_file"]).exists()


def test_readme_metrics_match_model_file():
    from credit_model import CreditModel

    test = CreditModel.load().spec["evaluation"]["test"]
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("accuracy", "precision", "recall", "roc_auc"):
        label = "ROC-AUC" if name == "roc_auc" else name
        assert f"{label} {test[name]:.3f}" in text, name
    assert f"F1 {test['f1']:.3f}" in text
