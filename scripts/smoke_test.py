"""After a deploy: wait for the Space to finish building, then call its API.

Usage: python scripts/smoke_test.py <owner/space-name>

Passes only if the live Space returns the same probability the evaluated model
gave a known test client. A green deploy step alone only proves the files were
uploaded; this proves the new version actually runs and answers correctly.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from gradio_client import Client
from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parents[1]
WAIT_SECONDS = 900
FAILED = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


def wait_until_running(space: str, sha: str | None) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    time.sleep(30)  # right after an upload the old version can still report RUNNING
    deadline = time.time() + WAIT_SECONDS
    while time.time() < deadline:
        runtime = api.get_space_runtime(space)
        stage = runtime.stage
        running_sha = (runtime.raw or {}).get("sha")
        print(f"  stage {stage}", flush=True)
        if stage in FAILED:
            raise SystemExit(f"Space failed to start: {stage}")
        if stage == "RUNNING" and (sha is None or running_sha in (None, sha)):
            return
        time.sleep(15)
    raise SystemExit(f"Space was not running after {WAIT_SECONDS} s.")


def main(space: str) -> None:
    sha = HfApi(token=os.environ.get("HF_TOKEN")).space_info(space).sha
    print(f"waiting for {space} at commit {sha} ...")
    wait_until_running(space, sha)
    case = pd.read_csv(HERE / "model" / "parity_cases.csv").iloc[0]
    from credit_model import CreditModel
    columns = CreditModel.load().raw_columns
    client = Client(space, token=os.environ.get("HF_TOKEN"))
    headline, _ = client.predict(*[float(case[c]) for c in columns], api_name="/score")
    print(headline)
    match = re.search(r"([0-9.]+)%", headline)
    if not match:
        raise SystemExit("The Space did not return a probability.")
    live = float(match.group(1)) / 100
    expected = round(float(case["expected_probability"]), 3)
    if abs(live - expected) > 0.0005:
        raise SystemExit(f"Live answer {live:.3f} differs from expected {expected:.3f}.")
    print(f"smoke test passed: live {live:.1%}, expected {expected:.1%}")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main(sys.argv[1])
