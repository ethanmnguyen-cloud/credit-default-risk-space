"""Open the page in a real browser (Chromium) and check that it scores correctly.

Usage: python scripts/browser_check.py <base-url> [--screenshot file.png]

CI runs this twice: against a local copy before deploying, and against the live
Space after deploying. It checks what the Node tests cannot see: that the page
loads its files, the form is wired to the model, and the browser's own maths gives
the evaluated probabilities for all 5,993 held-out test clients.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SELECTS = {"SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"}


def load_cases() -> list[dict[str, float]]:
    with open(ROOT / "model" / "parity_cases.csv", newline="") as handle:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def headline_percent(page) -> float:
    text = page.locator("#headline").inner_text()
    return float(text.rsplit(" ", 1)[1].rstrip("%"))


def fill(page, row: dict[str, float], columns: list[str]) -> None:
    for column in columns:
        value = row[column]
        text = str(int(value)) if value == int(value) else repr(value)
        if column in SELECTS:
            page.select_option(f"#{column}", text)
        else:
            page.fill(f"#{column}", text)


def check(base_url: str, screenshot: str | None = None) -> None:
    url = base_url.rstrip("/") + "/index.html"
    cases = load_cases()
    columns = json.loads((ROOT / "model" / "model.json").read_text())["inputs"]["raw_columns"]
    problems: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        page.on("pageerror", lambda error: problems.append(f"page error: {error}"))
        page.on("requestfailed", lambda request: problems.append(f"failed request: {request.url}"))
        page.goto(url, wait_until="load")
        page.wait_for_function("document.body.dataset.ready !== undefined", timeout=30_000)
        state = page.evaluate("document.body.dataset.ready")
        if state != "true":
            raise SystemExit(f"The page did not start: {page.locator('#loading').inner_text()}")

        # 1. The browser's own maths, on every held-out client.
        rows = [{c: r[c] for c in columns} for r in cases]
        live = page.evaluate("rows => rows.map(r => window.creditModel.predict(r).probability)", rows)
        gap = max(abs(a - r["expected_probability"]) for a, r in zip(live, cases))
        print(f"in-browser parity on {len(cases):,} clients: largest gap {gap:.1e}")
        if gap >= 1e-9:
            problems.append(f"in-browser parity gap {gap:.1e}")

        # 2. The form: type a few real test clients in and press the button.
        flagged = next(i for i, r in enumerate(cases) if r["expected_flag"] == 1)
        for i in (0, flagged, len(cases) - 1):
            fill(page, cases[i], columns)
            page.click("#estimate")
            shown = headline_percent(page)
            expected = cases[i]["expected_probability"] * 100
            print(f"form, client {int(cases[i]['ID'])}: shows {shown:.1f}%, expected {expected:.2f}%")
            if abs(shown - expected) > 0.05 + 1e-9:
                problems.append(f"form shows {shown}% for client {int(cases[i]['ID'])}")
            verdict = page.locator("#verdict").inner_text()
            if (verdict == "Not flagged") == bool(cases[i]["expected_flag"]):
                problems.append(f"wrong verdict '{verdict}' for client {int(cases[i]['ID'])}")

        # 3. Bad input is explained, not crashed on.
        page.fill("#LIMIT_BAL", "0")
        page.click("#estimate")
        if "LIMIT_BAL" not in page.locator("#error-message").inner_text():
            problems.append("a zero credit limit was not explained")

        # 4. The example buttons.
        scores = {}
        for button in page.locator("[data-example]").all():
            button.click()
            scores[button.inner_text()] = headline_percent(page)
        print(f"examples: {scores}")
        if not scores.get("Client two months behind", 0) > scores.get("On-time client", 100):
            problems.append(f"examples out of order: {scores}")
        if screenshot:
            page.screenshot(path=screenshot, full_page=True)
        browser.close()
    if problems:
        raise SystemExit("Browser check failed:\n  " + "\n  ".join(problems))
    print(f"browser check passed: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--screenshot")
    args = parser.parse_args()
    sys.exit(check(args.base_url, args.screenshot))
