// The deployed model must give the same answers as the model that was evaluated.
//
// model/parity_cases.csv holds the raw records of the 5,993 held-out test clients and
// the probability the evaluated scikit-learn pipeline gave each one (written by
// `model export`). If any step drifts (a rename, a recode, a feature formula, a
// scaling constant), these tests fail before the change can reach the Space.

import assert from "node:assert/strict";
import { test } from "node:test";

import { InputError } from "../credit_model.js";
import { CASES, MODEL, record } from "./helpers.js";

const close = (a, b, tolerance) =>
  assert.ok(Math.abs(a - b) <= tolerance, `${a} differs from ${b} by more than ${tolerance}`);

// ------------------------------------------------------------------ parity
test("parity with the evaluated model on every test client", () => {
  assert.equal(CASES.length, 5993);
  let gap = 0;
  CASES.forEach((row, i) => {
    gap = Math.max(gap, Math.abs(MODEL.predict(record(i)).probability - row.expected_probability));
  });
  assert.ok(gap < 1e-9, `largest gap ${gap.toExponential(2)}`);
});

test("flags match and reproduce the reported confusion matrix", () => {
  const counts = { tp: 0, fp: 0, fn: 0, tn: 0 };
  CASES.forEach((row, i) => {
    const flag = MODEL.predict(record(i)).flagged ? 1 : 0;
    assert.equal(flag, row.expected_flag, `client ${row.ID}`);
    const key = (flag === 1 ? (row.actual_default ? "tp" : "fp")
      : (row.actual_default ? "fn" : "tn"));
    counts[key] += 1;
  });
  const confusion = MODEL.spec.evaluation.test.confusion;
  for (const key of ["tp", "fp", "fn", "tn"]) assert.equal(counts[key], confusion[key], key);
});

test("contributions add up to the log-odds", () => {
  const result = MODEL.predict(record(7));
  const logOdds = result.contributions.reduce((sum, [, v]) => sum + v, 0) + MODEL.intercept;
  close(1 / (1 + Math.exp(-logOdds)), result.probability, 1e-12);
  assert.equal(result.contributions.length, 89);
});

test("text input from the form scores the same as numbers", () => {
  const asText = Object.fromEntries(Object.entries(record(0)).map(([k, v]) => [k, String(v)]));
  close(MODEL.predict(asText).probability, CASES[0].expected_probability, 1e-12);
});

// ------------------------------------------------------------------ protocol steps
test("undocumented codes are merged like training", () => {
  const merged = MODEL.predict(record(3, { EDUCATION: 4 })).probability;
  for (const code of [0, 5, 6]) {
    close(MODEL.predict(record(3, { EDUCATION: code })).probability, merged, 1e-12);
  }
  close(MODEL.predict(record(3, { MARRIAGE: 0 })).probability,
    MODEL.predict(record(3, { MARRIAGE: 3 })).probability, 1e-12);
});

test("PAY_RATIO falls back to 1 when nothing was owed", () => {
  assert.equal(MODEL.prepare(MODEL.validate(record(0, { BILL_AMT2: 0 }))).PAY_RATIO, 1);
});

test("MONTHS_LATE counts positive delays", () => {
  const changes = { PAY_0: 2, PAY_2: 1, PAY_3: 0, PAY_4: -1, PAY_5: -2, PAY_6: 3 };
  assert.equal(MODEL.prepare(MODEL.validate(record(0, changes))).MONTHS_LATE, 3);
});

test("a negative bill (overpayment) is allowed", () => {
  const p = MODEL.predict(record(0, { BILL_AMT1: -5000 })).probability;
  assert.ok(p > 0 && p < 1);
});

// ------------------------------------------------------------------ bad input
for (const [changes, message] of [
  [{ LIMIT_BAL: 0 }, /LIMIT_BAL/],
  [{ AGE: 12 }, /AGE/],
  [{ SEX: 3 }, /SEX/],
  [{ PAY_0: 1.5 }, /PAY_0/],
  [{ PAY_AMT1: -10 }, /cannot be negative/],
  [{ BILL_AMT3: "lots" }, /must be a number/],
  [{ BILL_AMT3: "" }, /must be a number/],
  [{ BILL_AMT3: null }, /must be a number/],
]) {
  test(`bad input ${JSON.stringify(changes)} is rejected with a clear message`, () => {
    assert.throws(() => MODEL.predict(record(0, changes)),
      (error) => error instanceof InputError && message.test(error.message));
  });
}

test("a missing input is named", () => {
  const row = record(0);
  delete row.AGE;
  assert.throws(() => MODEL.predict(row), (e) => e instanceof InputError && /AGE/.test(e.message));
});

// ------------------------------------------------------------------ the file itself
test("model file describes the evaluated model", () => {
  const { spec } = MODEL;
  assert.equal(spec.model.settings.C, 0.1);
  assert.equal(spec.model.settings.class_weight, "balanced");
  assert.equal(spec.model.coefficients.length, 89);
  assert.ok(spec.provenance.max_abs_diff_sklearn < 1e-9);
});
