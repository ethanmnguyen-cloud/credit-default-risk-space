// The example clients and factor labels shown on the page.

import assert from "node:assert/strict";
import { test } from "node:test";

import { EXAMPLES, label } from "../content.js";
import { MODEL } from "./helpers.js";

test("every example fills every raw field and scores", () => {
  for (const [name, example] of Object.entries(EXAMPLES)) {
    assert.deepEqual(Object.keys(example).sort(), [...MODEL.rawColumns].sort(), name);
    const p = MODEL.predict(example).probability;
    assert.ok(p > 0 && p < 1, name);
  }
});

test("the behind example scores higher than the on-time example", () => {
  assert.ok(MODEL.predict(EXAMPLES["Client two months behind"]).probability >
    MODEL.predict(EXAMPLES["On-time client"]).probability);
});

test("factor labels are readable", () => {
  assert.equal(label("nom__PAY_1_2"), "September status = 2");
  assert.equal(label("nom__PAY_6_-1"), "April status = -1");
  assert.equal(label("log__BILL_AMT1"), "September bill");
  assert.equal(label("log__PAY_AMT6"), "April payment");
  assert.equal(label("num__UTILIZATION"), "share of limit used");
  assert.equal(label("nom__EDUCATION_4"), "education = 4");
});

test("every model input has a readable label", () => {
  for (const name of MODEL.featureNames) assert.ok(!label(name).includes("__"), name);
});
