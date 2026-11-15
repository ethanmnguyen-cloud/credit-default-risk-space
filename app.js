// Page wiring: load model.json, build the form, score on submit.
// All scoring happens in credit_model.js, inside the visitor's browser.

import { CreditModel, InputError } from "./credit_model.js";
import { BILLS, EXAMPLES, MONTHS, PAY_STATUS, PAYMENTS, label } from "./content.js";

const $ = (id) => document.getElementById(id);

function element(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function fillSelect(select, codes, value) {
  for (const [code, text] of Object.entries(codes)) {
    select.append(element("option", { value: code }, `${code}: ${text}`));
  }
  select.value = String(value);
}

function buildForm(model) {
  const codes = model.codes;
  fillSelect($("SEX"), codes.SEX, Object.keys(codes.SEX)[0]);
  fillSelect($("EDUCATION"), codes.EDUCATION, Object.keys(codes.EDUCATION)[0]);
  fillSelect($("MARRIAGE"), codes.MARRIAGE, Object.keys(codes.MARRIAGE)[0]);
  PAY_STATUS.forEach((column, i) => {
    const select = element("select", { id: column, name: column });
    fillSelect(select, codes[column], 0);
    const wrap = element("label", {}, MONTHS[i]);
    wrap.append(select);
    $("status-fields").append(wrap);
  });
  for (const [fieldset, columns, min] of [["bill-fields", BILLS, null], ["payment-fields", PAYMENTS, 0]]) {
    columns.forEach((column, i) => {
      const attrs = { id: column, name: column, type: "number", step: "any", value: "0" };
      if (min !== null) attrs.min = String(min);
      const wrap = element("label", {}, MONTHS[i]);
      wrap.append(element("input", attrs));
      $(fieldset).append(wrap);
    });
  }
  for (const name of Object.keys(EXAMPLES)) {
    const button = element("button", { type: "button", "data-example": name }, name);
    button.addEventListener("click", () => {
      for (const [column, value] of Object.entries(EXAMPLES[name])) $(column).value = String(value);
      showResult(model);
    });
    $("example-buttons").append(button);
  }
}

function readForm(model) {
  return Object.fromEntries(model.rawColumns.map((c) => [c, $(c).value]));
}

function showResult(model) {
  const box = $("result");
  box.replaceChildren();
  let result;
  try {
    result = model.predict(readForm(model));
  } catch (error) {
    if (!(error instanceof InputError)) throw error;
    box.append(element("h2", {}, "Cannot score this client"));
    box.append(element("p", { class: "error", id: "error-message" }, error.message));
    return;
  }
  const percent = (result.probability * 100).toFixed(1);
  box.append(element("h2", { id: "headline" }, `Estimated chance of default: ${percent}%`));
  const verdict = element("p");
  verdict.append(element("span", { class: `verdict ${result.flagged ? "flagged" : "clear"}`, id: "verdict" },
    result.flagged ? "Flagged as likely to default" : "Not flagged"));
  verdict.append(` The model flags clients above ${Math.round(model.threshold * 100)}%.`);
  box.append(verdict);

  box.append(element("h3", {}, "What moved this estimate most"));
  const table = element("table", { id: "factors" });
  const head = element("tr");
  head.append(element("th", {}, "Factor"), element("th", {}, "Effect on risk"));
  table.append(head);
  for (const [name, value] of result.contributions.slice(0, 6)) {
    const direction = value > 0 ? "raises" : "lowers";
    const sign = value > 0 ? "+" : "";
    const row = element("tr");
    row.append(element("td", {}, label(name)),
      element("td", { class: direction }, `${direction} (${sign}${value.toFixed(2)})`));
    table.append(row);
  }
  box.append(table);
  box.append(element("p", { class: "note" },
    "Each effect compares this client with an average training client, on the log-odds " +
    "scale. Positive values push toward default. For example, paying nothing raises risk " +
    "because it is below the average payment."));
}

function modelCard(model) {
  const { test, cv, test_note: testNote, cv_note: cvNote } = model.spec.evaluation;
  const settings = model.spec.model.settings;
  const lines = [
    `Model. Logistic regression (C = ${settings.C}, ${settings.class_weight} class weights) on ` +
      `${model.featureNames.length} inputs built from ${model.rawColumns.length} raw fields.`,
    `Held-out test. ${testNote} Accuracy ${test.accuracy.toFixed(3)}, precision ` +
      `${test.precision.toFixed(3)}, recall ${test.recall.toFixed(3)}, F1 ${test.f1.toFixed(3)}, ` +
      `ROC-AUC ${test.roc_auc.toFixed(3)}.`,
  ];
  if (cv) {
    lines.push(`Cross-validation. ${cvNote} F1 ${cv.f1.mean.toFixed(3)} ± ${cv.f1.sd.toFixed(3)}, ` +
      `recall ${cv.recall.mean.toFixed(3)} ± ${cv.recall.sd.toFixed(3)}.`);
  }
  lines.push(`Data. ${model.spec.inputs.source}.`);
  lines.push("Use. A course project for AAI 6610 at Northeastern University. It is not a lending " +
    "tool and must not be used to make credit decisions about real people.");
  return lines.map((line) => element("p", {}, line));
}

async function main() {
  const response = await fetch("model/model.json");
  if (!response.ok) throw new Error(`Could not load the model (HTTP ${response.status}).`);
  const model = new CreditModel(await response.json());
  buildForm(model);
  $("model-card").append(...modelCard(model));
  $("client-form").addEventListener("submit", (event) => {
    event.preventDefault();
    showResult(model);
  });
  $("estimate").disabled = false;
  $("loading").textContent = "Fill in the form or pick an example, then press Estimate risk.";
  document.body.dataset.ready = "true";
  window.creditModel = model; // used by the browser checks in scripts/browser_check.py
}

main().catch((error) => {
  $("loading").textContent = `The app failed to start: ${error.message}`;
  $("loading").className = "error";
  document.body.dataset.ready = "error";
});
