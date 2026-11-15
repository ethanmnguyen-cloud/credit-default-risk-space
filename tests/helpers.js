// Shared test fixtures: the model file and the 5,993 held-out test clients.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { CreditModel } from "../credit_model.js";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
export const read = (path) => readFileSync(join(ROOT, path), "utf8");
export const MODEL = new CreditModel(JSON.parse(read("model/model.json")));

/** parity_cases.csv holds plain numbers only (no quotes or commas inside fields). */
export const CASES = (() => {
  const [header, ...lines] = read("model/parity_cases.csv").trim().split(/\r?\n/);
  const columns = header.split(",");
  return lines.map((line) => {
    const cells = line.split(",");
    return Object.fromEntries(columns.map((c, i) => [c, Number(cells[i])]));
  });
})();

/** Raw inputs of test client i, with optional changes. */
export function record(i = 0, changes = {}) {
  const row = Object.fromEntries(MODEL.rawColumns.map((c) => [c, CASES[i][c]]));
  return { ...row, ...changes };
}
