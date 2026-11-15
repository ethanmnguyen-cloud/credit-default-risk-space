// Catch configuration drift between the repository, the page and the Space.
//
// hub-sync uploads every file except the ones excluded in the workflow. If the page
// imports a file that is excluded (or missing), the tests pass here but the live
// Space breaks. These checks close that gap.

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { MODEL, ROOT, read } from "./helpers.js";

function frontMatter() {
  const block = read("README.md").split("---")[1];
  return Object.fromEntries(block.trim().split("\n").map((line) => {
    const cut = line.indexOf(":");
    return [line.slice(0, cut).trim(), line.slice(cut + 1).trim().replace(/^"|"$/g, "")];
  }));
}

/** The exclude patterns hub-sync is given in .github/workflows/ci-cd.yml. */
function excludePatterns() {
  const lines = read(".github/workflows/ci-cd.yml").split("\n");
  const start = lines.findIndex((l) => l.trim() === "exclude: |");
  assert.ok(start > 0, "exclude block not found in the workflow");
  const indent = lines[start + 1].search(/\S/);
  const patterns = [];
  for (const line of lines.slice(start + 1)) {
    if (line.trim() === "" || line.search(/\S/) < indent) break;
    patterns.push(line.trim());
  }
  return patterns;
}

const excluded = (path, patterns) => path.startsWith(".git") || patterns.some((p) =>
  p.endsWith("/**") ? path.startsWith(p.slice(0, -2)) : path === p);

/** Every local file the page loads: from index.html, then through JS imports and fetches. */
function pageFiles() {
  const found = new Set();
  const queue = [];
  for (const [, path] of read("index.html").matchAll(/(?:src|href)="([^"]+)"/g)) {
    if (!/^(https?:|#|mailto:)/.test(path)) queue.push(path);
  }
  while (queue.length) {
    const path = queue.shift().replace(/^\.\//, "");
    if (found.has(path)) continue;
    found.add(path);
    if (path.endsWith(".js") && existsSync(join(ROOT, path))) {
      const text = read(path);
      for (const [, ref] of text.matchAll(/from\s+"([^"]+)"/g)) queue.push(ref);
      for (const [, ref] of text.matchAll(/fetch\("([^"]+)"\)/g)) queue.push(ref);
    }
  }
  return [...found];
}

test("the README declares a static Space whose app file exists", () => {
  const meta = frontMatter();
  assert.equal(meta.sdk, "static");
  assert.ok(existsSync(join(ROOT, meta.app_file)), meta.app_file);
});

test("every file the page loads exists and is uploaded to the Space", () => {
  const patterns = excludePatterns();
  const files = pageFiles();
  for (const needed of ["app.js", "credit_model.js", "content.js", "style.css", "model/model.json"]) {
    assert.ok(files.includes(needed), `${needed} is not reachable from index.html`);
  }
  for (const path of files) {
    assert.ok(existsSync(join(ROOT, path)), `${path} is missing`);
    assert.ok(!excluded(path, patterns), `${path} is needed by the page but excluded from upload`);
  }
});

test("the large test file is not uploaded to the Space", () => {
  assert.ok(excluded("model/parity_cases.csv", excludePatterns()));
});

test("README metrics match the model file", () => {
  const { test: metrics } = MODEL.spec.evaluation;
  const text = read("README.md");
  for (const [name, key] of [["accuracy", "accuracy"], ["precision", "precision"],
    ["recall", "recall"], ["F1", "f1"], ["ROC-AUC", "roc_auc"]]) {
    assert.ok(text.includes(`${name} ${metrics[key].toFixed(3)}`), name);
  }
});
