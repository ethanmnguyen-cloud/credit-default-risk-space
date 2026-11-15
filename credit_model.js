// Credit default risk from a raw UCI client record, in plain JavaScript.
//
// The fitted logistic regression lives in model/model.json as plain numbers
// (written by `model export` in the training project). This module repeats the
// exact steps the model was trained after:
//
//   raw record -> rename PAY_0 to PAY_1 -> merge undocumented codes
//     -> add UTILIZATION, PAY_RATIO, MONTHS_LATE
//     -> impute, signed log, scale, one-hot -> weighted sum -> probability
//
// It runs in the visitor's browser (the Space is static, so there is no server)
// and in Node.js for the tests. tests/model.test.js checks that it gives the same
// probability as the evaluated scikit-learn model for all 5,993 held-out clients.

export class InputError extends Error {
  constructor(message) {
    super(message);
    this.name = "InputError";
  }
}

export class CreditModel {
  constructor(spec) {
    if (spec.format_version !== 1) {
      throw new Error(`Unsupported model format ${spec.format_version}.`);
    }
    this.spec = spec;
    this.rawColumns = spec.inputs.raw_columns;
    this.codes = spec.inputs.codes;
    this.protocol = spec.protocol;
    this.blocks = spec.preprocessing;
    this.coef = spec.model.coefficients;
    this.intercept = spec.model.intercept;
    this.threshold = spec.model.threshold;
    this.featureNames = spec.model.feature_names;
    if (this.coef.length !== this.featureNames.length) {
      throw new Error("Coefficient count does not match feature names.");
    }
  }

  /** Score one client. Throws InputError for anything the model cannot use. */
  predict(record) {
    const row = this.design(this.prepare(this.validate(record)));
    const parts = row.map((x, i) => x * this.coef[i]);
    const z = dot(row, this.coef) + this.intercept;
    const probability = sigmoid(z);
    const contributions = this.featureNames
      .map((name, i) => [name, parts[i]])
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    return { probability, flagged: probability > this.threshold, logOdds: z, contributions };
  }

  validate(record) {
    const missing = this.rawColumns.filter((c) => !(c in record));
    if (missing.length) throw new InputError(`Missing inputs: ${missing.join(", ")}.`);
    const out = {};
    for (const column of this.rawColumns) {
      const raw = record[column];
      const value = typeof raw === "string" && raw.trim() === "" ? NaN : Number(raw);
      if (raw === null || !Number.isFinite(value)) {
        throw new InputError(`${column} must be a number.`);
      }
      out[column] = value;
    }
    if (out.LIMIT_BAL <= 0) {
      throw new InputError("LIMIT_BAL (credit limit) must be greater than zero.");
    }
    if (out.AGE < 18 || out.AGE > 100) throw new InputError("AGE must be between 18 and 100.");
    for (const [column, allowed] of Object.entries(this.codes)) {
      if (!Number.isInteger(out[column]) || !(String(out[column]) in allowed)) {
        throw new InputError(
          `${column} must be one of the codes ${Object.keys(allowed).join(", ")}.`);
      }
    }
    for (const column of this.rawColumns.filter((c) => c.startsWith("PAY_AMT"))) {
      if (out[column] < 0) {
        throw new InputError("Payment amounts (PAY_AMT1 to PAY_AMT6) cannot be negative.");
      }
    }
    return out;
  }

  /** Rename, recode, and add the three row-wise features, as in training. */
  prepare(record) {
    const out = {};
    for (const [column, value] of Object.entries(record)) {
      out[this.protocol.rename[column] ?? column] = value;
    }
    for (const [column, rule] of Object.entries(this.protocol.recode)) {
      const text = String(Math.trunc(out[column]));
      out[column] = rule.codes.includes(text) ? rule.into : text;
    }
    for (const feature of this.protocol.features) {
      out[feature.name] = computeFeature(out, feature);
    }
    for (const column of this.protocol.nominal_columns) {
      out[column] = String(Math.trunc(Number(out[column])));
    }
    return out;
  }

  /** The 89 model inputs, in the order the coefficients expect. */
  design(row) {
    const values = [];
    for (const block of this.blocks) {
      if (block.kind === "numeric") {
        block.columns.forEach((column, i) => {
          let x = row[column];
          if (Number.isNaN(x)) x = block.medians[i];
          if (block.signed_log) x = Math.sign(x) * Math.log1p(Math.abs(x));
          values.push((x - block.means[i]) / block.scales[i]);
        });
      } else if (block.kind === "one_hot") {
        block.columns.forEach((column, i) => {
          for (const category of block.categories[i]) values.push(row[column] === category ? 1 : 0);
        });
      } else {
        throw new Error(`Unknown block kind ${block.kind}.`);
      }
    }
    return values;
  }
}

/** Same rules as modeling/features.py in the training project. */
export function computeFeature(row, spec) {
  if (spec.kind === "ratio") {
    const denominator = row[spec.denominator];
    if (denominator <= 0) {
      return spec.when_not_positive == null ? NaN : Number(spec.when_not_positive);
    }
    const value = row[spec.numerator] / denominator;
    return Number.isFinite(value) ? value : NaN;
  }
  if (spec.kind === "count_at_least") {
    const threshold = spec.threshold ?? 1;
    return spec.columns.filter((c) => Number(row[c]) >= threshold).length;
  }
  throw new Error(`Unknown feature kind ${spec.kind}.`);
}

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) total += a[i] * b[i];
  return total;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}
