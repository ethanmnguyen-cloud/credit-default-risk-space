// Words and example clients shown on the page. Kept apart from the DOM code so
// the tests can check them without a browser.

export const MONTHS = ["September", "August", "July", "June", "May", "April"]; // 2005, newest first
export const PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"];
export const BILLS = [1, 2, 3, 4, 5, 6].map((i) => `BILL_AMT${i}`);
export const PAYMENTS = [1, 2, 3, 4, 5, 6].map((i) => `PAY_AMT${i}`);

const sameStatus = (code) => Object.fromEntries(PAY_STATUS.map((c) => [c, code]));

// Two made-up clients for trying the app: one paying on time, one two months behind.
export const EXAMPLES = {
  "On-time client": {
    LIMIT_BAL: 200000, SEX: 2, EDUCATION: 2, MARRIAGE: 1, AGE: 35, ...sameStatus(0),
    BILL_AMT1: 50000, BILL_AMT2: 48000, BILL_AMT3: 47000,
    BILL_AMT4: 45000, BILL_AMT5: 44000, BILL_AMT6: 42000,
    PAY_AMT1: 2500, PAY_AMT2: 2400, PAY_AMT3: 2300,
    PAY_AMT4: 2200, PAY_AMT5: 2100, PAY_AMT6: 2000,
  },
  "Client two months behind": {
    LIMIT_BAL: 50000, SEX: 1, EDUCATION: 3, MARRIAGE: 2, AGE: 28,
    PAY_0: 2, PAY_2: 2, PAY_3: 2, PAY_4: 0, PAY_5: 0, PAY_6: 0,
    BILL_AMT1: 49000, BILL_AMT2: 48000, BILL_AMT3: 46000,
    BILL_AMT4: 44000, BILL_AMT5: 42000, BILL_AMT6: 40000,
    PAY_AMT1: 0, PAY_AMT2: 0, PAY_AMT3: 1500,
    PAY_AMT4: 1600, PAY_AMT5: 1600, PAY_AMT6: 1500,
  },
};

const FRIENDLY = {
  LIMIT_BAL: "credit limit", AGE: "age", UTILIZATION: "share of limit used",
  PAY_RATIO: "last payment vs. bill", MONTHS_LATE: "months behind (of 6)",
  SEX: "sex", EDUCATION: "education", MARRIAGE: "marital status",
};
MONTHS.forEach((month, i) => {
  FRIENDLY[`PAY_${i + 1}`] = `${month} status`;
  FRIENDLY[`BILL_AMT${i + 1}`] = `${month} bill`;
  FRIENDLY[`PAY_AMT${i + 1}`] = `${month} payment`;
});
const BY_LENGTH = Object.keys(FRIENDLY).sort((a, b) => b.length - a.length);

/** "nom__PAY_1_2" -> "September status = 2"; "log__BILL_AMT1" -> "September bill". */
export function label(feature) {
  const cut = feature.indexOf("__");
  const block = cut < 0 ? "" : feature.slice(0, cut);
  const rest = cut < 0 ? feature : feature.slice(cut + 2);
  if (block === "nom") {
    for (const column of BY_LENGTH) {
      if (rest.startsWith(`${column}_`)) return `${FRIENDLY[column]} = ${rest.slice(column.length + 1)}`;
    }
  }
  return FRIENDLY[rest] ?? rest;
}
