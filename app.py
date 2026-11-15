"""Gradio app: score one credit card client with the Module 8 logistic regression.

Runs on a free Hugging Face Space (CPU basic). The model is plain JSON, so the
app needs only NumPy, pandas and Gradio. The same function is exposed as an API
endpoint named /score.
"""

from __future__ import annotations

import gradio as gr

from credit_model import CreditModel, InputError

MODEL = CreditModel.load()
SPEC = MODEL.spec
CODES = SPEC["inputs"]["codes"]
EVAL = SPEC["evaluation"]
MONTHS = ["September", "August", "July", "June", "May", "April"]  # 2005, newest first
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Two made-up clients for trying the app: one paying on time, one two months behind.
EXAMPLES = {
    "On-time client": dict(LIMIT_BAL=200000, SEX=2, EDUCATION=2, MARRIAGE=1, AGE=35,
                           **dict.fromkeys(PAY_STATUS, 0),
                           BILL_AMT1=50000, BILL_AMT2=48000, BILL_AMT3=47000,
                           BILL_AMT4=45000, BILL_AMT5=44000, BILL_AMT6=42000,
                           PAY_AMT1=2500, PAY_AMT2=2400, PAY_AMT3=2300,
                           PAY_AMT4=2200, PAY_AMT5=2100, PAY_AMT6=2000),
    "Client two months behind": dict(LIMIT_BAL=50000, SEX=1, EDUCATION=3, MARRIAGE=2, AGE=28,
                                     PAY_0=2, PAY_2=2, PAY_3=2, PAY_4=0, PAY_5=0, PAY_6=0,
                                     BILL_AMT1=49000, BILL_AMT2=48000, BILL_AMT3=46000,
                                     BILL_AMT4=44000, BILL_AMT5=42000, BILL_AMT6=40000,
                                     PAY_AMT1=0, PAY_AMT2=0, PAY_AMT3=1500,
                                     PAY_AMT4=1600, PAY_AMT5=1600, PAY_AMT6=1500),
}

FRIENDLY = {"LIMIT_BAL": "credit limit", "AGE": "age", "UTILIZATION": "share of limit used",
            "PAY_RATIO": "last payment vs. bill", "MONTHS_LATE": "months behind (of 6)",
            "SEX": "sex", "EDUCATION": "education", "MARRIAGE": "marital status"}
FRIENDLY.update({f"PAY_{i}": f"{m} status" for i, m in zip([1, 2, 3, 4, 5, 6], MONTHS)})
FRIENDLY.update({f"BILL_AMT{i}": f"{m} bill" for i, m in zip(range(1, 7), MONTHS)})
FRIENDLY.update({f"PAY_AMT{i}": f"{m} payment" for i, m in zip(range(1, 7), MONTHS)})


def _label(feature: str) -> str:
    """'nom__PAY_1_2' -> 'September status = 2'; 'log__BILL_AMT1' -> 'September bill'."""
    block, _, rest = feature.partition("__")
    if block == "nom":
        for column in sorted(FRIENDLY, key=len, reverse=True):
            if rest.startswith(column + "_"):
                return f"{FRIENDLY[column]} = {rest[len(column) + 1:]}"
    return FRIENDLY.get(rest, rest)


def _choices(column: str) -> list[tuple[str, int]]:
    return [(f"{code}: {text}", int(code)) for code, text in CODES[column].items()]


def score(*values) -> tuple[str, str]:
    record = dict(zip(SPEC["inputs"]["raw_columns"], values))
    try:
        result = MODEL.predict(record)
    except InputError as error:
        return f"### Cannot score this client\n{error}", ""
    verdict = ("**Flagged as likely to default**" if result.flagged
               else "**Not flagged**")
    headline = (f"### Estimated chance of default: {result.probability:.1%}\n"
                f"{verdict} (the model flags clients above {MODEL.threshold:.0%}).")
    rows = ["| Factor | Effect on risk |", "|---|---|"]
    for name, value in result.contributions[:6]:
        direction = "raises" if value > 0 else "lowers"
        rows.append(f"| {_label(name)} | {direction} ({value:+.2f}) |")
    factors = ("#### What moved this estimate most\n" + "\n".join(rows) +
               "\n\nEach effect compares this client with an average training client, on the "
               "log-odds scale. Positive values push toward default. For example, paying "
               "nothing raises risk because it is below the average payment.")
    return headline, factors


def _model_card() -> str:
    test, cv = EVAL["test"], EVAL.get("cv", {})
    lines = [
        "**Model.** Logistic regression (C = 0.1, balanced class weights) on 89 inputs built "
        "from 23 raw fields. Trained on 23,972 clients from the UCI Default of Credit Card "
        "Clients data (Taiwan, 2005).",
        f"**Held-out test ({test['n']:,} clients, scored once).** Accuracy {test['accuracy']:.3f}, "
        f"precision {test['precision']:.3f}, recall {test['recall']:.3f}, F1 {test['f1']:.3f}, "
        f"ROC-AUC {test['roc_auc']:.3f}.",
    ]
    if cv:
        lines.append(f"**Five-fold cross-validation.** F1 {cv['f1']['mean']:.3f} ± "
                     f"{cv['f1']['sd']:.3f}, recall {cv['recall']['mean']:.3f} ± "
                     f"{cv['recall']['sd']:.3f}.")
    lines.append("**Use.** A course project for AAI 6610 at Northeastern University. It is not "
                 "a lending tool and must not be used to make credit decisions about real people.")
    return "\n\n".join(lines)


def build() -> gr.Blocks:
    with gr.Blocks(title="Credit Default Risk") as demo:
        gr.Markdown("# Credit Default Risk\nEstimate the chance that a credit card client "
                    "misses next month's payment. Amounts are in New Taiwan dollars.")
        inputs: dict[str, gr.components.Component] = {}
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Client")
                inputs["LIMIT_BAL"] = gr.Number(label="Credit limit", value=200000, minimum=1)
                inputs["AGE"] = gr.Number(label="Age", value=35, minimum=18, maximum=100,
                                          precision=0)
                for column, label in (("SEX", "Sex"), ("EDUCATION", "Education"),
                                      ("MARRIAGE", "Marital status")):
                    inputs[column] = gr.Dropdown(_choices(column), label=label,
                                                 value=int(next(iter(CODES[column]))))
            with gr.Column():
                gr.Markdown("#### Repayment status by month")
                for column, month in zip(PAY_STATUS, MONTHS):
                    inputs[column] = gr.Dropdown(_choices(column), label=month, value=0)
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Bill amount by month")
                for column, month in zip(BILLS, MONTHS):
                    inputs[column] = gr.Number(label=month, value=0)
            with gr.Column():
                gr.Markdown("#### Payment amount by month")
                for column, month in zip(PAYMENTS, MONTHS):
                    inputs[column] = gr.Number(label=month, value=0, minimum=0)
        ordered = [inputs[c] for c in SPEC["inputs"]["raw_columns"]]
        with gr.Row():
            buttons = {name: gr.Button(name, variant="secondary") for name in EXAMPLES}
            go = gr.Button("Estimate risk", variant="primary")
        headline = gr.Markdown()
        factors = gr.Markdown()
        go.click(score, inputs=ordered, outputs=[headline, factors], api_name="score")
        for name, button in buttons.items():
            values = [EXAMPLES[name][c] for c in SPEC["inputs"]["raw_columns"]]
            button.click(lambda v=values: v, outputs=ordered, api_visibility="private")
        with gr.Accordion("About this model", open=False):
            gr.Markdown(_model_card())
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()
