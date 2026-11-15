---
title: Credit Default Risk
emoji: 💳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Logistic regression for credit card default (course project)
---

# Credit Default Risk

A small web app that estimates the chance a credit card client misses next month's
payment. It serves the logistic regression evaluated in the Module 8 assignment of
AAI 6610 Machine Learning at Northeastern University.

**This is a course project, not a lending tool.** Do not use it to make decisions about
real people.

## The model

| | |
|---|---|
| Model | Logistic regression, C = 0.1, balanced class weights |
| Inputs | 23 raw fields, turned into 89 model inputs |
| Training data | 23,972 clients, UCI Default of Credit Card Clients (Taiwan, 2005) |
| Held-out test (5,993 clients) | accuracy 0.749, precision 0.452, recall 0.624, F1 0.524, ROC-AUC 0.771 |

The fitted model is stored as plain numbers in `model/model.json` and computed with
NumPy, so the app needs no scikit-learn and loads no pickled code.

## How it stays correct

Every push to `main` on GitHub runs the tests before anything is deployed:

1. **Test.** `tests/test_model.py` scores the raw records of all 5,993 test clients and
   requires the same probabilities the evaluated scikit-learn model gave (within 1e-9).
2. **Deploy.** The `huggingface/hub-sync` action uploads the app to this Space.
3. **Smoke test.** `scripts/smoke_test.py` waits for the Space to restart, calls its
   `/score` API, and checks the live answer.

## Data

Yeh, I.-C. (2009). *Default of credit card clients* [Data set]. UCI Machine Learning
Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0). The test-client records in
`model/parity_cases.csv` come from this data set.
