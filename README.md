---
title: Credit Default Risk
emoji: 💳
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Logistic regression for credit card default (course project)
---

# Credit Default Risk

A small web page that estimates the chance a credit card client misses next month's
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

The fitted model is stored as plain numbers in `model/model.json`. `credit_model.js`
repeats the training steps and computes the score in the visitor's browser, so the
Space is a free static site with no server, and nothing typed into the form leaves
the browser.

## How it stays correct

Every push to `main` on GitHub runs the checks before anything is deployed:

1. **Test.** `tests/model.test.js` scores the raw records of all 5,993 test clients and
   requires the same probabilities the evaluated scikit-learn model gave (within 1e-9).
   `scripts/browser_check.py` then opens the page in a real browser and checks it.
2. **Deploy.** The `huggingface/hub-sync` action uploads the page to this Space.
3. **Smoke test.** `scripts/smoke_test.py` waits until the live Space serves this exact
   commit, then runs the same browser check against it.

## Data

Yeh, I.-C. (2009). *Default of credit card clients* [Data set]. UCI Machine Learning
Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0). The test-client records in
`model/parity_cases.csv` come from this data set.
