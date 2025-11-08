# 🏦 Loan Default Prediction with Machine Learning

*A concise, end-to-end walkthrough of cleaning, exploring, modeling, and evaluating a binary loan-default predictor.*

## TL;DR
- **Goal:** Predict whether a loan **defaults** (target: `Defaulted`) using tabular borrower/loan features.
- **Approach** A sophisticated machine learning system for predicting loan defaults using **BorderlineSMOTE** oversampling, **Weight of Evidence** encoding, and **Random Forest** classification. The model achieves **91.2% Average Precision** and **97.4% ROC-AUC** on holdout data.


![Results at a glance](https://github.com/S-D-Willis/loan-default-pred/blob/4b9c6a8f53de8feace824755c5e6a1723ad85d5a/model_summary.png)


### Key Metrics (Holdout Set - 89,717 samples)

| Metric | Score | Business Impact |
|--------|-------|-----------------|
| **Average Precision** | 0.912 | Excellent ranking quality |
| **ROC-AUC** | 0.974 | Outstanding discrimination ability |
| **Precision @ 0.5** | 0.861 | 86% of flagged loans actually default |
| **Recall @ 0.5** | 0.846 | Catches 85% of all defaults |
| **F1 Score** | 0.853 | Strong balanced performance |
| **Accuracy** | 0.949 | 95% overall accuracy |


## Notebooks at a Glance

> The series is designed so each notebook can be read independently yet builds on the previous one. Outputs (cleaned data, fitted transformers, reports) are saved between steps.

### Loan-Default-Pred_0_INTRO.py
- Dataset overview and feature descriptions

### Loan-Default-Pred_1_CLEAN.py
- Type corrections
- Leakage removal
- Data entry issue control

### Loan-Default-Pred_2_EDA.py
- Visualization of train data
    - Bar charts
    - Violin plots
    - Dnsity histograms

### Loan-Default-Pred_3_EXPERIMENTS.ipynb
- Weight of Evidence encoding
- Up/down sampling
- Hyperparameter tuning

### Loan-Default-Pred_4_EVAL.ipynb
- Final model evaluation on holdout
