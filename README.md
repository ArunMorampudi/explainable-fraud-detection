# Explainable Fraud Detection Under Fixed FPR

Reproducible experimental pipeline for **explainable fraud detection under extreme class imbalance**, with:
- **fixed false-positive-rate (FPR) thresholding** (operationally meaningful alert budgets),
- **statistical robustness analysis** via bootstrap confidence intervals,
- **SHAP-based explanations** (global + local) for model decisions.

Repository: `explainable-fraud-detection` (branch: `main`).

---

## Title
**Explainable Fraud Detection Under Fixed FPR (PeerJ CS Reproducibility Package)**

---

## Description
This repository accompanies a PeerJ Computer Science manuscript draft (“Explainable Fraud Detection Under Fixed FPR”). It implements an end-to-end workflow to train and evaluate fraud detection models under **extreme class imbalance**, using a **fixed FPR constraint** to reflect real-world alert volume limits, and produces **interpretable explanations** of decisions using SHAP.

---

## Dataset Information

### Dataset used
The paper draft describes using the widely-used **Credit Card Fraud Detection** dataset (European cardholders; September 2013), with:
- 284,807 transactions
- 492 fraud cases (~0.172% fraud rate)
- PCA-transformed anonymized features (`V1`–`V28`) plus `Time`, `Amount`, and binary target label.

### Where to get it
You can obtain the dataset from Kaggle:
- Kaggle dataset: “Credit Card Fraud Detection” (mlg-ulb).
- URL : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 

> Notes:
> - This repo does **not** bundle the dataset (size reasons).
> - When you run main.py it will automatically download the dataset and save it to ./data/raw/creditcard.csv.

---

## Code Information

### Repository structure (top-level)
From the repository root:
- `main.py` — primary entry point/script.
- `config.yaml` — experiment configuration (paths, hyperparameters, thresholds).
- `requirements.txt` — Python dependencies.
- `results/` — output directory for generated artifacts (metrics, plots, etc.).
- `README.md` — this file.

---

## Requirements

### System environment (tested)
- **OS:** Windows 11
- **CPU:** Intel i7-12700K
- **RAM:** 32 GB
- **Python:** 3.10.9

### Python dependencies
Install all required libraries from `requirements.txt`:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Running the pipeline
Execute the code end-to-end:

```bash
python main.py
```
This will automatically download the dataset and run the entire analysis pipeline.

### Dataset Citations
- Worldline; Machine Learning Group, Université Libre de Bruxelles. (2016). *Credit Card Fraud Detection* (dataset). Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. 2015 IEEE Symposium Series on Computational Intelligence (SSCI), 159–166. https://doi.org/10.1109/SSCI.2015.33
