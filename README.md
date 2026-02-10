# Explainable fraud detection under fixed false positive rate

Reproducible experimental pipeline for **explainable fraud detection under extreme class imbalance**, with:
- **fixed false-positive-rate (FPR) thresholding** (operationally meaningful alert budgets),
- **statistical robustness analysis** via bootstrap confidence intervals,
- **SHAP-based explanations** (global + local) for model decisions.

Repository: `explainable-fraud-detection` (branch: `main`).

---

## Title

**Explainable fraud detection under fixed false positive rate**

---

## Description

This repository accompanies a PeerJ Computer Science manuscript draft ("Explainable Fraud Detection Under Fixed FPR"). It implements a complete end-to-end workflow to train, evaluate, and explain fraud detection models under **extreme class imbalance**, using a **fixed FPR constraint** to reflect real-world alert volume limits (operationally meaningful thresholds), and produces **interpretable explanations** of model decisions using SHAP values.

### Problem Context

**Objective**: Fraud detection systems operate under extreme class imbalance and strict operational alert budgets, where false positives create substantial review cost and customer friction. This study develops and evaluates an explainability-aware fraud detection pipeline explicitly assessed at an operationally constrained decision threshold (fixed false positive rate) and quantifies how explanation patterns behave for true and false alerts in the high-precision regime.

**Key Challenges Addressed**:
- **Extreme class imbalance**: 0.172% fraud rate (~492 frauds in 284,807 transactions)
- **Operational constraints**: Fixed FPR budget (≤1%) reflecting real-world alert volume limits
- **False positive costs**: Review costs and customer friction demand high precision
- **Interpretability**: SHAP-based explanations for regulatory compliance and stakeholder trust
- **Statistical reliability**: 1,000 stratified bootstrap resamples ensuring 95% confidence intervals

---

## Dataset Information

### Dataset Description

**Credit Card Fraud Detection Dataset** (European cardholders; September 2013)

- **Source**: Kaggle / Machine Learning Group, Université Libre de Bruxelles
- **Size**: 284,807 transactions from European cardholders
- **Fraud cases**: 492 fraudulent transactions (~0.172% fraud rate)
- **Features**:
  - 28 PCA-transformed anonymized numerical features (`V1`–`V28`)
  - `Time`: seconds elapsed between the first transaction and subsequent transactions
  - `Amount`: transaction amount in EUR
  - `Class`: binary target label (1 = fraud, 0 = legitimate)
- **Imbalance ratio**: ~579:1 (legitimate to fraudulent transactions)

### Data Access

You can obtain the dataset from Kaggle:
- **Kaggle dataset**: "Credit Card Fraud Detection" (mlg-ulb)
- **URL**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Note**: This repository does **not** bundle the dataset (file size reasons). When you run `main.py`, it will automatically download the dataset from Kaggle and save it to `./data/raw/creditcard.csv`.

---

## Code Information

### Repository Structure

```
explainable-fraud-detection/
├── main.py                          # Main entry point; orchestrates full pipeline
├── config.yaml                      # Configuration file (hyperparameters, thresholds, paths)
├── requirements.txt                 # Python dependencies and versions
├── README.md                        # This file
├── data/
│   └── raw/
│       └── creditcard.csv           # Credit Card Fraud Detection dataset (auto-downloaded)
└── results/
    ├── results.json                 # Aggregated metrics and model performance summaries
    ├── shap_logreg/                 # SHAP-based explanations for Logistic Regression
    ├── shap_rf/                     # SHAP-based explanations for Random Forest
    └── shap_xgb/                    # SHAP-based explanations for XGBoost
```

**Note**: SHAP visualizations are generated for Logistic Regression, Random Forest, and XGBoost. Histogram Gradient Boosting (HGB) is excluded from SHAP analysis due to known compatibility issues with the SHAP library.

### Key Modules and Components

**main.py** includes:
- `ensure_creditcard_csv()`: Automated dataset download and verification
- `build_models()`: Instantiation of four baseline classifiers with class-imbalance handling
- `time_based_split()` / `stratified_split()`: Train/validation/test data splitting strategies
- `choose_threshold_validation()`: Threshold selection under fixed FPR budget
- `bootstrap_test_metrics()`: Stratified bootstrap resampling for confidence intervals
- `run_repeated_cv()`: Repeated stratified K-fold cross-validation for robustness
- `compute_sample_weight_for_imbalance()`: Sample weighting strategy for extreme imbalance
- SHAP explainability pipeline: Global and local explanation generation and visualization

**config.yaml** specifies:
- Data split ratios and modes (time-based vs. stratified)
- Thresholding mode and FPR budget
- Evaluation metrics (precision@k, bootstrap resamples, CV folds)
- Explainability parameters (max explanations, background samples, SHAP settings)

---

## Usage Instructions

### Prerequisites
- **Python 3.10.9** (or compatible version; tested on 3.10.9)
- **pip** (Python package manager)

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ArunMorampudi/explainable-fraud-detection.git
   cd explainable-fraud-detection
   ```

2. **Create a Python virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

Execute the complete experimental pipeline end-to-end:

```bash
python main.py
```

**What this does**:
1. Automatically downloads the Credit Card Fraud Detection dataset from Kaggle (if not already present)
2. Loads and validates the dataset
3. Performs time-based or stratified train/validation/test splitting
4. Builds and trains four baseline classifiers:
   - Logistic Regression
   - Random Forest
   - Histogram Gradient Boosting
   - XGBoost
5. Applies sample weighting to handle extreme class imbalance
6. Selects decision thresholds using fixed FPR budget constraints
7. Evaluates models on the test set with standard metrics (PR-AUC, ROC-AUC, precision, recall)
8. Computes precision@k metrics for top-k high-risk predictions
9. Performs stratified bootstrap resampling (1000 resamples) to estimate confidence intervals
10. Runs repeated stratified K-fold cross-validation (5 folds × 3 repeats) for robustness
11. Generates SHAP-based explanations (global and local) for model decisions
12. Saves all results to `results/` directory (JSON metrics + visualizations per model)

### Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
experiment:
  random_seed: 42                    # Random seed for reproducibility

data:
  csv_path: data/raw/creditcard.csv  # Path to dataset
  split_mode: time                   # 'time' (temporal) or 'stratified' (random)
  train_fraction: 0.70               # Training set fraction
  val_fraction: 0.15                 # Validation set fraction (test = 1 - train - val)

thresholding:
  mode: fpr_budget                   # 'fpr_budget' or 'fbeta'
  target_fpr: 0.01                   # Target false-positive rate (1%)
  beta: 2.0                          # Beta for F-beta metric (if mode='fbeta')

evaluation:
  precision_at_k: [100, 500, 1000]   # Precision@k thresholds
  bootstrap_resamples: 1000          # Number of stratified bootstrap resamples
  cv_splits: 5                       # K-fold cross-validation (K=5)
  cv_repeats: 3                      # Number of CV repeats

explainability:
  enabled: true                      # Enable SHAP explanations
  max_explanations: 500              # Max number of instances to explain
  background_samples: 2000           # Background samples for SHAP
  top_k_attribution: 3               # Top-k features for local explanations

system:
  python_version: 3.10.9
  os: windows
```

### Output

After running `python main.py`, the `results/` directory will contain:

- **results.json**: Comprehensive metrics including:
  - PR-AUC and ROC-AUC scores
  - Decision thresholds and performance metrics
  - Confusion matrices (TP, FP, FN, TN)
  - Precision@k for various k values
  - Bootstrap confidence intervals (2.5%, 97.5% quantiles) for key metrics
  - Cross-validation robustness statistics

- **shap_logreg/, shap_rf/, shap_xgb/**: Directories containing SHAP visualizations and explanations for each model:
  - Summary plots (feature importance)
  - Instance-level local explanations
  - Force plots for decision explanation
  - *Note: No SHAP visualizations are generated for Histogram Gradient Boosting due to known SHAP library compatibility issues. HGB model evaluation metrics remain available in results.json.*

---

## Requirements

### System Environment (Tested)

- **OS**: Windows 11 (also compatible with macOS, Linux)
- **Python**: 3.10.9 (tested; compatible with 3.10.x)
- **CPU**: Intel i7-12700K or equivalent
- **RAM**: 32 GB (most analysis runs with 8 GB; bootstrap + SHAP more memory-intensive)
- **Storage**: ~1 GB (dataset + results)

### Python Dependencies

All required packages are listed in `requirements.txt`. Key dependencies include:

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.3.3 | Data manipulation and preprocessing |
| numpy | 2.2.6 | Numerical computations |
| scikit-learn | 1.7.2 | Machine learning models and metrics |
| xgboost | 3.1.3 | XGBoost classifier |
| shap | 0.49.1 | Model explainability (SHAP values) |
| matplotlib | 3.10.8 | Visualization |
| PyYAML | 6.0.3 | Configuration file parsing |
| kagglehub | 0.4.2 | Kaggle dataset download |
| tqdm | 4.67.3 | Progress bars |

### Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib PyYAML kagglehub tqdm
```

---

## Citations

### Dataset References

- **Worldline & ML Group, ULB** (2016). *Credit Card Fraud Detection* [Dataset]. Kaggle.
  - URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  - Retrieved: February 2026

- **Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G.** (2015).
  - *Calibrating Probability with Undersampling for Unbalanced Classification*.
  - In: 2015 IEEE Symposium Series on Computational Intelligence (SSCI), pp. 159–166.
  - https://doi.org/10.1109/SSCI.2015.33

---

## License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

### Citation

If you use this codebase in your research or work, please cite:

```bibtex
@misc{explainablefrauddetection2026,
  author = {Arun Morampudi},
  title = {Explainable Fraud Detection Under Fixed FPR},
  year = {2026},
  url = {https://github.com/ArunMorampudi/explainable-fraud-detection}
}
```

---

**Last Updated**: February 2026  
**Status**: Active repository for PeerJ CS submission
