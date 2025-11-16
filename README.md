# cst600-week03-gbm-<your-netid>

## Overview
Predict breast cancer diagnosis using Gradient Boosting (scikit-learn). This repo implements EDA, preprocessing, training, evaluation, and interpretation.

## Dataset
Place the Kaggle CSV downloaded to `data/raw/breast_cancer_data.csv`.
Source: https://www.kaggle.com/datasets/nancyalaswad90/breast-cancer-dataset

## Setup
1. Create & activate virtual environment (Windows example):
   ```bash
   python -m venv .venv
   .\\.venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python src/main.py --data-path data/raw/breast_cancer_data.csv --random-state 42
   ```

## Outputs
- `figures/` — saved EDA and evaluation plots
- `results/metrics.json` — test metrics
- `results/feature_importances.csv` — feature importance table

## Notes & Decisions
- Train/test split: 80/20 stratified on the target with reproducible `random_state`.
- GBM: `GradientBoostingClassifier` from scikit-learn. Default scaling is not applied (tree-based model).

## Reproducibility
Ensure `random_state` is set when running. Use the included pipeline so transformations are only fit on training data.
