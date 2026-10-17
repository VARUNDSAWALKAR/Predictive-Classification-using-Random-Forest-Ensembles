# Predictive Classification using Random Forest Ensembles

Breast cancer diagnosis (malignant vs benign) with a tuned `RandomForestClassifier`, built as an AIML minor project. The workflow covers EDA, preprocessing, hyperparameter justification, and a full evaluation with confusion matrix, precision/recall, F1, ROC-AUC and feature importance.

## Overview

| Item | Details |
|---|---|
| Dataset | [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), 569 samples, 30 numeric features |
| Target | `diagnosis`: M = Malignant (positive class, 1), B = Benign (0) |
| Model | `RandomForestClassifier` (scikit-learn) |
| Split | 80:20 stratified, `random_state=42` |
| Tuned hyperparameters | `n_estimators=200`, `max_depth=10`, `max_features='log2'` (5-fold stratified CV, F1) |

## Results (held-out test set, 114 patients)

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 95.61% | 100% | 88.10% | 93.67% | 0.9942 |

Confusion matrix: TN = 72, FP = 0, FN = 5, TP = 37. 5-fold CV on the full dataset gives F1 0.945 and AUC 0.993.

**Key takeaways**
- The model is never wrong when it says "malignant" (precision 100%), but it misses about 1 in 8 malignancies at the default 0.5 threshold.
- Lowering the threshold to 0.4 cuts false negatives from 5 to 3 at the cost of one false positive.
- Scaling was skipped: a `StandardScaler` pipeline gave identical predictions.
- Radius, perimeter and area are almost perfectly correlated, so feature importance should be read as a group effect.

## Visualizations

| | |
|---|---|
| ![Class distribution](plots/01_class_distribution.png) | ![Correlation heatmap](plots/03_correlation_heatmap.png) |
| ![Confusion matrix](plots/06_confusion_matrix.png) | ![ROC curve](plots/08_roc_curve.png) |
| ![Precision-recall trade-off](plots/07_precision_recall_tradeoff.png) | ![Feature importance](plots/09_feature_importance.png) |

All 10 plots are in [`plots/`](plots/) at 300 dpi.

## Project structure

```
.
├── RF_Minor_Project.ipynb         # commented notebook (main deliverable)
├── rf_random_forest_project.py    # same code as a script
├── Technical_Report.pdf           # full technical report
├── results.json                   # all reported numbers
├── requirements.txt
└── plots/                         # 10 high-resolution figures
```

## Workflow

1. **EDA:** class imbalance (62.7% benign / 37.3% malignant), feature distributions by class, correlation heatmap.
2. **Preprocessing:** drop the empty `Unnamed: 32` column and `id`, label-encode the target, stratified 80:20 split, train-only median imputer as a safety net.
3. **Model development:** baseline forest, then grid search over `n_estimators`, `max_depth` and `max_features`, with an out-of-bag error curve as a second check.
4. **Evaluation:** confusion-matrix heatmap, precision/recall with a threshold trade-off table, F1, ROC-AUC, impurity and permutation feature importance.
5. **Extras:** 5-fold CV on all data, scaling experiment, Gradient Boosting comparison.

## How to run

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
jupyter notebook RF_Minor_Project.ipynb
```

Or run the script directly: `python rf_random_forest_project.py`

If you download Kaggle's `data.csv` and place it next to the notebook, it is used automatically. Otherwise the code loads the identical dataset bundled with scikit-learn, so it runs offline too.

## Limitations and future work

- Small test set (114 rows): each patient moves recall by about 2.4 points.
- Tune the decision threshold on validation data to raise recall.
- Try Gradient Boosting / XGBoost / LightGBM, nested CV, probability calibration and SHAP values.
- Validate on an external dataset before any real-world use.

> This project is an educational exercise and not a diagnostic tool.
