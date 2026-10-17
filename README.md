# Breast Cancer Diagnosis using Random Forest

> A machine learning project that predicts whether a breast tumour is **malignant (cancerous)** or **benign (not cancerous)** from measurements of cell nuclei, using a **Random Forest** classifier.

This README explains the project in plain language, so you can understand what was done, why, and what the results mean without opening the code.

---

## 1. The problem in simple words

When a doctor takes a small tissue sample from a breast lump, the cells are photographed and measured: how large they are, how round, how irregular their borders are, and so on. The question is:

**Can a computer look at these measurements and tell whether the tumour is cancerous?**

This is a **classification problem**: the model must put each patient into one of two classes.

| Class | Meaning | Label used in code |
|---|---|---|
| Malignant (M) | Cancerous | 1 (the "positive" class) |
| Benign (B) | Not cancerous | 0 |

## 2. The data

- **Source:** [Breast Cancer Wisconsin (Diagnostic) dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) from Kaggle
- **Size:** 569 patients, 357 benign (62.7%) and 212 malignant (37.3%)
- **Features:** 30 numbers per patient. Ten cell properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given as its **mean**, **standard error** and **worst** (largest) value.

## 3. What is a Random Forest?

A **decision tree** is like a flowchart of yes/no questions ("Is the area larger than X? If yes, is the concavity larger than Y?...") that ends in a prediction. One tree can be unreliable.

A **Random Forest** builds many trees (here 200), each trained on a slightly different random sample of the data and features, and lets them **vote**. The majority vote is more accurate and more stable than any single tree. It also needs no feature scaling and can tell us which measurements mattered most.

## 4. What was done (step by step)

| Step | What happened | Why |
|---|---|---|
| **1. Explore the data (EDA)** | Checked class balance, plotted feature distributions, drew a correlation heatmap | To understand the data before modelling. Found that malignant cells tend to be larger and more irregular, and that many features are highly correlated |
| **2. Clean the data** | Dropped an empty column (`Unnamed: 32`) and the `id` column; converted the label M/B into 1/0 | Empty and ID columns carry no useful information; models need numbers |
| **3. Split the data** | 80% for training (455 patients), 20% for testing (114 patients), keeping the same malignant share in both | The test set is never seen during training, so it gives an honest score |
| **4. Skip feature scaling** | Did not use `StandardScaler` | Trees compare one feature at a time, so scaling changes nothing. This was verified: results were identical with and without it |
| **5. Choose settings** | Tried 40 combinations of `n_estimators`, `max_depth` and `max_features` using 5-fold cross-validation; picked **200 trees, max depth 10** | Too few trees is unstable, too many wastes time; too shallow underfits, too deep memorises. Cross-validation picks a balance without touching the test set |
| **6. Evaluate** | Measured the model on the unseen test set with the metrics below | To see how well it works on new patients |

## 5. Results

On the 114 test patients:

| Metric | Score | What it means here |
|---|---|---|
| **Accuracy** | 95.61% | 109 of 114 patients classified correctly |
| **Precision** | 100% | Every patient the model called "malignant" really was malignant (no false alarms) |
| **Recall** | 88.10% | The model found 37 of 42 real cancers, and missed 5 |
| **F1-score** | 93.67% | A single number balancing precision and recall |
| **ROC-AUC** | 0.9942 | Excellent ability to rank malignant cases above benign ones (1.0 = perfect, 0.5 = random guessing) |

### The confusion matrix

|  | Predicted benign | Predicted malignant |
|---|---|---|
| **Actually benign** | 72 (correct) | 0 (false alarm) |
| **Actually malignant** | **5 (missed cancer)** | 37 (correct) |

![Confusion matrix](plots/06_confusion_matrix.png)

### How to read these results

- **The model is cautious.** It never raised a false alarm, but it let 5 cancers pass as benign.
- **That matters.** In cancer screening, a missed cancer is far worse than an unnecessary follow-up test.
- **The cut-off can be changed.** By default the model says "malignant" when it is more than 50% sure. Lowering this to 40% reduces missed cancers from 5 to 3, at the cost of 1 false alarm.
- **Accuracy alone is misleading.** A useless model that always says "benign" would already be 62.7% accurate, which is why precision, recall, F1 and AUC are reported.

![Precision-recall trade-off](plots/07_precision_recall_tradeoff.png)

![ROC curve](plots/08_roc_curve.png)

### Which measurements matter most?

The most important features are `perimeter_worst`, `area_worst`, `concave points_worst`, `concave points_mean` and `radius_worst`. In other words, the model mainly looks at **how large the biggest nuclei are** and **how indented their outlines are**, which matches medical intuition that cancer cells are larger and more irregular.

![Feature importance](plots/09_feature_importance.png)

*Note:* radius, perimeter and area measure almost the same thing (correlation up to 0.998), so the model shares credit between them. Read their importance as one group, not individually.

## 6. How reliable is this?

- With only 114 test patients, **one patient changes recall by about 2.4 points**, so the exact numbers can shift on a different split.
- To check, 5-fold cross-validation on all 569 patients gave **F1 = 0.945 and AUC = 0.993**, confirming the model is consistently strong.
- The tuned model scored slightly *below* a default Random Forest on this particular test split (a difference of two patients). That is within noise.
- An untuned Gradient Boosting model performed similarly, so it is a good next experiment.

## 7. Limitations

- The data comes from a single source and has only 569 samples, so it may not generalise to other hospitals or equipment.
- A Random Forest is harder to explain than a single decision tree.
- **This is an educational project and must not be used for real medical diagnosis.**

## 8. Possible improvements

- Tune the decision threshold on validation data to catch more cancers.
- Try Gradient Boosting, XGBoost or LightGBM.
- Use nested cross-validation, calibrate probabilities, and add SHAP values to explain individual predictions.
- Test on an external dataset.

## 9. Repository contents

```
.
├── RF_Minor_Project.ipynb         # main notebook, fully commented, run this
├── rf_random_forest_project.py    # same code as a plain Python script
├── Technical_Report.pdf           # detailed report with all figures and discussion
├── results.json                   # every number reported above
├── requirements.txt               # Python libraries needed
└── plots/                         # 10 high-resolution figures (300 dpi)
```

## 10. How to run it

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
jupyter notebook RF_Minor_Project.ipynb
```

Or run the script: `python rf_random_forest_project.py`

If you download Kaggle's `data.csv` and put it next to the notebook, it is used automatically. If not, the code loads the identical copy of the dataset bundled with scikit-learn, so it also works offline. A fixed random seed (42) makes results reproducible.

## 11. Quick glossary

| Term | Meaning |
|---|---|
| **Classification** | Predicting which category something belongs to |
| **Training / test set** | Data the model learns from / data held back to check it honestly |
| **Cross-validation** | Splitting the training data into 5 parts and rotating which part is used for checking, for a steadier estimate |
| **Hyperparameter** | A setting chosen before training, such as the number of trees |
| **False positive / false negative** | A healthy patient flagged as sick / a sick patient missed |
| **Precision** | Of those flagged as sick, the share who really are |
| **Recall** | Of those who really are sick, the share we found |
| **F1-score** | The balance (harmonic mean) of precision and recall |
| **ROC-AUC** | How well the model separates the two classes across all possible cut-offs |
