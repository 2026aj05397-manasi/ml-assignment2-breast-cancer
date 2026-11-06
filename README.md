# Breast Tumour Classification - ML Assignment 2

**Course:** Machine Learning, M.Tech (AIML), BITS Pilani WILP
**Student:** Manasi Chalam | **BITS ID:** 2026AJ05397

## a. Problem statement

Given measurements computed from a digitised image of a fine-needle aspirate (FNA) of a breast mass, predict whether the tumour is **malignant (0)** or **benign (1)**. This is a binary classification task. Five classifiers are implemented on the same dataset, compared using six evaluation metrics, and exposed through an interactive Streamlit app that scores any uploaded test CSV with the saved models (no retraining).

## b. Dataset description

| Item | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic), WDBC |
| Source | UCI Machine Learning Repository (distributed with scikit-learn as `load_breast_cancer`) |
| Instances | 569 |
| Features | 30 numeric (minimum required: 12) |
| Target | `diagnosis`: 0 = malignant (212), 1 = benign (357) |
| Missing values | None |

The 30 features are the mean, standard error and "worst" (largest) value of 10 cell-nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

**Split:** 80/20 stratified train/test split (`random_state=42`): 455 training rows, 114 test rows (42 malignant, 72 benign). The test rows are saved in `test_data.csv`. Every model is a scikit-learn `Pipeline` with `StandardScaler` followed by the classifier, so scaling is stored inside each saved model.

## c. GitHub repository link



Repository layout:

```
project-folder/
|-- app.py                 Streamlit application
|-- train.py               trains, evaluates and saves all five models
|-- requirements.txt
|-- README.md
|-- test_data.csv          114-row hold-out test set (features + diagnosis)
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
    |-- metrics_summary.csv
```

Live app: https://ml-assignment2-breast-cancer-mmjdcymc7fnvhsyicsofbi.streamlit.app/

## d. Models used

Metrics on the 114-row test set (positive class = benign):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

Hyperparameters: Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=5`), KNN (`k=7`), Gaussian Naive Bayes (defaults), SVM (RBF kernel, `C=1.0`).

### Hypothesis

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Joint best model: only 2 errors out of 114 (1 false positive, 1 false negative) and the highest AUC (0.9954). The classes in this dataset are close to linearly separable once features are standardised, so a simple linear model works very well and is easy to interpret. |
| Decision Tree | Weakest model: 9 errors, lowest AUC (0.9163) and MCC (0.8341). A single depth-5 tree makes hard axis-aligned splits and, with only 455 training rows, is sensitive to the particular sample; it missed 6 benign cases. Its predicted probabilities are coarse, which hurts AUC. |
| KNN | Perfect recall (no benign case missed) but 3 malignant tumours were predicted benign, which lowers precision to 0.96. Strong overall (accuracy 0.9737) and it depends on feature scaling, which the pipeline provides. |
| Naive Bayes | Accuracy 0.9298 but AUC of 0.9868, so its ranking of cases is good while its probabilities are poorly calibrated. The independence assumption is violated because many features (radius, perimeter, area) are strongly correlated, which costs it accuracy and MCC. |
| Support Vector Machine | Ties Logistic Regression on every metric (2 errors, MCC 0.9623). The RBF kernel handles any mild non-linearity and the scaled features suit it well, but it gives no gain over the linear model on this dataset. |
| **Overall winner for this dataset** | **Logistic Regression** (tied with SVM on all point metrics, slightly higher AUC), preferred as the simpler and more interpretable model with equally strong performance. |

Note: results come from a single train/test split of a small test set (114 rows), so differences of one or two predictions between the top models are not statistically meaningful.

## How to run locally

```bash
pip install -r requirements.txt
python train.py          # regenerates model/*.joblib and test_data.csv
streamlit run app.py
```

## Streamlit app features

- CSV upload of test data (must include the 30 feature columns and `diagnosis`), or a one-click bundled sample
- Model selection dropdown (5 models)
- Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report
- Side-by-side comparison table of all five models on the same data

The app loads the saved models and only predicts on the uploaded data; it never retrains.
