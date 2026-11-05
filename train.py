"""
Training script - ML Assignment 2 (BITS Pilani M.Tech AIML)

Dataset : Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository
          569 instances, 30 numeric features, binary target
Models  : Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM
Output  : model/*.joblib, test_data.csv, model/metrics_summary.csv

Run:  python train.py
"""
import os

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

SEED = 42
TARGET_COL = "diagnosis"          # 0 = malignant, 1 = benign (sklearn encoding)
MODEL_DIR = "model"

# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
raw = load_breast_cancer(as_frame=True)
df = raw.frame.rename(columns={"target": TARGET_COL})
print(f"Dataset shape: {df.shape}  (features = {df.shape[1] - 1})")
print("Class balance:\n", df[TARGET_COL].value_counts().to_string(), "\n")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

# Test data is exported WITH the target column so the Streamlit app can score it
test_df = X_te.copy()
test_df[TARGET_COL] = y_te.values
test_df.to_csv("test_data.csv", index=False)

# ----------------------------------------------------------------------------
# 2. Define models (scaling is part of each pipeline, so the saved file is
#    self-contained and the app never has to retrain or re-fit a scaler)
# ----------------------------------------------------------------------------
def with_scaler(estimator):
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])

candidates = {
    "Logistic Regression": with_scaler(LogisticRegression(max_iter=1000, random_state=SEED)),
    "Decision Tree": with_scaler(DecisionTreeClassifier(max_depth=5, random_state=SEED)),
    "KNN": with_scaler(KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": with_scaler(GaussianNB()),
    "Support Vector Machine": with_scaler(SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)),
}

# ----------------------------------------------------------------------------
# 3. Train, evaluate, save
# ----------------------------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
rows = []
for name, pipe in candidates.items():
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_te, pred),
        "AUC": roc_auc_score(y_te, proba),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "MCC": matthews_corrcoef(y_te, pred),
    })
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join(MODEL_DIR, fname))

summary = pd.DataFrame(rows).round(4)
summary.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)
print(summary.to_string(index=False))
print("\nSaved models to ./model and test set to test_data.csv "
      f"({len(test_df)} rows).")
