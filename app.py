"""
Streamlit app - ML Assignment 2
Breast Cancer Wisconsin (Diagnostic): compare 5 pre-trained classifiers on a test CSV.

The app only LOADS saved models from ./model - it never retrains on uploaded data.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"
TARGET_COL = "diagnosis"
CLASS_NAMES = {0: "Malignant", 1: "Benign"}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Tumour Classifier Lab", page_icon="🔬", layout="wide")


@st.cache_resource
def load_model(file_name: str):
    return joblib.load(MODEL_DIR / file_name)


def score_model(model, X, y):
    """Return (metrics dict, predictions) for a fitted model on X, y."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }
    return metrics, pred


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🔬 Breast Tumour Classifier Lab")
st.caption(
    "Compare five pre-trained machine-learning models on the Breast Cancer "
    "Wisconsin (Diagnostic) dataset. Upload a test CSV, pick a model, and "
    "inspect its performance."
)

# ----------------------------------------------------------------------------
# Sidebar: data + model selection
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    use_sample = st.checkbox(
        "Use bundled sample test set", value=uploaded is None,
        help="Loads test_data.csv from the repository (114 rows).",
    )
    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
    st.markdown("---")
    st.caption(
        f"CSV must contain the 30 feature columns and a `{TARGET_COL}` target "
        "column (0 = malignant, 1 = benign)."
    )

if uploaded is not None:
    data = pd.read_csv(uploaded)
    source_label = f"uploaded file `{uploaded.name}`"
elif use_sample and SAMPLE_CSV.exists():
    data = pd.read_csv(SAMPLE_CSV)
    source_label = "bundled sample `test_data.csv`"
else:
    st.info("👈 Upload a CSV or tick the sample-data box to begin.")
    st.stop()

# ----------------------------------------------------------------------------
# Validate columns against what the model was trained on
# ----------------------------------------------------------------------------
model = load_model(MODEL_FILES[model_name])
expected_features = list(model.feature_names_in_)
missing = [c for c in expected_features + [TARGET_COL] if c not in data.columns]
if missing:
    st.error(f"The CSV is missing required column(s): {', '.join(missing)}")
    st.stop()

data = data.dropna(subset=expected_features + [TARGET_COL])
X_test = data[expected_features]
y_test = data[TARGET_COL].astype(int)

if y_test.nunique() < 2:
    st.error("The target column needs both classes (0 and 1) to compute AUC.")
    st.stop()

st.success(f"Loaded {len(data)} rows from {source_label}.")
with st.expander("Preview data"):
    st.dataframe(data.head(10), use_container_width=True)

# ----------------------------------------------------------------------------
# Metrics for selected model
# ----------------------------------------------------------------------------
metrics, predictions = score_model(model, X_test, y_test)

st.subheader(f"📊 {model_name}: evaluation metrics")
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

# ----------------------------------------------------------------------------
# Confusion matrix + classification report
# ----------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
        yticklabels=[CLASS_NAMES[0], CLASS_NAMES[1]],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_test, predictions, target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ----------------------------------------------------------------------------
# All models side by side
# ----------------------------------------------------------------------------
st.subheader("🏁 All models on this data")
rows = []
for name, file_name in MODEL_FILES.items():
    m, _ = score_model(load_model(file_name), X_test, y_test)
    rows.append({"ML Model Name": name, **m})
comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(
    comparison.style.highlight_max(axis=0, color="#cfe8cf"),
    use_container_width=True,
)
best = comparison["MCC"].idxmax()
st.caption(f"Highest MCC on this data: **{best}** ({comparison.loc[best, 'MCC']:.4f}).")
