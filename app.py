import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve)
from metrics_utils import compute_metrics

st.set_page_config(page_title="SpamShield | Model Lab", page_icon="🛡️", layout="wide")

MODELS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}
TARGET = "Class"


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("🛡️ SpamShield: Email Spam Classification Lab")
st.caption("UCI Spambase | 5 pre-trained classifiers | evaluated on the test CSV you upload (no retraining)")

with st.sidebar:
    st.header("Controls")
    chosen = st.selectbox("Select a classifier", list(MODELS))
    upload = st.file_uploader("Upload test CSV (features + 'Class')", type=["csv"])
    st.markdown("Class: **1 = spam**, **0 = not spam**")

if upload is None:
    st.info("Upload `test_data.csv` from the GitHub repo using the sidebar to start.")
    st.stop()

df = pd.read_csv(upload)
if TARGET not in df.columns:
    st.error("The uploaded CSV must contain the target column 'Class'.")
    st.stop()

model = load_model(MODELS[chosen])
X, y = df.drop(columns=[TARGET]), df[TARGET]

expected = list(getattr(model, "feature_names_in_", X.columns))
missing = [c for c in expected if c not in X.columns]
if missing:
    st.error(f"Missing required feature columns: {missing[:5]} ...")
    st.stop()
X = X[expected]

st.success(f"Loaded {len(df)} rows | model: {chosen}")
tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🧮 Confusion matrix & report", "📈 All models"])

with tab1:
    m = compute_metrics(model, X, y)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

with tab2:
    pred = model.predict(X)
    a, b = st.columns(2)
    with a:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ConfusionMatrixDisplay(confusion_matrix(y, pred),
                               display_labels=["Not spam", "Spam"]).plot(ax=ax, cmap="Blues")
        st.pyplot(fig)
    with b:
        st.subheader("Classification report")
        st.code(classification_report(y, pred, target_names=["Not spam", "Spam"]))

with tab3:
    rows = []
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    for name, path in MODELS.items():
        mdl = load_model(path)
        rows.append({"Model": name, **compute_metrics(mdl, X, y)})
        fpr, tpr, _ = roc_curve(y, mdl.predict_proba(X)[:, 1])
        ax2.plot(fpr, tpr, label=name)
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
    ax2.legend(fontsize=7)
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)
    st.pyplot(fig2)
