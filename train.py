"""Train 5 classifiers on Spambase and save them as .joblib pipelines."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics_utils import compute_metrics

SEED = 2026
data = pd.read_csv("data/spambase_full.csv")
features = data.drop(columns=["Class"])
target = data["Class"]

X_tr, X_te, y_tr, y_te = train_test_split(
    features, target, test_size=0.25, random_state=SEED, stratify=target)

zoo = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=SEED)),
}

os.makedirs("model", exist_ok=True)
rows = []
for name, est in zoo.items():
    est.fit(X_tr, y_tr)
    joblib.dump(est, "model/" + name.replace(" ", "_").lower() + ".joblib")
    rows.append({"Model": name, **compute_metrics(est, X_te, y_te)})

table = pd.DataFrame(rows).round(4)
print(table.to_string(index=False))
table.to_csv("results_table.csv", index=False)

test_out = X_te.copy()
test_out["Class"] = y_te.values
test_out.to_csv("test_data.csv", index=False)
print("\nTrain size:", len(X_tr), "| Test size:", len(X_te))
