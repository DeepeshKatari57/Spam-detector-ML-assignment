# SpamShield: Email Spam Classification with 5 ML Models

## a. Problem statement
Unwanted spam emails waste time and carry phishing risks. The goal is to build a binary classifier that decides whether an email is **spam (1)** or **not spam (0)** from word-frequency and character-frequency statistics. Five classification algorithms are trained on the same dataset, compared on six evaluation metrics, and demonstrated through an interactive Streamlit web app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository, *Spambase* dataset (Hewlett-Packard Labs).
- **Instances:** 4,601 emails (1,813 spam, 2,788 not spam). **Features:** 57 numeric features, plus 1 target column (`Class`).
- **Feature groups:** 48 word-frequency features (percentage of words matching a given word, e.g. `free`, `money`, `remove`), 6 character-frequency features (`;`, `(`, `[`, `!`, `$`, `#`) and 3 capital-letter run-length features (average, longest, total).
- **Missing values:** none.
- **Split:** 75% train (3,450 rows) / 25% test (1,151 rows), stratified, `random_state=2026`. The test split is saved as `test_data.csv` and is what the app evaluates on.
- **Preprocessing:** `StandardScaler` for Logistic Regression, KNN and SVM (stored inside each saved pipeline, so no separate preprocessing file is needed). Decision Tree and Gaussian Naive Bayes use raw features.

## c. GitHub repository link
GITHUB_LINK_HERE

Live Streamlit app: STREAMLIT_LINK_HERE

Repository layout: `app.py`, `train.py`, `metrics_utils.py`, `requirements.txt`, `README.md`, `test_data.csv`, `data/spambase_full.csv`, `model/` (5 `.joblib` pipelines).

## d. Models used

### Comparison table (evaluated on the 1,151-row test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9201 | 0.9670 | 0.9114 | 0.8833 | 0.8971 | 0.8320 |
| Decision Tree | 0.9262 | 0.9135 | 0.9055 | 0.9075 | 0.9065 | 0.8455 |
| KNN | 0.8992 | 0.9526 | 0.8876 | 0.8524 | 0.8697 | 0.7880 |
| Naive Bayes | 0.8271 | 0.9470 | 0.7094 | 0.9515 | 0.8128 | 0.6830 |
| Support Vector Machine | 0.9305 | 0.9720 | 0.9309 | 0.8899 | 0.9099 | 0.8540 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline: 92.0% accuracy and the second-best AUC (0.967). Spam is largely separable by a weighted sum of word frequencies, so a linear boundary works well. It is fast and interpretable, but its recall (0.883) is a little lower than the tree's, so it lets some spam through. |
| Decision Tree | Second-best accuracy (92.6%) and the highest recall among the non-Naive-Bayes models (0.9075), because it can isolate key words and capital-letter patterns with simple rules. Its AUC (0.9135) is the lowest of all models, since a single tree gives coarse probability scores. It may overfit without the depth limit (max_depth=12). |
| KNN | Weakest of the "good" models at 89.9% accuracy and MCC 0.788. With 57 features, distance becomes less meaningful (curse of dimensionality) and many features are sparse, so neighbours are noisy. Scaling was essential. Prediction is also slower because it stores all training data. |
| Naive Bayes | Lowest accuracy (82.7%) and lowest precision (0.709), but the highest recall (0.9515). It catches almost all spam but flags many legitimate emails as spam, which is costly in practice. The Gaussian and feature-independence assumptions do not fit skewed, correlated word-frequency data. Its AUC (0.947) shows the ranking of scores is still decent. |
| Support Vector Machine | Best overall: highest accuracy (93.05%), AUC (0.972), precision (0.931), F1 (0.910) and MCC (0.854). The RBF kernel captures non-linear interactions between features while scaling keeps features comparable. The trade-off is slower training and less interpretability. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF)**: best on 5 of 6 metrics, with Decision Tree close behind on accuracy and best recall among the balanced models. |

## How to run locally
```bash
pip install -r requirements.txt
python train.py          # retrains and re-saves models (optional)
streamlit run app.py     # then upload test_data.csv in the sidebar
```

## Streamlit app features
CSV upload (test data only) | model selection dropdown | metrics display (Accuracy, AUC, Precision, Recall, F1, MCC) | confusion matrix and classification report | all-model comparison table with ROC curves. Saved models are loaded, and no retraining happens on the uploaded data.
