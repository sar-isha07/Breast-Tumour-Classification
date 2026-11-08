import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

malignant_recall = make_scorer(
    recall_score,
    pos_label=0
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
}

scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "malignant_recall": malignant_recall
}

for name, model in models.items():

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    print("\n", name)
    print("-" * 40)

    for metric in scoring:
        scores = results["test_" + metric]

        print(
            metric,
            ":",
            round(scores.mean(), 4),
            "+/-",
            round(scores.std(), 4)
        )