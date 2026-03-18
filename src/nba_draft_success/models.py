from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn import ensemble, tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelName = Literal["knn", "dt", "rf", "mlp"]


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    confusion: np.ndarray
    report: str


def build_model(model: ModelName, *, random_state: int = 42) -> object:
    if model == "knn":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=30)),
            ]
        )
    if model == "dt":
        return tree.DecisionTreeClassifier(max_depth=3, random_state=random_state)
    if model == "rf":
        return ensemble.RandomForestClassifier(
            max_depth=5, n_estimators=150, random_state=random_state
        )
    if model == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(30, 30, 30),
                        max_iter=400,
                        alpha=1e-4,
                        solver="sgd",
                        verbose=False,
                        shuffle=True,
                        early_stopping=False,
                        tol=1e-4,
                        random_state=random_state,
                        learning_rate_init=0.1,
                        learning_rate="adaptive",
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {model!r}")


def fit_and_evaluate(
    X,
    y,
    *,
    model: ModelName,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[object, EvalResult]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    clf = build_model(model, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    conf = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    rep = classification_report(y_test, y_pred, labels=[0, 1, 2], digits=4)
    return clf, EvalResult(accuracy=acc, confusion=conf, report=rep)

