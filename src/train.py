"""Model definitions, cross-validation, hyperparameter tuning, training pipeline.

Avoids data leakage by putting StandardScaler INSIDE a Pipeline for
KNN and Logistic Regression, so scaling is fit only on training folds.
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV

from .preprocessing import RANDOM_STATE, FEATURE_COLS, separate_X_y, split_data, clean_data

MODELS_DIR = "models"
RESULTS_DIR = "results"


def get_models():
    """Return the 5 baseline models (some wrapped in a scaling Pipeline)."""
    models = {
        # Decision Tree: max_depth limits tree height (controls overfitting),
        # min_samples_split/leaf require enough samples per split/leaf.
        "decision_tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        # Random Forest: ensemble of trees. n_estimators = number of trees
        # (more = stabler, slower). max_depth caps each tree.
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        # KNN: distance-based -> MUST scale. n_neighbors=5 is a safe default.
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        # Logistic Regression: multiclass handled with multinomial loss
        # (softmax) using the lbfgs solver. C = inverse regularization
        # strength (small C = stronger regularization). Needs scaling.
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
            )),
        ]),
        # AdaBoost: boosting = sequentially fit weak learners (short trees),
        # each new learner focuses on previously misclassified samples.
        # n_estimators = boosting rounds, learning_rate shrinks each step.
        "adaboost": AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=RANDOM_STATE
        ),
    }
    return models


def get_param_grids():
    """Small, fast tuning grids (GridSearchCV with cv=5). Keys must match get_models()."""
    return {
        # Pipeline steps use the "clf__" prefix.
        "decision_tree": {
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 15, None],
            "min_samples_split": [2, 5],
        },
        "knn": {
            "clf__n_neighbors": [3, 5, 7, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
        "logistic_regression": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs"],
        },
        "adaboost": {
            "n_estimators": [50, 100],
            "learning_rate": [0.5, 1.0],
        },
    }


def cross_validate(model, X_train, y_train, cv=5):
    """k-fold cross-validation accuracy: mean + std. Same cv for every model."""
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    return {"mean_accuracy": float(scores.mean()), "std": float(scores.std()), "scores": scores}


def tune_model(model, param_grid, X_train, y_train, cv=5):
    """GridSearchCV tuning (no leakage: scaler stays inside the pipeline)."""
    grid = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


def train_and_save(train_path="data/train.csv"):
    """Full pipeline: load -> clean -> split -> baseline + CV + tune -> evaluate -> save.

    Returns the comparison DataFrame. Saves .pkl models, model_comparison.csv,
    confusion matrices and comparison plots under results/.
    """
    from .evaluate import evaluate_all, save_confusion_matrices, save_comparison_plots

    df = pd.read_csv(train_path)
    df, n_dupes = clean_data(df)
    print(f"Loaded {train_path}: shape={df.shape}, duplicates removed={n_dupes}")

    X, y = separate_X_y(df)
    X_train, X_valid, y_train, y_valid = split_data(X, y)
    print(f"Train: {X_train.shape}, Validation: {X_valid.shape}")

    models = get_models()
    param_grids = get_param_grids()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    trained = {}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        # 1) Cross-validation on training split (fair, same cv for all).
        cv = cross_validate(model, X_train, y_train)
        print(f"CV accuracy: {cv['mean_accuracy']:.4f} +/- {cv['std']:.4f}")
        # 2) Baseline fit + validation score.
        model.fit(X_train, y_train)
        # 3) Tune with a small grid, keep whichever is better on validation.
        best, best_params, best_cv = tune_model(model, param_grids[name], X_train, y_train)
        from sklearn.metrics import accuracy_score
        base_acc = accuracy_score(y_valid, model.predict(X_valid))
        tuned_acc = accuracy_score(y_valid, best.predict(X_valid))
        chosen = best if tuned_acc >= base_acc else model
        label = "tuned" if chosen is best else "baseline"
        print(f"Baseline val acc={base_acc:.4f}, tuned val acc={tuned_acc:.4f} -> using {label}")
        print(f"Best params: {best_params}")
        trained[name] = chosen
        # 4) Save model (pipeline includes scaler where needed).
        joblib.dump(chosen, os.path.join(MODELS_DIR, f"{name}.pkl"))
        rows.append({"model": name, "cv_mean": cv["mean_accuracy"], "cv_std": cv["std"],
                     "val_accuracy_baseline": base_acc, "val_accuracy_tuned": tuned_acc,
                     "best_params": str(best_params)})

    # 5) Shared evaluation on the SAME validation set -> fair comparison.
    comparison = evaluate_all(trained, X_valid, y_valid)
    comparison = comparison.merge(pd.DataFrame(rows), on="model", how="left")
    out_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    comparison.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    print(comparison.to_string(index=False))
    save_confusion_matrices(trained, X_valid, y_valid)
    save_comparison_plots(comparison)
    return comparison


if __name__ == "__main__":
    train_and_save()
