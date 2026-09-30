"""Basic tests: loading, preprocessing, training, prediction, model persistence.

Run:  python -m pytest tests/ -v
(All tests use a small synthetic sample or the real train.csv if present.)
"""
import os
import pandas as pd

from src.preprocessing import (
    FEATURE_COLS, TARGET_COL, separate_X_y, clean_data, split_data, load_data,
)
from src.evaluate import evaluate_model
from src.predict import predict_phone, DEFAULT_SPECS


def test_dataset_loading():
    train_df, _ = load_data()
    assert train_df.shape[0] > 0
    assert TARGET_COL in train_df.columns
    for col in FEATURE_COLS:
        assert col in train_df.columns
    # price classes must be 0..3
    assert set(train_df[TARGET_COL].unique()).issubset({0, 1, 2, 3})


def test_preprocessing():
    train_df, _ = load_data()
    cleaned, _ = clean_data(train_df)
    assert cleaned.duplicated().sum() == 0
    assert cleaned.isnull().sum().sum() == 0  # Kaggle set has no missing values
    X, y = separate_X_y(cleaned)
    assert list(X.columns) == FEATURE_COLS
    assert y.name == TARGET_COL
    X_tr, X_va, y_tr, y_va = split_data(X, y)
    assert len(X_tr) > len(X_va)
    # stratification keeps all 4 classes in both splits
    assert set(y_va.unique()) == {0, 1, 2, 3}


def test_models_train_and_predict():
    from src.train import get_models
    train_df, _ = load_data()
    X, y = separate_X_y(train_df)
    # tiny subsample so the test is fast
    X_small, _, y_small, _ = split_data(X, y, test_size=0.9)
    for name, model in get_models().items():
        model.fit(X_small, y_small)
        preds = model.predict(X_small.head(10))
        assert preds.shape == (10,)
        assert set(int(p) for p in preds).issubset({0, 1, 2, 3})


def test_evaluate_metrics_range():
    y_true = [0, 1, 2, 3, 0, 1]
    y_pred = [0, 1, 2, 2, 0, 3]
    m = evaluate_model(y_true, y_pred)
    for k in ["accuracy", "precision", "recall", "f1"]:
        assert 0.0 <= m[k] <= 1.0


def test_saved_models_loadable_and_prediction_shape():
    # Requires `python -m src.train` to have run at least once.
    from src.train import get_models
    import joblib
    sample = dict(DEFAULT_SPECS)
    sample.update({"ram": 4096, "battery_power": 3000, "int_memory": 64,
                   "pc": 16, "fc": 8, "four_g": 1, "three_g": 1})
    for name in get_models():
        path = os.path.join("models", f"{name}.pkl")
        assert os.path.exists(path), f"Missing {path} - run python -m src.train first"
        model = joblib.load(path)
        X = pd.DataFrame([{c: sample[c] for c in FEATURE_COLS}])
        preds = model.predict(X)
        assert preds.shape == (1,) and int(preds[0]) in (0, 1, 2, 3)


def test_predict_phone_function():
    result = predict_phone(dict(DEFAULT_SPECS), model_name="random_forest")
    assert result["class_number"] in (0, 1, 2, 3)
    assert result["label"] in ("Low Cost", "Medium Cost", "High Cost", "Very High Cost")
    if result["probabilities"] is not None:
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-6
