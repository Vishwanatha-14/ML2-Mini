"""Prediction system: load a saved model and predict price range for new phones.

Usage as a library:
    from src.predict import predict_phone
    predict_phone({"ram": 4096, "battery_power": 3000, ...}, model_name="random_forest")

Usage as a CLI (interactive):
    python -m src.predict --model random_forest
    python -m src.predict --model random_forest --ram 4096 --battery_power 3000 ...

The saved .pkl already contains the scaler (for KNN / logistic regression),
so new inputs go through the exact same preprocessing - no retraining needed.
"""

import argparse
import os
import joblib
import pandas as pd

from .preprocessing import FEATURE_COLS, PRICE_LABELS

MODELS_DIR = "models"
DEFAULT_MODEL = "random_forest"

# Sensible defaults (medians of the Kaggle dataset) so the user can press
# Enter to skip any prompt in interactive mode.
DEFAULT_SPECS = {
    "battery_power": 1230, "blue": 0, "clock_speed": 1.5, "dual_sim": 0,
    "fc": 4, "four_g": 0, "int_memory": 32, "m_dep": 0.5, "mobile_wt": 140,
    "n_cores": 4, "pc": 10, "px_height": 700, "px_width": 1200, "ram": 2048,
    "sc_h": 12, "sc_w": 6, "talk_time": 10, "three_g": 1,
    "touch_screen": 1, "wifi": 1,
}


def load_model(model_name=DEFAULT_MODEL):
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. Train first: python -m src.train"
        )
    return joblib.load(path)


def predict_phone(specs, model_name=DEFAULT_MODEL):
    """Predict price range from a dict of mobile specs.

    Returns dict with class_number, label, and per-class probabilities
    (or None when the model has no predict_proba).
    """
    model = load_model(model_name)
    row = {col: specs.get(col, DEFAULT_SPECS[col]) for col in FEATURE_COLS}
    X = pd.DataFrame([row], columns=FEATURE_COLS)
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = {int(i): float(p) for i, p in enumerate(model.predict_proba(X)[0])}
    return {"class_number": pred, "label": PRICE_LABELS[pred], "probabilities": proba}


def pretty_print(result):
    print(f"\nPredicted Price Range: {result['label']} (class {result['class_number']})")
    if result["probabilities"]:
        print("Probabilities:")
        for cls in [0, 1, 2, 3]:
            pct = result["probabilities"].get(cls, 0.0) * 100
            print(f"  {PRICE_LABELS[cls]}: {pct:.1f}%")


def interactive_prompt():
    print("Enter mobile specifications (press Enter for default in brackets):")
    specs = {}
    for col in FEATURE_COLS:
        raw = input(f"  {col} [{DEFAULT_SPECS[col]}]: ").strip()
        if raw == "":
            specs[col] = DEFAULT_SPECS[col]
        else:
            try:
                specs[col] = float(raw) if "." in raw else int(raw)
            except ValueError:
                print(f"  Invalid number, using default {DEFAULT_SPECS[col]}")
                specs[col] = DEFAULT_SPECS[col]
    return specs


def main():
    parser = argparse.ArgumentParser(description="Predict mobile price range.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=["decision_tree", "random_forest", "knn",
                                 "logistic_regression", "adaboost"])
    for col in FEATURE_COLS:  # allow any spec as --ram 4096 etc.
        parser.add_argument(f"--{col}", type=float, default=None)
    args = parser.parse_args()

    given = {c: getattr(args, c) for c in FEATURE_COLS if getattr(args, c) is not None}
    specs = dict(DEFAULT_SPECS)
    specs.update(given)
    if not given:
        specs = interactive_prompt()
    pretty_print(predict_phone(specs, model_name=args.model))


if __name__ == "__main__":
    main()
