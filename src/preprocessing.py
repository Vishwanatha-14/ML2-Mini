"""Shared preprocessing helpers for the mobile price classification project.

Beginner-friendly: every function has a clear job and no hidden side effects.
All randomness uses RANDOM_STATE = 42 so results are reproducible.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# All 20 input features from the Kaggle Mobile Price Classification dataset.
FEATURE_COLS = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi",
]

TARGET_COL = "price_range"

# Human-readable labels for the 4 classes.
PRICE_LABELS = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}

# Models that are distance-based and NEED feature scaling.
SCALED_MODELS = {"knn", "logistic_regression"}


def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    """Load train.csv (must contain price_range) and test.csv (may not).

    Returns (train_df, test_df_or_None). Raises FileNotFoundError with a
    helpful message telling the student where to put the files.
    """
    try:
        train_df = pd.read_csv(train_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            "Download the Kaggle 'Mobile Price Classification' dataset and place:\n"
            f"  - train.csv -> {train_path}\n"
            f"  - test.csv  -> {test_path}\n"
            "Kaggle: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification"
        )
    try:
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        test_df = None  # test.csv is optional (Kaggle test has no price_range)
    return train_df, test_df


def separate_X_y(df, target_col=TARGET_COL):
    """Split a dataframe into X (features) and y (target)."""
    missing = [c for c in FEATURE_COLS + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    X = df[FEATURE_COLS].copy()
    y = df[target_col].copy()
    return X, y


def clean_data(df):
    """Report and remove duplicate rows. Returns cleaned copy + duplicate count."""
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    return df, n_dupes


def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Stratified train/validation split (keeps class ratios equal).

    We use stratify=y so each price_range appears proportionally in both sets.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def needs_scaling(model_name):
    """True if the model is distance-based and needs StandardScaler."""
    return model_name in SCALED_MODELS
