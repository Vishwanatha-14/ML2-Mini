# Comparative Analysis of Machine Learning Algorithms for Mobile Price Range Classification

## 1. Problem Statement
Given a mobile phone's technical specifications (RAM, battery, camera, screen, connectivity, ...),
predict its price range: `0` = Low Cost, `1` = Medium Cost, `2` = High Cost, `3` = Very High Cost.
This is a **multiclass classification** problem with 4 balanced classes.

## 2. Objective
Train, evaluate, compare and interpret five classifiers on the same data and methodology:
Decision Tree, Random Forest, KNN, Logistic Regression, AdaBoost — then provide a reusable
prediction system.

## 3. Dataset
Kaggle **Mobile Price Classification** (https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification).
- `data/train.csv` — 2000 rows, 20 features + `price_range` (500 rows per class, no missing values, no duplicates).
- `data/test.csv` — optional (Kaggle test set has no label; not needed for training).
- The copy of `train.csv` in this repo came from the public mirror of the identical Kaggle file
  (https://github.com/arpita-maji/Mobile-Price-Classification). Overwrite it with the pristine
  Kaggle download and re-run — see `data/README_DATA.md`.

## 4. Features
`battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores,`
`pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi` → target `price_range`.

## 5. Algorithms
| Model | Key idea | Hyperparameters used |
|---|---|---|
| Decision Tree | Single axis-aligned tree | max_depth=10, min_samples_split=5, min_samples_leaf=2 |
| Random Forest | Bagged ensemble of trees, majority vote | n_estimators=100, max_depth=15, min_samples_split=5 |
| KNN | Vote of k nearest neighbours (distance-based → scaled) | Pipeline(StandardScaler + KNeighbors, k=5 baseline) |
| Logistic Regression | Multinomial (softmax) linear model via lbfgs (scaled) | Pipeline(StandardScaler + LogisticRegression C=1.0, max_iter=1000) |
| AdaBoost | Sequential boosting of weak trees | n_estimators=100, learning_rate=1.0 |

All use `random_state=42`. Tuning via small `GridSearchCV` (cv=5); scaler stays inside the
Pipeline so there is no data leakage; tuned model kept only if it beats baseline on validation.

## 6. Methodology
Dataset → cleaning (dup check) → EDA → stratified 80/20 train/validation split →
Pipeline training (scaling inside) → 5-fold CV → GridSearch tuning → same-set evaluation →
comparison + confusion matrices → joblib save → prediction CLI.

## 7. Preprocessing
- No missing values; 0 duplicate rows found.
- All features numeric → no encoding. Target never scaled.
- `StandardScaler` only for KNN / Logistic Regression, inside `Pipeline`.
- `train_test_split(test_size=0.2, stratify=y, random_state=42)` → 1600 train / 400 validation.

## 8. EDA (notebook `notebooks/mobile_price_classification.ipynb`)
- Classes balanced (500 each) → macro averaging is fair.
- `ram` rises steeply with price (correlation ≈ highest); `battery_power` overlaps heavily (weak alone).
- `pc`/`fc` rise slightly; screen/pixel area rise modestly; 4G/3G more common in higher classes.
- `px_height`/`px_width` correlate with each other (multicollinearity, harmless for trees).
- Full plots: `results/plots/eda_*.png` (after running the notebook) + `eda_correlation.png`.

## 9. Evaluation Metrics
Accuracy + macro-averaged Precision / Recall / F1 (equal weight per class, correct for balanced data),
`classification_report`, confusion matrix, 5-fold CV mean ± std. All models scored on the SAME
400-sample validation set.

## 10. Model Comparison (actual results from `python -m src.train`)

| Model | Accuracy | Precision | Recall | F1 Score | CV mean ± std |
|---|---|---|---|---|---|
| Decision Tree | 0.8500 | 0.8511 | 0.8500 | 0.8499 | 0.8231 ± 0.0269 |
| Random Forest | 0.8900 | 0.8901 | 0.8900 | 0.8900 | 0.8619 ± 0.0082 |
| KNN (tuned: k=11, weights=distance, metric=manhattan) | 0.6175 | 0.6211 | 0.6175 | 0.6187 | 0.4994 ± 0.0282 |
| Logistic Regression (tuned: C=10) | **0.9750** | **0.9752** | **0.9750** | **0.9750** | **0.9463 ± 0.0070** |
| AdaBoost (tuned: n=100, lr=0.5) | 0.7075 | 0.7072 | 0.7075 | 0.7067 | 0.4719 ± 0.0941 |

Source: `results/model_comparison.csv`. Confusion matrices: `results/confusion_matrices/*.png`.
Comparison bars + feature importance: `results/plots/`.

## 11. Results / Discussion
- **Logistic Regression wins clearly** (0.975 accuracy, tight CV ±0.007) — the classes are nearly
  linearly separable after scaling, and `ram` provides a strong linear signal.
- Random Forest (0.89) beats single Tree (0.85) as expected from ensembling; both rely mostly on
  `ram` (RF importance 0.51, Tree 0.64), then `battery_power`, `px_width/px_height`.
- AdaBoost baseline was poor (0.445) with unstable CV (±0.094); tuning (lr=0.5, n=100) lifted it to
  0.7075 — default depth-1 stumps underfit this 4-class problem.
- KNN is weakest (0.6175) even after tuning — distance in 20-D with mixed-scale specs generalises poorly.
- Middle classes (1, 2) are hardest for every model (see classification reports); extremes (0, 3) are easiest.

## 12. How to Install
```powershell
pip install -r requirements.txt
```

## 13. How to Run
```powershell
# 1. Put Kaggle files in data/ (train.csv required, test.csv optional)
# 2. Train + evaluate + save everything:
python -m src.train
# 3. Predict (interactive CLI):
python -m src.predict --model random_forest
#    or one-shot:
python -m src.predict --model logistic_regression --ram 4096 --battery_power 3000 --int_memory 64 --pc 16 --fc 8 --four_g 1 --three_g 1
# 4. Run tests:
python -m pytest tests/ -v
# 5. Open the notebook:
jupyter notebook notebooks/mobile_price_classification.ipynb
```

## 14. Example Prediction
Input: RAM=4096 MB, Battery=3000 mAh, Int Memory=64 GB, PC=16 MP, FC=8 MP, 4G=Yes, 3G=Yes, ...
```
Predicted Price Range: Very High Cost (class 3)
Low Cost: 1.0% | Medium Cost: 4.8% | High Cost: 15.4% | Very High Cost: 78.8%
```
(random_forest; probabilities from `predict_proba`; exact numbers vary by model).

## 15. Project Structure
```
data/ train.csv, README_DATA.md | notebooks/mobile_price_classification.ipynb
src/ preprocessing.py, train.py, evaluate.py, predict.py
models/ *.pkl (joblib, pipelines include scaler where needed)
results/ model_comparison.csv, confusion_matrices/, plots/
tests/ test_pipeline.py | requirements.txt, README.md, REPORT.md, .gitignore
```

## 16. Future Improvements
More specs (brand, chipset, 5G), larger/real market-price data, XGBoost/LightGBM,
neural nets, calibration, SHAP explanations, web/mobile deployment, price regression (not just bins).
