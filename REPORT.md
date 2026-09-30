# Academic Report — Mobile Price Range Classification

## Abstract
We build a multiclass classifier that maps 20 mobile-phone specifications to one of four price
ranges (0 Low … 3 Very High) using the Kaggle Mobile Price Classification dataset (2000 labelled
rows, balanced). Five algorithms — Decision Tree, Random Forest, KNN, Logistic Regression,
AdaBoost — are trained under one reproducible protocol (fixed seed 42, stratified 80/20 split,
scaling inside Pipelines, 5-fold CV, small-grid GridSearchCV) and compared with accuracy and
macro precision/recall/F1 plus confusion matrices. Logistic Regression achieved the best
validation accuracy (0.9750), followed by Random Forest (0.8900), Decision Tree (0.8500),
AdaBoost (0.7075) and KNN (0.6175). Feature importance shows `ram` dominates tree models.
A joblib-based prediction CLI is delivered.

## 1. Introduction
Smartphone prices span orders of magnitude while spec sheets grow ever longer. Mapping specs to
price bins helps buyers compare value, sellers position products, and analysts track market
segments. Manual rules do not scale across 20 interacting features, so supervised classification
is a natural fit: learn the spec→price mapping from labelled examples and generalise to new phones.

## 2. Problem Statement
Input: 20 numeric specs (battery, RAM, memory, cameras, screen, pixels, weight, connectivity flags…).
Output: `price_range ∈ {0,1,2,3}`. Balanced 4-class classification; success = high macro-averaged
scores and stable cross-validation on unseen phones from the same distribution.

## 3. Objectives (measurable)
1. Load and audit the dataset (shape, dtypes, missing, duplicates). 2. EDA with ≥8 visual analyses.
3. Leak-free preprocessing with scaling only where needed. 4. Train all 5 models reproducibly.
5. Report accuracy/precision/recall/F1 + confusion matrices on one shared validation set.
6. 5-fold CV mean ± std per model. 7. Light GridSearchCV tuning, baseline-vs-tuned comparison.
8. Feature-importance analysis. 9. Working `predict_phone()` + CLI with probabilities.

## 4. Literature Review
Decision trees (interpretable splits), Random Forests (bagging reduces tree variance), KNN
(non-parametric distance voting, needs scaling), multinomial Logistic Regression (linear softmax,
strong on separable tabular data), and AdaBoost (sequential reweighting of weak learners) are
textbook tabular classifiers. Prior student work on this exact Kaggle set typically finds tree
ensembles and logistic regression competitive, with RAM the dominant predictor — consistent with
our findings (logistic regression 0.975, RF 0.89). Deep models add little on 2000×20 tabular data.

## 5. Methodology
Dataset → duplicate/missing audit → EDA (distributions, box plots, crosstabs, correlation heatmap)
→ stratified 80/20 split (1600/400) → per-model Pipeline (StandardScaler only for KNN/LogReg) →
5-fold CV → GridSearchCV tuning (small grids) → shared-set evaluation → confusion matrices,
comparison table/plots → joblib persistence → CLI prediction. Seed 42 everywhere.

## 6. Algorithms
- **Decision Tree**: recursive feature-threshold splits maximising purity. Pros: interpretable, no
  scaling. Cons: high variance/overfit. Here max_depth=10 etc. → 0.8500.
- **Random Forest**: 100 bootstrapped trees vote. Pros: lower variance, importance scores.
  Cons: heavier, less interpretable. → 0.8900.
- **KNN**: k nearest training points vote (euclidean/manhattan). Pros: simple, non-linear.
  Cons: curse of dimensionality, needs scaling, slow at predict. Tuned k=11/distance/manhattan → 0.6175.
- **Logistic Regression**: linear scores + softmax, lbfgs optimisation, C regularisation.
  Pros: calibrated probabilities, fast, strong when classes are linearly separable. Cons: linear
  boundary. Tuned C=10 → 0.9750 (best).
- **AdaBoost**: sequential stumps reweighting mistakes. Pros: can boost weak learners.
  Cons: sensitive to noise, weak with un-tuned stumps on 4 classes. Baseline 0.445 → tuned 0.7075.

## 7. Results (executed, `results/model_comparison.csv`)
| Model | Acc | Prec | Rec | F1 | CV |
|---|---|---|---|---|---|
| Decision Tree | 0.8500 | 0.8511 | 0.8500 | 0.8499 | 0.8231±0.0269 |
| Random Forest | 0.8900 | 0.8901 | 0.8900 | 0.8900 | 0.8619±0.0082 |
| KNN | 0.6175 | 0.6211 | 0.6175 | 0.6187 | 0.4994±0.0282 |
| Logistic Regression | 0.9750 | 0.9752 | 0.9750 | 0.9750 | 0.9463±0.0070 |
| AdaBoost | 0.7075 | 0.7072 | 0.7075 | 0.7067 | 0.4719±0.0941 |
Top importances — RF: ram 0.5145, battery_power 0.0745, px_width 0.0582, px_height 0.0517,
mobile_wt 0.0370; Tree: ram 0.6393, battery_power 0.1218, px_height 0.0926, px_width 0.0875.
Middle classes (1,2) score lowest in every classification report; extremes are easiest.

## 8. Discussion
Logistic Regression is the clear winner on every metric with the tightest CV band — evidence the
bin boundaries are near-linear in scaled spec space (RAM acts almost like a linear separator).
Forests beat single trees (variance reduction) but cannot match the linear model here. AdaBoost's
huge baseline→tuned jump and wide CV band show instability of stumps on 4 classes. KNN's failure
illustrates distance concentration in 20-D. No model shows overfitting red flags beyond AdaBoost's
CV variance; tuning gains were modest except for KNN/AdaBoost which started from poor defaults.

## 9. Conclusion
All five models train and validate successfully under a fair protocol. For this dataset the
recommended model is tuned Logistic Regression (C=10), with Random Forest as an interpretable
runner-up. The saved pipelines + CLI (`python -m src.predict`) generalise the result to new phones.

## 10. Future Scope
Add brand/chipset/5G/camera-count features; larger and real market-price datasets; XGBoost/LightGBM
and calibrated neural nets; SHAP/permutation explanations; price regression instead of bins;
deployed web/mobile app with input validation and monitoring.
