# Health Insurance Cross-Sell Prediction Using Machine Learning

---

## 1. Title Page

- **Project Title:** Health Insurance Cross-Sell Prediction Using Machine Learning
- **Domain:** Supervised Machine Learning / Binary Classification / Insurance Analytics
- **Dataset:** Historical Health Insurance Customer Records (`381,109` rows × `12` columns)
- **Technology Stack:** Python `3.13.9`, Pandas `2.3.3`, NumPy `2.4.4`, Scikit-learn `1.7.2`, Imbalanced-learn `0.14.0`, Matplotlib `3.10.6`, Seaborn `0.13.2`, Joblib `1.5.2`, Streamlit `1.51.0`
- **Deliverables:** End-to-End Jupyter Notebooks (`01`–`06`), Saved Pipeline (`Models/best_model.pkl`), Prediction Module (`prediction.py`), Streamlit Web App (`app.py`), and Evaluation Artifacts (`Outputs/`)

---

## 2. Abstract

Insurance providers frequently seek cross-selling opportunities to increase customer lifetime value and reduce acquisition costs. This project develops a supervised binary classification pipeline to predict whether an existing health insurance policyholder is interested in purchasing vehicle insurance (`Response = 1`) or not (`Response = 0`). Using a real dataset of **381,109** customer records with **12** attributes and zero missing values, Exploratory Data Analysis (EDA) reveals substantial class imbalance (**87.74%** negative vs. **12.26%** positive) and strong predictive signals in prior vehicle insurance status (`Previously_Insured`), past vehicle damage (`Vehicle_Damage`), customer age (`Age`), vehicle age (`Vehicle_Age`), and outreach channel (`Policy_Sales_Channel`).

To prevent data leakage, an **80/20 stratified train-test split** (`304,887` train / `76,222` test) is applied before fitting a `ColumnTransformer` preprocessing pipeline. Five class-imbalance strategies are benchmarked via 3-fold stratified cross-validation on training data only, demonstrating that `class_weight='balanced'` improves Logistic Regression Recall from **0.0007** to **0.9366** while matching SMOTE and oversampling without inflating memory. Five classification pipelines (`DummyClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `GradientBoostingClassifier`) are trained and evaluated on the unseen test set. **Gradient Boosting Classifier** achieves the highest discrimination (**ROC-AUC = 0.8568**, **PR-AUC = 0.3676**). After hyperparameter tuning with `RandomizedSearchCV` and refitting on the full training set, the final model achieves **Accuracy = 0.7022**, **Precision = 0.2826**, **Recall = 0.9290**, **F1-score = 0.4334**, **ROC-AUC = 0.8567**, and **PR-AUC = 0.3679** at a `0.50` probability threshold. The end-to-end pipeline is serialized to `Models/best_model.pkl` and integrated into both `prediction.py` and an interactive Streamlit web application (`app.py`).

---

## 3. Introduction

Cross-selling—offering an additional product to an existing customer—is far more cost-effective than acquiring new customers from scratch. Because an insurer already holds demographic, policy, and contact channel data for its health insurance customers, predictive modeling can identify which policyholders are most receptive to vehicle insurance offers.

Rather than contacting every customer uniformly (which wastes telemarketing and campaign budgets on unlikely buyers, where baseline interest is only **12.26%**), Machine Learning enables targeted outreach ranked by estimated interest likelihood.

---

## 4. Problem Statement

Given historical customer demographics, vehicle attributes, prior vehicle insurance status, premium amount, and outreach channel details (`10` predictive features excluding customer `id`), predict the binary target `Response`:
- **`Response = 1`:** Customer is interested in purchasing vehicle insurance (`46,710` customers, `12.26%`).
- **`Response = 0`:** Customer is not interested (`334,399` customers, `87.74%`).

The core technical challenges are:
1. Handling strong **7.16 : 1 class imbalance** so models do not ignore the minority interested class.
2. Encoding high-cardinality numeric category codes (`Region_Code` with `53` codes and `Policy_Sales_Channel` with `155` codes) cleanly without target leakage.
3. Retaining genuine right-skewed high-premium customers (`Annual_Premium` up to `540,165.0`) while stabilizing linear and ensemble training.

---

## 5. Project Objectives

1. Inspect and understand the real dataset (`381,109` rows × `12` columns) without assumptions.
2. Perform comprehensive Exploratory Data Analysis (EDA) and save publication-ready charts.
3. Clean data types, validate domain boundaries, and handle outliers with domain justification.
4. Design a leakage-free `ColumnTransformer` preprocessing pipeline fitted strictly on training data.
5. Benchmark class-imbalance techniques using training-only stratified cross-validation.
6. Implement, evaluate, and compare `DummyClassifier`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `GradientBoostingClassifier`.
7. Tune the best model via Stratified CV (`RandomizedSearchCV`) without touching the test set.
8. Explain predictions via tree-based and permutation feature importance and configurable threshold analysis.
9. Deploy the serialized pipeline (`Models/best_model.pkl`) in `prediction.py` and a Streamlit web app (`app.py`).

---

## 6. Existing System

In traditional manual or rule-based marketing systems:
- Marketing teams either contact **all customers** indiscriminately (achieving roughly **12.26%** baseline hit rate and high call/email waste) or apply static manual heuristics.
- Static rules fail to capture multi-feature interactions (such as `Previously_Insured × Vehicle_Damage × Age × Policy_Sales_Channel`) and cannot dynamically rank leads or adjust outreach volume when campaign budgets change.

---

## 7. Proposed System

The proposed Machine Learning system replaces blind outreach with an automated, probability-scoring pipeline:
- **Leakage-Free Preprocessing:** Imputation, `log1p` + `StandardScaler`, `OrdinalEncoder`, and frequency-grouped `OneHotEncoder` bundled inside a single Scikit-learn `Pipeline`.
- **Imbalance-Aware Ensemble:** Tuned `GradientBoostingClassifier` trained with balanced class/sample weights so the model prioritizes catching interested leads (**92.90% Recall** at `0.50` threshold).
- **Configurable Decision Threshold:** Business teams can raise the threshold (e.g., to `0.70`) for higher Precision (`33.26%`) or lower it (e.g., `0.40`) for maximum Recall (`96.55%`).
- **Interactive Interface:** A Streamlit web UI (`app.py`) and Python API (`prediction.py`) powered by `Models/best_model.pkl`.

---

## 8. Dataset Description

- **Source File:** `Dataset/train.csv` (`20.44 MB` on disk; `85.22 MB` deep RAM in Pandas)
- **Dimensions:** `381,109` rows × `12` columns (`0` missing values, `0` exact duplicate rows)
- **Separate Test File:** None provided; an 80/20 stratified split holds out `76,222` unseen customers for final evaluation.

| # | Column | Raw Dtype | Domain Role | Summary / Unique Values |
|---|---|---|---|---|
| 1 | `id` | `int64` | Identifier (Dropped) | `1` to `381,109` (`381,109` unique) |
| 2 | `Gender` | `object` | Categorical | `Male` (54.08%), `Female` (45.92%) |
| 3 | `Age` | `int64` | Numerical | Min `20`, Median `36`, Mean `38.82`, Max `85` |
| 4 | `Driving_License` | `int64` | Binary (`0`/`1`) | `1` (99.79%), `0` (0.21%) |
| 5 | `Region_Code` | `float64` | Categorical Code | `53` unique integer codes (`0` to `52`) |
| 6 | `Previously_Insured` | `int64` | Binary (`0`/`1`) | `0` (54.18%), `1` (45.82%) |
| 7 | `Vehicle_Age` | `object` | Ordinal Categorical | `1-2 Year` (52.56%), `< 1 Year` (43.24%), `> 2 Years` (4.20%) |
| 8 | `Vehicle_Damage` | `object` | Binary Categorical | `Yes` (50.49%), `No` (49.51%) |
| 9 | `Annual_Premium` | `float64` | Numerical | Min `2,630`, Median `31,669`, Mean `30,564.39`, Max `540,165` |
| 10 | `Policy_Sales_Channel` | `float64` | Categorical Code | `155` unique integer codes (`1` to `163`) |
| 11 | `Vintage` | `int64` | Numerical | Min `10`, Median `154`, Mean `154.35`, Max `299` |
| 12 | `Response` | `int64` | **Target Variable** | `0` (`334,399`, `87.74%`), `1` (`46,710`, `12.26%`) |

---

## 9. System Requirements

### Hardware Requirements (Minimum Recommended)
- **Processor:** Multi-core x64 CPU (Intel Core i5 / AMD Ryzen 5 or higher)
- **RAM:** 8 GB minimum (16 GB recommended for smooth cross-validation on 300k+ rows)
- **Storage:** 500 MB free disk space for dataset, notebooks, graphs, and saved models

### Software Requirements
- **Operating System:** Windows 10 / 11 (verified on Windows), Linux, or macOS
- **Python Version:** Python `3.10+` (verified on Python `3.13.9` via Anaconda)
- **IDE / Environment:** VS Code or Jupyter Notebook / JupyterLab

---

## 10. Technologies Used

| Library | Verified Version | Purpose |
|---|---|---|
| `pandas` | `2.3.3` | Data loading, tabular analysis, crosstabs |
| `numpy` | `2.4.4` | Numerical operations and `log1p` transformation |
| `matplotlib` | `3.10.6` | Static charts, ROC/PR curves, threshold plots |
| `seaborn` | `0.13.2` | Histograms, boxplots, heatmaps |
| `scikit-learn` | `1.7.2` | Pipelines, encoders, scalers, CV, classifiers, metrics |
| `imbalanced-learn` | `0.14.0` | `ImbPipeline`, `RandomOverSampler`, `SMOTE`, `RandomUnderSampler` |
| `joblib` | `1.5.2` | Serialization of `Models/best_model.pkl` |
| `streamlit` | `1.51.0` | Interactive web prediction UI (`app.py`) |

---

## 11. Project Architecture

```text
Raw CSV (Dataset/train.csv: 381,109 rows)
   │
   ▼
Data Cleaning (Type casting Region_Code & Policy_Sales_Channel; Domain & Outlier checks)
   │
   ▼
80/20 Stratified Train-Test Split (random_state=42, stratify=y)
   ├──► Training Set (304,887 rows)
   │       ├──► Phase 6: 3-Fold Stratified CV Imbalance Comparison
   │       ├──► Phase 8: Fit ColumnTransformer + 5 Candidate Models
   │       └──► Phase 10: RandomizedSearchCV (3-Fold CV) -> Refit Full Train -> best_model.pkl
   │
   └──► Unseen Test Set (76,222 rows)
           ├──► Phase 9: Final Metrics, Confusion Matrices, ROC & PR Curves
           └──► Phase 11: Permutation Importance & Threshold Trade-off
   │
   ▼
Deployment: Models/best_model.pkl ──► prediction.py ──► app.py (Streamlit UI)
```

---

## 12. Data Preprocessing

Implemented in [03_Data_Cleaning_Preprocessing.ipynb](Notebooks/03_Data_Cleaning_Preprocessing.ipynb):
1. **Cleaning Decisions:**
   - **Missing Values:** `0` missing values; median/most-frequent `SimpleImputer` steps remain inside the pipeline for robustness on new inputs.
   - **Duplicates:** `0` exact duplicates. `269` feature-identical rows (excluding `id`, mostly at minimum premium `2,630.0`) represent distinct customers and were retained.
   - **Type Casting:** Verified 100% of `Region_Code` and `Policy_Sales_Channel` values are whole numbers and cast them to clean string category codes (`'28'`, `'152'`, etc.).
   - **Outlier Retention:** `Age` and `Vintage` had `0` IQR outliers. `Annual_Premium` had `10,320` (`2.71%`) upper-tail values above `61,892.5`, which showed a **higher** interest rate (`15.07%` vs `12.18%`). All rows were retained, and `np.log1p` + `StandardScaler` was applied to `Annual_Premium`.
2. **Feature Engineering & Multicollinearity:**
   - Dropped `id`.
   - `Gender` & `Vehicle_Damage`: `OneHotEncoder(drop='if_binary')`.
   - `Vehicle_Age`: `OrdinalEncoder` (`< 1 Year` → `0`, `1-2 Year` → `1`, `> 2 Years` → `2`).
   - `Region_Code` & `Policy_Sales_Channel`: `OneHotEncoder(handle_unknown='ignore', min_frequency=0.01)` producing compact binary columns for frequent codes (`>= 1%`) and an `infrequent` group, yielding **44 engineered features** with zero target leakage.
   - **VIF Check:** All numerical/binary features had VIF between `1.0000` and `1.0760`, confirming no multicollinearity.

---

## 13. Exploratory Data Analysis

Thirteen EDA visualizations were generated in [02_Exploratory_Data_Analysis.ipynb](Notebooks/02_Exploratory_Data_Analysis.ipynb) and saved under `Outputs/graphs/` (`01_target_distribution.png` through `13_interested_vs_not_interested_comparison.png`):
- **Target Imbalance (`01_target_distribution.png`):** `334,399` (`87.74%`) Not Interested vs. `46,710` (`12.26%`) Interested.
- **Previously Insured (`07_previously_insured_vs_response.png`):** Customers without prior vehicle insurance (`0`) have a **22.55%** interest rate (`46,552` interested), whereas already-insured customers (`1`) have only **0.09%** (`158` interested).
- **Vehicle Damage (`06_vehicle_damage_vs_response.png`):** Prior vehicle damage (`Yes`) corresponds to **23.77%** interest vs. **0.52%** for `No`.
- **Combined Segment (`13_interested_vs_not_interested_comparison.png`):** `Previously_Insured = 0` & `Vehicle_Damage = Yes` yields **25.01%** interest, vs. `0.04%` for `Previously_Insured = 1` & `Vehicle_Damage = No`.
- **Age & Vehicle Age (`04_age_vs_response.png`, `05_vehicle_age_distribution.png`):** Interest peaks in ages `31–40` (**21.02%**) and `41–50` (**21.10%**), and rises with vehicle age (`< 1 Year`: **4.37%**, `1-2 Year`: **17.38%**, `> 2 Years`: **29.37%**).
- **Channel & Region (`09_region_code_distribution.png`, `10_policy_sales_channel_distribution.png`):** Channel `152` (`35.37%` volume) has **2.86%** interest, while channels `26` (`19.94%`), `124` (`18.91%`), `157` (`26.84%`), and `163` (`30.42%`) convert much higher.
- **Vintage & Correlation (`11_vintage_distribution.png`, `12_correlation_heatmap.png`):** `Vintage` averages are nearly identical (`154.11` vs `154.38` days, correlation `-0.0011`).

---

## 14. Machine Learning Algorithms

1. **Dummy Classifier (`strategy='most_frequent'`):** Baseline predicting `0` for all customers.
2. **Logistic Regression (`class_weight='balanced'`):** Linear log-odds classifier; fast and interpretable, though constrained to linear boundaries.
3. **Decision Tree Classifier (`max_depth=10, min_samples_leaf=50, class_weight='balanced'`):** Non-linear rule-based tree; captures feature interactions (`Previously_Insured` + `Vehicle_Damage`), though single trees have higher variance.
4. **Random Forest Classifier (`n_estimators=100, max_depth=12, min_samples_leaf=20, class_weight='balanced'`):** Bagged ensemble of 100 trees reducing variance and improving probability ranking.
5. **Gradient Boosting Classifier (`sample_weight='balanced'`):** Sequential boosting ensemble fitting shallow trees to residual errors; achieves strongest tabular discrimination.

---

## 15. Model Training

Implemented in [04_Model_Training.ipynb](Notebooks/04_Model_Training.ipynb) on `X_train` (`304,887` rows: `87.74%` negative, `12.26%` positive):

### Phase 6: Class Imbalance Comparison (3-Fold Stratified CV on Training Set)

| Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | CV Time (s) |
|---|---|---|---|---|---|---|---|
| No Balancing | 0.8773 | 0.2718 | 0.0007 | 0.0014 | 0.8476 | 0.3337 | 8.10 |
| **`class_weight='balanced'`** | 0.6911 | 0.2760 | **0.9366** | **0.4264** | 0.8473 | 0.3323 | 8.23 |
| RandomOverSampler | 0.6910 | 0.2760 | 0.9366 | 0.4263 | 0.8473 | 0.3321 | 11.39 |
| SMOTE | 0.6883 | 0.2743 | 0.9379 | 0.4245 | 0.8473 | 0.3315 | 14.78 |
| RandomUnderSampler | 0.6927 | 0.2769 | 0.9354 | 0.4273 | 0.8472 | 0.3321 | 5.98 |

`class_weight='balanced'` (and balanced `sample_weight` for Gradient Boosting) was chosen because it resolves the near-zero Recall (`0.0007` → `0.9366`) without discarding majority data or synthesizing artificial one-hot rows.

---

## 16. Model Evaluation

Evaluated in [05_Model_Evaluation.ipynb](Notebooks/05_Model_Evaluation.ipynb) on the unseen test set (`76,222` customers, `9,342` positives):

| Algorithm | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Dummy Classifier (Baseline) | 0.8774 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1226 |
| Logistic Regression | 0.6933 | 0.2774 | 0.9360 | 0.4279 | 0.8485 | 0.3342 |
| Decision Tree | 0.7055 | 0.2831 | 0.9160 | 0.4326 | 0.8503 | 0.3471 |
| Random Forest | 0.6993 | 0.2806 | 0.9291 | 0.4310 | 0.8543 | 0.3599 |
| **Gradient Boosting** | 0.7015 | 0.2820 | 0.9286 | **0.4326** | **0.8568** | **0.3676** |

Visual outputs: `Outputs/graphs/16_confusion_matrices.png` and `Outputs/graphs/17_roc_and_pr_curves.png`.

### Hyperparameter Tuning (Before vs. After on Gradient Boosting)
`RandomizedSearchCV` (`n_iter=8`, 3-fold Stratified CV on a 100k stratified training subset) selected `n_estimators=150`, `learning_rate=0.08`, `max_depth=4`, `min_samples_split=50`, `min_samples_leaf=20`, `subsample=1.0`, which was then refitted on all `304,887` training rows:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Gradient Boosting (Before Tuning) | 0.7015 | 0.2820 | 0.9286 | 0.4326 | **0.8568** | 0.3676 |
| **Gradient Boosting (After Tuning)** | **0.7022** | **0.2826** | **0.9290** | **0.4334** | 0.8567 | **0.3679** |

---

## 17. Results and Discussion

- **Accuracy Trap Avoided:** Dummy Classifier achieved `87.74%` accuracy with `0%` Recall. Balanced Gradient Boosting catches **`92.90%`** of interested customers while achieving **`28.26%`** Precision (**2.30×** higher conversion density than the `12.26%` base rate).
- **Threshold Flexibility (`Outputs/graphs/20_threshold_tradeoff.png`):**
  - `threshold = 0.30`: `35,550` flagged, Precision `0.2571`, Recall `0.9783`, F1 `0.4072`
  - `threshold = 0.50`: `30,712` flagged, Precision `0.2826`, Recall `0.9290`, F1 `0.4334`
  - `threshold = 0.70`: `20,275` flagged, Precision `0.3326`, Recall `0.7218`, F1 `0.4553`
  - `threshold = 0.80`: `4,848` flagged, Precision `0.4088`, Recall `0.2122`, F1 `0.2794`

---

## 18. Feature Importance

Analyzed in [06_Model_Explainability.ipynb](Notebooks/06_Model_Explainability.ipynb) (`Outputs/graphs/18_feature_importance.png`, `Outputs/graphs/19_permutation_importance.png`):
- **Permutation Importance (ROC-AUC drop on `X_test`):**
  1. `Previously_Insured`: **`0.1882`**
  2. `Vehicle_Damage`: **`0.0562`**
  3. `Policy_Sales_Channel`: **`0.0326`**
  4. `Age`: **`0.0310`**
  5. `Region_Code`: **`0.0049`**
  6. `Vehicle_Age`: **`0.0038`**
  7. `Annual_Premium`: **`0.0011`**
  8. `Gender`: **`0.0003`**
  9. `Driving_License`: **`0.0001`**
  10. `Vintage`: **`-0.00005`** (negligible impact)
- **Association vs. Causation:** These rankings measure statistical contribution to model predictions on historical data, not causal proof that changing one attribute causes a customer to buy.

---

## 19. Limitations

1. **No External Out-of-Time Test File:** Evaluation relies on a static 20% stratified split; temporal drift across future years is not measured.
2. **Moderate Precision at Default Threshold (`0.50`):** `28.26%` Precision means roughly 7 in 10 flagged customers still decline, though this is adjustable via higher thresholds.
3. **Anonymized Codes:** `Region_Code` and `Policy_Sales_Channel` lack human-readable labels (e.g., city names or channel names like "App" vs. "Agent").
4. **Rare Category Grouping:** Codes representing `< 1%` of training data are grouped into an `infrequent` bucket during one-hot encoding.

---

## 20. Future Scope

1. Benchmark gradient-boosted libraries (`XGBoost`, `LightGBM`, `CatBoost`) with native categorical handling.
2. Apply probability calibration (`CalibratedClassifierCV` / Platt scaling) when training with balanced sample weights so raw probabilities directly reflect true empirical conversion rates.
3. Add SHAP summary/waterfall plots for individual customer-level explanations in the Streamlit UI.
4. Integrate campaign cost and expected policy revenue into a profit-maximizing threshold optimizer.

---

## 21. Conclusion

This project successfully built, validated, tuned, explained, and deployed an end-to-end Machine Learning classification system for **Health Insurance Cross-Sell Prediction**. By pairing careful data inspection and leakage-free preprocessing with balanced **Gradient Boosting**, the final pipeline achieves **0.8567 ROC-AUC**, **0.3679 PR-AUC**, and **92.90% Recall** on `76,222` unseen customers, providing a practical, interpretable tool (`prediction.py` and `app.py`) for targeted insurance marketing.

---

## 22. References

1. Pedregosa et al., *"Scikit-learn: Machine Learning in Python"*, Journal of Machine Learning Research (JMLR 12), 2011 (`scikit-learn 1.7.2` documentation).
2. Lemaître, Nogueira, Aridas, *"Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning"*, JMLR 18, 2017 (`imblearn 0.14.0`).
3. Friedman, J. H., *"Greedy Function Approximation: A Gradient Boosting Machine"*, Annals of Statistics, 2001.
4. McKinney, W., *"Data Structures for Statistical Computing in Python"* (`pandas 2.3.3`).
5. Hunter, J. D., *"Matplotlib: A 2D Graphics Environment"* (`matplotlib 3.10.6`), and Waskom, M., *"Seaborn: Statistical Data Visualization"* (`seaborn 0.13.2`).
