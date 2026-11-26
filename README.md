# Health Insurance Cross-Sell Prediction Using Machine Learning

A supervised binary classification project that predicts whether an existing health insurance customer is likely to be interested in purchasing vehicle insurance (`Response = 1`) or not (`Response = 0`).

## Project Structure

```text
Insurance_Cross_Sell_Prediction/
├── Dataset/
│   └── train.csv                                  # 381,109 rows x 12 columns
├── Notebooks/
│   ├── 01_Data_Understanding.ipynb                # Phase 1: Shape, types, missing values, summary stats
│   ├── 02_Exploratory_Data_Analysis.ipynb         # Phase 2: 13 EDA graphs & behavioral findings
│   ├── 03_Data_Cleaning_Preprocessing.ipynb       # Phases 3-5 & 7: Cleaning, VIF, 80/20 split, ColumnTransformer
│   ├── 04_Model_Training.ipynb                    # Phases 6 & 8: Imbalance comparison & 5 model pipelines
│   ├── 05_Model_Evaluation.ipynb                  # Phases 9 & 10: Test metrics, ROC/PR curves, tuning & best_model.pkl
│   └── 06_Model_Explainability.ipynb              # Phase 11: Feature importance, permutation importance, thresholds
├── Models/
│   ├── candidate_models.joblib                    # Trained baseline/candidate pipelines
│   └── best_model.pkl                             # Final tuned Gradient Boosting end-to-end pipeline
├── Outputs/
│   ├── graphs/                                    # 20 saved PNG visualizations
│   └── evaluation_results/                        # Saved CSV metric & importance tables
├── app.py                                         # Streamlit web application using best_model.pkl
├── prediction.py                                  # Reusable prediction module with configurable threshold
├── PROJECT_REPORT.md                              # Complete 22-section college academic project report
├── requirements.txt                               # Python dependencies
└── README.md                                      # Project documentation
```

## Setup & Installation

1. Ensure Python 3.10+ (tested on Python `3.13.9` via Anaconda) is installed.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Project

1. **Run Notebooks in Order (`Notebooks/01` → `06`)** using VS Code or Jupyter Notebook to reproduce data inspection, EDA, preprocessing, training, evaluation, tuning, and explainability.
2. **Test Predictions on New Customer Data (`prediction.py`):**
   ```bash
   python prediction.py
   ```
3. **Launch the Streamlit Web Application (`app.py`):**
   ```bash
   streamlit run app.py
   ```

## Model Summary (Verified on 76,222 Unseen Test Customers)

- **Selected Model:** Tuned `GradientBoostingClassifier` inside an end-to-end `ColumnTransformer` + `Pipeline` (`Models/best_model.pkl`)
- **Test Performance (at `threshold = 0.50`):**
  - **Accuracy:** `0.7022`
  - **Precision (`Response = 1`):** `0.2826`
  - **Recall (`Response = 1`):** `0.9290`
  - **F1-score (`Response = 1`):** `0.4334`
  - **ROC-AUC:** `0.8567`
  - **PR-AUC:** `0.3679`
