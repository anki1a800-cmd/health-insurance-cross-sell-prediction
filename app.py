"""
Streamlit Web Application for Health Insurance Cross-Sell Prediction.

Uses the exact saved preprocessing + tuned Gradient Boosting pipeline
(`Models/best_model.pkl`) via `prediction.py`.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from prediction import load_trained_pipeline, predict_customer_interest

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "Outputs" / "evaluation_results"

st.set_page_config(
    page_title="Health Insurance Cross-Sell Predictor",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def get_model():
    """Load the serialized Scikit-learn pipeline once and cache it."""
    return load_trained_pipeline()


st.title("Health Insurance Cross-Sell Prediction")
st.markdown(
    "Predict whether an existing health insurance customer is likely to be interested "
    "in purchasing **vehicle insurance** using the trained **Tuned Gradient Boosting** pipeline."
)

tab_predict, tab_insights = st.tabs(["Predict Customer Interest", "Model Performance & Insights"])

with tab_predict:
    st.subheader("Enter Customer & Policy Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=44, step=1)
        driving_license_label = st.selectbox("Has Driving License?", ["Yes (1)", "No (0)"])
        driving_license = 1 if driving_license_label.startswith("Yes") else 0
        region_code = st.number_input("Region Code (0 - 52)", min_value=0, max_value=100, value=28, step=1)

    with col2:
        previously_insured_label = st.selectbox(
            "Already Has Vehicle Insurance? (Previously_Insured)",
            ["No (0)", "Yes (1)"],
        )
        previously_insured = 1 if previously_insured_label.startswith("Yes") else 0
        vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"], index=2)
        vehicle_damage = st.selectbox("Previous Vehicle Damage?", ["Yes", "No"])

    with col3:
        annual_premium = st.number_input(
            "Annual Health Premium", min_value=500.0, max_value=1000000.0, value=40454.0, step=500.0
        )
        policy_sales_channel = st.number_input(
            "Policy Sales Channel Code (1 - 163)", min_value=1, max_value=200, value=26, step=1
        )
        vintage = st.number_input("Vintage (Days with Company)", min_value=0, max_value=400, value=217, step=1)

    st.divider()
    threshold = st.slider(
        "Classification Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Lower threshold -> higher Recall (catches more interested leads, more outreach calls). "
            "Higher threshold -> higher Precision (fewer wasted calls, may miss some leads)."
        ),
    )

    if st.button("Predict Vehicle Insurance Interest", type="primary"):
        try:
            model = get_model()
            result = predict_customer_interest(
                Gender=gender,
                Age=age,
                Driving_License=driving_license,
                Region_Code=region_code,
                Previously_Insured=previously_insured,
                Vehicle_Age=vehicle_age,
                Vehicle_Damage=vehicle_damage,
                Annual_Premium=annual_premium,
                Policy_Sales_Channel=policy_sales_channel,
                Vintage=vintage,
                threshold=threshold,
                model=model,
            )

            score = result["estimated_interest_score"]
            st.markdown("### Prediction Result")

            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", result["prediction_label"])
            m2.metric("Predicted Class", str(result["predicted_class"]))
            m3.metric("Estimated Interest Score", result["estimated_interest_pct"])

            st.progress(min(max(score, 0.0), 1.0))

            if result["predicted_class"] == 1:
                st.success(
                    f"Customer is classified as **Likely Interested** at threshold `{threshold:.2f}` "
                    f"(estimated interest score: **{result['estimated_interest_pct']}**)."
                )
            else:
                st.info(
                    f"Customer is classified as **Likely Not Interested** at threshold `{threshold:.2f}` "
                    f"(estimated interest score: **{result['estimated_interest_pct']}**)."
                )

            st.caption(
                "Note: This score reflects a model-estimated likelihood based on historical customer patterns, "
                "not a guaranteed purchase probability."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab_insights:
    st.subheader("Verified Test-Set Evaluation Results (76,222 Unseen Customers)")

    comp_path = EVAL_DIR / "model_comparison.csv"
    tuning_path = EVAL_DIR / "tuning_comparison.csv"
    perm_path = EVAL_DIR / "permutation_importance.csv"

    if comp_path.exists():
        st.markdown("**Model Comparison Table**")
        st.dataframe(pd.read_csv(comp_path).round(4), use_container_width=True, hide_index=True)

    if tuning_path.exists():
        st.markdown("**Gradient Boosting: Before vs. After Tuning**")
        st.dataframe(pd.read_csv(tuning_path).round(4), use_container_width=True, hide_index=True)

    if perm_path.exists():
        st.markdown("**Permutation Importance (ROC-AUC Drop on Unseen Test Set)**")
        st.dataframe(pd.read_csv(perm_path).round(5), use_container_width=True, hide_index=True)
