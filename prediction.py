"""
Reusable Prediction Module for Health Insurance Cross-Sell Prediction.

Loads the saved preprocessing + tuned Gradient Boosting pipeline (`Models/best_model.pkl`)
and predicts whether an existing health insurance customer is likely to be interested
in purchasing vehicle insurance.

Note: Predicted scores represent model-estimated likelihood based on historical patterns,
not a guaranteed purchase probability.
"""

from pathlib import Path
from typing import Any, Dict, Union

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "Models" / "best_model.pkl"
VALID_GENDERS = {"Male", "Female"}
VALID_VEHICLE_AGES = {"< 1 Year", "1-2 Year", "> 2 Years"}
VALID_VEHICLE_DAMAGE = {"Yes", "No"}


def _to_clean_code(value: Union[int, float, str], field_name: str) -> str:
    """Convert numeric or string region/channel codes (e.g., 28 or 28.0) to '28'."""
    try:
        numeric_val = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a numeric code, got {value!r}.") from exc
    if not numeric_val.is_integer():
        raise ValueError(f"{field_name} must represent an integer code, got {value!r}.")
    return str(int(numeric_val))


def load_trained_pipeline(model_path: Union[str, Path] = DEFAULT_MODEL_PATH):
    """Load the saved Scikit-learn preprocessing + classifier pipeline."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found at: {path}")
    return joblib.load(path)


def predict_customer_interest(
    Gender: str,
    Age: Union[int, float],
    Driving_License: int,
    Region_Code: Union[int, float, str],
    Previously_Insured: int,
    Vehicle_Age: str,
    Vehicle_Damage: str,
    Annual_Premium: float,
    Policy_Sales_Channel: Union[int, float, str],
    Vintage: int,
    threshold: float = 0.50,
    model=None,
) -> Dict[str, Any]:
    """
    Predict vehicle insurance interest for a single health insurance customer.

    Parameters
    ----------
    Gender : str
        'Male' or 'Female'.
    Age : int or float
        Customer age in years (must be > 0).
    Driving_License : int
        1 if customer holds a driving licence, 0 otherwise.
    Region_Code : int, float, or str
        Customer region identifier code (e.g., 28).
    Previously_Insured : int
        1 if customer already has vehicle insurance, 0 otherwise.
    Vehicle_Age : str
        '< 1 Year', '1-2 Year', or '> 2 Years'.
    Vehicle_Damage : str
        'Yes' or 'No'.
    Annual_Premium : float
        Annual health insurance premium amount (> 0).
    Policy_Sales_Channel : int, float, or str
        Outreach/sales channel code (e.g., 26, 152).
    Vintage : int
        Number of days the customer has been with the company (>= 0).
    threshold : float, default=0.50
        Decision cutoff between 0.0 and 1.0.
        Lower threshold -> higher Recall (catches more leads, more outreach calls).
        Higher threshold -> higher Precision (fewer wasted calls, may miss some leads).
    model : sklearn Pipeline, optional
        Pre-loaded model pipeline. If None, loads Models/best_model.pkl.

    Returns
    -------
    dict
        Contains predicted_class (0 or 1), estimated_interest_score (float),
        estimated_interest_pct (str), prediction_label, and threshold_used.
    """
    if Gender not in VALID_GENDERS:
        raise ValueError(f"Gender must be one of {VALID_GENDERS}, got {Gender!r}.")
    if Vehicle_Age not in VALID_VEHICLE_AGES:
        raise ValueError(f"Vehicle_Age must be one of {VALID_VEHICLE_AGES}, got {Vehicle_Age!r}.")
    if Vehicle_Damage not in VALID_VEHICLE_DAMAGE:
        raise ValueError(f"Vehicle_Damage must be one of {VALID_VEHICLE_DAMAGE}, got {Vehicle_Damage!r}.")
    if Driving_License not in (0, 1):
        raise ValueError(f"Driving_License must be 0 or 1, got {Driving_License!r}.")
    if Previously_Insured not in (0, 1):
        raise ValueError(f"Previously_Insured must be 0 or 1, got {Previously_Insured!r}.")
    if Age <= 0:
        raise ValueError(f"Age must be positive, got {Age!r}.")
    if Annual_Premium <= 0:
        raise ValueError(f"Annual_Premium must be positive, got {Annual_Premium!r}.")
    if Vintage < 0:
        raise ValueError(f"Vintage must be non-negative, got {Vintage!r}.")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold!r}.")

    pipeline = model if model is not None else load_trained_pipeline()

    input_df = pd.DataFrame(
        [
            {
                "Gender": Gender,
                "Age": float(Age),
                "Driving_License": int(Driving_License),
                "Region_Code": _to_clean_code(Region_Code, "Region_Code"),
                "Previously_Insured": int(Previously_Insured),
                "Vehicle_Age": Vehicle_Age,
                "Vehicle_Damage": Vehicle_Damage,
                "Annual_Premium": float(Annual_Premium),
                "Policy_Sales_Channel": _to_clean_code(Policy_Sales_Channel, "Policy_Sales_Channel"),
                "Vintage": int(Vintage),
            }
        ]
    )

    interest_score = float(pipeline.predict_proba(input_df)[0, 1])
    predicted_class = int(interest_score >= threshold)
    prediction_label = "Likely Interested" if predicted_class == 1 else "Likely Not Interested"

    return {
        "predicted_class": predicted_class,
        "estimated_interest_score": round(interest_score, 4),
        "estimated_interest_pct": f"{interest_score * 100:.2f}%",
        "prediction_label": prediction_label,
        "threshold_used": threshold,
    }


if __name__ == "__main__":
    loaded_model = load_trained_pipeline()
    print(f"Successfully loaded model pipeline from: {DEFAULT_MODEL_PATH}\n")

    sample_customers = [
        {
            "desc": "Profile A (Uninsured, damaged older vehicle, middle-aged)",
            "data": dict(
                Gender="Male",
                Age=44,
                Driving_License=1,
                Region_Code=28,
                Previously_Insured=0,
                Vehicle_Age="> 2 Years",
                Vehicle_Damage="Yes",
                Annual_Premium=40454.0,
                Policy_Sales_Channel=26,
                Vintage=217,
            ),
        },
        {
            "desc": "Profile B (Already insured, undamaged new vehicle, young adult)",
            "data": dict(
                Gender="Female",
                Age=24,
                Driving_License=1,
                Region_Code=11,
                Previously_Insured=1,
                Vehicle_Age="< 1 Year",
                Vehicle_Damage="No",
                Annual_Premium=27496.0,
                Policy_Sales_Channel=152,
                Vintage=90,
            ),
        },
    ]

    for item in sample_customers:
        res_default = predict_customer_interest(**item["data"], threshold=0.50, model=loaded_model)
        res_strict = predict_customer_interest(**item["data"], threshold=0.70, model=loaded_model)
        print(item["desc"])
        print(f"  Threshold 0.50 -> {res_default}")
        print(f"  Threshold 0.70 -> {res_strict}\n")
