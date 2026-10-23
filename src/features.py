"""Feature engineering shared by training and inference.

Kept as a plain module-level function so the saved sklearn pipeline can be
unpickled anywhere `src` is importable (Streamlit app, tests, tools).
"""
import numpy as np
import pandas as pd

RAW_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
NUMERIC = ["age", "bmi", "children", "smoker_bmi"]
CATEGORICAL = ["sex", "smoker", "region", "bmi_category", "age_band"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bmi_category, age_band and the smoker x BMI interaction."""
    out = df[RAW_COLUMNS].copy()
    out["bmi_category"] = pd.cut(
        out["bmi"], bins=[0, 18.5, 25, 30, np.inf],
        labels=["under", "normal", "over", "obese"], right=False,
    ).astype(str)
    out["age_band"] = pd.cut(
        out["age"], bins=[0, 25, 35, 45, 55, 65, np.inf],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"], right=False,
    ).astype(str)
    # Smoking multiplies the effect of high BMI in this data, so give tree and
    # linear models the product directly.
    out["smoker_bmi"] = (out["smoker"] == "yes").astype(float) * out["bmi"]
    return out
