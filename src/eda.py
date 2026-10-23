"""Brief EDA on insurance.csv: prints summary stats, saves 3 plots to reports/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SMOKER_COLORS = {"no": "#2a78b8", "yes": "#d1495b"}


def main():
    df = pd.read_csv(ROOT / "data" / "insurance.csv")
    print(f"Rows: {len(df)}, columns: {list(df.columns)}")
    print(f"Missing values: {int(df.isna().sum().sum())}, duplicates: {int(df.duplicated().sum())}")
    print(df.describe().round(2).to_string())
    print("\nMedian charges by smoker:\n", df.groupby("smoker")["charges"].median().round(0).to_string())
    print("\nSkewness raw vs log:", round(df["charges"].skew(), 2), round(np.log(df["charges"]).skew(), 2))

    # 1. Target distribution, raw vs log
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["charges"], bins=40, color="#2a78b8")
    axes[0].set(title="Annual charges (USD)", xlabel="USD", ylabel="People")
    axes[1].hist(np.log(df["charges"]), bins=40, color="#2a78b8")
    axes[1].set(title="log(charges)", xlabel="log USD")
    fig.tight_layout()
    fig.savefig(REPORTS / "eda_1_charges_distribution.png", dpi=120)
    plt.close(fig)

    # 2. Age effect split by smoker
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, g in df.groupby("smoker"):
        ax.scatter(g["age"], g["charges"], s=10, alpha=0.6, color=SMOKER_COLORS[s], label=f"smoker={s}")
    ax.set(title="Charges rise with age; smokers sit on a higher band", xlabel="Age", ylabel="Charges (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "eda_2_age_smoker.png", dpi=120)
    plt.close(fig)

    # 3. BMI effect split by smoker (shows the interaction)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, g in df.groupby("smoker"):
        ax.scatter(g["bmi"], g["charges"], s=10, alpha=0.6, color=SMOKER_COLORS[s], label=f"smoker={s}")
    ax.axvline(30, color="grey", ls="--", lw=1)
    ax.text(30.3, df["charges"].max() * 0.95, "BMI 30 (obese)", color="grey")
    ax.set(title="BMI matters mainly for smokers (interaction)", xlabel="BMI", ylabel="Charges (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "eda_3_bmi_smoker.png", dpi=120)
    plt.close(fig)
    print("\nSaved 3 plots to reports/")


if __name__ == "__main__":
    main()
