"""Train and compare annual medical cost models on insurance.csv.

Steps: 80/20 split stratified on smoker -> for each model try raw and log target
with 5-fold CV -> tune the top 2 with RandomizedSearchCV (30 iters) -> report
test metrics for all -> SHAP plot -> save best pipeline to models/cost_model.pkl.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.features import CATEGORICAL, NUMERIC, add_features

warnings.filterwarnings("ignore", message=".*feature names.*")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

BASE_MODELS = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                            random_state=SEED, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              random_state=SEED, n_jobs=-1, verbose=-1),
}

SEARCH_SPACES = {
    "RandomForest": {
        "model__n_estimators": randint(200, 700),
        "model__max_depth": [3, 4, 5, 6, 8, 10, None],
        "model__min_samples_leaf": randint(1, 25),
        "model__max_features": uniform(0.3, 0.7),
    },
    "XGBoost": {
        "model__n_estimators": randint(100, 900),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 7),
        "model__min_child_weight": randint(1, 15),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_lambda": loguniform(0.1, 20),
    },
    "LightGBM": {
        "model__n_estimators": randint(100, 900),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__num_leaves": randint(4, 32),
        "model__min_child_samples": randint(5, 60),
        "model__subsample": uniform(0.6, 0.4),
        "model__subsample_freq": [1],
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_lambda": loguniform(0.1, 20),
    },
}


def make_pipeline(model, log_target: bool):
    """Feature engineering + encoding + model, optionally on log(charges)."""
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    pipe = Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", prep),
        ("model", model),
    ])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)
    return pipe


def prefix_space(space, log_target):
    # Hyperparameters sit one level deeper when wrapped in TransformedTargetRegressor.
    return {(f"regressor__{k}" if log_target else k): v for k, v in space.items()}


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def main():
    df = pd.read_csv(ROOT / "data" / "insurance.csv").drop_duplicates()
    X, y = df.drop(columns="charges"), df["charges"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=X["smoker"], random_state=SEED)
    print(f"Train {len(X_tr)} / test {len(X_te)} (1 duplicate row dropped)\n")

    # 1. Raw vs log target for each model, chosen on 5-fold CV RMSE (train only).
    # RMSE is the selection metric because a premium needs the *mean* cost;
    # log models predict closer to the median, which lowers MAE but biases low.
    print("== 5-fold CV on train: raw vs log target ==")
    cv_rows, chosen = [], {}
    for name, model in BASE_MODELS.items():
        best = None
        for log_target in (False, True):
            res = cross_validate(make_pipeline(model, log_target), X_tr, y_tr, cv=CV,
                                 scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"))
            row = {"model": name, "target": "log" if log_target else "raw",
                   "cv_R2": res["test_r2"].mean(),
                   "cv_RMSE": -res["test_neg_root_mean_squared_error"].mean(),
                   "cv_MAE": -res["test_neg_mean_absolute_error"].mean()}
            cv_rows.append(row)
            if best is None or row["cv_RMSE"] < best["cv_RMSE"]:
                best = row
        chosen[name] = best["target"] == "log"
    cv_df = pd.DataFrame(cv_rows)
    print(cv_df.round(3).to_string(index=False))
    print("Chosen target per model:", {k: ("log" if v else "raw") for k, v in chosen.items()}, "\n")

    # 2. Tune the top 2 (by CV RMSE) among models that have hyperparameters.
    best_cv = cv_df.loc[cv_df.groupby("model")["cv_RMSE"].idxmin()].sort_values("cv_RMSE")
    top2 = [m for m in best_cv["model"] if m in SEARCH_SPACES][:2]
    print("== Tuning top 2:", top2, "==")
    fitted, tuned_cv = {}, {}
    for name in top2:
        log_target = chosen[name]
        search = RandomizedSearchCV(
            make_pipeline(BASE_MODELS[name], log_target),
            prefix_space(SEARCH_SPACES[name], log_target),
            n_iter=30, cv=CV, scoring="neg_root_mean_squared_error",
            random_state=SEED, n_jobs=-1)
        search.fit(X_tr, y_tr)
        key = f"{name} (tuned)"
        fitted[key] = search.best_estimator_
        tuned_cv[key] = -search.best_score_
        params = {k.split("__")[-1]: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in search.best_params_.items()}
        print(f"{key}: CV RMSE {-search.best_score_:.1f}  params {params}")
    print()

    # 3. Fit untuned versions and score everything on train and test.
    for name, model in BASE_MODELS.items():
        fitted[name] = make_pipeline(model, chosen[name]).fit(X_tr, y_tr)
    rows = []
    for key, est in fitted.items():
        base = key.replace(" (tuned)", "")
        tr, te = metrics(y_tr, est.predict(X_tr)), metrics(y_te, est.predict(X_te))
        cv_rmse = tuned_cv.get(key, best_cv.set_index("model").loc[base, "cv_RMSE"])
        rows.append({"model": key, "target": "log" if chosen[base] else "raw",
                     "cv_RMSE": cv_rmse, "train_R2": tr["R2"],
                     "test_R2": te["R2"], "test_MAE": te["MAE"], "test_RMSE": te["RMSE"],
                     "test_MAPE_%": te["MAPE"], "R2_gap": tr["R2"] - te["R2"]})
    table = pd.DataFrame(rows).sort_values("cv_RMSE").reset_index(drop=True)
    print("== Test-set comparison (best model picked on CV RMSE, not test) ==")
    print(table.round(3).to_string(index=False))

    best_name = table.loc[0, "model"]
    best = fitted[best_name]
    print(f"\nBest model: {best_name}")

    # 4. SHAP on the best model (tree models only; values are on the model's target scale).
    pipe = best.regressor_ if isinstance(best, TransformedTargetRegressor) else best
    Xt = pipe[:-1].transform(X_te)
    feat_names = [n.split("__", 1)[1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer.shap_values(Xt)
    plt.figure()
    shap.summary_plot(sv, pd.DataFrame(Xt, columns=feat_names), plot_type="bar", show=False, max_display=12)
    scale = "log(charges)" if isinstance(best, TransformedTargetRegressor) else "USD charges"
    plt.title(f"SHAP importance, {best_name} (mean |SHAP| on {scale})")
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "shap_importance.png", dpi=120)
    plt.close()
    importance = pd.Series(np.abs(sv).mean(axis=0), index=feat_names).sort_values(ascending=False)
    print("\nTop SHAP drivers:\n", importance.head(8).round(3).to_string())

    # 5. Persist. Residual spread on the test set (log scale) lets premium.py
    # turn a mean cost into a distribution and cap it at the sum insured.
    log_resid = np.log(y_te) - np.log(np.clip(best.predict(X_te), 1, None))
    joblib.dump(best, ROOT / "models" / "cost_model.pkl")
    model_meta = {
        "best_model": best_name,
        "log_residual_sigma_test": float(log_resid.std()),
        "train_mean_charges_usd": float(y_tr.mean()),
        "data_rows": len(df),
    }
    (ROOT / "models" / "cost_model_meta.json").write_text(json.dumps(model_meta, indent=2))
    print(f"\nLog-residual sigma (test): {model_meta['log_residual_sigma_test']:.3f}")
    out = {
        "best_model": best_name,
        "target": "log" if chosen[best_name.replace(' (tuned)', '')] else "raw",
        "n_train": len(X_tr), "n_test": len(X_te),
        "cv_table": cv_df.to_dict(orient="records"),
        "comparison": table.to_dict(orient="records"),
        "shap_top": importance.head(8).to_dict(),
    }
    (ROOT / "reports" / "model_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nSaved models/cost_model.pkl, reports/model_metrics.json, reports/shap_importance.png")


if __name__ == "__main__":
    main()
