"""Validate term_life.py against published term quotes and calibrate insured mortality.

Same approach as validate.py: uncalibrated round first; if any quote is more
than 40% off, fit one insured_mortality_factor per country (WHO population
mortality -> insured-lives mortality) and report leave-one-out deviations.
Also reports how the fitted factor and premium respond to the assumed loss ratio.
"""
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

from src import premium, term_life

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "data" / "country_config.json"
THRESHOLD = 0.40


def _set(code, key, value, extra=None):
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    cfg[code]["term_life"][key].update({"value": value, **(extra or {})})
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    premium.load_config.cache_clear()


def _model(q, code):
    return term_life.estimate_term_premium(q["age"], q["sex"], q["smoker"], q["cover"],
                                           q["term_years"], code)["annual_premium"]


def _fit(qs, code):
    """Factor that best matches the quotes in log terms (iterated: the fee and survival are non-linear)."""
    f = 1.0
    for _ in range(6):
        _set(code, "insured_mortality_factor", f)
        f *= math.exp(sum(math.log(q["annual_premium"] / _model(q, code)) for q in qs) / len(qs))
    return f


def run():
    quotes = json.loads((ROOT / "data" / "term_quotes.json").read_text(encoding="utf-8"))
    rows, factors, sens = [], {}, []
    for code in premium.COUNTRIES:
        qs = quotes[code]
        _set(code, "insured_mortality_factor", 1.0)
        for q in qs:
            m = _model(q, code)
            rows.append({"country": code, "id": q["id"], "profile": q["label"], "quote": q["annual_premium"],
                         "model_raw": m, "dev_raw_%": (m / q["annual_premium"] - 1) * 100})
        if max(abs(r["dev_raw_%"]) for r in rows if r["country"] == code) <= THRESHOLD * 100:
            factors[code] = 1.0
            continue
        for q in qs:
            _set(code, "insured_mortality_factor", _fit([o for o in qs if o is not q], code))
            m = _model(q, code)
            r = next(r for r in rows if r["id"] == q["id"])
            r.update({"model_loo": m, "dev_loo_%": (m / q["annual_premium"] - 1) * 100})
        f = _fit(qs, code)
        factors[code] = f
        _set(code, "insured_mortality_factor", round(f, 4), {
            "source": "Fitted by src/validate_term.py: scales WHO population mortality to insured-lives "
                      "mortality so premiums match the quotes in fitted_on (geometric-mean fit).",
            "fitted_on": [q["id"] for q in qs], "quotes_file": "data/term_quotes.json",
            "fitted": date.today().isoformat()})
        for q in qs:
            m = _model(q, code)
            r = next(r for r in rows if r["id"] == q["id"])
            r.update({"model_cal": m, "dev_cal_%": (m / q["annual_premium"] - 1) * 100})

        # Loss-ratio sensitivity: refit under each assumed loss ratio, then restore.
        tl = premium.load_config()[code]["term_life"]
        base_lr = tl["life_loss_ratio"]["value"]
        for case, lr in {"low": tl["life_loss_ratio"]["sensitivity"]["low"], "base": base_lr,
                         "high": tl["life_loss_ratio"]["sensitivity"]["high"]}.items():
            _set(code, "life_loss_ratio", lr)
            fl = _fit(qs, code)
            sens.append({"country": code, "case": case, "loss_ratio": lr, "insured_mortality_factor": round(fl, 3),
                         "premium_" + qs[0]["id"]: _model(qs[0], code)})
        _set(code, "life_loss_ratio", base_lr)
        _set(code, "insured_mortality_factor", round(f, 4))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("== Term life validation (annual premium, local currency) ==")
    print(df[["country", "id", "quote", "model_raw", "dev_raw_%", "model_loo", "dev_loo_%",
              "model_cal", "dev_cal_%"]].round(1).to_string(index=False))
    print("\nInsured mortality factors:", {k: round(v, 4) for k, v in factors.items()})
    summary = (df.assign(abs_raw=df["dev_raw_%"].abs(), abs_loo=df["dev_loo_%"].abs(), abs_cal=df["dev_cal_%"].abs())
                 .groupby("country")[["abs_raw", "abs_loo", "abs_cal"]].mean())
    summary.columns = ["MAPE_raw_%", "MAPE_leave_one_out_%", "MAPE_calibrated_in_sample_%"]
    print("\nMean absolute % deviation per country:\n", summary.round(1).to_string())
    print("\nLife loss-ratio sensitivity (factor refitted each time):")
    print(pd.DataFrame(sens).to_string(index=False))
    out = {"rows": df.to_dict(orient="records"), "summary": summary.reset_index().to_dict(orient="records"),
           "insured_mortality_factors": factors, "loss_ratio_sensitivity": sens}
    (ROOT / "reports" / "term_validation_results.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nSaved reports/term_validation_results.json; factors written to data/country_config.json")


if __name__ == "__main__":
    run()
