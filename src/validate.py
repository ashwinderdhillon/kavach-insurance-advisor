"""Validate premium.py against published market premiums and calibrate if needed.

Round 1: uncalibrated model (calibration_factor = 1) vs each cited quote.
If any quote in a country deviates by more than 40%, fit ONE scalar per
country (geometric mean of quote / model) - the ML model is never touched.
Out-of-sample honesty: leave-one-out deviation, where each quote is predicted
with a factor fitted on the other quotes only.
Writes the factor back into data/country_config.json and results to reports/.
"""
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

from src import premium

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "data" / "country_config.json"
THRESHOLD = 0.40

REASONS = {
    "USA": "insurance.csv charges (mean $13,270) are well above what insurers actually pay per "
           "working-age adult, and ACA benchmark silver plans pay only about 70% of covered costs. "
           "Both push the raw model above market premiums.",
    "IND": "National health spend per person averages over everyone, including the many who use "
           "low-cost public facilities or skip care. People with private health insurance mostly use "
           "private hospitals, which cost several times more, so the raw scaling understates insured claims.",
    "KEN": "Same effect as India, stronger: the US model puts a healthy 30-year-old at about a third "
           "of national spend per person, but Kenyan insurers price the same person at about 2.5x "
           "Kenya's spend per person, because private cover buyers use private hospitals and the "
           "plans include extras (maternity, regional cover, critical-illness lump sums).",
}


def _set_factor(code, value, extra=None):
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    entry = {"value": value, "unit": "multiplier", "source_type": "calibration"}
    entry.update(extra or {"source": "Uncalibrated default; set by src/validate.py against cited market quotes."})
    cfg[code]["calibration_factor"] = entry
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    premium.load_config.cache_clear()


def _model_premium(q, code):
    group = q["profile"].get("group")
    if group:
        return round(premium.group_premium(code, group == "family", year=q["quote_year"]), 0)
    return premium.estimate_premium(q["profile"], code, q["sum_insured"], year=q["quote_year"])["annual_premium"]


def _gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _fit(quotes, code, start=1.0):
    """Scalar that best matches quotes in log terms (2 passes: the sum-insured cap is mildly non-linear)."""
    factor = start
    for _ in range(3):
        _set_factor(code, factor)
        factor *= _gm([q["annual_premium"] / _model_premium(q, code) for q in quotes])
    return factor


def _inflate_old_quotes(quotes):
    """Bring quotes flagged inflate_to_current_year to today's prices with the country's medical inflation."""
    notes = []
    for code in premium.COUNTRIES:
        rate = premium.load_config()[code]["medical_inflation_pct"]["value"] / 100
        for q in quotes[code]:
            if not q.get("inflate_to_current_year"):
                continue
            years = premium.CURRENT_YEAR - q["quote_year"]
            q["original_premium"], q["original_year"] = q["annual_premium"], q["quote_year"]
            q["annual_premium"] = round(q["annual_premium"] * (1 + rate) ** years, 0)
            q["quote_year"] = premium.CURRENT_YEAR
            notes.append(f"{q['id']}: {q['original_premium']:,} ({q['original_year']}) x (1 + {rate:.1%})^{years} "
                         f"= {q['annual_premium']:,.0f} ({premium.CURRENT_YEAR})")
    return notes


def _set_loss_ratio(code, value):
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    cfg[code]["loss_ratio"]["value"] = value
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    premium.load_config.cache_clear()


def _loss_ratio_sensitivity(code, qs, probe):
    """Refit the country under each loss ratio in the sensitivity range.

    Because the factor is refitted to the same quotes, premiums barely move;
    what changes is the factor and the implied expected claims.
    """
    cfg = premium.load_config()[code]
    base_lr, base_entry = cfg["loss_ratio"]["value"], dict(cfg["calibration_factor"])
    out = []
    for label, lr in cfg["loss_ratio_sensitivity"]["value"].items():
        _set_loss_ratio(code, lr)
        f = _fit(qs, code)
        r = premium.estimate_premium(probe, code, 1_000_000)
        out.append({"case": label, "loss_ratio": lr, "calibration_factor": round(f, 3),
                    "probe_expected_claims": r["expected_annual_claims"], "probe_premium": r["annual_premium"]})
    _set_loss_ratio(code, base_lr)
    _set_factor(code, base_entry.pop("value"), base_entry)
    return out


def run():
    quotes = json.loads((ROOT / "data" / "validation_quotes.json").read_text(encoding="utf-8"))
    adjustments = _inflate_old_quotes(quotes)
    if adjustments:
        print("Inflation adjustment of older quotes (country medical inflation from country_config.json):")
        print("\n".join("  " + a for a in adjustments), "\n")
    rows, calib = [], {}
    for code in premium.COUNTRIES:
        _set_factor(code, 1.0)
        # Only independent published quotes are fitted and scored; derived ones
        # (official factors applied to a published figure) are shown for shape only.
        all_qs = quotes[code]
        qs = [q for q in all_qs if q["type"] == "published"]
        for q in all_qs:
            m = _model_premium(q, code)
            rows.append({"country": code, "id": q["id"], "profile": q["label"], "type": q["type"],
                         "quote_year": q["quote_year"], "quote": q["annual_premium"],
                         "model_raw": m, "dev_raw_%": (m / q["annual_premium"] - 1) * 100})
        worst = max(abs(r["dev_raw_%"]) for r in rows
                    if r["country"] == code and r["type"] == "published") / 100
        if worst <= THRESHOLD:
            calib[code] = None
            continue
        # Leave-one-out: fit on the other quotes, predict the held-out one.
        for q in qs:
            others = [o for o in qs if o["id"] != q["id"]]
            f = _fit(others, code)
            _set_factor(code, f)
            m = _model_premium(q, code)
            r = next(r for r in rows if r["id"] == q["id"])
            r["model_loo"] = m
            r["dev_loo_%"] = (m / q["annual_premium"] - 1) * 100
        f = _fit(qs, code)
        calib[code] = f
        _set_factor(code, round(f, 4), {
            "source": "Fitted by src/validate.py: geometric mean of (published quote / uncalibrated model "
                      "premium) over the quotes listed in fitted_on. Adjusts only the calibration layer; "
                      "the ML model is unchanged.",
            "fitted_on": [q["id"] for q in qs],
            "quotes_file": "data/validation_quotes.json",
            "reason": REASONS[code],
            "fitted": date.today().isoformat(),
        })
        for q in all_qs:
            m = _model_premium(q, code)
            r = next(r for r in rows if r["id"] == q["id"])
            r["model_cal"] = m
            r["dev_cal_%"] = (m / q["annual_premium"] - 1) * 100

    df = pd.DataFrame(rows)
    scored = df[df["type"] == "published"]
    print("Independent published quotes per country:", scored.groupby("country").size().to_dict())
    pd.set_option("display.width", 200)
    print("== Validation vs published premiums (annual, local currency) ==")
    print(df[["country", "id", "type", "quote_year", "quote", "model_raw", "dev_raw_%",
              "model_loo", "dev_loo_%", "model_cal", "dev_cal_%"]].round(1).to_string(index=False))
    print("\nCalibration factors:", {k: (round(v, 4) if v else "not needed") for k, v in calib.items()})

    summary = (scored.assign(abs_raw=scored["dev_raw_%"].abs(), abs_loo=scored["dev_loo_%"].abs(),
                             abs_cal=scored["dev_cal_%"].abs())
                 .groupby("country")[["abs_raw", "abs_loo", "abs_cal"]].mean())
    summary.columns = ["MAPE_raw_%", "MAPE_leave_one_out_%", "MAPE_calibrated_in_sample_%"]
    print("\nMean absolute % deviation per country (published quotes only):\n", summary.round(1).to_string())

    # Kenya loss-ratio sensitivity (only 2022 is verified from a primary IRA document).
    probe = {"age": 30, "sex": "male", "bmi": 24, "children": 0, "smoker": False}
    sens = _loss_ratio_sensitivity("KEN", [q for q in quotes["KEN"] if q["type"] == "published"], probe)
    print("\nKenya loss-ratio sensitivity (refit each case; probe = age 30, KES 1M cover, 2026):")
    print(pd.DataFrame(sens).to_string(index=False))

    out = {"rows": df.to_dict(orient="records"), "summary": summary.reset_index().to_dict(orient="records"),
           "calibration_factors": calib, "quote_adjustments": adjustments,
           "kenya_loss_ratio_sensitivity": sens, "threshold_pct": THRESHOLD * 100}
    (ROOT / "reports" / "validation_results.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nSaved reports/validation_results.json; calibration written to data/country_config.json")


if __name__ == "__main__":
    run()
