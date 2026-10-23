"""Country-calibrated medical cost and premium estimates (chatbot tool layer).

The ML model (models/cost_model.pkl) predicts annual US medical charges from
age, sex, BMI, children, smoker and US region. This module turns that into a
local-currency cost and premium for USA, India and Kenya:

    cost_LCU  = model_USD x (H_country_LCU / H_USA_USD)       # health-spend scaling
                x (1 + medical_inflation)^(this_year - H_year)  # trend to today
                x calibration_factor                            # fitted in validate.py
    premium   = cost / loss_ratio x age_band_loading x SI factor
    SI factor = exp(f(ln(SI / SI_ref)))                         # fitted in fit_si_curve.py

USA headline premiums follow ACA rating rules (age curve + tobacco cap) because
US law forbids pricing individual cover on health status; plans have no annual
cap, so the cover level is the metal tier (KFF 2026 market ratios vs benchmark silver).

Profiles are dicts: {"age": 30, "sex": "male", "bmi": 24, "children": 0,
"smoker": False}. A family is {"members": [profile, profile, ...]}; members
under 18 are priced as children.
"""
import json
import math
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = ("USA", "IND", "KEN")
US_REGIONS = ("northeast", "northwest", "southeast", "southwest")
ALIASES = {"US": "USA", "UNITED STATES": "USA", "IN": "IND", "INDIA": "IND", "KE": "KEN", "KENYA": "KEN"}
CURRENT_YEAR = date.today().year


# ---------------------------------------------------------------- loading
@lru_cache(maxsize=1)
def _model():
    return joblib.load(ROOT / "models" / "cost_model.pkl")


@lru_cache(maxsize=1)
def _model_meta():
    return json.loads((ROOT / "models" / "cost_model_meta.json").read_text())


@lru_cache(maxsize=1)
def load_config():
    return json.loads((ROOT / "data" / "country_config.json").read_text(encoding="utf-8"))


def _country(country: str) -> dict:
    code = ALIASES.get(country.strip().upper(), country.strip().upper())
    if code not in COUNTRIES:
        raise ValueError(f"Unsupported country {country!r}; use one of {COUNTRIES}")
    return code, load_config()[code]


def _v(cfg, key):
    return cfg[key]["value"]


# ---------------------------------------------------------------- profiles
def _normalise(p: dict) -> dict:
    smoker = p.get("smoker", False)
    if isinstance(smoker, str):
        smoker = smoker.strip().lower() in ("yes", "y", "true", "1")
    return {
        "age": float(p["age"]),
        "sex": str(p.get("sex", "female")).lower(),
        "bmi": float(p.get("bmi", 25.0)),
        "children": int(p.get("children", 0)),
        "smoker": bool(smoker),
        "region": p.get("region"),
    }


def _members(profile: dict) -> list:
    return [_normalise(m) for m in profile.get("members", [profile])]


def _model_usd(person: dict, notes: list) -> float:
    """Model prediction in USD for one adult; averages over US regions if none given."""
    age = person["age"]
    if age > 64:
        notes.append(f"Age {age:.0f} is above the training range (18-64); priced as 64.")
        age = 64
    regions = [person["region"]] if person["region"] in US_REGIONS else US_REGIONS
    rows = pd.DataFrame([{
        "age": max(age, 18), "sex": person["sex"], "bmi": person["bmi"], "children": person["children"],
        "smoker": "yes" if person["smoker"] else "no", "region": r} for r in regions])
    return float(_model().predict(rows).mean())


def _person_usd(person: dict, cfg: dict, notes: list) -> float:
    if person["age"] < 18:
        # Children were not in the training data: price as a healthy 18-year-old
        # times the CMS child/18 age-curve ratio.
        base = _model_usd({**person, "age": 18, "bmi": 21.7, "smoker": False, "children": 0}, notes)
        return base * _v(cfg, "child_cost_ratio")
    return _model_usd(person, notes)


# ---------------------------------------------------------------- scaling
def scaling_factor(country: str) -> dict:
    """USD-model -> local currency multiplier at the health-spend reference year."""
    code, cfg = _country(country)
    us = load_config()["USA"]["worldbank"]
    wb = cfg["worldbank"]
    h_c_usd = wb["health_spend_per_capita_usd"]["value"]
    fx_h = wb["wb_fx_lcu_per_usd_health_year"]["value"]
    ppp = wb["ppp_conversion_factor_lcu_per_intl_usd"]["value"]
    h_us = us["health_spend_per_capita_usd"]["value"]
    h_c_ppp = h_c_usd * fx_h / ppp                 # international $ per person
    ratio_ppp = h_c_ppp / h_us                     # real health resources vs USA
    return {
        "health_spend_ppp_intl_usd": h_c_ppp,
        "ratio_vs_usa_ppp": ratio_ppp,
        "lcu_per_model_usd": ratio_ppp * ppp,      # == H_c_LCU / H_USA
        "base_year": wb["health_spend_per_capita_usd"]["year"],
    }


def _trend(cfg, from_year, to_year):
    return (1 + _v(cfg, "medical_inflation_pct") / 100) ** (to_year - from_year)


# ---------------------------------------------------------------- public tools
def predict_annual_cost(profile: dict, country: str, year: int = CURRENT_YEAR) -> float:
    """Expected annual medical cost in local currency (whole family if 'members' given)."""
    return _cost_detail(profile, country, year)["annual_cost"]


def _cost_detail(profile, country, year=CURRENT_YEAR):
    code, cfg = _country(country)
    notes = []
    s = scaling_factor(code)
    trend = _trend(cfg, s["base_year"], year)
    calib = _v(cfg, "calibration_factor")
    per_member = []
    for m in _members(profile):
        usd = _person_usd(m, cfg, notes)
        per_member.append(usd * s["lcu_per_model_usd"] * trend * calib)
    return {"country": code, "currency": cfg["currency"], "year": year,
            "annual_cost": float(sum(per_member)), "per_member": per_member,
            "scaling": s, "trend_factor": trend, "calibration_factor": calib, "notes": notes}


def _curve_f(form, coef, x, age):
    if form == "power":
        return coef["b"] * x
    if form == "quad_log":
        return coef["b1"] * x + coef["b2"] * x * x
    return (coef["b"] + coef["c_age_per_decade"] * (age - 40) / 10) * x


def sum_insured_factor(country: str, sum_insured: float = None, age: float = 40) -> float:
    """Premium multiplier for sum_insured relative to the country's reference sum insured.

    Fitted in src/fit_si_curve.py. Beyond the fitted range the curve continues
    along its boundary tangent in log-log space rather than bending further.
    """
    code, cfg = _country(country)
    if sum_insured is None or "sum_insured_curve" not in cfg:
        return 1.0
    if sum_insured <= 0:
        raise ValueError("sum_insured must be positive")
    c = cfg["sum_insured_curve"]["value"]
    ref, (lo, hi) = c["reference_sum_insured"], c["fitted_range"]
    x = math.log(sum_insured / ref)
    xb = min(max(x, math.log(lo / ref)), math.log(hi / ref))
    f = lambda v: _curve_f(c["form"], c["coef"], v, age)
    slope = (f(xb + 1e-6) - f(xb - 1e-6)) / 2e-6
    return math.exp(f(xb) + slope * (x - xb))


def metal_tier_factor(tier: str = "silver") -> float:
    tiers = load_config()["USA"]["metal_tier_factor"]["value"]
    key = str(tier).strip().lower()
    if key not in tiers:
        raise ValueError(f"Unknown metal tier {tier!r}; use one of {tuple(tiers)}")
    return tiers[key]


def _age_loading(cfg, age):
    bands = cfg["age_band_loading"]["value"]
    if age >= 60:
        return bands["60+"]
    if age >= 45:
        return bands["45-59"]
    return bands["0-44"]


@lru_cache(maxsize=1)
def _pool_predictions():
    df = pd.read_csv(ROOT / "data" / "insurance.csv")
    return df, _model().predict(df.drop(columns="charges"))


def _aca_pool(cfg, year):
    """Base rate per unit of (age factor x tobacco factor) for the US risk pool.

    Pool = the insurance.csv population. Total pool premium is set to the sum of
    risk premiums, then shared out using only the factors ACA allows.
    """
    df, preds = _pool_predictions()
    s = scaling_factor("USA")
    risk = preds * s["lcu_per_model_usd"] * _trend(cfg, s["base_year"], year) \
        * _v(cfg, "calibration_factor") / _v(cfg, "loss_ratio")
    tob = _v(cfg["rating_rules"], "tobacco_max_ratio")
    units = sum(_aca_age_factor(cfg, a) * (tob if sm == "yes" else 1.0)
                for a, sm in zip(df["age"], df["smoker"]))
    return float(risk.sum() / units)


def _aca_age_factor(cfg, age):
    curve = cfg["rating_rules"]["age_curve"]["value"]
    age = int(age)
    if age <= 14:
        return curve["0-14"]
    if age >= 64:
        return curve["64+"]
    return curve[str(age)]


def estimate_premium(profile: dict, country: str, sum_insured: float = None,
                     year: int = CURRENT_YEAR, metal_tier: str = "silver") -> dict:
    """Estimated annual premium in local currency plus every assumption used.

    sum_insured is in local currency; None prices the reference sum insured.
    USA ignores sum_insured (ACA plans have no annual cap) and uses metal_tier.
    """
    code, cfg = _country(country)
    detail = _cost_detail(profile, code, year)
    members = _members(profile)
    lr = _v(cfg, "loss_ratio")

    # Risk-based premium: expected claims / loss ratio x age loading x cover-level factor.
    # Floaters are priced like insurers do: eldest member in full + an add-on share of the others.
    expected = detail["annual_cost"]
    eldest = max(range(len(members)), key=lambda i: members[i]["age"])
    addon = _v(cfg, "floater_addon_factor") if "floater_addon_factor" in cfg else 1.0
    floater_cost = detail["per_member"][eldest] + addon * (expected - detail["per_member"][eldest])
    oldest = members[eldest]["age"]
    loading = _age_loading(cfg, oldest)
    si_factor = sum_insured_factor(code, sum_insured, oldest) if code != "USA" else 1.0
    risk_premium = floater_cost / lr * loading * si_factor

    result = {
        "country": code, "currency": cfg["currency"], "symbol": cfg["symbol"], "year": year,
        "annual_premium": round(risk_premium, 0),
        "monthly_premium": round(risk_premium / 12, 0),
        "expected_annual_claims": round(expected, 0),
        "sum_insured": sum_insured,
        "sum_insured_factor": round(si_factor, 4),
        "members": len(members),
        "floater_addon_factor": addon if len(members) > 1 else None,
        "method": "risk-based: expected claims / loss ratio x age-band loading x sum-insured factor",
    }
    if code != "USA":
        result["reference_sum_insured"] = cfg["sum_insured_curve"]["value"]["reference_sum_insured"]

    if code == "USA":
        # ACA community rating: same base rate for everyone in the pool, varied
        # only by the CMS age curve and the tobacco surcharge (max 1.5x).
        base = _aca_pool(cfg, year)
        tob = _v(cfg["rating_rules"], "tobacco_max_ratio")
        max_kids = _v(cfg["rating_rules"], "max_children_rated")
        kids = sorted([m for m in members if m["age"] < 21], key=lambda m: -m["age"])
        rated = [m for m in members if m["age"] >= 21] + kids[:max_kids]
        tier = metal_tier_factor(metal_tier)
        aca = tier * sum(base * _aca_age_factor(cfg, m["age"]) * (tob if m["smoker"] else 1.0) for m in rated)
        result.update({
            "annual_premium": round(aca, 0), "monthly_premium": round(aca / 12, 0),
            "risk_based_premium": round(risk_premium, 0),
            "metal_tier": str(metal_tier).lower(), "metal_tier_factor": tier,
            "method": "ACA community rating: pool base rate x CMS age factor x tobacco factor x metal-tier factor",
        })

    result["assumptions"] = {
        "model": f"{_model_meta()['best_model']} trained on Kaggle insurance.csv (US, ages 18-64)",
        "health_spend_scaling_lcu_per_model_usd": round(detail["scaling"]["lcu_per_model_usd"], 4),
        "health_spend_ratio_vs_usa_ppp": round(detail["scaling"]["ratio_vs_usa_ppp"], 4),
        "health_spend_year": detail["scaling"]["base_year"],
        "medical_inflation_pct": _v(cfg, "medical_inflation_pct"),
        "trend_factor_to_year": round(detail["trend_factor"], 4),
        "calibration_factor": detail["calibration_factor"],
        "loss_ratio": lr,
        "loss_ratio_source": cfg["loss_ratio"]["source"],
        "age_band_loading": loading,
        "cover_level": (cfg["sum_insured_curve"]["value"] if code != "USA"
                        else {"metal_tier_factor_source": cfg["metal_tier_factor"]["source"]}),
        "notes": detail["notes"] + ([cfg["loss_ratio"]["flag"]] if cfg["loss_ratio"].get("flag") else [])
                 + ([] if code == "USA" else ["Scaled estimate from US data; not an insurer quote."]),
    }
    return result


def group_premium(country: str, family_of_four: bool = False, year: int = CURRENT_YEAR) -> float:
    """Average employer (group) premium: the pool's mean expected cost / group loss ratio.

    Group cover is priced on the whole workforce, so the premium is the average
    over a working-age pool (insurance.csv adults) rather than one person.
    Family of four = 2 pool-average adults + 2 children.
    """
    code, cfg = _country(country)
    s = scaling_factor(code)
    to_lcu = s["lcu_per_model_usd"] * _trend(cfg, s["base_year"], year) * _v(cfg, "calibration_factor")
    _, preds = _pool_predictions()
    adult = float(preds.mean()) * to_lcu
    lr = _v(cfg, "loss_ratio_group") if "loss_ratio_group" in cfg else _v(cfg, "loss_ratio")
    if not family_of_four:
        return adult / lr
    child = _person_usd(_normalise({"age": 8}), cfg, []) * to_lcu
    return (2 * adult + 2 * child) / lr


def project_cost(profile: dict, country: str, years: int) -> dict:
    """Annual medical cost today and in `years` years at the country's medical inflation.

    Age is held fixed so the projection isolates inflation; call
    predict_annual_cost with an older age to see the ageing effect.
    """
    code, cfg = _country(country)
    today = predict_annual_cost(profile, code)
    rate = _v(cfg, "medical_inflation_pct") / 100
    path = [round(today * (1 + rate) ** t, 0) for t in range(years + 1)]
    return {"country": code, "currency": cfg["currency"], "medical_inflation_pct": rate * 100,
            "start_year": CURRENT_YEAR, "years": years,
            "cost_today": round(today, 0), "cost_in_n_years": path[-1], "yearly_path": path,
            "source": cfg["medical_inflation_pct"]["source"]}


if __name__ == "__main__":
    demo = {"age": 30, "sex": "male", "bmi": 24, "children": 0, "smoker": False}
    for c, si in (("USA", None), ("IND", 1_000_000), ("KEN", 1_000_000)):
        r = estimate_premium(demo, c, si)
        print(c, r["currency"], "cost", r["expected_annual_claims"], "premium", r["annual_premium"])
