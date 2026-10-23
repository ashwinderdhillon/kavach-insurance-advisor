"""Term life premium estimates from WHO life tables (chatbot tool layer).

    q_pop(x)     annual death probability from WHO nqx (5-year groups, latest year)
    q_ns, q_s    split of population mortality into never/current smokers:
                 q_ns = q_pop / (1 + p x (RR - 1)),  q_s = RR x q_ns
                 p = WHO smoking prevalence by sex, RR = Jha et al. NEJM 2013
    q_insured    = insured_mortality_factor x q_ns or q_s   (fitted in validate_term.py;
                   people who buy term cover die less often than the population)
    net premium  = PV(expected death claims over the term) / PV(premiums paid while alive)
                   discounted at the country's policy interest rate
    premium      = net premium / life loss ratio + fixed yearly policy fee
"""
import json
import math
from functools import lru_cache
from pathlib import Path

from src.premium import _country

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _who():
    return json.loads((ROOT / "data" / "who_life_tables.json").read_text())


def _v(node):
    return node["value"]


def _q_population(country, sex, age):
    """Annual probability of death at integer age from the WHO abridged life table."""
    table = _who()[country]
    if age >= 85:
        m = table["nmx_85plus"][sex]
        return 1 - math.exp(-m)
    groups = sorted(int(k) for k in table["nqx"][sex] if int(k) < 85)
    start = max(g for g in groups if g <= age)
    width = {0: 1, 1: 4}.get(start, 5)
    nqx = table["nqx"][sex][str(start)]
    return 1 - (1 - nqx) ** (1 / width)


def _q_insured(code, cfg, sex, smoker, age):
    tl = cfg["term_life"]
    rr = _v(tl["smoker_mortality_ratio"])[sex]
    p = _who()[code]["smoking_prevalence"][sex]["value"]
    q_ns = _q_population(code, sex, age) / (1 + p * (rr - 1))
    q = q_ns * rr if smoker else q_ns
    return min(1.0, q * _v(tl["insured_mortality_factor"]))


def estimate_term_premium(age: int, sex: str, smoker: bool, cover: float, term_years: int,
                          country: str) -> dict:
    """Level annual premium for a pure term policy, in local currency, plus assumptions.

    cover is the sum assured in local currency; premiums are paid yearly in advance
    while the insured is alive, and the death benefit is paid at the end of the year of death.
    """
    code, cfg = _country(country)
    sex = "female" if str(sex).lower().startswith("f") else "male"
    if isinstance(smoker, str):
        smoker = smoker.strip().lower() in ("yes", "y", "true", "1")
    age, term_years = int(age), int(term_years)
    if not (18 <= age <= 75) or term_years < 1 or age + term_years > 100:
        raise ValueError("Age must be 18-75 and the term must end by age 100")

    tl = cfg["term_life"]
    i = cfg["policy_interest_rate_pct"]["value"] / 100
    v = 1 / (1 + i)
    alive, pv_claims, pv_annuity = 1.0, 0.0, 0.0
    for t in range(term_years):
        q = _q_insured(code, cfg, sex, smoker, age + t)
        pv_annuity += alive * v ** t
        pv_claims += alive * q * v ** (t + 1)
        alive *= 1 - q
    net = cover * pv_claims / pv_annuity
    lr = _v(tl["life_loss_ratio"])
    fee = _v(tl["policy_fee"])
    gross = net / lr + fee

    return {
        "country": code, "currency": cfg["currency"], "symbol": cfg["symbol"],
        "annual_premium": round(gross, 0), "monthly_premium": round(gross / 12, 0),
        "expected_annual_claims_cost": round(net, 0),
        "probability_of_death_during_term": round(1 - alive, 5),
        "cover": cover, "term_years": term_years,
        "assumptions": {
            "mortality": f"WHO GHO life table {_who()[code]['year']} ({sex}), split by smoking status",
            "smoking_prevalence": _who()[code]["smoking_prevalence"][sex]["value"],
            "smoker_mortality_ratio": _v(tl["smoker_mortality_ratio"])[sex],
            "insured_mortality_factor": _v(tl["insured_mortality_factor"]),
            "discount_rate_pct": i * 100,
            "life_loss_ratio": lr,
            "policy_fee": fee,
            "notes": ["Population life tables adjusted to insured lives with a single calibrated factor; "
                      "not an insurer quote. Educational estimate, not licensed advice."],
        },
    }


if __name__ == "__main__":
    for c, cover in (("USA", 500_000), ("IND", 10_000_000), ("KEN", 5_000_000)):
        r = estimate_term_premium(30, "male", False, cover, 20, c)
        print(c, r["currency"], r["annual_premium"], "P(death in term)", r["probability_of_death_during_term"])
