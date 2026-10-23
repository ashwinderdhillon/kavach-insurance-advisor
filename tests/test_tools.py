"""Tests for the chatbot tools (src/tools.py) with realistic profiles from all three countries."""
import inspect
import json
import math
from pathlib import Path

import pytest

from src import premium, term_life, tools

ROOT = Path(__file__).resolve().parents[1]

# Personas: two per country.
MARIA = dict(country="USA", age=34, sex="female", bmi=27, smoker=False, annual_income=42_000, state="CA",
             household_size=3, employment_type="salaried")  # single mother, 2 kids, no employer plan
TOM = dict(country="USA", age=58, sex="male", bmi=31, smoker=True, annual_income=95_000, state="TX")
PRIYA = dict(country="IND", age=32, sex="female", bmi=23, smoker=False, annual_income=1_800_000,
             employment_type="salaried")  # Bengaluru, married, 1 child, Rs 40 L home loan
RAMESH = dict(country="IND", age=45, sex="male", bmi=21, smoker=False, annual_income=120_000,
              employment_type="farmer")  # rural Maharashtra
WANJIRU = dict(country="KEN", age=29, sex="female", bmi=24, smoker=False, annual_income=360_000,
               employment_type="informal")  # Nairobi boda-boda rider
OTIENO = dict(country="KEN", age=41, sex="male", bmi=28, smoker=False, annual_income=1_200_000,
              employment_type="salaried")  # 3 children


def _ok(r):
    """Every tool result is a JSON-serialisable dict with sources and the disclaimer."""
    assert isinstance(r, dict) and "error" not in r, r
    json.dumps(r)
    assert r["sources"] and r["disclaimer"]
    return r


# ---------------------------------------------------------------- sum-insured curve (the FIX FIRST item)
@pytest.mark.parametrize("country", ["IND", "KEN"])
def test_premium_rises_with_sum_insured(country):
    sis = [300_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000, 20_000_000]
    prems = [tools.predict_health_premium(35, "male", 25, 0, False, country, sum_insured=s)["annual_premium"]
             for s in sis]
    assert all(b > a for a, b in zip(prems, prems[1:])), prems


@pytest.mark.parametrize("country", ["IND", "KEN"])
def test_si_factor_is_one_at_reference_and_concave_in_level(country):
    ref = premium.load_config()[country]["sum_insured_curve"]["value"]["reference_sum_insured"]
    assert premium.sum_insured_factor(country, ref) == pytest.approx(1.0)
    # 10x the cover costs much less than 10x the premium (claims near the cap are rare).
    assert 1.3 < premium.sum_insured_factor(country, 10 * ref) < 4


def test_india_curve_matches_published_ratios():
    """Rs 1 Cr vs Rs 10 L: brokers show 2.2-2.6x for age 30 (nyvo); the fitted curve must land near that."""
    r = premium.sum_insured_factor("IND", 10_000_000) / premium.sum_insured_factor("IND", 1_000_000)
    assert 1.8 < r < 2.8


def test_kenya_curve_reproduces_old_mutual_table():
    """Old Mutual AfyaImara principal 30-40: KES 10M costs 1.84x the KES 1M premium (64,692 / 35,223)."""
    r = premium.sum_insured_factor("KEN", 10_000_000) / premium.sum_insured_factor("KEN", 1_000_000)
    assert r == pytest.approx(64_692 / 35_223, rel=0.15)


def test_usa_metal_tiers_follow_market_prices():
    p = {t: tools.predict_health_premium(40, "male", 25, 0, False, "USA", metal_tier=t)["annual_premium"]
         for t in ("bronze", "silver", "gold", "platinum")}
    assert p["bronze"] < p["gold"] <= p["silver"] < p["platinum"]  # silver loading: gold ~ silver
    assert p["bronze"] / p["silver"] == pytest.approx(456 / 625, rel=0.01)
    # sum_insured is ignored in the USA (no annual cap under the ACA)
    assert tools.predict_health_premium(40, "male", 25, 0, False, "USA", sum_insured=1)["annual_premium"] == p["silver"]


def test_si_curve_fit_quality_and_provenance():
    q = json.loads((ROOT / "data" / "sum_insured_quotes.json").read_text(encoding="utf-8"))
    cfg = premium.load_config()
    for code in ("IND", "KEN"):
        curve = cfg[code]["sum_insured_curve"]
        ids = {s["id"] for s in q[code]}
        assert set(curve["fitted_on"]) == ids
        chosen = curve["fit_stats"][curve["value"]["form"]]
        assert chosen["monotone"] and chosen["loso_mape_pct"] < 15
        for s in q[code]:
            assert s["source_url"].startswith("http") and len(s["points"]) >= 2


# ---------------------------------------------------------------- 1-2 premium tools
@pytest.mark.parametrize("p,si", [(MARIA, None), (TOM, None), (PRIYA, 1_000_000), (RAMESH, 500_000),
                                  (WANJIRU, 500_000), (OTIENO, 1_000_000)])
def test_predict_health_premium_personas(p, si):
    r = _ok(tools.predict_health_premium(p["age"], p["sex"], p["bmi"], 0, p["smoker"], p["country"], sum_insured=si))
    assert r["annual_premium"] > 0 and r["monthly_premium"] == pytest.approx(r["annual_premium"] / 12, abs=1)
    assert r["currency"] == {"USA": "USD", "IND": "INR", "KEN": "KES"}[p["country"]]


def test_family_floater_costs_more_than_individual():
    solo = tools.predict_health_premium(32, "female", 23, 1, False, "IND", 1_500_000)
    fam = _ok(tools.predict_health_premium(32, "female", 23, 1, False, "IND", 1_500_000,
                                           family_members=[{"age": 34, "sex": "male"}, {"age": 4}]))
    assert fam["annual_premium"] > solo["annual_premium"] and fam["policy_type"] == "family floater"


@pytest.mark.parametrize("p,cover", [(MARIA, 500_000), (PRIYA, 20_000_000), (OTIENO, 5_000_000)])
def test_predict_term_premium_matches_module(p, cover):
    r = _ok(tools.predict_term_premium(p["age"], p["sex"], p["smoker"], cover, 25, p["country"]))
    direct = term_life.estimate_term_premium(p["age"], p["sex"], p["smoker"], cover, 25, p["country"])
    assert r["annual_premium"] == direct["annual_premium"]
    assert r["cover_ends_at_age"] == p["age"] + 25


# ---------------------------------------------------------------- 3-5 projections
def test_cover_decay_matches_claude_md_example():
    """CLAUDE.md: Rs 5 lakh at ~13% medical inflation is worth about Rs 1.5 lakh in 10 years."""
    r = _ok(tools.project_cover_decay(500_000, "IND", 10))
    assert len(r["series"]) == 11 and r["series"][0]["real_value_today_prices"] == 500_000
    assert 140_000 < r["real_value_after_years"] < 160_000
    vals = [s["real_value_today_prices"] for s in r["series"]]
    assert all(b < a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("country", ["USA", "KEN"])
def test_cover_decay_other_countries(country):
    r = _ok(tools.project_cover_decay(100_000, country, 20))
    assert r["cover_needed_after_years"] > 100_000 > r["real_value_after_years"]


def test_term_cover_needed_formula_by_hand():
    cfg = premium.load_config()["IND"]
    i = cfg["policy_interest_rate_pct"]["value"] / 100
    pi = cfg["worldbank"]["cpi_inflation_pct"]["value"] / 100
    r_ = (1 + i) / (1 + pi) - 1
    expect = 600_000 * (1 - (1 + r_) ** -20) / r_ + 4_000_000 + 1_500_000 - 500_000 - 2_000_000
    r = _ok(tools.calc_term_cover_needed(600_000, 20, 4_000_000, 1_500_000, 500_000, 2_000_000, "IND"))
    assert r["cover_needed"] == pytest.approx(expect, abs=1)
    assert r["cover_needed_rounded_up"] >= r["cover_needed"]


def test_term_cover_needed_floors_at_zero_and_country_rates_differ():
    assert tools.calc_term_cover_needed(10_000, 5, liquid_assets=10_000_000, country="USA")["cover_needed"] == 0
    us = tools.calc_term_cover_needed(30_000, 15, country="USA")["real_discount_rate_pct"]
    ke = tools.calc_term_cover_needed(30_000, 15, country="KEN")["real_discount_rate_pct"]
    assert ke > us  # Kenya's policy rate is far above its inflation


@pytest.mark.parametrize("p,cover", [(PRIYA, 15_000_000), (TOM, 250_000), (OTIENO, 5_000_000)])
def test_cost_of_delay(p, cover):
    r = _ok(tools.cost_of_delay(p["age"], 5, p["country"],
                                {"sex": p["sex"], "smoker": p["smoker"], "cover": cover, "term_years": 15}))
    t = r["term_life"]
    assert t["annual_premium_if_delayed"] > t["annual_premium_now"]
    assert t["extra_over_policy_term"] == t["extra_per_year"] * 10
    assert 0 < t["probability_of_death_while_uninsured"] < 1
    assert "net_cost_of_delay_present_value" in t
    assert r["health"]["entry_premium_if_delayed"] > r["health"]["entry_premium_now"]


def test_cost_of_delay_india_waiting_period_shift():
    r = tools.cost_of_delay(30, 3, "IND", {"cover": 10_000_000, "term_years": 30})
    h = r["health"]
    assert h["pre_existing_waiting_ends_year_if_delayed"] - h["pre_existing_waiting_ends_year_now"] == 3


@pytest.mark.parametrize("country", ["USA", "IND", "KEN"])
def test_risk_of_waiting_present_and_grows_with_delay(country):
    short = tools.cost_of_delay(35, 2, country, {"term_years": 25})["risk_of_waiting"]
    long_ = tools.cost_of_delay(35, 10, country, {"term_years": 25})
    p = long_["risk_of_waiting"]["probability_of_developing"]
    assert 0 < short["probability_of_developing"]["either"] < p["either"] < 1
    assert p["either"] >= max(p["diabetes"], p["hypertension"])
    assert long_["risk_of_waiting"]["consequences"]["health_insurance"]
    assert all(s["source"] for s in long_["risk_of_waiting"]["sources"])


def test_cost_of_delay_never_recommends_waiting():
    """Young healthy buyer: the pure premium maths favours waiting, so the insurability risk must sit beside it."""
    r = tools.cost_of_delay(32, 5, "IND", {"sex": "female", "cover": 20_000_000, "term_years": 28})
    assert r["term_life"]["net_cost_of_delay_present_value"] < 0
    assert r["price_comparison_favours_waiting"] is True
    assert r["verdict"].startswith("Buy now") and "not guaranteed" in r["verdict"]
    assert "%" in r["verdict"] and r["risk_of_waiting"]["probability_of_developing"]["either"] > 0
    for p in (MARIA, TOM, PRIYA, RAMESH, WANJIRU, OTIENO):
        v = tools.cost_of_delay(p["age"], 3, p["country"], {"term_years": 15})["verdict"]
        assert v.startswith("Buy now") and "better to wait" not in v.lower()


def test_family_floater_is_eldest_plus_addon():
    """Floater = eldest member's premium + addon x each other member's premium (Care Supreme illustrations)."""
    fam = [{"age": 35, "sex": "male", "bmi": 24}, {"age": 34, "sex": "female", "bmi": 24}, {"age": 5}]
    for code in ("IND", "KEN"):
        d = premium.load_config()[code]["floater_addon_factor"]["value"]
        solo = [premium.estimate_premium(m, code, 1_000_000)["annual_premium"] for m in fam]
        floater = premium.estimate_premium({"members": fam}, code, 1_000_000)
        assert floater["annual_premium"] == pytest.approx(solo[0] + d * (solo[1] + solo[2]), abs=3)
        assert floater["annual_premium"] < sum(solo) and floater["floater_addon_factor"] == d
    # USA: ACA family premium is the sum of members, no floater discount
    us = [premium.estimate_premium(m, "USA")["annual_premium"] for m in fam]
    assert premium.estimate_premium({"members": fam}, "USA")["annual_premium"] == pytest.approx(sum(us), abs=3)


def test_cost_of_delay_rejects_delay_longer_than_term():
    r = tools.call_tool("cost_of_delay", {"age": 30, "delay_years": 10, "country": "IND",
                                          "profile": {"term_years": 5}})
    assert "error" in r


# ---------------------------------------------------------------- 6 schemes
def _status(r, sid):
    return next(s["status"] for s in r["schemes"] if s["id"] == sid)


def test_india_schemes():
    priya = _ok(tools.check_govt_schemes("IND", 32, 1_800_000, "salaried", False, has_bank_account=True))
    assert _status(priya, "pmjay") == "not_eligible"
    assert _status(priya, "pmjjby") == "eligible" and _status(priya, "pmsby") == "eligible"
    ramesh = tools.check_govt_schemes("IND", 45, 120_000, "farmer", True)
    assert _status(ramesh, "pmjay") == "check"          # SECC listing must be confirmed on the portal
    assert _status(ramesh, "pmjjby") == "check"         # bank account unknown
    senior = tools.check_govt_schemes("IND", 72, 2_500_000, "retired", False)
    assert _status(senior, "pmjay") == "eligible"       # Vay Vandana: 70+ regardless of income
    assert _status(senior, "pmjjby") == "not_eligible" and _status(senior, "pmsby") == "not_eligible"


def test_usa_schemes():
    ca = _ok(tools.check_govt_schemes("USA", 30, 18_000, "informal", False, household_size=1, state="CA"))
    assert _status(ca, "medicaid_expansion") == "eligible"        # 18,000 / 15,960 = 113% FPL
    tx = tools.check_govt_schemes("USA", 30, 18_000, "informal", False, household_size=1, state="TX")
    assert _status(tx, "medicaid_expansion") == "not_eligible"    # Texas did not expand
    maria = tools.check_govt_schemes("USA", MARIA["age"], MARIA["annual_income"], "salaried", False,
                                     household_size=3, state="CA", has_affordable_employer_coverage=False)
    assert _status(maria, "aca_premium_tax_credit") == "eligible"
    assert _status(maria, "medicare") == "not_eligible"          # 34, no disability mentioned
    assert maria["income_pct_of_poverty_line"]["2025"] == pytest.approx(100 * 42_000 / (15_650 + 2 * 5_500), abs=0.1)
    tom = tools.check_govt_schemes("USA", TOM["age"], TOM["annual_income"], "salaried", False, state="TX")
    assert _status(tom, "aca_premium_tax_credit") == "not_eligible"  # above 400% FPL: the cliff is back
    assert _status(tools.check_govt_schemes("USA", 67, 30_000, "retired", False), "medicare") == "eligible"


def test_kenya_sha_and_contribution():
    w = _ok(tools.check_govt_schemes("KEN", WANJIRU["age"], WANJIRU["annual_income"], "informal", False))
    sha = w["schemes"][0]
    assert sha["status"] == "eligible"
    assert sha["estimated_contribution"]["monthly_kes"] == pytest.approx(0.0275 * 30_000)
    low = tools.check_govt_schemes("KEN", 60, 60_000, "farmer", True)["schemes"][0]
    assert low["estimated_contribution"]["monthly_kes"] == 300  # statutory minimum


def test_every_scheme_rule_is_sourced():
    data = json.loads((ROOT / "data" / "schemes.json").read_text(encoding="utf-8"))
    for code in premium.COUNTRIES:
        for s in data[code]["schemes"]:
            for path in s["paths"]:
                for rule in path["rules"]:
                    assert rule["source"] and rule["source_url"].startswith("http"), (s["id"], rule)
                    assert rule["strength"] in ("hard", "proxy")


# ---------------------------------------------------------------- 7 recommendations
def test_recommend_needs_more_info_when_profile_thin():
    r = _ok(tools.recommend_cover_types({"country": "IND", "age": 30}))
    assert r["status"] == "need_more_info" and r["recommendations"] == [] and "dependants" in r["missing"]


def _priya_profile():
    return {"country": "IND", "age": 32, "dependants": 2, "annual_income": 1_800_000, "employment_type": "salaried",
            "has_health_cover": True, "health_cover_source": "employer", "has_life_cover": False,
            "debts": 4_000_000, "owns_vehicle": True, "parents_dependent": True, "parents_age": 62,
            "rural": False, "has_bank_account": True}


def test_recommend_priya_follows_priority_order():
    r = _ok(tools.recommend_cover_types(_priya_profile()))
    cats = [x["category"] for x in r["recommendations"]]
    prios = [x["priority"] for x in r["recommendations"]]
    assert prios == sorted(prios) and cats[0] == "Health"
    assert "Life" in cats and "Assets" in cats
    assert any("employer" in x["reason"].lower() for x in r["recommendations"])   # status-quo bias alert
    assert any("parents" in x["product"].lower() for x in r["recommendations"])
    assert "ULIP" in r["avoid"][0]["product"]
    assert "PMJJBY (Pradhan Mantri Jeevan Jyoti Bima Yojana)" in r["schemes_first"]["eligible"]


def test_recommend_kenyan_rider_and_indian_farmer():
    rider = {"country": "KEN", "age": 29, "dependants": 1, "annual_income": 360_000, "employment_type": "informal",
             "has_health_cover": False, "has_life_cover": False, "debts": 0, "owns_vehicle": True,
             "occupation_risk": "high"}
    r = tools.recommend_cover_types(rider)["recommendations"]
    inc = next(x for x in r if x["category"] == "Income protection")
    assert inc["priority"] == 2 and inc["urgency"] == "high"
    assert any(x["category"] == "Micro / funeral" for x in r)
    farmer = {"country": "IND", "age": 45, "dependants": 3, "annual_income": 120_000, "employment_type": "farmer",
              "has_health_cover": False, "has_life_cover": False, "debts": 50_000, "owns_vehicle": False}
    f = tools.recommend_cover_types(farmer)["recommendations"]
    live = next(x for x in f if x["category"] == "Livelihood")
    assert "PMFBY" in live["product"] and "basis risk" in live["reason"]


def test_recommend_usa_young_single_no_life():
    p = {"country": "USA", "age": 26, "dependants": 0, "annual_income": 55_000, "employment_type": "salaried",
         "has_health_cover": False, "has_life_cover": False, "debts": 0, "owns_vehicle": False, "rents": True}
    cats = [x["category"] for x in tools.recommend_cover_types(p)["recommendations"]]
    assert "Life" not in cats and cats[0] == "Health" and "Assets" in cats  # renters


# ---------------------------------------------------------------- 8 protection score
def test_protection_score_india_covered_vs_uncovered():
    prof = {"country": "IND", "age": 32, "annual_income": 1_800_000, "annual_expenses": 700_000, "dependants": 2,
            "years_support_needed": 20, "debts": 4_000_000, "liquid_assets": 500_000, "family_size": 3}
    good = _ok(tools.protection_score(prof, {"health_sum_insured": 2_500_000, "life_cover": 20_000_000,
                                             "annual_premiums": 45_000, "health_premium": 30_000}))
    bad = tools.protection_score(prof, {"health_sum_insured": 300_000, "life_cover": 0, "annual_premiums": 120_000,
                                        "policy_features": {"plan_type": "endowment", "room_rent_cap": "1% of SI"}})
    assert 0 <= bad["score_today"] < good["score_today"] <= 100
    b = good["breakdown"]
    assert b["health_adequacy"]["year_10"] <= b["health_adequacy"]["today"]   # cover decay
    assert bad["breakdown"]["red_flags"]["count"] == 2
    assert good["score_year_10"] <= good["score_today"]


def test_protection_score_affordability_flags_lapse_risk_kenya():
    prof = {"country": "KEN", "age": 29, "annual_income": 360_000, "dependants": 0, "family_size": 1}
    r = _ok(tools.protection_score(prof, {"health_sum_insured": 1_000_000, "annual_premiums": 60_000}))
    a = r["breakdown"]["affordability"]
    assert a["lapse_risk"] and a["today"] < 100 and a["premium_to_income_year_10"] > a["premium_to_income_today"]
    assert r["breakdown"]["life_adequacy"]["today"] == 100  # no dependants or debts


def test_protection_score_usa_uses_oop_max():
    prof = {"country": "USA", "age": 34, "annual_income": 42_000, "annual_expenses": 30_000, "dependants": 2,
            "years_support_needed": 16, "liquid_assets": 3_000, "family_size": 3}
    insured = tools.protection_score(prof, {"has_health_plan": True, "life_cover": 300_000, "annual_premiums": 3_000})
    none = tools.protection_score(prof, {"has_health_plan": False})
    assert insured["breakdown"]["health_adequacy"]["today"] > 50 > none["breakdown"]["health_adequacy"]["today"]


def test_score_weights_sum_to_one():
    for code in premium.COUNTRIES:
        w = premium.load_config()[code]["advisor"]["protection_score_weights"]["value"]
        assert sum(w.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- 9 red flags
def test_red_flags_india_bad_policy():
    r = _ok(tools.check_red_flags({"room_rent_cap": "1% of sum insured per day", "copay_pct": 30,
                                   "ped_wait_months": 48, "disease_sublimits": True, "plan_type": "ULIP"}, "IND"))
    ids = {f["id"] for f in r["flags"]}
    assert {"room_rent_cap", "heavy_copay", "ped_wait_above_legal_max", "disease_sublimits",
            "investment_linked"} <= ids
    assert r["red_flag_score"] == 0


def test_red_flags_clean_term_plan():
    r = tools.check_red_flags({"plan_type": "term", "copay_pct": 0, "ped_wait_months": 24}, "IND")
    assert r["flags"] == [] and r["red_flag_score"] == 100


def test_red_flags_usa_and_kenya():
    us = tools.check_red_flags({"oop_max": 15_000, "deductible": 7_500, "emergency_fund": 2_000}, "USA")
    assert {"not_aca_compliant", "deductible_above_savings"} <= {f["id"] for f in us["flags"]}
    ke = tools.check_red_flags({"plan_type": "whole life", "restoration": False}, "KEN")
    assert {"investment_linked", "no_restoration"} == {f["id"] for f in ke["flags"]}


# ---------------------------------------------------------------- 10 accident + motor
def test_accident_cover_by_country():
    ind = _ok(tools.accident_cover_needed(1_800_000, "high", "IND"))
    assert ind["cover_target"] == 18_000_000 and "PMSBY" in ind["base_layer"]
    assert tools.accident_cover_needed(1_800_000, "low", "IND")["cover_target"] == 9_000_000
    ke = tools.accident_cover_needed(360_000, "high", "KEN")
    assert ke["cover_target"] == 8 * 360_000 and ke["priority_rank"] == 2   # WIBA 96 months
    us = tools.accident_cover_needed(60_000, "medium", "USA")
    assert us["monthly_benefit_target"] == pytest.approx(3_000)            # 60% of $5,000/month


def test_accident_rejects_bad_risk():
    assert "error" in tools.call_tool("accident_cover_needed",
                                      {"annual_income": 1, "occupation_risk": "extreme", "country": "IND"})


def test_motor_check_india_depreciation():
    r = _ok(tools.motor_check("car", 2.5, "IND", vehicle_value=800_000))
    assert r["idv_depreciation_pct"] == 30 and r["estimated_idv"] == 560_000
    assert any("Zero-depreciation" in a for a in r["advice"])
    old = tools.motor_check("two_wheeler", 7, "IND")
    assert old["idv_depreciation_pct"] is None and not any("Zero-depreciation" in a for a in old["advice"])


def test_motor_check_kenya_and_usa():
    ke = _ok(tools.motor_check("motorcycle", 2, "KEN"))
    assert ke["third_party"]["mandatory"] and "Cap. 405" in ke["advice"][0]
    us = _ok(tools.motor_check("car", 12, "USA"))
    assert any("third-party only" in a for a in us["advice"])


# ---------------------------------------------------------------- schemas + dispatch
def _schemas():
    return json.loads((ROOT / "data" / "tool_schemas.json").read_text(encoding="utf-8"))["tools"]


def test_schemas_cover_every_tool_and_match_signatures():
    schemas = {s["name"]: s for s in _schemas()}
    assert set(schemas) == set(tools.TOOLS)
    for name, fn in tools.TOOLS.items():
        sig = inspect.signature(fn).parameters
        props = schemas[name]["parameters"]["properties"]
        required = set(schemas[name]["parameters"]["required"])
        assert set(props) == set(sig), name
        no_default = {p for p, v in sig.items() if v.default is inspect.Parameter.empty}
        assert no_default <= required <= set(props), name
        assert len(schemas[name]["description"]) > 40


def test_call_tool_dispatch_and_errors():
    r = tools.call_tool("project_cover_decay", {"cover": 1_000_000, "country": "KEN", "years": 5})
    assert r["tool"] == "project_cover_decay"
    assert "error" in tools.call_tool("nope", {})
    assert "error" in tools.call_tool("predict_term_premium", {"age": 30, "sex": "male", "smoker": False,
                                                               "cover": 1, "term_years": 10, "country": "France"})
