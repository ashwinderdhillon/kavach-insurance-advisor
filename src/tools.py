"""Tools the chatbot calls (function calling). Every tool returns a plain,
JSON-serialisable dict holding the result plus the assumptions and sources used.

Country numbers come from data/country_config.json; scheme rules from
data/schemes.json; schemas for the LLM are in data/tool_schemas.json.
Use call_tool(name, args) from the agent loop: it validates the name and turns
bad input into {"error": ...} instead of an exception.
"""
import json
import math
from functools import lru_cache
from pathlib import Path

from src import premium, term_life
from src.premium import CURRENT_YEAR, _country

ROOT = Path(__file__).resolve().parents[1]
DISCLAIMER = "Educational guidance, not licensed insurance advice. Confirm terms with the insurer before buying."
INVESTMENT_LINKED = {"endowment", "money_back", "ulip", "whole_life", "return_of_premium", "guaranteed_savings"}


# ---------------------------------------------------------------- helpers
@lru_cache(maxsize=1)
def _schemes():
    return json.loads((ROOT / "data" / "schemes.json").read_text(encoding="utf-8"))


def _v(node):
    return node["value"]


def _adv(cfg, key):
    return cfg["advisor"][key]


def _src(node, what):
    """One citation entry from a config field."""
    return {"what": what, "source": node.get("source"), "url": node.get("source_url"),
            "year": node.get("year"), "kind": node.get("source_type", "data")}


def _clean(obj):
    """Make numpy scalars and tuples JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _bool(x):
    if isinstance(x, str):
        return x.strip().lower() in ("yes", "y", "true", "1")
    return bool(x)


def _rates(cfg):
    i = _v(cfg["policy_interest_rate_pct"]) / 100
    pi = _v(cfg["worldbank"]["cpi_inflation_pct"]) / 100
    return i, pi


def _wrap(tool, code, cfg, result, sources, assumptions=None, notes=None):
    out = {"tool": tool, "country": code, "currency": cfg["currency"], "symbol": cfg["symbol"], **result,
           "assumptions": assumptions or {}, "sources": sources, "notes": notes or [], "disclaimer": DISCLAIMER}
    return _clean(out)


# ---------------------------------------------------------------- 1. health premium
def predict_health_premium(age, sex, bmi, children, smoker, country, sum_insured=None,
                           metal_tier="silver", family_members=None):
    """Annual health premium. sum_insured in local currency (India/Kenya); metal_tier for the USA."""
    code, cfg = _country(country)
    person = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker}
    profile = {"members": [person] + list(family_members)} if family_members else person
    r = premium.estimate_premium(profile, code, sum_insured, metal_tier=metal_tier)
    keys = ("annual_premium", "monthly_premium", "expected_annual_claims", "sum_insured", "sum_insured_factor",
            "reference_sum_insured", "metal_tier", "metal_tier_factor", "risk_based_premium", "method")
    result = {k: r[k] for k in keys if k in r}
    result["policy_type"] = "family floater" if family_members else "individual"
    sources = [_src(cfg["worldbank"]["health_spend_per_capita_usd"], "health spend scaling"),
               _src(cfg["medical_inflation_pct"], "medical inflation (trend to today)"),
               _src(cfg["loss_ratio"], "loss ratio"),
               _src(cfg["calibration_factor"], "calibration to market quotes")]
    sources.append(_src(cfg["metal_tier_factor"], "metal tier factor") if code == "USA"
                   else _src(cfg["sum_insured_curve"], "sum-insured scaling curve"))
    return _wrap("predict_health_premium", code, cfg, result, sources, r["assumptions"], r["assumptions"]["notes"])


# ---------------------------------------------------------------- 2. term premium
def predict_term_premium(age, sex, smoker, cover, term_years, country):
    code, cfg = _country(country)
    r = term_life.estimate_term_premium(age, sex, smoker, cover, term_years, code)
    tl = cfg["term_life"]
    result = {k: r[k] for k in ("annual_premium", "monthly_premium", "expected_annual_claims_cost",
                                "probability_of_death_during_term", "cover", "term_years")}
    result["cover_ends_at_age"] = int(age) + int(term_years)
    who = term_life._who()[code]
    sources = [{"what": "mortality", "source": r["assumptions"]["mortality"], "url": who["source_url_nqx"],
                "year": who["year"], "kind": "data"},
               _src(tl["smoker_mortality_ratio"], "smoker mortality ratio"),
               _src(tl["insured_mortality_factor"], "insured-lives adjustment"),
               _src(tl["life_loss_ratio"], "life loss ratio"),
               _src(cfg["policy_interest_rate_pct"], "discount rate")]
    return _wrap("predict_term_premium", code, cfg, result, sources, r["assumptions"], r["assumptions"]["notes"])


# ---------------------------------------------------------------- 3. cover decay
def project_cover_decay(cover, country, years=10):
    """Real value of a fixed health cover as medical prices rise: C / (1 + pi_m)^t."""
    code, cfg = _country(country)
    if cover <= 0 or years < 1:
        raise ValueError("cover must be positive and years >= 1")
    pm = _v(cfg["medical_inflation_pct"]) / 100
    series = [{"year": CURRENT_YEAR + t, "t": t,
               "real_value_today_prices": round(cover / (1 + pm) ** t, 0),
               "cover_needed_to_match_today": round(cover * (1 + pm) ** t, 0)} for t in range(int(years) + 1)]
    end = series[-1]
    result = {"cover": cover, "years": int(years), "medical_inflation_pct": pm * 100,
              "real_value_after_years": end["real_value_today_prices"],
              "value_lost_pct": round(100 * (1 - 1 / (1 + pm) ** years), 1),
              "cover_needed_after_years": end["cover_needed_to_match_today"],
              "half_life_years": round(math.log(2) / math.log(1 + pm), 1),
              "series": series}
    notes = ["Formula: C_real = C / (1 + medical inflation)^t (CLAUDE.md section 5).",
             "Medical inflation is the insurer cost trend (price + use), so it overstates pure price rises."]
    return _wrap("project_cover_decay", code, cfg, result,
                 [_src(cfg["medical_inflation_pct"], "medical inflation")], notes=notes)


# ---------------------------------------------------------------- 4. term cover needed
def calc_term_cover_needed(annual_expenses, years_needed, debts=0, goals=0, liquid_assets=0,
                           existing_cover=0, country="IND"):
    """Needs-based life cover: E x annuity(r, n) + D + G - L - X, r = (1+i)/(1+pi) - 1."""
    code, cfg = _country(country)
    if annual_expenses < 0 or years_needed < 0:
        raise ValueError("annual_expenses and years_needed must be non-negative")
    i, pi = _rates(cfg)
    r = (1 + i) / (1 + pi) - 1
    n = float(years_needed)
    annuity = n if abs(r) < 1e-9 else (1 - (1 + r) ** -n) / r
    income_need = annual_expenses * annuity
    gross = income_need + debts + goals
    need = max(0.0, gross - liquid_assets - existing_cover)
    step = 10 ** max(0, int(math.log10(need)) - 1) if need >= 10 else 1
    result = {"cover_needed": round(need, 0),
              "cover_needed_rounded_up": math.ceil(need / step) * step if need else 0,
              "breakdown": {"income_replacement": round(income_need, 0), "debts": debts, "goals": goals,
                            "less_liquid_assets": -liquid_assets, "less_existing_cover": -existing_cover},
              "real_discount_rate_pct": round(r * 100, 3), "annuity_factor": round(annuity, 3),
              "inputs": {"annual_expenses": annual_expenses, "years_needed": years_needed}}
    assumptions = {"interest_rate_pct": i * 100, "cpi_inflation_pct": round(pi * 100, 3),
                   "formula": "Need = E x (1 - (1+r)^-n) / r + D + G - L - X (CLAUDE.md section 5)"}
    notes = ["Expenses are assumed to rise with general inflation and the payout to earn the policy rate.",
             "Use the family's expenses WITHOUT the earner (their own costs drop out)."]
    sources = [_src(cfg["policy_interest_rate_pct"], "interest rate"),
               _src(cfg["worldbank"]["cpi_inflation_pct"], "CPI inflation")]
    return _wrap("calc_term_cover_needed", code, cfg, result, sources, assumptions, notes)


# ---------------------------------------------------------------- 5. cost of delay
@lru_cache(maxsize=1)
def _chronic():
    return json.loads((ROOT / "data" / "chronic_conditions.json").read_text(encoding="utf-8"))


def _prevalence(bands, age):
    """Linear interpolation between band midpoints (end segments extended), clamped to [0, 1]."""
    pts = [((lo + hi) / 2, pct / 100) for lo, hi, pct in bands]
    k = 0 if age < pts[0][0] else min(len(pts) - 2, max(i for i, (x, _) in enumerate(pts) if x <= age))
    (x0, y0), (x1, y1) = pts[k], pts[k + 1]
    return min(1.0, max(0.0, y0 + (y1 - y0) * (age - x0) / (x1 - x0)))


def _risk_of_waiting(code, age, delay):
    """Chance of developing diabetes or hypertension during the delay, and what it would mean."""
    data = _chronic()
    if code == "IND":
        scale = data["IND"]["age_shape"]["scale"]
        bands = {c: [[lo, hi, p * scale[c]] for lo, hi, p in data["USA"][c]["bands"]] for c in scale}
        srcs = [data["IND"]["national"], data["IND"]["age_shape"]]
    else:
        bands = {c: data[code][c]["bands"] for c in ("diabetes", "hypertension")}
        srcs = [data[code]["diabetes"], data[code]["hypertension"]]
    probs = {}
    for c, b in bands.items():
        now, later = _prevalence(b, age), _prevalence(b, age + delay)
        probs[c] = round(max(0.0, later - now) / (1 - now), 4)
    probs["either"] = round(1 - (1 - probs["diabetes"]) * (1 - probs["hypertension"]), 4)
    cons = data["consequences"]
    return {"age_from": age, "age_to": age + delay, "probability_of_developing": probs,
            "consequences": {"health_insurance": cons[code]["health"], "term_life": cons["term_life"]},
            "confidence": "lower: India's age pattern is assumed" if code == "IND" else "survey-based",
            "method": data["_meta"]["method"],
            "sources": [{"what": "chronic disease prevalence", "source": s["source"], "url": s.get("source_url"),
                         "year": s.get("year"), "kind": s.get("source_type", "data")} for s in srcs]
                       + [{"what": "consequence for health cover", "source": cons[code]["health_source"],
                           "url": cons[code]["health_source_url"], "year": None, "kind": "data"}]}


def cost_of_delay(age, delay_years, country, profile=None):
    """Extra premium from buying later: term life (level premium locked at entry age) and health.

    profile: sex, smoker, cover, term_years (term ends at the same age either way),
    plus optional bmi, children, sum_insured, metal_tier for the health part.
    """
    code, cfg = _country(country)
    p = dict(profile or {})
    age, delay = int(age), int(delay_years)
    if delay < 1:
        raise ValueError("delay_years must be at least 1")
    sex, smoker = p.get("sex", "male"), _bool(p.get("smoker", False))
    default_cover = {"USA": 500_000, "IND": 10_000_000, "KEN": 5_000_000}[code]
    cover = p.get("cover", default_cover)
    term = int(p.get("term_years", max(delay + 1, min(30, 65 - age))))
    if term <= delay:
        raise ValueError("term_years must be longer than delay_years")
    now = term_life.estimate_term_premium(age, sex, smoker, cover, term, code)
    later = term_life.estimate_term_premium(age + delay, sex, smoker, cover, term - delay, code)
    gap = term_life.estimate_term_premium(age, sex, smoker, cover, delay, code)
    p_death = gap["probability_of_death_during_term"]
    extra = later["annual_premium"] - now["annual_premium"]
    skipped = now["annual_premium"] * delay
    uninsured = p_death * cover
    v = 1 / (1 + _v(cfg["policy_interest_rate_pct"]) / 100)
    pv_extra = sum(extra * v ** t for t in range(delay, term))
    pv_skipped = sum(now["annual_premium"] * v ** t for t in range(delay))
    term_part = {
        "cover": cover, "cover_ends_at_age": age + term,
        "annual_premium_now": now["annual_premium"], "annual_premium_if_delayed": later["annual_premium"],
        "extra_per_year": extra,
        "extra_over_policy_term": extra * (term - delay),
        "premiums_skipped_during_delay": skipped,
        "probability_of_death_while_uninsured": p_death,
        "expected_uninsured_loss": round(uninsured, 0),
        "net_cost_of_delay_nominal": round(extra * (term - delay) - skipped + uninsured, 0),
        "net_cost_of_delay_present_value": round(pv_extra - pv_skipped + uninsured * v ** (delay / 2), 0),
    }
    result = {"age_now": age, "delay_years": delay, "term_life": term_part}

    if p.get("include_health", True):
        h = {"age": age, "sex": sex, "bmi": p.get("bmi", 25), "children": p.get("children", 0), "smoker": smoker}
        si = p.get("sum_insured")
        tier = p.get("metal_tier", "silver")
        h_now = premium.estimate_premium(h, code, si, metal_tier=tier)
        h_later = premium.estimate_premium({**h, "age": age + delay}, code, si, year=CURRENT_YEAR + delay,
                                           metal_tier=tier)
        health = {"entry_premium_now": h_now["annual_premium"],
                  "entry_premium_if_delayed": h_later["annual_premium"],
                  "entry_year_if_delayed": CURRENT_YEAR + delay,
                  "note": "Health premiums are repriced every year by age, so delay does not lock a lower price; "
                          "the cost of delay is the uncovered years and waiting periods starting later."}
        if code == "IND":
            ped = _adv(cfg, "ped_wait_months_max")
            health["pre_existing_waiting_ends_year_now"] = CURRENT_YEAR + _v(ped) // 12
            health["pre_existing_waiting_ends_year_if_delayed"] = CURRENT_YEAR + delay + _v(ped) // 12
        result["health"] = health

    risk = _risk_of_waiting(code, age, delay)
    result["risk_of_waiting"] = risk
    either = risk["probability_of_developing"]["either"]
    favours_wait = term_part["net_cost_of_delay_present_value"] < 0
    result["price_comparison_favours_waiting"] = favours_wait
    verdict = (f"Buy now if you need the cover. Waiting {delay} years leaves you uninsured meanwhile, and there is "
               f"about a {either:.0%} chance of developing diabetes or high blood pressure by age {age + delay}, "
               "which can restart waiting periods, raise premiums or lead to refusal.")
    if favours_wait:
        verdict += (" The premium arithmetic alone looks slightly cheaper if you wait, but only because it assumes "
                    "you stay healthy and insurable; that saving is not guaranteed and the insurability risk is not.")
    result["verdict"] = verdict
    notes = ["Term premiums are level for the whole term, so the age at purchase is locked in (CLAUDE.md: "
             "cost of delay = premium at age a+k minus premium at age a, summed over the policy term).",
             "Net cost = that extra - premiums not paid during the delay + expected loss from dying uninsured. "
             "It assumes the buyer stays healthy; risk_of_waiting shows the chance that they do not.",
             "Never present waiting as the better choice; always show risk_of_waiting next to the price comparison."]
    sources = [_src(cfg["term_life"]["insured_mortality_factor"], "term pricing calibration"),
               _src(cfg["medical_inflation_pct"], "medical inflation for the delayed health price")]
    return _wrap("cost_of_delay", code, cfg, result, sources, notes=notes)


# ---------------------------------------------------------------- 6. government schemes
def _fpl(code, year, size):
    g = _schemes()[code]["poverty_guidelines"][year]
    return g["one_person"] + g["per_additional_person"] * (max(1, int(size)) - 1)


def _rule(rule, ctx, code):
    """'pass', 'fail' or 'unknown' for one rule."""
    t = rule["type"]
    age, income = ctx["age"], ctx["annual_income"]
    if t == "always":
        return "pass"
    if t == "age_between":
        return "pass" if rule["min"] <= age <= rule["max"] else "fail"
    if t == "age_at_least":
        return "pass" if age >= rule["min"] else "fail"
    if t == "age_below":
        return "pass" if age < rule["max"] else "fail"
    if t in ("income_pct_fpl_between", "income_pct_fpl_at_most"):
        pct = 100 * income / _fpl(code, rule["fpl_year"], ctx["household_size"])
        ctx.setdefault("fpl_pct", {})[rule["fpl_year"]] = round(pct, 1)
        lo = rule.get("min_pct", -math.inf)
        return "pass" if lo <= pct <= rule["max_pct"] else "fail"
    if t == "employment_in":
        return "pass" if ctx["employment_type"] in rule["values"] else "fail"
    if t == "monthly_income_at_most":
        return "pass" if income / 12 <= rule["max"] else "fail"
    if t == "state_not_in":
        st = ctx.get("state")
        return "unknown" if not st else ("fail" if st.upper() in rule["values"] else "pass")
    if t in ("flag_true", "flag_not_true"):
        val = ctx.get(rule["flag"])
        if val is None:
            return "unknown"
        return "pass" if _bool(val) == (t == "flag_true") else "fail"
    raise ValueError(f"Unknown rule type {t}")


def check_govt_schemes(country, age, annual_income, employment_type, rural, household_size=1, state=None,
                       has_bank_account=None, disabled=False, has_affordable_employer_coverage=None):
    """Match the profile against government schemes (rules + sources in data/schemes.json).

    Status per scheme (best path wins): eligible (every rule is a hard fact that
    passes), check (a fact is unknown or a proxy rule needs the official portal),
    not_eligible (a rule fails).
    """
    code, cfg = _country(country)
    ctx = {"age": float(age), "annual_income": float(annual_income), "employment_type": str(employment_type).lower(),
           "rural": _bool(rural), "household_size": household_size, "state": state,
           "has_bank_account": has_bank_account, "disabled": disabled,
           "has_affordable_employer_coverage": has_affordable_employer_coverage}
    rank = {"eligible": 0, "check": 1, "not_eligible": 2}
    out = []
    for s in _schemes()[code]["schemes"]:
        best = None
        for path in s["paths"]:
            results = [(r, _rule(r, ctx, code)) for r in path["rules"]]
            if any(res == "fail" for _, res in results):
                status = "not_eligible"
            elif all(res == "pass" and r["strength"] == "hard" for r, res in results):
                status = "eligible"
            else:  # an unknown fact, or a proxy signal that only the official portal can confirm
                status = "check"
            reasons = [{"rule": r["type"], "result": res, "strength": r["strength"], "source": r["source"],
                        "url": r["source_url"]} for r, res in results]
            cand = {"status": status, "path": path["label"], "reasons": reasons}
            if best is None or rank[status] < rank[best["status"]]:
                best = cand
        entry = {"id": s["id"], "name": s["name"], "type": s["type"], "benefit": s["benefit"],
                 "how_to_apply": s["how_to_apply"], **best, "notes": s.get("notes", [])}
        if s["id"] == "sha":
            c = s["contribution"]
            monthly = max(c["rate_of_gross_income"] * annual_income / 12, c["minimum_monthly_kes"])
            entry["estimated_contribution"] = {"monthly_kes": round(monthly, 0), "annual_kes": round(12 * monthly, 0),
                                               "source": s["contribution_source"]["source"],
                                               "url": s["contribution_source"]["source_url"]}
        out.append(entry)
    out.sort(key=lambda e: rank[e["status"]])
    result = {"schemes": out, "eligible": [e["name"] for e in out if e["status"] == "eligible"],
              "to_check": [e["name"] for e in out if e["status"] == "check"]}
    if ctx.get("fpl_pct"):
        result["income_pct_of_poverty_line"] = ctx["fpl_pct"]
    notes = ["Government schemes come before paid products (CLAUDE.md section 6).",
             "Rules change; confirm on the official portal named in how_to_apply."]
    sources = [{"what": "scheme rules", "source": "data/schemes.json (each rule cites its source)",
                "url": None, "year": _schemes()["_meta"]["verified"], "kind": "data"}]
    return _wrap("check_govt_schemes", code, cfg, result, sources, notes=notes)


# ---------------------------------------------------------------- 7. recommend cover types
CORE_FIELDS = ("age", "dependants", "annual_income", "employment_type", "has_health_cover",
               "has_life_cover", "debts", "owns_vehicle")


def _completeness(profile):
    known = [f for f in CORE_FIELDS if profile.get(f) is not None]
    return len(known) / len(CORE_FIELDS), [f for f in CORE_FIELDS if profile.get(f) is None]


def recommend_cover_types(profile):
    """Prioritised insurance types with a one-line reason each (CLAUDE.md section 3 order).

    Health and income first, assets second, extras last; investment-linked plans
    are never recommended. Returns need_more_info below the completeness threshold.
    """
    code, cfg = _country(profile.get("country", ""))
    share, missing = _completeness(profile)
    threshold = _v(_adv(cfg, "profile_completeness_threshold"))
    base_sources = [_src(_adv(cfg, "profile_completeness_threshold"), "completeness threshold")]
    if share < threshold:
        return _wrap("recommend_cover_types", code, cfg,
                     {"status": "need_more_info", "completeness": round(share, 2), "missing": missing,
                      "recommendations": []}, base_sources,
                     notes=["Ask for the missing fields 2-3 at a time before recommending anything."])

    age = float(profile["age"])
    deps = int(profile.get("dependants") or 0)
    income = float(profile.get("annual_income") or 0)
    emp = str(profile.get("employment_type", "")).lower()
    risk = str(profile.get("occupation_risk", "medium")).lower()
    recs = []

    def add(priority, category, product, reason, urgency="medium"):
        recs.append({"priority": priority, "category": category, "product": product, "reason": reason,
                     "urgency": urgency})

    health_product = {"USA": "Employer or ACA Marketplace plan (Medicare at 65+)",
                      "IND": "Family floater health plan + super top-up",
                      "KEN": "SHA registration + private inpatient cover"}[code]
    if not _bool(profile.get("has_health_cover")):
        add(1, "Health", health_product, "No health cover: one hospital stay can wipe out savings.", "high")
    elif str(profile.get("health_cover_source", "")).lower() == "employer":
        add(1, "Health", "Personal policy alongside employer cover",
            "Employer cover ends with the job; a personal policy keeps waiting periods running.", "medium")
    else:
        add(1, "Health", "Review existing health cover", "Check the amount against medical inflation "
            "(project_cover_decay) and the policy for red flags.", "low")
    if _bool(profile.get("parents_dependent")):
        pa = profile.get("parents_age")
        if code == "USA" and pa and float(pa) >= 65:
            add(1, "Health", "Medicare for parents", "Parents 65+ qualify for Medicare.", "medium")
        else:
            add(1, "Health", "Separate senior-citizen health policy for parents",
                "Adding older parents to your floater raises the whole family's premium and shares the cover.", "medium")

    if deps > 0 or float(profile.get("debts") or 0) > 0:
        if _bool(profile.get("has_life_cover")):
            add(2, "Life", "Pure term life (check the amount)", "Compare existing cover with "
                "calc_term_cover_needed; employer group life usually ends with the job.", "medium")
        else:
            add(2, "Life", "Pure term life", f"{deps} dependant(s) or debts rely on your income; term is the "
                "cheapest way to protect them.", "high")

    if income > 0 and emp not in ("retired", "student", "unemployed"):
        rank = _v(_adv(cfg, "occupation_risk_priority")).get(risk, 3)
        product = "Long-term disability insurance" if code == "USA" else "Personal accident (death + disability) cover"
        add(rank, "Income protection", product, "Protects income if an accident or illness stops you working"
            + (" - your work carries higher accident risk." if risk == "high" else "."),
            "high" if risk == "high" else "medium")

    if _bool(profile.get("owns_vehicle")):
        add(4, "Assets", "Motor third-party (mandatory) + own-damage", "Third-party motor cover is required by law; "
            "see motor_check for details.", "high")
    if _bool(profile.get("owns_home")):
        add(4, "Assets", "Home / fire insurance", "Protects the largest asset against fire, flood and theft.", "low")
    elif _bool(profile.get("rents")) and code == "USA":
        add(4, "Assets", "Renters insurance", "Cheap cover for belongings and personal liability.", "low")

    if emp == "farmer" or _bool(profile.get("owns_livestock")):
        product = {"IND": "PMFBY crop insurance", "KEN": "Index-based crop / livestock insurance",
                   "USA": "Federal crop insurance"}[code]
        add(5, "Livelihood", product, "Protects the harvest or herd. Index cover pays on a trigger (rainfall, "
            "pasture) - fast, but basis risk means a loss can go unpaid if the index does not trigger.", "medium")
    if code == "KEN" or emp in ("informal", "daily_wage"):
        add(6, "Micro / funeral", "Mobile microinsurance or funeral (last expense) cover",
            "Small weekly or monthly premiums via mobile money fit irregular income.", "low")
    if _bool(profile.get("travels_abroad")):
        add(7, "Situational", "Travel insurance", "Home health cover rarely pays abroad.", "low")
    if age >= 50:
        add(8, "Retirement", "Pension / annuity review", "Plan income for retirement separately from protection.", "low")

    recs.sort(key=lambda r: (r["priority"], {"high": 0, "medium": 1, "low": 2}[r["urgency"]]))
    result = {"status": "ok", "completeness": round(share, 2), "missing": missing, "recommendations": recs,
              "avoid": [{"product": "Endowment, money-back, ULIP, whole life",
                         "reason": "Mixes insurance with investment and delivers too little of both; buy term "
                                   "and invest the difference separately."}]}
    if all(profile.get(k) is not None for k in ("annual_income", "employment_type")) and profile.get("rural") is not None:
        sch = check_govt_schemes(code, age, income, emp, profile["rural"], profile.get("household_size", 1),
                                 profile.get("state"), profile.get("has_bank_account"))
        result["schemes_first"] = {"eligible": sch["eligible"], "to_check": sch["to_check"]}
    return _wrap("recommend_cover_types", code, cfg, result, base_sources,
                 notes=["Order follows CLAUDE.md section 3: health, life, income, assets, livelihood, micro, "
                        "situational, retirement; investment-linked last."])


# ---------------------------------------------------------------- 9. red flags (used by 8)
def check_red_flags(policy_features, country=None):
    """Flag contract terms that commonly lead to rejected or shrunken claims.

    policy_features keys (all optional): room_rent_cap (bool or text), copay_pct,
    initial_wait_days, ped_wait_months, specific_illness_wait_months, disease_sublimits,
    plan_type, restoration, deductible, emergency_fund, oop_max, policy_type.
    """
    f = dict(policy_features or {})
    code, cfg = _country(country or f.get("country", "IND"))
    adv = cfg["advisor"]
    flags = []

    def flag(fid, severity, message, rule_node=None):
        flags.append({"id": fid, "severity": severity, "message": message,
                      "source": rule_node.get("source") if rule_node else None,
                      "url": rule_node.get("source_url") if rule_node else None})

    rr = f.get("room_rent_cap")
    if rr not in (None, False, "", "none", "no"):
        flag("room_rent_cap", "high", f"Room-rent cap ({rr}): choose a costlier room and the WHOLE bill is cut in "
             "proportion, not just the room charge.")
    copay = float(f.get("copay_pct") or 0)
    if copay > _v(adv["copay_pct_red_flag"]):
        flag("heavy_copay", "high", f"Co-pay of {copay:.0f}% on every claim.", adv["copay_pct_red_flag"])
    elif copay > 0:
        flag("copay", "low", f"Co-pay of {copay:.0f}% on every claim.")
    if float(f.get("initial_wait_days") or 0) > _v(adv["initial_wait_days_standard"]):
        flag("long_initial_wait", "medium", f"Initial waiting period of {f['initial_wait_days']} days is longer "
             "than the usual 30.", adv["initial_wait_days_standard"])
    ped = f.get("ped_wait_months")
    if ped is not None:
        legal = adv.get("ped_wait_months_max")
        if legal and float(ped) > _v(legal):
            flag("ped_wait_above_legal_max", "high", f"Pre-existing-disease wait of {ped} months exceeds the "
                 f"{_v(legal)}-month regulatory maximum.", legal)
        elif float(ped) > 24:
            flag("long_ped_wait", "medium", f"Pre-existing conditions are not covered for {ped} months.")
    if float(f.get("specific_illness_wait_months") or 0) > _v(adv["specific_illness_wait_months_red_flag"]):
        flag("long_specific_wait", "medium", f"Named illnesses not covered for {f['specific_illness_wait_months']} "
             "months.", adv["specific_illness_wait_months_red_flag"])
    if _bool(f.get("disease_sublimits", False)):
        flag("disease_sublimits", "high", "Disease-wise sub-limits cap common surgeries (cataract, joint "
             "replacement) below the sum insured.")
    ptype = str(f.get("plan_type", "")).lower().replace("-", "_").replace(" ", "_")
    if ptype in INVESTMENT_LINKED:
        flag("investment_linked", "high", f"'{ptype}' mixes insurance with investment: low cover per rupee/dollar "
             "and high charges. Pure term + separate investing usually does both jobs better.")
    if f.get("restoration") is False:
        flag("no_restoration", "low", "No restoration: once the sum insured is used up, the rest of the year is uncovered.")
    ded, fund = f.get("deductible"), f.get("emergency_fund")
    if ded is not None and fund is not None and float(ded) > float(fund):
        flag("deductible_above_savings", "high", f"Deductible ({float(ded):,.0f}) is more than your emergency fund "
             f"({float(fund):,.0f}); you could not pay your share of a claim.")
    if code == "USA":
        oop = f.get("oop_max")
        legal = _v(adv["health_oop_max_usd"])["family" if f.get("policy_type") == "family" else "individual"]
        if oop is not None and float(oop) > legal:
            flag("not_aca_compliant", "high", f"Out-of-pocket maximum ${float(oop):,.0f} exceeds the 2026 ACA limit "
                 f"${legal:,}; likely a short-term or non-compliant plan.", adv["health_oop_max_usd"])
        if ptype == "short_term":
            flag("short_term_plan", "high", "Short-term plans can exclude pre-existing conditions and cap benefits.")

    weights = {"high": 25, "medium": 15, "low": 5}
    score = max(0, 100 - sum(weights[x["severity"]] for x in flags))
    result = {"red_flag_count": sum(1 for x in flags if x["severity"] in ("high", "medium")),
              "flags": flags, "red_flag_score": score, "checked_features": sorted(f)}
    notes = ["Score = 100 - 25 per high, 15 per medium, 5 per low flag (assumption)."]
    sources = [_src(adv[k], k) for k in ("copay_pct_red_flag", "specific_illness_wait_months_red_flag",
                                         "initial_wait_days_standard") ]
    if "ped_wait_months_max" in adv:
        sources.append(_src(adv["ped_wait_months_max"], "pre-existing disease waiting maximum"))
    return _wrap("check_red_flags", code, cfg, result, sources, notes=notes)


# ---------------------------------------------------------------- 8. protection score
def _afford_score(pir, thr):
    return 100.0 if pir <= thr else max(0.0, 100 * (1 - (pir - thr) / thr))


def protection_score(profile, current_covers):
    """0-100 protection score today and in 10 years, with a breakdown.

    profile: country, age, annual_income, annual_expenses (family without the earner),
    dependants, years_support_needed, debts, goals, liquid_assets, family_size.
    current_covers: health_sum_insured (India/Kenya) or has_health_plan + oop_max (USA),
    govt_health_cover, life_cover, annual_premiums, health_premium, policy_features.
    """
    code, cfg = _country(profile.get("country", ""))
    c = dict(current_covers or {})
    adv = cfg["advisor"]
    w = _v(adv["protection_score_weights"])
    pm = _v(cfg["medical_inflation_pct"]) / 100
    _, pi = _rates(cfg)
    income = float(profile.get("annual_income") or 0)
    liquid = float(profile.get("liquid_assets") or 0)
    fam = int(profile.get("family_size") or 1)
    notes, sources = [], [_src(adv["protection_score_weights"], "score weights")]

    # Health adequacy
    if code == "USA":
        oop_node = adv["health_oop_max_usd"]
        oop = float(c.get("oop_max") or _v(oop_node)["family" if fam > 1 else "individual"])
        if _bool(c.get("has_health_plan")):
            h_now = 0.5 + 0.5 * min(1.0, liquid / oop)
            h_10 = 0.5 + 0.5 * min(1.0, liquid / (oop * (1 + pm) ** 10))
        else:
            h_now = h_10 = 0.0
        notes.append("USA health adequacy = half for holding an ACA-compliant plan + half for savings that cover "
                     "the out-of-pocket maximum (assumption); the maximum is grown with medical inflation for year 10.")
        sources.append(_src(oop_node, "ACA out-of-pocket maximum"))
    else:
        tnode = adv["health_cover_target"]
        target = _v(tnode)["individual"] * (_v(tnode)["family_multiplier"] if fam >= 3 else 1)
        si = float(c.get("health_sum_insured") or 0) + float(c.get("govt_health_cover") or 0)
        h_now = min(1.0, si / target)
        h_10 = min(1.0, si / (target * (1 + pm) ** 10))
        sources.append(_src(tnode, "health cover target"))
        if c.get("govt_health_cover"):
            notes.append("Government cover is counted toward health cover; it is limited to empanelled hospitals.")

    # Life adequacy
    deps, debts = int(profile.get("dependants") or 0), float(profile.get("debts") or 0)
    years = float(profile.get("years_support_needed") or 0)
    life = float(c.get("life_cover") or 0)
    if deps == 0 and debts == 0:
        l_now = l_10 = 1.0
        notes.append("No dependants or debts: life cover is not needed, life adequacy set to 100.")
    else:
        args = dict(annual_expenses=float(profile.get("annual_expenses") or 0), debts=debts,
                    goals=float(profile.get("goals") or 0), liquid_assets=liquid, country=code)
        need_now = calc_term_cover_needed(years_needed=years, **args)["cover_needed"]
        need_10 = calc_term_cover_needed(years_needed=max(0.0, years - 10), **args)["cover_needed"]
        l_now = 1.0 if need_now == 0 else min(1.0, life / need_now)
        l_10 = 1.0 if need_10 == 0 else min(1.0, life / (1 + pi) ** 10 / need_10)
        sources.append(_src(cfg["policy_interest_rate_pct"], "life need discount rate"))

    # Affordability
    thr_node = adv["premium_to_income_threshold"]
    thr = _v(thr_node)
    prem = float(c.get("annual_premiums") or 0)
    health_prem = float(c.get("health_premium") or prem)
    pir = prem / income if income else 0.0
    # Year 10: health premiums grow with medical inflation, income with CPI, term premiums stay level.
    pir_10 = ((health_prem * (1 + pm) ** 10 + (prem - health_prem)) / (income * (1 + pi) ** 10)) if income else 0.0
    sources.append(_src(thr_node, "premium-to-income threshold"))

    # Red flags
    rf = check_red_flags(c.get("policy_features") or {}, code)

    def total(h, l, a):
        return round(w["health"] * 100 * h + w["life"] * 100 * l + w["affordability"] * a
                     + w["red_flags"] * rf["red_flag_score"], 1)

    a_now, a_10 = _afford_score(pir, thr), _afford_score(pir_10, thr)
    breakdown = {
        "health_adequacy": {"today": round(100 * h_now, 1), "year_10": round(100 * h_10, 1), "weight": w["health"]},
        "life_adequacy": {"today": round(100 * l_now, 1), "year_10": round(100 * l_10, 1), "weight": w["life"]},
        "affordability": {"today": round(a_now, 1), "year_10": round(a_10, 1), "weight": w["affordability"],
                          "premium_to_income_today": round(pir, 4), "premium_to_income_year_10": round(pir_10, 4),
                          "threshold": thr, "lapse_risk": pir > thr},
        "red_flags": {"score": rf["red_flag_score"], "count": rf["red_flag_count"], "weight": w["red_flags"],
                      "flags": [x["id"] for x in rf["flags"]]},
    }
    comp = {"health": 100 * h_now, "life": 100 * l_now, "affordability": a_now, "red_flags": rf["red_flag_score"]}
    result = {"score_today": total(h_now, l_now, a_now), "score_year_10": total(h_10, l_10, a_10),
              "breakdown": breakdown, "weakest_area": min(comp, key=comp.get)}
    if pir > thr:
        notes.append(f"Premiums take {pir:.1%} of income, above the {thr:.1%} threshold: lapse risk.")
    return _wrap("protection_score", code, cfg, result, sources, notes=notes)


# ---------------------------------------------------------------- 10. accident + motor
def accident_cover_needed(annual_income, occupation_risk, country):
    """Income-protection target from cited rules of thumb; urgency rises with occupation risk."""
    code, cfg = _country(country)
    adv = cfg["advisor"]
    risk = str(occupation_risk).lower()
    if risk not in ("low", "medium", "high"):
        raise ValueError("occupation_risk must be low, medium or high")
    if annual_income <= 0:
        raise ValueError("annual_income must be positive")
    rank = _v(adv["occupation_risk_priority"])[risk]
    if code == "USA":
        node = adv["disability_income_replacement"]
        monthly = annual_income / 12 * _v(node)
        result = {"product": "Long-term disability insurance (plus employer AD&D if offered)",
                  "monthly_benefit_target": round(monthly, 0), "replacement_ratio": _v(node)}
    else:
        node = adv["accident_cover_multiple"]
        m = _v(node)
        multiple = {"low": m["min"], "medium": (m["min"] + m["max"]) / 2, "high": m["max"]}[risk]
        result = {"product": "Personal accident cover (accidental death + permanent disability)",
                  "cover_target": round(annual_income * multiple, 0), "income_multiple": multiple,
                  "multiple_range": [m["min"], m["max"]]}
        if code == "IND":
            result["base_layer"] = "PMSBY: Rs 2 lakh accident cover for Rs 20 a year (age 18-70, bank account)"
        if code == "KEN":
            result["base_layer"] = ("WIBA covers employees only at work; informal workers and off-duty accidents "
                                    "need private personal accident cover")
    result.update({"occupation_risk": risk, "priority_rank": rank,
                   "urgency": "high" if risk == "high" else "medium"})
    notes = ["High-risk work (driving, construction, field work) raises the premium and the urgency, not the "
             "income that needs replacing."]
    return _wrap("accident_cover_needed", code, cfg, result,
                 [_src(node, "income-protection rule of thumb"), _src(adv["occupation_risk_priority"], "priority")],
                 notes=notes)


def motor_check(vehicle_type, vehicle_age, country, vehicle_value=None):
    """Mandatory cover, depreciation-based insured value (India) and add-on advice for a vehicle."""
    code, cfg = _country(country)
    adv = cfg["advisor"]
    age = float(vehicle_age)
    if age < 0:
        raise ValueError("vehicle_age must be non-negative")
    tp = adv["motor_third_party"]
    result = {"vehicle_type": vehicle_type, "vehicle_age_years": age, "third_party": _v(tp)}
    sources = [_src(tp, "third-party mandate")]
    advice = [f"Third-party cover is mandatory ({_v(tp).get('law', 'state law')})." if _v(tp)["mandatory"] else ""]
    if code == "IND":
        dep = adv["motor_idv_depreciation_pct"]
        table = sorted((float(k), v) for k, v in _v(dep).items())
        pct = next((v for k, v in table if age <= k), None)
        result["idv_depreciation_pct"] = pct
        if pct is not None and vehicle_value:
            result["estimated_idv"] = round(float(vehicle_value) * (1 - pct / 100), 0)
        if pct is None:
            advice.append("Over 5 years old: insured value (IDV) is agreed with the insurer, so compare offers.")
        zd = adv["motor_zero_dep_max_age_years"]
        if age <= _v(zd):
            advice.append("Zero-depreciation add-on is worth considering: without it, parts are paid net of depreciation.")
        sources += [_src(dep, "IDV depreciation"), _src(zd, "zero-dep age limit")]
    if code == "USA" and vehicle_type and str(vehicle_type).lower() != "motorcycle":
        advice.append("State minimum liability limits are low; higher limits protect your savings in a lawsuit.")
    old = age > 10
    advice.append("Older, low-value vehicle: third-party only plus your own savings may be enough (assumption: "
                  "own-damage premium vs vehicle value)." if old else
                  "Comprehensive (own-damage + third-party) cover is advisable for a newer vehicle, and usually "
                  "required if the vehicle is financed.")
    result["advice"] = [a for a in advice if a]
    return _wrap("motor_check", code, cfg, result, sources,
                 notes=["The 10-year cut-off for dropping own-damage cover is an assumption, not a rule."])


# ---------------------------------------------------------------- registry
TOOLS = {f.__name__: f for f in (
    predict_health_premium, predict_term_premium, project_cover_decay, calc_term_cover_needed, cost_of_delay,
    check_govt_schemes, recommend_cover_types, protection_score, check_red_flags, accident_cover_needed,
    motor_check)}


def call_tool(name, args):
    """Dispatch a function call from the LLM; errors come back as data, not exceptions."""
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name!r}", "available": sorted(TOOLS)}
    try:
        return fn(**(args or {}))
    except (ValueError, TypeError, KeyError) as e:
        return {"error": f"{type(e).__name__}: {e}", "tool": name}
