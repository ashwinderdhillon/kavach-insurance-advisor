"""Build data/country_config.json from World Bank data + hand-verified cited figures.

Every numeric field is a dict: {value, unit, year, source, source_url, fetched, note}.
Fields we could not verify from a primary document carry "flag" so the app and
report can show them as lower-confidence. Nothing here is invented: if a value
cannot be sourced it must be None with a flag.

Run order: fetch_worldbank.py -> build_config.py -> fit_si_curve.py -> validate.py
-> validate_term.py (the last three add fitted values).
"""
import json
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()

WTW_URL = "https://www.wtwco.com/en-cm/insights/2025/10/2026-global-medical-trends-survey"
WTW_PDF = "https://www.moncey-assurances.com/doc/TW-2026.pdf"
WTW_NOTE = ("WTW 2026 Global Medical Trends Survey, Figure 4 (p.8-10), gross medical trend "
            f"projected for 2026. Copy of the report PDF read at {WTW_PDF}. This is insurer-reported "
            "medical cost trend (price + utilisation), not a consumer price index.")


def field(value, unit, year, source, source_url, note=None, flag=None, fetched=TODAY):
    d = {"value": value, "unit": unit, "year": year, "source": source,
         "source_url": source_url, "fetched": fetched}
    if note:
        d["note"] = note
    if flag:
        d["flag"] = flag
    return d


def fred_usd_inr():
    """Latest USD/INR from the Federal Reserve H.10 series via FRED (no API key needed)."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS"
    rows = [r.split(",") for r in requests.get(url, timeout=30).text.strip().splitlines()[1:]]
    rows = [r for r in rows if r[1] not in ("", ".")]
    day, val = rows[-1]
    return field(float(val), "INR per USD", int(day[:4]),
                 "Federal Reserve H.10 noon buying rate (FRED series DEXINUS)", url,
                 note=f"Observation date {day}.")


MANUAL = {
    "USA": {
        "medical_inflation_pct": field(9.6, "% per year", 2026, "WTW 2026 Global Medical Trends Survey",
                                       WTW_URL, note=WTW_NOTE),
        "loss_ratio": field(0.85, "claims / premium", 2024,
                            "KFF, Health Insurer Financial Performance in 2024 (pub. 23 Feb 2026), NAIC data",
                            "https://www.kff.org/medicare/health-insurer-financial-performance/",
                            note="'Simple loss ratios were around 85% in individual market, 88% in the fully "
                                 "insured (group) market'. Individual market used because the app prices "
                                 "individual/family cover."),
        "loss_ratio_group": field(0.88, "claims / premium", 2024,
                                  "KFF, Health Insurer Financial Performance in 2024 (pub. 23 Feb 2026), NAIC data",
                                  "https://www.kff.org/medicare/health-insurer-financial-performance/",
                                  note="Fully insured group market; used for employer (group) premiums."),
        "policy_interest_rate_pct": field(3.875, "% per year", 2026,
                                          "Federal Reserve FOMC statement, 16 Sep 2026",
                                          "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm",
                                          note="Target range 3.75-4.00%; midpoint stored."),
        "fx_lcu_per_usd": field(1.0, "USD per USD", 2026, "Identity (home currency)", None),
    },
    "IND": {
        "medical_inflation_pct": field(12.9, "% per year", 2026, "WTW 2026 Global Medical Trends Survey",
                                       WTW_URL, note=WTW_NOTE),
        "loss_ratio": field(0.6806, "claims / premium", 2025,
                            "IRDAI Annual Report 2024-25, Table I.14 (segment-wise incurred claims ratio), "
                            "Health, standalone health insurers",
                            "https://lifeinscouncil.org/component/IRDAI%20Annual%20Report%202024-25.pdf",
                            note="FY2024-25. Standalone health insurers (68.06%) used because they dominate "
                                 "retail individual/family policies; the all-insurer health ICR is 85.34% "
                                 "(public sector 99.84%, private general 83.46%), driven by group and "
                                 "government business."),
        "loss_ratio_all_insurers": field(0.8534, "claims / premium", 2025,
                                         "IRDAI Annual Report 2024-25, Table I.14, Health, all insurers",
                                         "https://lifeinscouncil.org/component/IRDAI%20Annual%20Report%202024-25.pdf"),
        "policy_interest_rate_pct": field(5.50, "% per year", 2026,
                                          "RBI Monetary Policy Statement 2026-27, MPC resolution 5-7 Oct 2026",
                                          "https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid=63742",
                                          note="Policy repo rate raised 25 bps to 5.50%."),
        "fx_lcu_per_usd": None,  # filled from FRED below
    },
    "KEN": {
        "medical_inflation_pct": field(9.5, "% per year", 2026, "WTW 2026 Global Medical Trends Survey",
                                       WTW_URL, note=WTW_NOTE),
        "loss_ratio": field(0.746, "net incurred claims / net earned premium", 2022,
                            "IRA Kenya, Insurance Industry Annual Report 2022 (pub. Sept 2023), net incurred "
                            "claims ratios section (PDF p.58, Table 27)",
                            "https://libraryir.parliament.go.ke/bitstreams/2b2982e3-7efa-4c8c-89b9-558694bb988f/download",
                            note="Verified in the report PDF (Parliament of Kenya library copy): 'Motor Private, "
                                 "Medical and Motor Commercial classes registered loss ratios above the global "
                                 "benchmark at 80.3%, 74.6% and 70.6% respectively.' Latest year verified from a "
                                 "primary document; see loss_ratio_sensitivity for newer, unverified figures."),
        "loss_ratio_sensitivity": {
            "value": {"low": 0.746, "mid": 0.81, "high": 1.018},
            "unit": "net incurred claims / net earned premium",
            "source_type": "assumption",
            "source": "Sensitivity range. low = IRA 2022 annual report (verified). mid = IRA Q2-2025 "
                      "industry release, medical 81.0% (search-indexed text; ira.go.ke blocks download). "
                      "high = 2025 medical-only ratio 101.8% from the 2026 Kenya Medical Industry Analytics "
                      "report as summarised by Willow Health Media (secondary).",
            "source_url": "https://willowhealthmedia.org/private-health-insurance-in-kenya-same-ward-same-procedure-different-prices/",
            "note": "Premiums are calibrated to market quotes, so changing the loss ratio changes the "
                    "calibration factor and the expected-claims figure, not the fitted premium.",
        },
        "policy_interest_rate_pct": field(8.75, "% per year", 2026,
                                          "Central Bank of Kenya MPC press release, 7 Oct 2026",
                                          "https://www.centralbank.go.ke/2026/10/07/mpc-retains-the-cbr-at-8-75-percent-3/",
                                          note="Central Bank Rate retained at 8.75%."),
        "fx_lcu_per_usd": field(129.91, "KES per USD", 2026, "Central Bank of Kenya daily KES exchange rates",
                                "https://www.centralbank.go.ke/home/forex-2/",
                                note="US dollar rate posted 22-10-2026."),
    },
}

META = {
    "USA": {"name": "United States", "currency": "USD", "symbol": "$"},
    "IND": {"name": "India", "currency": "INR", "symbol": "₹"},
    "KEN": {"name": "Kenya", "currency": "KES", "symbol": "KSh"},
}

# ACA rating rules: in the US individual market the premium may vary only by age
# (federal default curve, max 3:1), tobacco (max 1.5:1), area and family size.
US_RATING_RULES = {
    "age_curve": field(
        {"0-14": 0.765, "15": 0.833, "16": 0.859, "17": 0.885, "18": 0.913, "19": 0.941, "20": 0.970,
         "21": 1.000, "22": 1.000, "23": 1.000, "24": 1.000, "25": 1.004, "26": 1.024, "27": 1.048,
         "28": 1.087, "29": 1.119, "30": 1.135, "31": 1.159, "32": 1.183, "33": 1.198, "34": 1.214,
         "35": 1.222, "36": 1.230, "37": 1.238, "38": 1.246, "39": 1.262, "40": 1.278, "41": 1.302,
         "42": 1.325, "43": 1.357, "44": 1.397, "45": 1.444, "46": 1.500, "47": 1.563, "48": 1.635,
         "49": 1.706, "50": 1.786, "51": 1.865, "52": 1.952, "53": 2.040, "54": 2.135, "55": 2.230,
         "56": 2.333, "57": 2.437, "58": 2.548, "59": 2.603, "60": 2.714, "61": 2.810, "62": 2.873,
         "63": 2.952, "64+": 3.000},
        "premium ratio vs age 21", 2018,
        "CMS, Final Guidance Regarding Age Curves and State Reporting (16 Dec 2016), Appendix I",
        "https://www.cms.gov/CCIIO/Resources/Regulations-and-Guidance/Downloads/Final-Guidance-Regarding-Age-Curves-and-State-Reporting-12-16-16.pdf",
        note="Federal default standard age curve, effective plan years from 2018; states may use their own."),
    "tobacco_max_ratio": field(1.5, "max premium ratio tobacco vs non-tobacco", 2026,
                               "45 CFR 147.102(a)(1)(iv)",
                               "https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-B/part-147/section-147.102",
                               note="Federal ceiling; some states set a lower ratio or ban tobacco rating."),
    "max_children_rated": field(3, "children under 21 charged per family", 2026, "45 CFR 147.102(c)(1)",
                                "https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-B/part-147/section-147.102"),
}

# Child cost relative to an 18-year-old: the ML model only saw ages 18-64, so a
# child's cost is the 18-year-old prediction times this CMS age-curve ratio.
CHILD_COST_RATIO = field(round(0.765 / 0.913, 4), "child (0-14) cost / age-18 cost", 2018,
                         "Derived from CMS federal default age curve (0-14: 0.765, 18: 0.913)",
                         US_RATING_RULES["age_curve"]["source_url"])

# Age-band loading on top of expected claims. Neutral (1.0) by default: the ML
# model already prices age. validate.py may change these only with a stated reason.
AGE_BAND_LOADING = {
    "value": {"0-44": 1.0, "45-59": 1.0, "60+": 1.0},
    "unit": "multiplier on expected claims / loss ratio",
    "source": "Model assumption (neutral). See reports/results.md, calibration section.",
    "source_type": "assumption",
}


JHA_URL = "https://www.nejm.org/doi/full/10.1056/NEJMsa1211128"

# Term life inputs. Mortality comes from data/who_life_tables.json (fetch_who.py).
TERM_LIFE = {
    "smoker_mortality_ratio": field(
        {"male": 2.8, "female": 3.0}, "all-cause death rate, current vs never smokers", 2013,
        "Jha P et al., 21st-Century Hazards of Smoking and Benefits of Cessation in the United States, "
        "NEJM 2013;368:341-50 (PMID 23343063)", JHA_URL,
        note="Abstract: 'hazard ratio for women, 3.0; 99% CI 2.7 to 3.3; hazard ratio for men, 2.8; "
             "99% CI 2.4 to 3.1', ages 25-79. US cohort, applied to all three countries."),
    "life_loss_ratio": {
        "value": 0.65, "unit": "expected claims / premium",
        "sensitivity": {"low": 0.55, "high": 0.75},
        "source_type": "assumption",
        "source": "Assumption: no regulator publishes a term-life loss ratio for these markets (IRDAI and "
                  "IRA report claim settlement ratios, which measure something else). Premiums are "
                  "calibrated to market quotes, so this splits the premium between claims and loading "
                  "but does not change the fitted premium; see reports/results.md.",
    },
    "insured_mortality_factor": {
        "value": 1.0, "unit": "multiplier on WHO population mortality", "source_type": "calibration",
        "source": "Uncalibrated default; set by src/validate_term.py against cited term quotes.",
    },
}
POLICY_FEE = {  # fixed yearly expense loading per policy, local currency
    "USA": 60, "IND": 600, "KEN": 600,
}


# Family floater: eldest member's premium + this share of each other member's individual premium.
CARE_PROSPECTUS = ("https://cms.careinsurance.com/cms/public/uploads/download_center/"
                   "care-supreme---prospectus-cum-sales-literature.pdf")
FLOATER = {
    "USA": field(1.0, "share of each extra member's premium", 2026,
                 "45 CFR 147.102(c)(1): ACA family premiums are the sum of each member's premium (max 3 children "
                 "under 21 charged); no floater discount",
                 "https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-B/part-147/section-147.102"),
    "IND": field(0.5085, "share of each extra member's premium", 2026,
                 "Care Supreme prospectus (UIN CHIHLIP27061V032627, Apr 2026), Annexure V premium illustrations, "
                 "Zone 1, Rs 5 L: floater vs individual premiums. Ages 51+46: 32,881 vs 23,450 + 18,546 (0.509); "
                 "60+56+17: 50,727 vs 31,113 + 31,113 + 5,046 (0.542); 64+61: 54,509 vs 36,965 x 2 (0.475). Mean.",
                 CARE_PROSPECTUS),
    "KEN": field(0.692, "share of each extra member's premium", 2022,
                 "Mean of two Kenyan insurer tables at KES 1M: AAR SME family shared M+1 39,073 vs individual 31-40 "
                 "24,059 (0.624; 2017); Old Mutual AfyaImara spouse 26,785 vs principal 35,223, ages 30-40 (0.760; 2022)",
                 "https://cactus.co.ke/downloads/AAR_Business_Medical_Plan_SME_Brochure.pdf",
                 note="Same brochures as validation quotes KE-3 / KE-6, so Kenya's family validation is not fully "
                      "independent; the AAR M+3 row (= KE-6) was not used. Old Mutual: "
                      "https://www.oldmutual.co.ke/om-docs/blt94368202535313ed/Afyaimara_Family_Cover_Brochure.pdf"),
}


def assumption(value, unit, source, note=None):
    d = {"value": value, "unit": unit, "source_type": "assumption", "source": source}
    if note:
        d["note"] = note
    return d


# Advisor rules used by src/tools.py. Real thresholds carry a source URL; judgement
# calls are labelled source_type "assumption" with the reasoning in "source".
WHO_FP_URL = "https://www.who.int/data/gho/data/themes/topics/financial-protection"
PIR_WHO = ("Assumption by analogy with WHO SDG 3.8.2, which treats household health spending above 10% of "
           "consumption as catastrophic; no regulator sets a premium-to-income ceiling here.")
SCORE_WEIGHTS_NOTE = ("Assumption reflecting the binding constraint per region (CLAUDE.md section 2): "
                      "cost in the USA, adequacy and mis-selling in India, affordability in Kenya.")
COMMON_ADVISOR = {
    "profile_completeness_threshold": assumption(
        0.7, "share of core profile fields answered",
        "Assumption: no recommendation until at least 70% of the core Layer-A fields are known (CLAUDE.md section 6)."),
    "cooling_off_premium_months_of_income": assumption(
        1.0, "annual premium / monthly income",
        "Assumption: suggest sleeping on any purchase whose annual premium exceeds one month of income (CLAUDE.md layer F)."),
    "copay_pct_red_flag": assumption(
        20, "% co-payment per claim", "Assumption: co-pay above 20% of every claim is flagged as heavy."),
    "specific_illness_wait_months_red_flag": assumption(
        24, "months", "Assumption: specific-illness waiting periods longer than 24 months are flagged as long."),
    "initial_wait_days_standard": assumption(
        30, "days", "Assumption: 30-day initial waiting period is the market norm; longer is flagged."),
    "occupation_risk_priority": assumption(
        {"low": 3, "medium": 3, "high": 2}, "priority rank for accident/disability cover",
        "Assumption: high-risk occupations (driving, construction, field work) move accident cover up one rank."),
}
ADVISOR = {
    "USA": {
        "premium_to_income_threshold": field(
            0.0996, "annual premiums / annual income", 2026,
            "IRS Rev. Proc. 2025-25: 2026 ACA employer-coverage affordability percentage (9.96%)",
            "https://www.irs.gov/pub/irs-drop/rp-25-25.pdf",
            note="Legally applies to the employee's self-only contribution; used here as the lapse-risk flag for "
                 "total premiums. Value confirmed via LCW, HUB and EY summaries of the Rev. Proc."),
        "health_oop_max_usd": field(
            {"individual": 10600, "family": 21200}, "USD per year", 2026,
            "ACA maximum annual limitation on cost sharing for plan year 2026",
            "https://www.healthcare.gov/glossary/out-of-pocket-maximum-limit/",
            note="Confirmed via Hylant and HUB International 2026 summaries ($10,600 self-only / $21,200 other)."),
        "protection_score_weights": assumption(
            {"health": 0.45, "life": 0.25, "affordability": 0.20, "red_flags": 0.10}, "weights summing to 1",
            SCORE_WEIGHTS_NOTE),
        "free_look_days": field(
            {"min": 10, "max": 30}, "days", 2024,
            "NAIC model law chart LI-30 (life insurance disclosure provisions): state free-look periods",
            "https://content.naic.org/sites/default/files/model-law-chart-li-30-life-insurance-disclosure-provisions.pdf",
            note="Set by state law; 10 days is the common minimum, many states require 30 days for seniors or "
                 "replacements. ACA health plans have no free look; use the plan's cancellation rules."),
        "disability_income_replacement": field(
            0.60, "share of annual earnings", 2014,
            "BLS National Compensation Survey, Beyond the Numbers 'Disability insurance plans' (2014 data): "
            "median long-term disability benefit is 60% of annual earnings",
            "https://www.bls.gov/opub/btn/volume-4/disability-insurance-plans.htm"),
        "motor_third_party": field(
            {"mandatory": True, "exception": "New Hampshire (financial-responsibility rule instead of a mandate)"},
            "rule", 2026, "State auto insurance laws; New Hampshire is the only state without a general mandate",
            "https://insurify.com/car-insurance/new-hampshire/minimum-requirements/",
            note="Secondary source; minimum liability limits vary by state."),
    },
    "IND": {
        "premium_to_income_threshold": assumption(0.10, "annual premiums / annual income", PIR_WHO,
                                                  note=f"WHO financial protection: {WHO_FP_URL}"),
        "health_cover_target": assumption(
            {"individual": 1_000_000, "family_multiplier": 1.5}, "INR sum insured",
            "Assumption: Rs 10 lakh per person (the sum-insured curve's reference point and the cover level broker "
            "comparisons centre on), x1.5 for a floater of 3+ members. Grown with medical inflation for the "
            "year-10 view."),
        "protection_score_weights": assumption(
            {"health": 0.35, "life": 0.30, "affordability": 0.15, "red_flags": 0.20}, "weights summing to 1",
            SCORE_WEIGHTS_NOTE),
        "free_look_days": field(
            30, "days", 2024,
            "IRDAI Master Circular on Protection of Policyholders' Interests (5 Sep 2024): 30-day free look for "
            "life and health policies (was 15 days)",
            "https://www.gicouncil.in/news-media/gic-in-the-news/how-irdai-s-new-master-circular-help-life-health-insurance-policyholders/",
            note="Secondary summary; general (motor) policies not confirmed."),
        "ped_wait_months_max": field(
            36, "months", 2024,
            "IRDAI Master Circular on Health Insurance Business, 29 May 2024 (IRDAI/HLT/CIR/MISC/77/05/2024): "
            "maximum pre-existing-disease waiting period cut from 48 to 36 months; moratorium cut from 8 to 5 years",
            "https://nyvo.in/health-insurance/irdai-master-circular-2024",
            note="Secondary summary of the circular."),
        "accident_cover_multiple": field(
            {"min": 5, "max": 10}, "x annual income", 2024,
            "Insurer underwriting caps for personal accident: SBI General caps death/PTD cover at 120x monthly or "
            "10x annual gross income; Business Standard quotes underwriters suggesting 60-100x monthly income",
            "https://content.sbigeneral.in/uploads/48a081ad76ca416b9acf1b2efeb56f21.pdf"),
        "motor_third_party": field(
            {"mandatory": True, "law": "Motor Vehicles Act 1988, section 146"}, "rule", 1988,
            "Motor Vehicles Act 1988 s.146: no motor vehicle may be used in a public place without third-party "
            "insurance", "https://www.bajajallianz.com/blog/motor-insurance-articles/zero-depreciation-car-insurance-after-5-years.html",
            note="Statute summarised by insurer material; own-damage cover is optional."),
        "motor_idv_depreciation_pct": field(
            {"0.5": 5, "1": 15, "2": 20, "3": 30, "4": 40, "5": 50}, "% of ex-showroom price, by max vehicle age (years)",
            2024, "India Motor Tariff General Regulation 8 IDV depreciation schedule (insurer copy)",
            "https://www.iffcotokio.co.in/content/dam/iffcotokio/iffco-pdf/sites/default/files/IDV-Depreciation-Table.pdf",
            note="Beyond 5 years IDV is agreed between insurer and insured."),
        "motor_zero_dep_max_age_years": field(
            5, "years", 2026, "Zero-depreciation add-on is typically sold for vehicles up to 5 years old (some insurers 7)",
            "https://www.91wheels.com/buyers-guide/zero-depreciation-car-insurance-explained-benefits-limitations-claim-rules-and-whether-its-worth-it",
            note="Insurer practice, not regulation."),
    },
    "KEN": {
        "premium_to_income_threshold": assumption(0.10, "annual premiums / annual income", PIR_WHO,
                                                  note=f"WHO financial protection: {WHO_FP_URL}"),
        "health_cover_target": assumption(
            {"individual": 1_000_000, "family_multiplier": 1.5}, "KES inpatient limit",
            "Assumption: KES 1M inpatient limit (Old Mutual AfyaImara option II and the sum-insured curve's "
            "reference point), x1.5 for families of 3+. SHA cover is counted separately as the base layer."),
        "protection_score_weights": assumption(
            {"health": 0.40, "life": 0.20, "affordability": 0.30, "red_flags": 0.10}, "weights summing to 1",
            SCORE_WEIGHTS_NOTE),
        "free_look_days": field(
            30, "days", 2026,
            "Insurance Regulatory Authority FAQ: a life policy can be cancelled within 30 days of receiving the "
            "policy document for a refund of premium less withholding tax", "https://ira.go.ke/index.php/about-us/faq-s",
            note="Stated for life policies; general policies not confirmed."),
        "accident_cover_multiple": field(
            {"min": 8, "max": 8}, "x annual income", 2007,
            "Work Injury Benefits Act 2007 (No. 13 of 2007): permanent total disablement compensated at 96 months' earnings",
            "https://wageindicator.org/en-ke/work-in-kenya/labour-law/sick-leave/work-injury-benefits",
            note="Statutory employer benchmark used as the income-replacement target. Sources disagree on the death "
                 "benefit (60 vs 96 months), so only the disablement figure is used."),
        "motor_third_party": field(
            {"mandatory": True, "law": "Insurance (Motor Vehicles Third Party Risks) Act, Cap. 405, s.4(1)"}, "rule", 2022,
            "Cap. 405 s.4(1): no person may use a motor vehicle on a road without third-party cover; IRA confirms it "
            "is compulsory", "https://new.kenyalaw.org/akn/ke/act/1945/12"),
    },
}


def build():
    wb = json.loads((ROOT / "data" / "worldbank_raw.json").read_text())
    MANUAL["IND"]["fx_lcu_per_usd"] = fred_usd_inr()
    cfg = {"_meta": {
        "built": TODAY,
        "description": "Country calibration inputs for the premium layer. Every number has a source.",
        "health_spend_scaling": "cost_LCU = model_USD x (H_c_PPP / H_USA) x PPP_c, where "
                                "H_c_PPP = H_c_USD x FX_c(health year) / PPP_c. Algebraically this equals "
                                "model_USD x H_c_LCU / H_USA.",
    }}
    for c in ("USA", "IND", "KEN"):
        entry = dict(META[c])
        entry["worldbank"] = wb[c]
        entry.update(MANUAL[c])
        entry["age_band_loading"] = AGE_BAND_LOADING
        entry["child_cost_ratio"] = CHILD_COST_RATIO
        if c == "USA":
            entry["rating_rules"] = US_RATING_RULES
        entry["calibration_factor"] = {
            "value": 1.0, "unit": "multiplier", "source_type": "calibration",
            "source": "Uncalibrated default; set by src/validate.py against cited market quotes.",
        }
        entry["floater_addon_factor"] = FLOATER[c]
        entry["advisor"] = {**json.loads(json.dumps(COMMON_ADVISOR)), **ADVISOR[c]}
        entry["term_life"] = json.loads(json.dumps(TERM_LIFE))
        entry["term_life"]["policy_fee"] = {
            "value": POLICY_FEE[c], "unit": f"{META[c]['currency']} per policy per year",
            "source_type": "assumption",
            "source": "Assumption: fixed yearly admin cost per policy (expense loading), roughly USD 6-60. "
                      "Small next to typical premiums; shown separately in every quote's assumptions.",
        }
        cfg[c] = entry
    (ROOT / "data" / "country_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    for c in ("USA", "IND", "KEN"):
        e = cfg[c]
        print(f"{c}: med_infl {e['medical_inflation_pct']['value']}%  LR {e['loss_ratio']['value']}  "
              f"rate {e['policy_interest_rate_pct']['value']}%  FX {e['fx_lcu_per_usd']['value']}"
              + (f"  FLAG: {e['loss_ratio']['flag'][:40]}..." if e['loss_ratio'].get('flag') else ""))
    print("Wrote data/country_config.json")


if __name__ == "__main__":
    build()
