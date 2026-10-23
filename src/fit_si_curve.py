"""Fit how health premiums scale with sum insured (India, Kenya) and metal tier (USA).

Data: data/sum_insured_quotes.json - the same plan and profile priced at several
sums insured. Only ratios within a series are used, so quote year and plan level
cancel out:

    ln P(s, SI) = a_s + f(x),   x = ln(SI / SI_ref)

a_s is a free intercept per series; f is shared by the country. Three shapes are
compared by leave-one-series-out (LOSO) error on within-series ratios:

    power      f = b x
    quad_log   f = b1 x + b2 x^2            (bends in log-log space)
    power_age  f = (b + c (age - 40) / 10) x (slope varies with age)

The chosen curve is written to country_config.json as sum_insured_curve and
premium.py prices a policy at SI as premium(SI_ref) x exp(f(x)). Outside the
fitted range the curve continues along its boundary tangent (log-log).

USA: ACA plans have no annual cap, so cover level is the metal tier. Tier factors
are the KFF 2026 market averages relative to benchmark silver; see results.md for
why a smooth actuarial curve is not used.

Run order: build_config -> fit_si_curve -> validate -> validate_term.
"""
import json
import math
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "data" / "country_config.json"
QUOTES = ROOT / "data" / "sum_insured_quotes.json"
SI_REF = {"IND": 1_000_000, "KEN": 1_000_000}
FORMS = ("power", "quad_log", "power_age")


def _features(form, x, age):
    if form == "power":
        return [x]
    if form == "quad_log":
        return [x, x * x]
    return [x, x * (age - 40) / 10]


def _fit(series, code, form):
    """Least squares with one intercept per series; returns shared coefficients."""
    ids = [s["id"] for s in series]
    rows, y = [], []
    for s in series:
        for si, p in s["points"]:
            x = math.log(si / SI_REF[code])
            dummies = [1.0 if s["id"] == i else 0.0 for i in ids]
            rows.append(dummies + _features(form, x, s["age"]))
            y.append(math.log(p))
    coef, *_ = np.linalg.lstsq(np.array(rows), np.array(y), rcond=None)
    return [float(c) for c in coef[len(ids):]]


def _f(form, coef, x, age=40.0):
    return float(np.dot(coef, _features(form, x, age)))


def _ratio_errors(series, code, form, coef):
    """|predicted/actual - 1| for every point's premium relative to the series anchor.

    Anchor = the point nearest SI_ref, so a held-out series is predicted from one
    observed premium plus the curve.
    """
    errs = []
    for s in series:
        xs = [math.log(si / SI_REF[code]) for si, _ in s["points"]]
        k = min(range(len(xs)), key=lambda i: abs(xs[i]))
        p0 = s["points"][k][1]
        for i, (si, p) in enumerate(s["points"]):
            if i == k:
                continue
            pred = p0 * math.exp(_f(form, coef, xs[i], s["age"]) - _f(form, coef, xs[k], s["age"]))
            errs.append(abs(pred / p - 1))
    return errs


def _monotone(form, coef, xmin, xmax, ages=(0, 40, 80)):
    grid = np.linspace(xmin, xmax, 50)
    return all(np.all(np.diff([_f(form, coef, x, a) for x in grid]) >= -1e-9) for a in ages)


def fit_country(code, series):
    xs = [math.log(si / SI_REF[code]) for s in series for si, _ in s["points"]]
    xmin, xmax = min(xs), max(xs)
    out = {}
    for form in FORMS:
        coef = _fit(series, code, form)
        loso = []
        for s in series:
            rest = [o for o in series if o["id"] != s["id"]]
            loso += _ratio_errors([s], code, form, _fit(rest, code, form))
        out[form] = {"coef": coef,
                     "in_sample_mape_pct": 100 * float(np.mean(_ratio_errors(series, code, form, coef))),
                     "loso_mape_pct": 100 * float(np.mean(loso)),
                     "monotone": _monotone(form, coef, xmin, xmax)}
    best = min((f for f in FORMS if out[f]["monotone"]), key=lambda f: out[f]["loso_mape_pct"])
    per_series = {s["id"]: round(_fit([s], code, "power")[0], 3) for s in series if len(s["points"]) >= 2}
    return best, out, (xmin, xmax), per_series


def _curve_entry(code, form, coef, xr, stats, series, extra):
    si_min, si_max = (round(SI_REF[code] * math.exp(v)) for v in xr)
    names = {"power": ["b"], "quad_log": ["b1", "b2"], "power_age": ["b", "c_age_per_decade"]}[form]
    return {
        "value": {"form": form, "coef": dict(zip(names, [round(c, 5) for c in coef])),
                  "reference_sum_insured": SI_REF[code], "fitted_range": [si_min, si_max]},
        "unit": "premium(SI) = premium(SI_ref) x exp(f(ln(SI/SI_ref)))",
        "source_type": "calibration",
        "source": "Fitted by src/fit_si_curve.py on same-plan, same-profile premiums at different sums insured "
                  "(series intercepts free, curve shared). Form chosen by leave-one-series-out error.",
        "fitted_on": [s["id"] for s in series],
        "quotes_file": "data/sum_insured_quotes.json",
        "fit_stats": {f: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items() if k != "coef"}
                      for f, s in stats.items()},
        **extra,
        "fitted": date.today().isoformat(),
    }


def us_tiers(q):
    m = q["metal_tiers"]["monthly_premium"]
    av = q["actuarial_value"]["value"]
    bench = m["silver_benchmark"]
    factors = {"bronze": m["bronze"] / bench, "silver": 1.0, "gold": m["gold"] / bench}
    # No platinum figure in the KFF table: scale gold by the regulatory AV ratio.
    factors["platinum"] = factors["gold"] * av["platinum"] / av["gold"]
    actuarial = {t: av[t] / av["silver"] for t in av}
    return {
        "value": {t: round(v, 4) for t, v in factors.items()},
        "unit": "annual premium ratio vs benchmark silver",
        "year": 2026,
        "source": q["metal_tiers"]["source"] + "; platinum = gold x AV ratio 0.90/0.80 (45 CFR 156.140), derived",
        "source_url": q["metal_tiers"]["source_url"],
        "fetched": "2026-10-23",
        "note": "Market prices, not actuarial value: silver plans are 'silver-loaded' with the cost of cost-sharing "
                "reductions, so lowest-cost gold ($615) costs about the same as benchmark silver ($625). "
                "Pure AV ratios would be " + ", ".join(f"{t} {v:.3f}" for t, v in actuarial.items()) + ".",
        "actuarial_value": av,
        "standard_silver_deductible_usd": q["standard_silver_deductible_usd"]["value"],
        "standard_silver_deductible_source_url": q["standard_silver_deductible_usd"]["source_url"],
    }


def run():
    q = json.loads(QUOTES.read_text(encoding="utf-8"))
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    report = {}
    for code in ("IND", "KEN"):
        series = q[code]
        best, stats, xr, per_series = fit_country(code, series)
        extra = {"per_series_power_slope": per_series}
        if code == "IND":
            # Sensitivity: the nyvo series are broker-scaled from a 10L anchor; refit on Ditto (chart) series only.
            chart = [s for s in series if "scaled" not in s["quality"]]
            extra["sensitivity_chart_series_only"] = {
                "series": [s["id"] for s in chart],
                "power_slope": round(_fit(chart, code, "power")[0], 3),
                "range": "15L-50L only"}
        cfg[code]["sum_insured_curve"] = _curve_entry(code, best, stats[best]["coef"], xr, stats, series, extra)
        report[code] = {"chosen": best, "stats": stats, "per_series_power_slope": per_series,
                        "fitted_range": cfg[code]["sum_insured_curve"]["value"]["fitted_range"], **extra}
        print(f"\n== {code}: {len(series)} series, {sum(len(s['points']) for s in series)} points ==")
        for f, s in stats.items():
            print(f"  {f:10s} coef {np.round(s['coef'], 4)}  in-sample {s['in_sample_mape_pct']:.1f}%  "
                  f"LOSO {s['loso_mape_pct']:.1f}%  monotone {s['monotone']}" + ("   <- chosen" if f == best else ""))
        print("  per-series power slope:", per_series)
        if "sensitivity_chart_series_only" in extra:
            print("  chart-only sensitivity:", extra["sensitivity_chart_series_only"])
    cfg["USA"]["metal_tier_factor"] = us_tiers(q["USA"])
    report["USA"] = cfg["USA"]["metal_tier_factor"]
    print("\n== USA metal tier factors (vs benchmark silver) ==", cfg["USA"]["metal_tier_factor"]["value"])
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "reports" / "si_curve_results.json").write_text(json.dumps(report, indent=2))
    print("\nWrote sum_insured_curve / metal_tier_factor to data/country_config.json and reports/si_curve_results.json")


if __name__ == "__main__":
    run()
