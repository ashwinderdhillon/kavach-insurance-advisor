"""Fetch WHO Global Health Observatory life tables and smoking prevalence.

Writes data/who_life_tables.json:
  - nqx (LIFE_0000000030): probability of dying between ages x and x+n,
    by sex and 5-year age group, latest year available per country
  - nMx (LIFE_0000000029): central death rate, used only for the open 85+ group
  - current tobacco smoking prevalence, age-standardised (M_Est_smk_curr_std), by sex
"""
import json
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://ghoapi.azureedge.net/api/"
COUNTRIES = ("USA", "IND", "KEN")
SEXES = {"SEX_MLE": "male", "SEX_FMLE": "female"}
SMOKING_YEAR = 2022  # latest WHO estimate year that is not a projection for USA and KEN


def _get(code, country):
    url = f"{API}{code}?$filter=SpatialDim eq '{country}'"
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return url, r.json()["value"]
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"WHO GHO API unreachable for {code} {country}")


def _age_start(dim2):
    # 'AGEGROUP_YEARS30-34' -> 30, 'AGEGROUP_YEARS85PLUS' -> 85, 'AGEGROUP_YEARS00-01' -> 0
    tail = dim2.replace("AGEGROUP_YEARS", "")
    return int(tail[:2]) if not tail.startswith("85") else 85


def fetch():
    today = date.today().isoformat()
    out = {"_meta": {
        "source": "WHO Global Health Observatory (GHO) OData API",
        "fetched": today,
        "indicators": {
            "LIFE_0000000030": "nqx - probability of dying between ages x and x+n",
            "LIFE_0000000029": "nMx - age-specific death rate between ages x and x+n",
            "M_Est_smk_curr_std": "Estimate of current tobacco smoking prevalence (%) (age-standardized)",
        },
    }}
    for c in COUNTRIES:
        url_q, q_rows = _get("LIFE_0000000030", c)
        url_m, m_rows = _get("LIFE_0000000029", c)
        url_s, s_rows = _get("M_Est_smk_curr_std", c)
        year = max(r["TimeDim"] for r in q_rows if r["Dim1"] in SEXES)
        entry = {"year": year, "source_url_nqx": url_q, "source_url_nmx": url_m, "fetched": today,
                 "nqx": {}, "nmx_85plus": {}, "smoking_prevalence": {}}
        for dim, sex in SEXES.items():
            rows = [r for r in q_rows if r["TimeDim"] == year and r["Dim1"] == dim]
            entry["nqx"][sex] = {str(_age_start(r["Dim2"])): r["NumericValue"]
                                 for r in sorted(rows, key=lambda r: _age_start(r["Dim2"]))}
            m85 = [r for r in m_rows if r["TimeDim"] == year and r["Dim1"] == dim
                   and r["Dim2"].endswith("85PLUS")]
            entry["nmx_85plus"][sex] = m85[0]["NumericValue"] if m85 else None
            s = [r for r in s_rows if r["TimeDim"] == SMOKING_YEAR and r["Dim1"] == dim]
            entry["smoking_prevalence"][sex] = {
                "value": s[0]["NumericValue"] / 100 if s else None, "year": SMOKING_YEAR,
                "source": "WHO GHO M_Est_smk_curr_std", "source_url": url_s, "fetched": today,
                "note": (s[0].get("Comments") or "") if s else "not found",
            }
        out[c] = entry
        print(f"{c}: life table year {year}; male 30-34 nqx {entry['nqx']['male'].get('30')}; "
              f"smoking prevalence M/F {entry['smoking_prevalence']['male']['value']}/"
              f"{entry['smoking_prevalence']['female']['value']} ({SMOKING_YEAR})")
    (ROOT / "data" / "who_life_tables.json").write_text(json.dumps(out, indent=2))
    print("Wrote data/who_life_tables.json")


if __name__ == "__main__":
    fetch()
