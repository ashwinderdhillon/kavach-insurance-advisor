"""Fetch the latest non-empty World Bank values for USA, IND, KEN.

Writes data/worldbank_raw.json; build_config.py merges it into country_config.json.
Every value keeps its year, the exact API URL it came from and the fetch date.
"""
import json
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = ["USA", "IND", "KEN"]
INDICATORS = {
    "SH.XPD.CHEX.PC.CD": "health_spend_per_capita_usd",
    "SH.XPD.OOPC.CH.ZS": "out_of_pocket_pct_health_spend",
    "FP.CPI.TOTL.ZG": "cpi_inflation_pct",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "PA.NUS.PPP": "ppp_conversion_factor_lcu_per_intl_usd",
    # Extra: official exchange rate in the same year, needed to convert the
    # USD health spend back to local currency before applying PPP.
    "PA.NUS.FCRF": "wb_official_fx_lcu_per_usd",
}
API = "https://api.worldbank.org/v2/country/{c}/indicator/{i}?format=json&mrnev=1"


def fetch():
    today = date.today().isoformat()
    out = {c: {} for c in COUNTRIES}
    for code, field in INDICATORS.items():
        for c in COUNTRIES:
            url = API.format(c=c, i=code)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload[1] if len(payload) > 1 and payload[1] else []
            row = rows[0] if rows else None
            out[c][field] = {
                "value": row["value"] if row else None,
                "year": int(row["date"]) if row else None,
                "unit": code,
                "source": f"World Bank WDI, indicator {code}",
                "source_url": url,
                "fetched": today,
            }
            v = out[c][field]
            print(f"{c} {field:40s} {v['value']!s:>22} ({v['year']})")
    # The FX used to turn health spend (USD) back into local currency must be
    # from the same year as the health spend, not the latest year.
    for c in COUNTRIES:
        year = out[c]["health_spend_per_capita_usd"]["year"]
        url = f"https://api.worldbank.org/v2/country/{c}/indicator/PA.NUS.FCRF?format=json&date={year}"
        rows = requests.get(url, timeout=30).json()[1] or []
        out[c]["wb_fx_lcu_per_usd_health_year"] = {
            "value": rows[0]["value"] if rows else None, "year": year, "unit": "PA.NUS.FCRF",
            "source": "World Bank WDI, indicator PA.NUS.FCRF", "source_url": url, "fetched": today,
        }
        print(f"{c} {'wb_fx_lcu_per_usd_health_year':40s} {out[c]['wb_fx_lcu_per_usd_health_year']['value']!s:>22} ({year})")
    (ROOT / "data" / "worldbank_raw.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    fetch()
