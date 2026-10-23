"""Sanity tests for the term life module and its WHO data."""
import json
from pathlib import Path

import pytest

from src import premium, term_life

ROOT = Path(__file__).resolve().parents[1]
COVER = {"USA": 500_000, "IND": 10_000_000, "KEN": 5_000_000}
EXPECTED_CURRENCY = {"USA": "USD", "IND": "INR", "KEN": "KES"}


def _p(country, **kw):
    args = {"age": 35, "sex": "male", "smoker": False, "cover": COVER[country], "term_years": 20}
    args.update(kw)
    return term_life.estimate_term_premium(country=country, **args)


def test_who_tables_present_and_sourced():
    who = json.loads((ROOT / "data" / "who_life_tables.json").read_text())
    for c in premium.COUNTRIES:
        assert who[c]["year"] >= 2019
        assert who[c]["source_url_nqx"].startswith("https://ghoapi")
        for sex in ("male", "female"):
            nqx = who[c]["nqx"][sex]
            assert all(0 < v <= 1 for v in nqx.values())
            assert 0 < who[c]["smoking_prevalence"][sex]["value"] < 1


@pytest.mark.parametrize("country", premium.COUNTRIES)
def test_local_currency_and_positive(country):
    r = _p(country)
    assert r["currency"] == EXPECTED_CURRENCY[country]
    assert r["annual_premium"] > 0
    assert 0 < r["probability_of_death_during_term"] < 1
    assert {"insured_mortality_factor", "life_loss_ratio", "policy_fee"} <= set(r["assumptions"])


@pytest.mark.parametrize("country", premium.COUNTRIES)
def test_smoker_pays_more(country):
    assert _p(country, smoker=True)["annual_premium"] > _p(country)["annual_premium"]


@pytest.mark.parametrize("country", premium.COUNTRIES)
def test_older_pays_more(country):
    assert _p(country, age=50)["annual_premium"] > _p(country, age=30)["annual_premium"]


@pytest.mark.parametrize("country", premium.COUNTRIES)
def test_more_cover_and_longer_term_cost_more(country):
    base = _p(country)["annual_premium"]
    assert _p(country, cover=2 * COVER[country])["annual_premium"] > base
    assert _p(country, term_years=30)["annual_premium"] > base


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        term_life.estimate_term_premium(10, "male", False, 100_000, 20, "USA")
    with pytest.raises(ValueError):
        term_life.estimate_term_premium(30, "male", False, 100_000, 20, "France")


def test_term_quotes_are_sourced():
    q = json.loads((ROOT / "data" / "term_quotes.json").read_text(encoding="utf-8"))
    for c in premium.COUNTRIES:
        assert 2 <= len(q[c]) <= 3
        for item in q[c]:
            assert item["source"] and item["source_url"].startswith("http")
