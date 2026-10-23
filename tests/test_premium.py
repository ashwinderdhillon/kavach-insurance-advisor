"""Sanity tests for the cost model, premium layer and country config."""
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src import premium

ROOT = Path(__file__).resolve().parents[1]
BASE = {"age": 35, "sex": "male", "bmi": 26, "children": 1, "smoker": False}
EXPECTED_CURRENCY = {"USA": "USD", "IND": "INR", "KEN": "KES"}


@pytest.fixture(scope="module")
def model():
    return joblib.load(ROOT / "models" / "cost_model.pkl")


def test_model_loads_and_predicts(model):
    X = pd.DataFrame([{"age": 30, "sex": "female", "bmi": 22, "children": 0,
                       "smoker": "no", "region": "northeast"}])
    assert model.predict(X).shape == (1,)


def test_predictions_positive(model):
    df = pd.read_csv(ROOT / "data" / "insurance.csv").drop(columns="charges")
    assert (model.predict(df) > 0).all()


@pytest.mark.parametrize("country", premium.COUNTRIES)
def test_smoker_costs_more(country):
    assert premium.predict_annual_cost({**BASE, "smoker": True}, country) > \
        premium.predict_annual_cost(BASE, country)


@pytest.mark.parametrize("country", premium.COUNTRIES)
def test_older_costs_more(country):
    assert premium.predict_annual_cost({**BASE, "age": 60}, country) > \
        premium.predict_annual_cost({**BASE, "age": 25}, country)


@pytest.mark.parametrize("country", premium.COUNTRIES)
def test_local_currency_and_positive_premium(country):
    r = premium.estimate_premium(BASE, country, sum_insured=1_000_000)
    assert r["currency"] == EXPECTED_CURRENCY[country]
    assert r["annual_premium"] > 0
    assert "loss_ratio" in r["assumptions"] and "calibration_factor" in r["assumptions"]


@pytest.mark.parametrize("country", ["IND", "KEN"])
def test_higher_sum_insured_costs_more(country):
    low = premium.estimate_premium(BASE, country, sum_insured=500_000)["annual_premium"]
    high = premium.estimate_premium(BASE, country, sum_insured=5_000_000)["annual_premium"]
    assert high > 1.2 * low
    assert premium.estimate_premium(BASE, country)["sum_insured_factor"] == 1.0  # None -> reference cover


@pytest.mark.parametrize("country", premium.COUNTRIES)
def test_project_cost_grows_with_inflation(country):
    p = premium.project_cost(BASE, country, 10)
    assert len(p["yearly_path"]) == 11
    assert p["cost_in_n_years"] > p["cost_today"]


def test_family_costs_more_than_one_adult():
    fam = {"members": [BASE, {**BASE, "sex": "female"}, {"age": 6, "bmi": 16}]}
    for c in premium.COUNTRIES:
        assert premium.predict_annual_cost(fam, c) > premium.predict_annual_cost(BASE, c)


def test_country_aliases():
    assert premium.estimate_premium(BASE, "India")["currency"] == "INR"
    with pytest.raises(ValueError):
        premium.estimate_premium(BASE, "France")


def _numeric_fields(node, path=""):
    """Yield (path, dict) for every config entry that carries a 'value'."""
    if isinstance(node, dict):
        if "value" in node:
            yield path, node
            return
        for k, v in node.items():
            yield from _numeric_fields(v, f"{path}.{k}")


def test_config_has_no_unsourced_numbers():
    cfg = json.loads((ROOT / "data" / "country_config.json").read_text(encoding="utf-8"))
    fields = [f for c in premium.COUNTRIES for f in _numeric_fields(cfg[c], c)]
    assert len(fields) > 20
    for path, f in fields:
        assert f.get("source"), f"{path} has no source"
        if f["value"] is None:
            assert f.get("flag"), f"{path} is null without a flag"
        # Real-world data (not assumptions/calibration) must also be traceable to a URL and a year.
        if f.get("source_type") not in ("assumption", "calibration") and "Identity" not in f["source"]:
            assert f.get("source_url"), f"{path} has no source_url"
            assert f.get("year"), f"{path} has no year"


def test_validation_quotes_are_sourced():
    q = json.loads((ROOT / "data" / "validation_quotes.json").read_text(encoding="utf-8"))
    for c in premium.COUNTRIES:
        assert len(q[c]) >= 3
        for item in q[c]:
            assert item["source"] and item["source_url"].startswith("http")
