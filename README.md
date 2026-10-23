# Kavach: AI insurance advisor

Kavach is a chatbot that acts as a personal insurance advisor for households in the **USA, India and Kenya**. It speaks each country's major languages, and it is meant to advise, not just answer questions. It finds the **protection gap**: the cover a household needs compared with the cover it holds, today and in 10 years once medical inflation is counted. Then it walks the user through closing that gap.

Team project on the theme "AI in daily lives". Built on a zero budget: free data, free APIs, free hosting.

> Educational guidance, not licensed financial advice. All premiums are model estimates, not quotes.

## What it does

- **Advises in stages.** It asks about you (age, dependants, income, debts, savings, existing cover), diagnoses your gaps with a 0–100 protection score, and recommends cover types and amounts, with government schemes first. It also compares options, stress-tests them (cover decay, cost of delay) and ends with a calm-decision checklist covering the free-look period and a cooling-off suggestion.
- **Gets every number from a tool.** The LLM (Gemini) only talks. Every premium, score and projection comes from 11 Python tools (`src/tools.py`) backed by a trained cost model and cited data. Under each reply, a **"How I calculated this"** panel shows the tool, its inputs and its sources.
- **Speaks 8 languages:**
  - USA: English, Spanish
  - India: English, Hindi, Tamil, Bengali, Telugu, Marathi
  - Kenya: English, Swahili
- **Has a Simple mode** for elderly or new readers, with short sentences and no jargon.
- **Draws charts:** cover decay, cost of delay, and a protection-score gauge (Plotly).
- **Has safeguards:**
  - No recommendation until the profile is at least 70% complete.
  - It never names a "best" insurer.
  - Investment-linked plans are never a first pick.
  - Prompt-injection attempts are refused.
  - Health details stay in the browser session and are never stored. There is no database.

## Run it locally

You need Python 3.12 and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```powershell
git clone https://github.com/ashwinderdhillon/kavach-insurance-advisor.git
cd kavach-insurance-advisor
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`. It is git-ignored, so never commit it.

```toml
GEMINI_API_KEY = "your-gemini-key"
# Optional second fallback provider (free key at console.groq.com):
# GROQ_API_KEY = "your-groq-key"
```

Then start the app:

```powershell
streamlit run app.py
```

It opens at http://localhost:8501.

**Tests** use mocked LLM responses, so they make no network calls and need no key:

```powershell
pip install -r requirements-dev.txt
pytest
```

## Deploy (Streamlit Community Cloud, free)

1. At [share.streamlit.io](https://share.streamlit.io), click **Create app**, pick this repo, branch `main`, file `app.py`.
2. Under **Advanced settings**, choose Python 3.12, and paste the same lines as your local `secrets.toml` into **Secrets**.
3. Click Deploy. Only `requirements.txt` is installed. Training-only packages are in `requirements-dev.txt`.

## How the free AI quota is used

`src/llm.py` puts every model behind one small interface, so the provider can be swapped (for example to the Claude API) by editing one file. It tries models in this order, moving on whenever one has used up its daily free quota:

1. Gemini Flash models (3.8 → 3.5)
2. Gemini Flash-Lite
3. Groq, only if `GROQ_API_KEY` is set

A question costs 1 LLM call and a tool turn usually costs 2. The quota is per Google project and shared by every user of a deployment.

## Project layout

| Path | What it is |
|---|---|
| `app.py` | Streamlit UI: sidebar (country, language, Simple mode, reset), chat, profile panel, charts |
| `src/agent.py` | Agent loop: up to 6 tool calls per turn; profile capture; compact history |
| `src/llm.py` | Provider layer: Gemini (google-genai) with Flash-Lite and Groq fallbacks; retry on rate limits |
| `src/prompts.py` | Base system prompt, one persona per country, greetings |
| `src/tools.py` | The 11 calculation tools the chatbot calls |
| `src/premium.py`, `src/term_life.py` | Health premium model (country-calibrated) and term life pricing |
| `src/charts.py` | Plotly charts |
| `data/*.json` | Country config, schemes, quotes, WHO and World Bank data; every value carries its source |
| `models/cost_model.pkl` | Trained XGBoost cost model (sklearn pipeline, needed at runtime) |
| `reports/results.md` | Full write-up of the model, calibration, validation and limitations |
| `src/train.py`, `src/build_config.py`, `src/fit_si_curve.py`, `src/validate*.py`, `src/fetch_*.py` | Training and data-building scripts (need `requirements-dev.txt`) |
| `tests/` | pytest suite: tools, models, agent, LLM layer and UI, all offline |

To rebuild the data and model: `python -m src.fetch_worldbank`, then `python -m src.fetch_who`, then `python -m src.build_config`, then `python -m src.train`, then `python -m src.fit_si_curve`, then `python -m src.validate`, then `python -m src.validate_term`.

## Data sources

Each figure in `data/country_config.json` records its value, year, source URL and fetch date. The main sources are:

- **Cost model training data:**
  - Kaggle *Medical Cost Personal Dataset* (`insurance.csv`): 1,337 US adults and their annual medical charges.
  - The tuned XGBoost model scores test R² 0.919 and test MAE $2,112, with 5-fold CV RMSE about 4,700.
- **Macro data:**
  - World Bank API: health spend per person, out-of-pocket share, CPI, GDP and PPP.
  - WHO Global Health Observatory: life tables (2023) and smoking prevalence.
- **Medical inflation:** WTW 2026 Global Medical Trends Survey.
- **Loss ratios:**
  - KFF / NAIC (USA)
  - IRDAI Annual Report 2024-25 (India)
  - IRA Kenya Insurance Industry Annual Report 2022 (Kenya)
- **Interest and exchange rates:** Fed FOMC, RBI MPC and CBK MPC rates; FRED and CBK exchange rates (Oct 2026).
- **Premium calibration and validation quotes:**
  - USA: KFF 2026 ACA benchmark and employer surveys, MoneyGeek term averages.
  - India: insurer prospectuses (Care Supreme), HDFC ERGO and nyvo.in broker data.
  - Kenya: Old Mutual and AAR rate tables, plus indicative broker term prices.
- **Smoker mortality:** Jha et al., NEJM 2013.
- **Chronic disease prevalence** (used for the "risk of waiting"):
  - USA: NCHS and CDC
  - India: ICMR-INDIAB-17
  - Kenya: STEPS 2015
- **Schemes and rules:**
  - USA: IRS, CMS and KFF (ACA, Medicare, Medicaid)
  - India: PIB / DFS (PM-JAY, PMJJBY, PMSBY), IRDAI regulations, India Motor Tariff
  - Kenya: SHA rules, Cap. 405, WIBA

## Limitations

The full list is in `reports/results.md` section 6.

- **The training data is from the USA.** India and Kenya premiums reuse the US *shape* of risk (age, BMI, smoker). They are scaled by health spending and calibrated on only 3 quotes (India) and 6 (Kenya), so treat them as estimates.
- **Smoker pricing** follows US costs. Indian and Kenyan insurers load smokers far less, so for smokers there the output is an expected *cost*, not a market price.
- **Quote quality varies.**
  - The Indian quotes come from broker pages.
  - The Kenyan rate tables date from 2017 and 2022, and the Kenyan term quotes name no insurer.
  - The US figures are national averages.
- **Scope:** health and term life are the core. Motor, accident and scheme checks are rule-based. Policy-wording RAG, insurer claim ratios and crop insurance are not built yet.
- **Rules change.** Scheme premiums, eligibility and thresholds were checked in October 2026. Verify them with the insurer or official portal before acting.
- **Some thresholds are judgement calls** and are labelled `assumption` in the config: score weights, the premium-to-income warning level, and the 70% completeness gate.
- **The free AI tier** allows only a small number of messages per day per deployment. When it runs out, the app shows a friendly message until the midnight Pacific reset.
- **LLMs can still make mistakes** despite the rules. Check every number in the "How I calculated this" panel.
