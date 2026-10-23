# Premium model: results

*Everything below was produced by the scripts in `src/`. All numbers were printed by actual runs on 23 Oct 2026.*

## 1. The dataset

- **Source:** Kaggle `insurance.csv`. It has 1,338 US adults aged 18–64 and records age, sex, BMI, children, smoker, US region and **annual medical charges** (USD). One duplicate row was removed, leaving 1,337.
- **Shape of the target:** charges are skewed. The median is $9,382 and the mean $13,270. Skew is 1.52, and falls to −0.09 after a log transform.
- **Biggest single effect:** smoking. The median charge is $34,456 for smokers against $7,345 for non-smokers, about 4.7×. Cost also rises steadily with age, and high BMI raises cost **mainly for smokers**.

Plots: `eda_1_charges_distribution.png`, `eda_2_age_smoker.png`, `eda_3_bmi_smoker.png`.

## 2. Model comparison

**Setup:**
- **Split:** 80/20, stratified on smoker, `random_state=42`. That gives 1,069 rows for training and 268 for test.
- **Features:** the 6 raw columns plus `bmi_category`, `age_band` and `smoker × BMI`.
- **Target:** each model was tried on raw charges and on log charges. The version with the better 5-fold CV RMSE was kept. RMSE is the right yardstick here because a premium needs the **mean** cost, and log models drift toward the median, which runs low.
- **Tuning:** the top 2 models got `RandomizedSearchCV` (30 iterations, 5-fold).

| Model | Target | CV RMSE | Train R² | **Test R²** | **Test MAE** | Test RMSE | Test MAPE | Train−test R² gap |
|---|---|---|---|---|---|---|---|---|
| **XGBoost (tuned)** | raw | **4,692** | 0.868 | **0.919** | **$2,112** | $3,426 | 29.1% | −0.051 |
| RandomForest (tuned) | log | 4,753 | 0.874 | 0.927 | $1,585 | $3,252 | 15.5% | −0.053 |
| RandomForest | log | 4,853 | 0.958 | 0.912 | $1,765 | $3,557 | 19.7% | +0.046 |
| XGBoost | raw | 4,930 | 0.923 | 0.902 | $2,303 | $3,762 | 30.6% | +0.022 |
| LinearRegression | raw | 4,954 | 0.836 | 0.899 | $2,508 | $3,823 | 31.0% | −0.062 |
| LightGBM | log | 5,072 | 0.932 | 0.901 | $2,132 | $3,784 | 21.4% | +0.031 |

**Best model: tuned XGBoost on raw charges** (saved as `models/cost_model.pkl`, preprocessing included).

**Why this one:**
- It has the lowest cross-validated error. I chose on CV, not on test, so the test set never influenced the choice.
- It predicts the mean cost directly, which is what premium maths needs.
- Its tuned settings are shallow (depth 2, slow learning rate), which keeps it from memorising the data.

**Two honest notes:**
- **RandomForest (tuned, log target) did slightly better on the test set** (R² 0.927, much lower MAPE). Its log target makes it good at typical cases but biased low on the mean, and choosing it after seeing test scores would be cherry-picking.
- **Test R² is higher than train R² for the tuned models.** That is not leakage. The test split happens to be easier, and the 5-fold CV estimate (RMSE ≈ 4,700, about R² 0.84) is the more reliable expectation for new people. The untuned trees do overfit a little: RandomForest scores train R² 0.96 against test 0.91.

## 3. Top cost drivers (SHAP, `shap_importance.png`)

| Rank | Feature | Mean \|SHAP\| (USD) | Plain meaning |
|---|---|---|---|
| 1 | smoker × BMI | 5,304 | A heavy smoker costs far more, and the gap widens with BMI |
| 2 | age | 2,877 | Older means more costly, steadily |
| 3 | smoker (no) | 1,914 | Being a non-smoker pulls cost down |
| 4 | children | 760 | Small effect |
| 5 | BMI | 685 | Matters mostly through the smoker interaction |

Sex and US region barely matter.

## 4. Country calibration: how a US model prices India and Kenya

All inputs are in `data/country_config.json`. Each one records its value, year, source, URL and fetch date.

| Input | USA | India | Kenya | Source |
|---|---|---|---|---|
| Health spend per person (USD, 2023) | 13,473 | 84.7 | 85.0 | World Bank SH.XPD.CHEX.PC.CD |
| Same, PPP-adjusted (intl $) | 13,473 | 348 | 273 | WB health spend × WB FX 2023 ÷ PA.NUS.PPP (2025) |
| Out-of-pocket share of health spend (2023) | 10.9% | 43.9% | 24.2% | World Bank SH.XPD.OOPC.CH.ZS |
| CPI inflation | 2.9% (2024) | 2.4% (2025) | 4.1% (2025) | World Bank FP.CPI.TOTL.ZG |
| GDP per person (USD, 2025) | 90,027 | 2,702 | 2,363 | World Bank NY.GDP.PCAP.CD |
| Medical inflation, 2026 | 9.6% | 12.9% | 9.5% | WTW 2026 Global Medical Trends Survey (gross trend) |
| Health loss ratio | 85% individual / 88% group (2024) | 68.06% (FY25, standalone health insurers) | 74.6% (2022, medical class) | KFF / NAIC; IRDAI Annual Report 2024-25 Table I.14; IRA Kenya Insurance Industry Annual Report 2022 |
| Policy interest rate | 3.75–4.00% | 5.50% | 8.75% | Fed FOMC 16 Sep 2026; RBI MPC 7 Oct 2026; CBK MPC 7 Oct 2026 |
| Exchange rate per USD | 1 | 94.96 (16 Oct 2026) | 129.91 (22 Oct 2026) | FRED DEXINUS (Fed H.10); CBK |

**Kenya's loss ratio is now checked against the IRA's own report.** The IRA Insurance Industry Annual Report 2022 (p.58, via the Parliament of Kenya library copy) says: *"Motor Private, Medical and Motor Commercial classes registered loss ratios above the global benchmark at 80.3%, 74.6% and 70.6% respectively."* Newer figures could not be checked in a primary document, so they form a sensitivity range:
- 81% for H1-2025, from IRA release text seen only in search results
- 101.8% for 2025, from a secondary analytics report

Because the premium is calibrated to market quotes, the loss ratio does not change the premium. It only changes how much of the premium is counted as expected claims. Refitting under each case gives:

| Loss ratio | Calibration factor | Expected claims, age 30, KES 1 M | Premium |
|---|---|---|---|
| 74.6% (verified) | 5.55 | KES 29,349 | KES 39,341 |
| 81% | 6.02 | KES 31,867 | KES 39,342 |
| 101.8% | 7.57 | KES 40,050 | KES 39,342 |

**Method** (`src/premium.py`):

1. **Cost in local currency** = US model cost × (country's PPP health spend ÷ US health spend) × PPP factor. This simplifies to US model cost × (country health spend in local currency ÷ US health spend in USD).
2. **Trend to today:** multiply by (1 + medical inflation) for each year since 2023.
3. **Calibration factor:** one number per country, fitted in step 5 below.
4. **Premium** = expected claims ÷ loss ratio × age-band loading × **sum-insured factor** (section 5b). The loading is neutral (1.0) and labelled as an assumption in the config.
5. **USA only:** US law (ACA) forbids pricing individual cover on health. The pool's total cost is shared out using only the legal factors: the CMS federal age curve and a tobacco surcharge of at most 1.5×. The function returns both this regulated premium and the pure risk premium. ACA plans have no annual cap, so the cover level is the **metal tier** (section 5b).
6. **Children** (not in the training data) are priced as a healthy 18-year-old × 0.838, the CMS age-curve ratio for ages 0–14 against 18.
7. **Family floaters** (India, Kenya) are priced the way insurers price them: the eldest member's premium in full, plus a fixed share of each other member's individual premium (`floater_addon_factor`).
   - **India: 0.509.** This is the mean of the three floater-vs-individual illustrations in Care Supreme's own prospectus (Annexure V, Apr 2026, Zone 1, ₹5 L):
     - ages 51 + 46: ₹32,881 vs ₹23,450 + ₹18,546, an add-on share of 0.509
     - ages 60 + 56 + 17: ₹50,727 vs ₹31,113 + ₹31,113 + ₹5,046, a share of 0.542
     - ages 64 + 61: ₹54,509 vs 2 × ₹36,965, a share of 0.475
   - **Kenya: 0.692.** This is the mean of AAR's family-shared M+1 rate vs its individual rate (0.624) and Old Mutual's spouse rate vs its principal rate (0.760).
   - **USA: 1.0.** ACA rules price a family as the sum of its members (45 CFR 147.102(c)), so there is no floater discount.

## 5. Real-world validation

The quotes are in `data/validation_quotes.json`, each with its source. **Only independent published quotes are used for fitting and scoring: 3 for the USA, 3 for India, 6 for Kenya.**
- **USA:** two different markets.
  - The KFF 2026 average benchmark silver premium (ACA individual market) for a 40-year-old: $625/month.
  - The KFF 2026 Employer Health Benefits Survey averages (published 22 Oct 2026, 1,937 firms): $9,804 for single cover and $28,303 for a family of four.
  - Employer premiums are modelled as group cover: the average expected cost of a working-age pool ÷ the group loss ratio (88%, KFF).
  - Three more US rows (age 30, a 45-year-old smoker, an ACA family) were *derived* from the $625 figure with the CMS age curve and tobacco cap. They are shown only to illustrate shape and are left out of every score.
- **India:** insurer premiums from broker comparison pages, taking the median of 3–4 plans for each profile.
- **Kenya:** two insurers' published rate tables, both for KES 1 M inpatient cover.
  - Old Mutual AfyaImara (retail, 2022). Priced in the quote's own year.
  - AAR Business Medical Plan (SME, 2017). **Inflation adjustment:** AAR's 2017 prices were brought to 2026 using Kenya's medical inflation from `country_config.json` (9.5%/yr, WTW): × 1.095⁹ = × 2.263, so 16,801 → 38,024, 34,693 → 78,518 and 59,734 → 135,191. Applying one 2026 trend rate to 2017–2025 is itself an assumption; WTW reports 8.2–9.5% for 2024–26.

**Round 1 (uncalibrated)** was off by more than 40% in every country, so **only the calibration factor was adjusted. The ML model was not touched.** One factor was fitted per country as the geometric mean of quote ÷ model. To keep the test honest, the **leave-one-out** column predicts each quote with a factor fitted on the *other* quotes only.

| Country | Profile | Quote | Uncalibrated | Dev. | Leave-one-out | **Dev.** | Final calibrated | Dev. |
|---|---|---|---|---|---|---|---|---|
| USA | ACA, age 40 non-smoker (KFF) | $7,500 | $15,753 | +110% | $8,458 | **+13%** | $8,125 | +8% |
| USA | Employer, single (KFF EHBS) | $9,804 | $20,039 | +104% | $10,614 | **+8%** | $10,336 | +5% |
| USA | Employer, family of four (KFF EHBS) | $28,303 | $48,030 | +70% | $23,181 | **−18%** | $24,774 | −12% |
| *USA (derived, not scored)* | *ACA age 30 / 45 smoker / family* | *$6,661 / $12,712 / $23,322* | | | | | *$7,216 / $13,771 / $25,266* | *+8% each* |
| India | 30 NS, ₹10 L | ₹10,350 | ₹5,017 | −52% | ₹10,702 | **+3%** | ₹10,583 | +2% |
| India | Family 35/34 + child, ₹15 L | ₹25,522 | ₹13,861 | −46% | ₹31,296 | **+23%** | ₹29,239 | +15% |
| India | Couple 63/62, ₹15 L | ₹68,239 | ₹27,614 | −60% | ₹53,819 | **−21%** | ₹58,249 | −15% |
| Kenya | Old Mutual, 30, KES 1 M (2022) | 35,223 | 4,934 | −86% | 26,018 | **−26%** | 27,365 | −22% |
| Kenya | Old Mutual, 45 (2022) | 39,385 | 9,249 | −77% | 54,082 | **+37%** | 51,298 | +30% |
| Kenya | Old Mutual, family 35/33 + 2 kids (2022) | 96,212 | 17,356 | −82% | 96,271 | **+0.1%** | 96,261 | +0.1% |
| Kenya | AAR, age 18–30 (2017 → 2026) | 38,024 | 7,043 | −81% | 39,272 | **+3%** | 39,061 | +3% |
| Kenya | AAR, age 41–50 (2017 → 2026) | 78,518 | 13,297 | −83% | 72,830 | **−7%** | 73,749 | −6% |
| Kenya | AAR, family M+3 (2017 → 2026) | 135,191 | 24,952 | −82% | 139,041 | **+3%** | 138,391 | +2% |

**Mean absolute deviation, which works as a validation accuracy score:**

| Country | Quotes | Uncalibrated | Leave-one-out (out of sample) | Calibrated (in sample) | Calibration factor |
|---|---|---|---|---|---|
| USA | 3 | 94.7% | **13.0%** | 8.7% | 0.516 |
| India | 3 | 52.2% | **15.7%** | 10.5% | 2.109 |
| Kenya | 6 | 81.8% | **12.8%** | 10.6% | 5.546 |

**History of changes to this check:**
- An earlier version scored the USA on one published number plus three derived from it, which gave a circular 0%. The derived rows are now excluded and two employer-market quotes were added.
- Kenya was first fitted on Old Mutual alone (31.6% leave-one-out). Adding AAR's inflation-adjusted table brought it to 19.9%.
- **The sum-insured curve (section 5b) made India's check worse: 22.5% → 28.1% leave-one-out.** Two of the three Indian quotes are at ₹15 L, so they now cost 1.16× the ₹10 L price, and the family quote IN-2 moves from +38% to +49%. A sensitivity run shows the curve is not the main problem. With a flat curve the error is 22.5%, with the Ditto-only slope (0.184) it is 25.3%, and with the fitted slope it is 28.1%. IN-2 is 27–49% too high in every case. Families are over-priced in Kenya too (KE-3 +15%, KE-6 +18%), so the cause is **family-floater pricing**: insurers price a floater well below the sum of its members. The curve was kept as fitted. Tuning it to three validation quotes would be fitting to the test set.
- **Floater pricing (method step 7) fixed this.** Leave-one-out error fell from 28.1% to **15.7%** for India and from 19.9% to **12.8%** for Kenya.
  - India's add-on share comes from Care's prospectus, which is independent of the Ditto and nyvo validation quotes.
  - Kenya's add-on share comes from the same two brochures as KE-3 and KE-6, so its family fit is not fully independent. As a check, Kenya was re-run with India's share (0.509): leave-one-out error is **16.5%**, still better than 19.9%. KE-3 and KE-6 then land about 10% low instead of 15–18% high.
  - What remains is mostly the age gradient: Kenyan 30-year-olds are priced −22% and 45-year-olds +30% (limitation 3). The Indian senior couple (IN-3) is −15%.

**Why each country needed calibrating:**
- **USA (factor 0.52):** insurance.csv charges (mean $13,270) are higher than what insurers actually pay per working-age adult, and ACA silver plans pay only about 70% of covered costs.
- **India (2.11) and Kenya (5.55):** national health spend per person is an average over everyone, including the many who use cheap public clinics or skip care. People who buy private cover use private hospitals that cost several times more.
- **Kenya in numbers:**
  - The US model says a healthy 30-year-old costs $4,570, about **34%** of US spend per person. In the US, older people account for much of the national average.
  - Old Mutual's price for a 30-year-old (KES 35,223 × 74.6% loss ratio) implies insurer claims of KES 26,276. That is **2.4×** Kenya's spend per person (KES 10,850 in 2022).
  - So the raw scaling puts this person at about a third of the national average, while the market prices them at 2.4× it. That is a 7.1× gap for this profile, and about 5.2× averaged over the 6 quotes.
  - Product richness also adds to the gap. The quoted plan includes maternity, East-Africa-wide cover and a KES 750,000 critical-illness lump sum.

## 5b. How premiums scale with cover (`src/fit_si_curve.py`)

**The problem:** the earlier version capped expected claims at the sum insured with a lognormal spread learned from insurance.csv (σ = 0.30). That spread is so narrow that claims almost never reach ₹5 L, so the premium was **identical from ₹5 L to ₹1 Cr** (₹9,497 for a 30-year-old at every level). Real premiums rise with cover, though far less than in proportion.

**Data** (`data/sum_insured_quotes.json`): the same plan and profile priced at several cover levels, so the plan, profile and year cancel out and only the ratio is used.

| Country | Series | Points | Source |
|---|---|---|---|
| India | 4 | 12 | HDFC ERGO Optima Secure+ (Ditto, 2026): age 25, 2A, 2A+1C and parents (60, 60) at ₹15 L / ₹25 L / ₹50 L |
| India | 6 | 17 | Care Supreme, ABHI Activ One Max, HDFC Optima Secure Plus (nyvo.in, Oct 2026): age 30 at ₹5 L / ₹10 L / ₹25 L / ₹1 Cr, age 45 at ₹25 L / ₹1 Cr |
| Kenya | 6 | 30 | Old Mutual AfyaImara insurer rate table (2022): 5 principal age bands + child at KES 0.5 M / 1 M / 3 M / 5 M / 10 M |
| Kenya | 7 | 77 | AAR SME insurer rate table (2017): 6 age bands + family M+3, 11 limits from KES 0.5 M to 10 M |
| USA | 1 | 4 tiers | KFF 2026 average marketplace premium for a 40-year-old, by metal tier |

**What I could not get:** Care Health's official rate chart blocks automated access (HTTP 403), and HDFC ERGO's full prospectus copy online is truncated before the rate annexure. One search summary quoted Care rate-chart figures (₹10,882 / ₹12,292 / ₹17,483 / ₹25,598), but a follow-up search found no page containing them, so they were **not used**. nyvo.in states that only its ₹10 L figures come from its premium database and the other tiers are its own extrapolation. Those 6 series are therefore labelled "broker-scaled" in the data file.

**Method:** ln P = a_series + f(x), with x = ln(SI / SI_ref). Each series gets its own intercept; the curve f is shared per country. SI_ref is ₹10 L / KES 1 M, the cover level the calibration quotes sit at. Three shapes were compared on leave-one-series-out (LOSO) error, where each held-out series is predicted from one observed premium plus the curve:

| Country | Power: f = b·x | Quadratic in log-log | Power with age term | Chosen |
|---|---|---|---|---|
| India | b = 0.374, LOSO **12.5%** | b₁ = 0.474, b₂ = −0.048, LOSO 12.7% | b = 0.363, c = −0.017, LOSO 12.6% | **power** (simplest, best) |
| Kenya | b = 0.239, LOSO 9.0% | b₁ = 0.165, b₂ = +0.044, LOSO **8.2%** | b = 0.237, LOSO 8.8% | **quadratic** |

All three shapes are monotone in both countries. Beyond the fitted range (₹5 L–₹1 Cr, KES 0.5–10 M) the curve continues along its end tangent.

**Results for a 30-year-old male non-smoker:**

| | Model | Published |
|---|---|---|
| India ₹5 L / ₹10 L / ₹25 L / ₹1 Cr | ₹8,169 / ₹10,583 / ₹14,903 / ₹25,016 (factor 0.77 / 1 / 1.41 / 2.36) | nyvo medians ₹6,400 / ₹10,350 / ₹15,100 / ₹24,900 |
| Kenya KES 0.5 M / 1 M / 3 M / 10 M | 35,831 / 39,341 / 49,711 / 72,494 (factor 0.91 / 1 / 1.26 / 1.84) | Old Mutual 30–40 (2022 prices) 33,863 / 35,223 / 52,342 / 64,692 |
| USA bronze / silver / gold / platinum, age 40 | $5,928 / $8,125 / $7,995 / $8,995 | KFF × 12: $5,472 / $7,500 / $7,380 / n/a |

After the floater fix (method step 7), the Indian model matches nyvo within 2% at ₹10 L, ₹25 L and ₹1 Cr. It is 28% high at ₹5 L, where nyvo's own figure is an extrapolation.

**What the curve says:** doubling cover in India raises the premium by about 30% (2^0.374). Going from ₹10 L to ₹1 Cr costs 2.4×, not 10×. In Kenya, KES 10 M costs 1.84× KES 1 M, against 1.84× in Old Mutual's own table (64,692 / 35,223). This is the "buy a big cover cheaply" argument for the advisor, made with real numbers.

**Sensitivity and caveats:**
- **India's slope depends on which source you trust.** The 4 Ditto series alone give b = 0.184 (₹1 Cr would be 1.53× ₹10 L instead of 2.36×), but they span only ₹15 L–₹50 L. The young-adult Ditto series have slopes of 0.12–0.14, while the parents (60, 60) series has 0.34. So cover costs more per step for older buyers, as theory predicts: big claims are likelier for them. The age-interaction model did not improve LOSO error, because the nyvo series show no age difference (the broker scaled them all the same way).
- **Kenya's plans change more than the limit.** Old Mutual's higher options also upgrade the room (general ward → en-suite) and add air evacuation. Every Old Mutual option includes the same KES 750,000 critical-illness lump sum, and every AAR option the same KES 1 M accident cover. The fixed parts flatten the curve slightly.
- **USA: market tiers, not an actuarial curve.** Actuarial value (45 CFR 156.140) says gold should cost 80/70 = 1.14× silver. In the 2026 market, lowest-cost gold ($615) costs about the same as benchmark silver ($625), because insurers "silver-load" the cost of cost-sharing reductions onto silver plans. The tool uses the market ratios (bronze 0.730, silver 1, gold 0.984), since that is what buyers pay. Platinum (no KFF figure) is gold × 0.90/0.80 = 1.107, labelled as derived. The agent can turn this into advice: gold often costs no more than silver, with a lower deductible (standard silver deductible in 2026: $5,304, KFF).

## 6. Limitations (please read before quoting numbers)

1. **US-origin training data.** Age, BMI and smoker effects are learned from US adults, so the *shape* of risk in India and Kenya is assumed to match the US. Only the *level* is calibrated, with 3 quotes each for the USA and India and 6 for Kenya.
2. **Smoker pricing.** The model makes a 45-year-old smoker about 4.7× the cost of a 30-year-old non-smoker. Indian and Kenyan retail insurers price smoking far less steeply or not at all, and I found no published smoker quote there to test against. For smokers in India and Kenya, read the output as expected *cost*, not a market premium.
3. **Age gradient.** Kenyan rate tables are flatter by age than the model: the quoted 30-year-old is underestimated by 28% and the 45-year-old overestimated by 21%. Fitting age loadings on a handful of quotes would be overfitting, so the loadings stay neutral.
4. **Sum-insured scaling is now fitted to published prices (section 5b), with caveats.**
   - India's curve leans on broker-scaled tiers (nyvo). The insurer-chart series alone imply a flatter curve.
   - One curve per country ignores the steeper slope seen for older buyers.
   - **Family floaters** now use one add-on share per country (eldest member + 0.51 / 0.69 × the others). Real insurers vary this share by age and family size; Care's three illustrations range from 0.475 to 0.542. The Indian family quote IN-2 is still +15%.
5. **Ages 64 and over** are priced as 64, and **children** use a CMS ratio. Both are outside the training data.
6. **Quote quality.**
   - The Indian quotes come from broker comparison pages, not insurer filings.
   - The Kenyan rate tables are from 2022 (Old Mutual) and 2017 (AAR, an SME group product), trended with WTW medical inflation.
   - The US quotes are national averages (ACA benchmark and employer survey), not individual insurer quotes.
7. **Kenya's loss ratio** is verified for 2022 (74.6%) but newer years are not. This does not change premiums (see the sensitivity table), only the expected-claims figure.
8. **Medical inflation** is WTW's insurer-reported cost trend (price plus usage). That is higher than consumer medical CPI and is the right input for premiums, but it overstates pure price inflation.

## 7. Term life premiums (`src/term_life.py`)

**Tool:** `estimate_term_premium(age, sex, smoker, cover, term_years, country)` returns a level yearly premium in local currency, plus every assumption used.

**How it works:**

1. **Death rates:** WHO Global Health Observatory life tables, **2023** (latest year), by sex and 5-year age group (indicator `LIFE_0000000030`, nqx). They are converted to yearly probabilities. The open 85+ group uses WHO's death rate (`LIFE_0000000029`).
2. **Smokers vs non-smokers:** population death rates mix both groups, so they are split using two inputs:
   - smoking prevalence by sex (WHO `M_Est_smk_curr_std`, 2022): US 18.7% / 12.5%, India 18.0% / 1.6%, Kenya 14.6% / 1.1% (men / women)
   - the smoker death-rate multiplier from **Jha et al., NEJM 2013**: current smokers die at **2.8×** (men) and **3.0×** (women) the rate of never-smokers
3. **Insured lives:** people who buy term cover pass medical checks and are richer than average, so they die less often than the population. One factor per country (`insured_mortality_factor`) scales WHO death rates down. It was fitted to market quotes, as for health.
4. **Premium** = present value of expected death claims over the term ÷ present value of premiums paid while alive (discounted at the policy rate), ÷ life loss ratio, + a fixed yearly policy fee.

**Two inputs are assumptions, labelled as such in the config:**
- **Life loss ratio (65%):** no regulator in these markets publishes a term-life loss ratio. IRDAI and IRA report claim *settlement* ratios, which measure something else.
- **Policy fee:** $60 / ₹600 / KES 600 a year.

Because the mortality factor is refitted, the loss ratio does not move premiums. At 55%, 65% and 75%, the fitted 30-year-old US premium is $486, $487 and $486; only the factor changes (0.23 / 0.27 / 0.31).

**Validation** (`src/validate_term.py`, quotes in `data/term_quotes.json`):

| Country | Profile | Quote / yr | Uncalibrated | Dev. | Leave-one-out | **Dev.** | Calibrated | Dev. |
|---|---|---|---|---|---|---|---|---|
| USA | M30 NS, $500k, 20 yr | $456 | $1,655 | +263% | $501 | **+10%** | $486 | +7% |
| USA | M45 NS, $500k, 20 yr | $1,080 | $3,958 | +267% | $1,125 | **+4%** | $1,109 | +3% |
| USA | M40 smoker, $500k, 20 yr | $2,328 | $7,673 | +230% | $2,021 | **−13%** | $2,123 | −9% |
| India | M30 NS, ₹1 Cr, to age 60 | ₹9,528 | ₹56,222 | +490% | ₹10,832 | **+14%** | ₹10,382 | +9% |
| India | M45 NS, ₹1 Cr, to age 60 | ₹22,008 | ₹104,912 | +377% | ₹17,268 | **−22%** | ₹18,760 | −15% |
| India | F30 NS, ₹1 Cr, to age 60 | ₹8,100 | ₹46,934 | +479% | ₹9,042 | **+12%** | ₹8,722 | +8% |
| Kenya | ~28 NS, KES 5 M, 20 yr | 11,000 | 24,981 | +127% | 11,078 | **+1%** | 11,052 | +0.5% |
| Kenya | 35 NS, KES 5 M, 20 yr | 18,000 | 41,599 | +131% | 18,361 | **+2%** | 18,239 | +1% |
| Kenya | ~48 NS, KES 5 M, 20 yr | 42,500 | 95,560 | +125% | 41,363 | **−3%** | 41,743 | −2% |

| Country | Uncalibrated | Leave-one-out | Calibrated | Insured mortality factor |
|---|---|---|---|---|
| USA | 253% | **9.1%** | 6.0% | 0.27 |
| India | 449% | **15.6%** | 10.5% | 0.17 |
| Kenya | 128% | **1.8%** | 1.2% | 0.43 |

**Quote sources:**
- **USA:** MoneyGeek 2026 averages (quotes from 30 insurers, average health).
- **India:** nyvo.in medians of 5 insurers' plans (May 2026 data, nil GST).
- **Kenya:** two broker/blog pages (GetCovered Kenya, Keryl Kelonye) giving indicative prices with **no insurer named**. These are the weakest quotes in this project.

**What the factors mean:**
- Insured Americans die at about 27% of the population rate.
- Indian term buyers die at about 17% of the population rate. India's gap is largest because online term buyers are a small, urban, affluent group, while the national table includes everyone.
- Kenya's near-perfect fit (1.8%) says more about the smooth, broker-made Kenyan price ranges than about the model.

**Term life limitations:**
- One insured-mortality factor per country assumes the insured-to-population ratio is the same at every age. Real selection wears off over time and differs by age.
- The smoker multiplier is from a US cohort and is applied to all three countries. Smoker quotes were only found for the USA (−9% after calibration). India's figure also ignores smokeless tobacco.
- Kenyan quotes are indicative ranges (midpoints used), with ages assigned to band midpoints and sex assumed male.
- Only level-premium pure term is modelled: no return-of-premium, riders or limited-pay plans.

## 8. Chatbot tools (`src/tools.py`, Phase 2)

Eleven functions the chatbot calls. Each returns a plain JSON-serialisable dict with the result, the assumptions and a `sources` list. `call_tool(name, args)` dispatches LLM function calls and returns `{"error": ...}` instead of raising. `data/tool_schemas.json` declares all eleven in a JSON-Schema subset that works for Gemini `function_declarations` and maps one-to-one to Claude `input_schema`. A test checks that the schemas match the Python signatures.

| Tool | What it does | Main inputs from config / data |
|---|---|---|
| `predict_health_premium` | Wraps `premium.py`; sum-insured curve (India/Kenya) or metal tier (USA); optional family floater | calibration, loss ratio, SI curve, tier factors |
| `predict_term_premium` | Wraps `term_life.py` | WHO life tables, insured-mortality factor |
| `project_cover_decay` | C / (1 + π_m)^t, yearly series for the chart, half-life | WTW medical inflation |
| `calc_term_cover_needed` | Needs-based formula from CLAUDE.md section 5, r = (1+i)/(1+π) − 1 | policy rate, CPI |
| `cost_of_delay` | Term premium now vs later (same end age), skipped premiums, risk of dying uninsured, net nominal and present-value cost; delayed health entry price and waiting periods | term model, medical inflation |
| `check_govt_schemes` | Rules engine over `data/schemes.json`: PM-JAY, PMJJBY, PMSBY / Medicare, Medicaid expansion, ACA tax credit / SHA with contribution estimate | 2025 and 2026 poverty guidelines, KFF expansion list (Oct 2026) |
| `recommend_cover_types` | Priority order from CLAUDE.md section 3, one-line reasons, "avoid" list, schemes first; refuses below 70% profile completeness | completeness threshold, occupation-risk rank |
| `protection_score` | 0–100 today and in year 10: health adequacy, life adequacy, premium-to-income, red flags | weights and thresholds per country |
| `check_red_flags` | Room-rent cap, co-pay, waiting periods (IRDAI 36-month PED maximum), sub-limits, investment-linked plans, deductible above savings, non-ACA-compliant US plan | advisor rules |
| `accident_cover_needed` | India 5–10× income (insurer caps), Kenya 8× (WIBA 96 months), USA 60% income-replacement disability (BLS) | advisor rules |
| `motor_check` | Third-party mandate per country, India IDV depreciation (GR-8) and zero-dep advice | advisor rules |

**Where the rule numbers come from.** Every number sits in `country_config.json` under `advisor`, or in `schemes.json`, each with its source. Real rules carry a URL and year, for example:
- ACA 2026 affordability 9.96% (IRS Rev. Proc. 2025-25)
- 2026 out-of-pocket maximum $10,600 / $21,200
- IRDAI 30-day free look and 36-month PED cap
- PM-JAY 70+ expansion (Cabinet, 11 Sep 2024)
- PMJJBY ₹436 / ages 18–50 (PIB, May 2025) and PMSBY ₹20 (DFS, 2022)
- SHA 2.75% of income, minimum KES 300 a month
- Kenya Cap. 405 and the India Motor Tariff depreciation table

Judgement calls are marked `source_type: "assumption"` with the reasoning:
- score weights
- premium-to-income threshold of 10% for India and Kenya, by analogy with WHO SDG 3.8.2
- health cover targets of ₹10 L / KES 1 M
- the 70% completeness gate
- red-flag cut-offs

**Unresolved rules, handled conservatively:**
- **PM-JAY's ₹10,000-a-month exclusion** comes only from secondary sources, which disagree on whether it applies per member or per household. It is a `proxy` rule, so it can only return "check the portal", never "eligible".
- **WIBA's death benefit** is reported as 60 months' earnings in one source and 96 in another. Only the disablement figure (96 months), on which they agree, is used.
- **Kenya's free look** is confirmed only for life policies (IRA FAQ).

**Examples from the personas in `tests/test_tools.py`:**
- **India, 32, ₹18 L income, 2 dependants, ₹40 L loan, ₹5 L health cover, ₹50 L term, an endowment plan with a room-rent cap:**
  - Protection score 47.5 today, 41.0 in year 10.
  - Weakest area is health: adequacy 33 today, 10 in year 10 as ₹5 L decays.
  - Life need is ₹1.54 Cr against ₹50 L held.
- **₹5 L of Indian health cover is worth ₹1.49 L in 10 years.** This matches the CLAUDE.md example of about ₹1.5 L.
- **Cost of delay for a 32-year-old woman with a ₹2 Cr term to 60, waiting 5 years:**
  - The premium rises ₹3,717 a year, or ₹85.5 k over the remaining term.
  - But ₹90.1 k of premiums are skipped, and the expected loss from dying uninsured is ₹30.6 k.
  - Net: +₹26 k nominal, **−₹15.8 k in present value**.

  For young buyers the price effect of waiting is small. The honest case for buying now is the uninsured gap and the risk of becoming uninsurable, and the tool's notes tell the agent to say so rather than quote only the extra premium.

**Risk of waiting (insurability).** `cost_of_delay` now always returns `risk_of_waiting` and a `verdict`.

`risk_of_waiting` gives the chance of developing diabetes or hypertension during the delay. It is computed from age-specific prevalence (`data/chronic_conditions.json`) as (prevalence at the later age − prevalence now) ÷ (1 − prevalence now). The two conditions are then combined assuming they are independent. The data sources:

| Country | Hypertension by age | Diabetes by age | Confidence |
|---|---|---|---|
| USA | NCHS Data Brief 511, 2021–23: 23.4% / 52.5% / 71.6% | CDC 2017 release (2015 data): 4% / 17% / 25% | survey-based |
| Kenya | STEPS 2015: 17.7% at 18–29, 58.3% at 60–69 (secondary report) | STEPS 2015, Mohamed et al. 2018, Table 3: 0.6% / 2.3% / 7.1% / 6.0% | survey-based |
| India | ICMR-INDIAB-17 (Lancet D&E 2023), national 35.5% | ICMR-INDIAB-17, national 11.4% | **lower**: the paper has no age table, so the US age pattern is scaled to India's national figures (an assumption) |

What the result means depends on the country:
- **India:** a new condition becomes a pre-existing disease, which can be excluded for up to 36 months and may bring a loading or a refusal.
- **USA:** ACA plans cannot refuse cover or charge more, but you can only enrol at open enrolment.
- **Kenya:** a 1-year wait plus a KES 400 k sub-limit at Old Mutual, or a 6-month wait at AAR.
- **Term life, all countries:** cover is medically underwritten, so a condition can mean a loading or a decline. The size of a loading is not modelled.

Example chances of developing either condition over a 5-year wait:
- India, age 32: 9%
- India, age 45: 10%
- USA, age 35: 13%
- Kenya, age 35: 8%

The `verdict` always starts "Buy now if you need the cover". When the pure premium arithmetic favours waiting, the tool sets `price_comparison_favours_waiting: true` and puts the insurability risk in the same sentence. For example, the 32-year-old Indian woman above, whose present-value comparison is −₹15.8 k, sees: *"…about a 9% chance of developing diabetes or high blood pressure by age 37 … The premium arithmetic alone looks slightly cheaper if you wait, but only because it assumes you stay healthy and insurable…"*. A test checks that no persona's verdict recommends waiting.
- **USA, household of 3 in California on $42 k (158% of the 2025 poverty line):** eligible for the ACA premium tax credit. The same person in Texas on $18 k is not eligible for Medicaid (non-expansion state).
- **Kenya, informal rider on KES 360 k a year:** SHA contribution of KES 825 a month. Accident cover target is KES 2.9 M (8× income), ranked priority 2 because the work is high-risk.

**Tests:** `pytest` runs 89 tests, all passing on 23 Oct 2026. They include tests that floater pricing equals the eldest member plus the add-on share (with no discount in the USA), that `risk_of_waiting` is present in all three countries and grows with the delay, and that no verdict recommends waiting. They cover:
- every tool with personas from all three countries
- monotone sum-insured pricing and the Old Mutual ratio
- US silver loading
- hand-checked needs formula
- each scheme's eligible, check and not-eligible paths, including the 400% poverty-line cliff and Texas Medicaid
- red flags and score weights
- schema/signature agreement and error dispatch

**Reproduce:** run these in order, then `pytest`:
1. `python -m src.eda`
2. `python -m src.train`
3. `python -m src.fetch_worldbank`
4. `python -m src.fetch_who`
5. `python -m src.build_config`
6. `python -m src.fit_si_curve`
7. `python -m src.validate`
8. `python -m src.validate_term`

`build_config` rebuilds the config from scratch, so the last three must run after it, in that order: `validate` fits calibration on top of the sum-insured curve.
