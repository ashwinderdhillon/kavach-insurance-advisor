# AI Insurance Advisor — Project Context for Claude Code

## Project summary
- Team project, topic: "AI in daily lives".
- Product: a website with an AI insurance advisor chatbot that acts as a personalised agent (not plain Q&A) for USA, India and Kenya, in each country's major languages.
- Stack: Python, scikit-learn / XGBoost / LightGBM, Streamlit, deployed free on Streamlit Community Cloud via GitHub.
- Chatbot LLM: Google Gemini API free tier (Flash model) with function calling. Keep the LLM provider swappable so it can be switched to the Claude API later.
- Budget: zero. Use only free data, free APIs and free hosting.
- Secrets: API keys live in .streamlit/secrets.toml, which must be in .gitignore. Never commit keys.

## Build phases
1. Model: train and test the cost/premium model, country calibration with real cited data.
2. Tools + region config: tools.py functions the chatbot calls.
3. Website: Streamlit chat UI + agent loop.
4. User testing with 3 personas.
5. Deploy to Streamlit Cloud.
6. Presentation.

---

# AI Insurance Advisor — Parameter Framework

## 1. Core problem

The agent closes the **protection gap**: the difference between the cover a household needs and the cover it actually holds, measured today and projected forward with inflation. People buy the wrong product, in the wrong amount, from the wrong insurer, at the wrong time, and find out only at claim time.

**The 5 gaps**

| Gap | What goes wrong | Economic root cause |
| --- | --- | --- |
| Awareness | User doesn't know what they need or what the state gives free | Information asymmetry, low financial literacy |
| Adequacy | Cover too small; inflation erodes it every year | Anchoring on today's prices, no indexation |
| Affordability | Premium too high for income, policy lapses | Liquidity constraints, income volatility |
| Fine print | Exclusions, caps and waiting periods trigger claim rejection | Complex contracts, asymmetric expertise |
| Trust | Agents push high-commission products | Misaligned incentives (principal-agent problem) |

**Behavioural traps the agent must counter**

- **Present bias**: the premium hurts today, the benefit feels far away, so people underinsure.
- **Probability neglect**: "it won't happen to me" for illness, death and disaster.
- **Return-seeking**: buying endowment or ULIP plans because they "give money back", which mixes investment with protection and delivers too little of both.
- **Status-quo bias**: relying on employer cover that ends with the job.

**Success measure**: a household that would face **catastrophic health spending** (out-of-pocket costs above 10% or 25% of household consumption, the WHO SDG 3.8.2 thresholds) is protected against it, today and in 10 years.

## 2. Three-region lens

The same agent runs three advisor personas, because the binding constraint differs: cost in the USA, adequacy and trust in India, access and affordability in Kenya.

| Dimension | USA (developed) | India (developing) | Kenya (low-income) |
| --- | --- | --- | --- |
| Binding constraint | Very high health costs, plan complexity | Underinsurance, medical inflation, mis-selling | Most people uninsured, informal income |
| Core products | Employer/ACA health, Medicare, Medicaid, term life, disability, auto, renters | Term life, health + super top-up, motor TP, PM-JAY, PMJJBY, PMSBY, PMFBY | SHA (public health), mobile microinsurance, funeral cover, index crop/livestock |
| Jargon to decode | Deductible, copay, out-of-pocket max, HSA, metal tiers, HMO vs PPO | Sum insured, waiting period, co-pay, room-rent cap, claim settlement ratio | Pay-as-you-go premium, group (chama) cover, payout trigger |
| Public safety net | Medicare (65+), Medicaid (low income), ACA subsidies | PM-JAY family cover, state schemes | SHA |
| Tax angle | HSA, employer pre-tax premiums | Sec 80C (life), 80D (health) | Limited |
| Payment rail | Card, payroll | UPI | M-Pesa |
| Channel | Web | Web + WhatsApp | WhatsApp, USSD, voice |
| Languages | English, Spanish | English, Hindi, Tamil, Bengali, Telugu, Marathi | English, Swahili |
| Regulator | State departments + NAIC | IRDAI | IRA Kenya |
| Tone | Efficient, numbers-first | Educational, anti-mis-selling | Very simple, voice-friendly, small amounts |

Kenya can be swapped for Nigeria (English, Hausa, Yoruba) without changing the framework.

## 3. Insurance types and priority order

The agent recommends in a fixed order: protect health and income first, assets second, extras last. Investment-linked insurance is never a first recommendation.

| Priority | Category | Products | Who needs it |
| --- | --- | --- | --- |
| 1 | Health | Indemnity, family floater, super top-up, critical illness (lump sum), hospital cash | Everyone |
| 2 | Life | Term (pure protection) | Anyone with dependants or debt |
| 3 | Income protection | Personal accident, disability (short/long term) | Earners, especially manual or driving jobs |
| 4 | Assets | Motor (third-party is mandatory), home, renters, fire, flood | Owners |
| 5 | Livelihood | Crop, livestock, index-based insurance | Farmers, pastoralists |
| 6 | Micro / funeral | Small daily or weekly cover via mobile, burial cover | Low-income, informal workers |
| 7 | Situational | Travel, liability, cyber, pet | As needed |
| 8 | Retirement | Annuities, pension products | Near retirement |
| Last | Investment-linked | Endowment, money-back, ULIP, whole life | Rarely; separate insurance from investing |

**Economist's note on index insurance (Kenya, rural India):** it pays on a trigger such as low rainfall, so claims are fast and cheap. The catch is **basis risk**: a farmer can lose a crop while the index doesn't trigger. The agent must explain this.

## 4. Parameters (layers A–F)

Six layers feed every recommendation: the person, the policy, the insurer, the region, time, and decision quality. Tier says what the 4-hour MVP collects (MVP) and what waits (P2).

### A. The person (household risk profile)

| Parameter | Why it matters | Tier |
| --- | --- | --- |
| Age | Drives premium, cover need, eligibility, years to retirement | MVP |
| Country, state, urban/rural | Sets prices, schemes, hospital access, channel | MVP |
| Language, literacy level | Sets reply language and reading level | MVP |
| Dependants (count, ages), aging parents | Main driver of life cover need | MVP |
| Spouse income | Second earner lowers the cover needed | MVP |
| Monthly income, income band | Affordability and cover-need math | MVP |
| Employment type: salaried, self-employed, informal, farmer | Income stability; employer cover; scheme eligibility | MVP |
| Income volatility (fixed vs seasonal) | Seasonal earners need flexible premium frequency | P2 |
| Emergency fund (months of expenses) | Sets how high a deductible the household can carry | MVP |
| Outstanding loans and EMIs | Debt must be covered so family doesn't inherit it | MVP |
| Assets: vehicle, home, crops, livestock | Which asset covers are needed | P2 |
| Smoker, BMI, chronic conditions (session only, never stored) | Premium, waiting periods, exclusions | MVP |
| Family medical history | Critical-illness priority | P2 |
| Occupation risk (desk, field, driving, construction) | Accident and disability priority | P2 |
| Existing cover: employer, government, private, amounts | Avoids duplicate buying; finds the gap | MVP |
| Risk appetite, past claim experience, trust level | Tone, deductible choice, insurer choice | P2 |
| Device and payment rail (smartphone/feature phone; card/UPI/M-Pesa) | Product format and premium frequency | P2 |
| Goals: education, home, retirement age | Future cover timeline | MVP |

### B. The policy (contract terms)

**Health**

| Parameter | What the agent checks | Tier |
| --- | --- | --- |
| Sum insured | Against projected bill size in user's city | MVP |
| Premium and premium growth by age band | Affordability now and later | MVP |
| Deductible / co-pay % | Out-of-pocket per claim vs emergency fund | MVP |
| Room-rent cap | Red flag: a cap shrinks the whole bill proportionally | MVP |
| Disease sub-limits | Red flag for common surgeries | MVP |
| Waiting periods: initial, pre-existing, specific illness | Gap before cover actually starts | MVP |
| Exclusions | Read via RAG, explained in plain language | MVP |
| Network hospitals near user, cashless access | Real usability | P2 |
| Restoration, no-claim bonus | Partial inflation hedge | MVP |
| Lifelong renewability, max entry age, portability | Long-term safety | P2 |
| USA only: HMO/PPO, out-of-pocket max, drug formulary, HSA eligibility | Total annual cost, not just premium | MVP (USA) |

**Life (term)**

| Parameter | What the agent checks | Tier |
| --- | --- | --- |
| Cover amount | Against needs-based target (section 5) | MVP |
| Policy term | Until youngest dependant is independent or loans end | MVP |
| Premium-paying term, level vs rising premium | Affordability over time | MVP |
| Riders: critical illness, premium waiver, accidental death | Value vs a separate cheaper policy | P2 |
| Payout mode: lump sum vs monthly income | Fit for financially inexperienced dependants | P2 |

**Motor, crop, micro (Phase 2):** insured vehicle value, zero-depreciation, no-claim bonus; payout trigger, basis risk, payout speed, premium frequency.

### C. The insurer (counterparty quality)

| Parameter | Why it matters | Tier |
| --- | --- | --- |
| Claim settlement ratio | Share of claims paid | MVP |
| Amount settlement ratio | Share of claimed money paid; often more revealing | P2 |
| Incurred claim ratio (health) | Too low = stingy; too high = possibly unsustainable | P2 |
| Solvency ratio | Can the insurer survive a shock (counterparty risk) | P2 |
| Complaints per 10,000 policies | Service quality | P2 |
| Claim turnaround time | Speed when it matters | P2 |

### D. The region (macro and policy environment)

| Parameter | Used for | Tier |
| --- | --- | --- |
| General CPI inflation | Real value of life cover and income | MVP |
| Medical inflation | Cover decay, future bill projection | MVP |
| Interest rate / discount rate | Present value in cover-need math | MVP |
| Currency, exchange rate | Display and cross-country comparison | MVP |
| Health spending per person | Scaling the premium model across countries | MVP |
| Out-of-pocket share of health spending | How exposed an uninsured household is | MVP |
| Insurance penetration (% of GDP) | Context and awareness messaging | P2 |
| Public schemes and eligibility rules | Free or subsidised cover first | MVP |
| Mandatory covers by law | Compliance prompts | MVP |
| Tax benefits | Net cost of premium | P2 |
| Climate and disaster exposure (flood, drought, heat) | Property and crop priority | P2 |
| Regulator and grievance route | What to do if a claim is denied | MVP |

### E. Time (future-proofing)

| Parameter | Used for | Tier |
| --- | --- | --- |
| Projection horizon (5, 10, 20 years) | All forward views | MVP |
| Premium step-ups at age bands | Cost of delaying purchase | MVP |
| Real value of cover over time | Cover decay meter | MVP |
| Expected life events: marriage, children, home loan, retirement, job change | Timeline of top-ups | MVP |
| Employer cover end date | Status-quo bias alert | MVP |
| Renewal and repricing risk | Long-term product safety | P2 |

### F. Decision quality (the anti-rush layer)

| Parameter | Rule of thumb the agent applies | Tier |
| --- | --- | --- |
| Premium-to-income ratio | Flag when total premiums take too large a share of income (lapse risk) | MVP |
| Coverage adequacy score | Actual ÷ needed cover, now and in 10 years | MVP |
| Overlap check | No paying twice for one risk | MVP |
| Red-flag count | Room-rent cap, heavy co-pay, long waits, investment mix | MVP |
| Profile completeness / confidence | No recommendation below a set completeness | MVP |
| Free-look reminder | Cancellation window after purchase; check local rules | MVP |
| Cooling-off prompt | Suggest sleeping on any purchase above a threshold | MVP |

## 5. Derived scores and formulas

Symbols: C = cover, π = inflation (π_m medical), t = years, i = interest rate.

1. Cover decay (real value of today's health cover in t years): C_real = C / (1 + π_m)^t
   Example: ₹5 lakh at 13% medical inflation is worth about ₹1.5 lakh in 10 years.
2. Future hospital bill: B_t = B_0 × (1 + π_m)^t
3. Term life cover needed (needs-based, inflation-adjusted):
   r = (1 + i) / (1 + π) − 1
   Need = E × (1 − (1 + r)^(−n)) / r + D + G − L − X
   E = family's annual expenses without the earner, n = years support is needed, D = debts, G = goals (education etc.), L = liquid assets, X = existing life cover.
4. Coverage adequacy score: Adequacy = min(1, actual cover / needed cover) × 100. Shown for today and for year 10.
5. Affordability (premium-to-income ratio): PIR = total annual premiums / annual income. Thresholds set per country in the region config; above the threshold the agent warns about lapse risk.

Cost of delay: predicted premium at age a + k minus premium at age a, summed over the policy term. Turns "I'll buy later" into a rupee or dollar figure.

Overall protection score (0–100): weighted blend of health adequacy, life adequacy, affordability and red-flag count. Weights live in the region config so each country can tune them.

Premium from expected claims: premium ≈ expected annual claims / country loss ratio, plus an age-band loading. insurance.csv gives medical costs, not premiums.

## 6. Agent decision process

The agent moves through six stages in order and never recommends a product before stage 3 is complete.

1. **Discover**: asks Layer A questions conversationally, 2–3 at a time, in the user's language.
2. **Diagnose**: shows the 5 gaps with scores ("health adequacy 40%, affordability fine, 2 red flags").
3. **Prescribe**: recommends the type and amount of cover first, government schemes before paid products.
4. **Compare**: shortlists options on Layer B and C parameters, red flags highlighted.
5. **Stress-test**: shows cover decay, cost of delay, and what-ifs (job loss, illness, new child, rate rise).
6. **Decide calmly**: summary card, questions to ask the insurer, free-look reminder, cooling-off prompt.

**Safeguards**

- No recommendation while profile completeness is below the set threshold; the agent asks for what's missing.
- Every factual claim about a scheme or policy comes from a retrieved document, with a citation.
- Health details stay in the session and are never stored.
- Disclaimer in every summary: educational guidance, not licensed advice.
- The agent never names a "best" insurer; it shows the parameters and lets the user choose.

## 7. Data sources and model mapping

| Component | Method | Data source | Layers fed |
| --- | --- | --- | --- |
| Cost/premium predictor | XGBoost / LightGBM regression | Kaggle insurance.csv (age, BMI, smoker, children, charges), scaled per country by health spending per person (PPP-adjusted) | A, D |
| Cost-of-delay curve | Same model, age swept | Model output | A, E |
| Inflation projection | Compound formula; optional Prophet forecast | MoSPI CPI health sub-index (India), BLS CPI medical care (USA), KNBS CPI (Kenya) | D, E |
| Cover-need and adequacy | Formulas in section 5 | User inputs + region config | A, D, E, F |
| Scheme matcher | Rules engine | Official scheme pages (PM-JAY, PMJJBY, PMSBY, Medicaid, SHA) | A, D |
| Insurer quality | Lookup table | IRDAI annual report; NAIC; IRA Kenya | C |
| Policy decoder | RAG (Chroma + embeddings) | Sample policy wordings, regulator glossaries | B |
| Macro context | Static config refreshed from API | World Bank API, WHO Global Health Expenditure Database | D |
| Conversation, languages, explanation | Gemini Flash (free tier) via API with function calling; swappable to Claude | System prompt per region | All |

**Known limitation:** insurance.csv is US data. India and Kenya premiums are scaled estimates, labelled as such in the app.

## 8. MVP scope and open assumptions

The MVP covers health and term life in all three regions, with the future-proofing layer as the differentiator. Motor, property, crop and RAG are Phase 2.

| Hour | Build |
| --- | --- |
| 0–1 | Train model; write region config JSON (macro metrics, schemes, thresholds, weights) |
| 1–2 | Tools: predict_premium, project_cover_decay, calc_term_cover_needed, check_govt_schemes |
| 2–3 | Streamlit chat UI, country + language picker, agent loop |
| 3–4 | Charts (cover decay, cost of delay, protection score); deploy to Streamlit Cloud |

**Open assumptions to verify before the demo**

- [ ] Current medical inflation figure per country
- [ ] Scheme premiums, cover amounts and eligibility (they change)
- [ ] Premium-to-income thresholds per country
- [ ] Free-look period per country
- [ ] Choice of Kenya vs Nigeria as the low-income market
