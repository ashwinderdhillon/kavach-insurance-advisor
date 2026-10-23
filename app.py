"""Kavach: AI insurance advisor (Streamlit). Run with:  streamlit run app.py

Everything lives in session state; nothing is written to disk or a database.
"""
import streamlit as st

from src import agent, charts, prompts
from src.llm import LLMError, get_llm
from src.premium import _country

st.set_page_config(page_title="Kavach · AI insurance advisor", page_icon="🛡️", layout="centered")

COUNTRY_LABELS = {"USA": "🇺🇸 United States", "IND": "🇮🇳 India", "KEN": "🇰🇪 Kenya"}
LANGUAGE_LABELS = {"English": "English", "Spanish": "Español (Spanish)", "Hindi": "हिन्दी (Hindi)",
                   "Tamil": "தமிழ் (Tamil)", "Bengali": "বাংলা (Bengali)", "Telugu": "తెలుగు (Telugu)",
                   "Marathi": "मराठी (Marathi)", "Swahili": "Kiswahili (Swahili)"}
FIELD_LABELS = {"annual_income": "Income / year", "spouse_income": "Spouse income / year",
                "annual_expenses": "Family expenses / year (without you)", "liquid_assets": "Savings",
                "health_sum_insured": "Health cover", "life_cover": "Life cover",
                "annual_premiums": "Premiums / year", "bmi": "BMI"}
MONEY_FIELDS = {"annual_income", "spouse_income", "annual_expenses", "liquid_assets", "debts",
                "health_sum_insured", "life_cover", "annual_premiums"}
FOOTER = (f"**{prompts.DISCLAIMER}** Premiums are model estimates, not quotes: India and Kenya figures are "
          "scaled from US cost data and calibrated to published prices. Always verify with the insurer.")
SIMPLE_CSS = """<style>
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size: 1.3rem; line-height: 1.65; }
[data-testid="stChatInput"] textarea { font-size: 1.2rem; }
</style>"""


def reset_chat():
    for k in ("chat", "transcript", "profile", "score", "error"):
        st.session_state.pop(k, None)


@st.cache_resource(show_spinner=False)
def load_llm():
    try:
        secrets = dict(st.secrets)
    except Exception:  # no secrets file at all
        secrets = {}
    return get_llm(secrets)


def md(text):
    """Escape $ so Streamlit does not render currency as LaTeX."""
    return text.replace("$", "\\$")


def fmt(key, value, symbol):
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if key in MONEY_FIELDS and isinstance(value, (int, float)):
        return f"{symbol}{value:,.0f}"
    return str(value).replace("_", " ")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    country = st.selectbox("Country", list(COUNTRY_LABELS), index=1, format_func=COUNTRY_LABELS.get, key="country")
    language = st.selectbox("Language", prompts.LANGUAGES[country], format_func=LANGUAGE_LABELS.get,
                            key=f"language_{country}")
    simple = st.toggle("Simple mode", key="simple",
                       help="Larger text, shorter replies and very simple words.")
    st.button("Reset chat", on_click=reset_chat, icon="🔄", width="stretch")
    st.divider()
    st.caption("🔒 Your answers, including health details, stay in this browser session and are cleared when you "
               "reset or close the tab. To write replies, messages are sent to Google's Gemini API (or Groq as a "
               "backup); on free tiers these services may use them to improve their products, so please don't "
               "share names, ID numbers or policy numbers.")
    try:
        _llm = load_llm()
        left = _llm.available()
        st.caption(f"🤖 AI model: {_llm.active or (left[0] if left else 'none')} · "
                   f"{len(left)} of {len(_llm.models)} models not yet out of today's free quota")
    except LLMError:
        st.caption("🤖 AI model: not configured (add GEMINI_API_KEY to .streamlit/secrets.toml)")

if st.session_state.get("active_country") != country:  # a new country means new currency, schemes and rules
    reset_chat()
    st.session_state.active_country = country
for key, default in (("chat", []), ("transcript", []), ("profile", {})):
    st.session_state.setdefault(key, default)
if simple:
    st.markdown(SIMPLE_CSS, unsafe_allow_html=True)
_, cfg = _country(country)
symbol = cfg["symbol"]

# ---------------------------------------------------------------- header + profile panel
st.title("🛡️ Kavach")
st.caption("Your personal AI insurance advisor: find the gaps in your cover, today and 10 years from now.")

profile = st.session_state.profile
share, missing = agent.completeness(profile)
score = st.session_state.get("score")
cols = st.columns([3, 2]) if score else [st.container()]
with cols[0].expander(f"Your profile · {share:.0%} complete", expanded=bool(profile) and not score):
    st.progress(share, text="Kavach recommends cover once the profile is at least 70% complete.")
    if profile:
        rows = [f"- **{FIELD_LABELS.get(k, k.replace('_', ' ').capitalize())}:** {fmt(k, v, symbol)}"
                + (" *(session only)*" if k in agent.HEALTH_FIELDS else "") for k, v in profile.items()]
        st.markdown(md("\n".join(rows)))
    else:
        st.caption("Nothing yet. Kavach fills this in as you chat.")
    if missing:
        st.caption("Still needed: " + ", ".join(m.replace("_", " ") for m in missing))
if score:
    cols[1].plotly_chart(charts.protection_gauge(score, height=220), key="score-gauge",
                         config={"displayModeBar": False})


# ---------------------------------------------------------------- chat
def render_assistant(entry, idx):
    st.markdown(md(entry["content"]))
    for j, call in enumerate(entry.get("tool_log", [])):
        fig = charts.figure_for(call["name"], call["result"])
        if fig is not None:
            st.plotly_chart(fig, key=f"chart-{idx}-{j}", config={"displayModeBar": False})
    if entry.get("tool_log"):
        with st.expander("How I calculated this"):
            for j, call in enumerate(entry["tool_log"], 1):
                st.markdown(f"**{j}. `{call['name']}`**")
                st.caption("Inputs")
                st.json(call["args"], expanded=False)
                res = call["result"]
                if "error" in res:
                    st.error(res["error"])
                    continue
                srcs = {(s.get("source"), s.get("url"), s.get("year")) for s in res.get("sources", [])}
                if srcs:
                    st.caption("Sources")
                    st.markdown(md("\n".join(
                        f"- {src or 'Model'}" + (f" ({yr})" if yr else "") + (f" · [link]({url})" if url else "")
                        for src, url, yr in sorted(srcs, key=lambda s: str(s[0])))))


with st.chat_message("assistant", avatar="🛡️"):
    st.markdown(prompts.GREETINGS[language])
for i, entry in enumerate(st.session_state.chat):
    with st.chat_message(entry["role"], avatar="🛡️" if entry["role"] == "assistant" else None):
        if entry["role"] == "assistant":
            render_assistant(entry, i)
        else:
            st.markdown(md(entry["content"]))

if st.session_state.get("error"):
    st.warning(st.session_state.error)

if prompt := st.chat_input("Type your message…"):
    st.session_state.pop("error", None)
    with st.chat_message("user"):
        st.markdown(md(prompt))
    with st.chat_message("assistant", avatar="🛡️"):
        try:
            with st.spinner("Kavach is working it out…"):
                turn = agent.run_turn(load_llm(), prompts.build_system_prompt(country, language, simple),
                                      st.session_state.transcript, prompt, profile, country)
        except LLMError as e:
            st.session_state.error = f"{e}\n\nYour message was not sent: “{prompt}”"
        else:
            st.session_state.chat += [{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": turn.text, "tool_log": turn.tool_log}]
            st.session_state.transcript += turn.messages
            scores = [c["result"] for c in turn.tool_log if c["name"] == "protection_score" and "error" not in c["result"]]
            if scores:
                st.session_state.score = scores[-1]
    st.rerun()

st.divider()
st.caption(md(FOOTER))
