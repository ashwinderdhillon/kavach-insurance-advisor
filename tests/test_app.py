"""UI smoke tests with Streamlit's AppTest (no LLM call is made)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import prompts

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def app():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_first_render_shows_header_greeting_and_footer():
    at = app()
    assert at.title[0].value == "🛡️ Kavach"
    assert at.chat_message[0].markdown[0].value == prompts.GREETINGS["English"]
    assert any(prompts.DISCLAIMER in c.value for c in at.main.caption)


def test_language_list_follows_country_and_greeting_follows_language():
    at = app()
    assert at.selectbox(key="language_IND").options == [
        "English", "हिन्दी (Hindi)", "தமிழ் (Tamil)", "বাংলা (Bengali)", "తెలుగు (Telugu)", "मराठी (Marathi)"]
    at.selectbox(key="country").set_value("KEN").run()
    assert at.selectbox(key="language_KEN").options == ["English", "Kiswahili (Swahili)"]
    at.selectbox(key="language_KEN").set_value("Swahili").run()
    assert at.chat_message[0].markdown[0].value == prompts.GREETINGS["Swahili"]


def test_country_change_resets_the_session():
    at = app()
    at.session_state["profile"] = {"age": 40}
    at.session_state["chat"] = [{"role": "user", "content": "hi"}]
    at.selectbox(key="country").set_value("USA").run()
    assert at.session_state["profile"] == {} and at.session_state["chat"] == []
    assert at.selectbox(key="language_USA").options == ["English", "Español (Spanish)"]


def test_saved_reply_renders_charts_and_calculation_log():
    from src import tools
    at = app()
    decay = tools.project_cover_decay(500000, "IND", 10)
    score = tools.protection_score({"country": "IND", "annual_income": 1_800_000}, {"health_sum_insured": 500000})
    at.session_state["chat"] = [
        {"role": "user", "content": "How is my cover doing?"},
        {"role": "assistant", "content": "It costs $5 and $10.", "tool_log": [
            {"name": "project_cover_decay", "args": {"cover": 500000, "country": "IND"}, "result": decay}]}]
    at.session_state["score"] = score
    at.session_state["profile"] = {"age": 32, "annual_income": 1_800_000, "smoker": False}
    at.run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 2           # decay chart in the chat + score gauge in the panel
    assert any(e.label == "How I calculated this" for e in at.expander)
    assert "\\$5" in at.chat_message[2].markdown[0].value  # currency is escaped, not LaTeX
