"""Agent loop, profile handling, prompts and charts with a scripted fake LLM (no network, no key)."""
import pytest

from src import agent, charts, prompts, tools
from src.llm import LLMError, LLMReply, ToolCall


class FakeLLM:
    """Returns scripted replies in order and records what it was called with."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system, messages, tools=None, allow_tools=True):
        self.calls.append({"messages": list(messages), "allow_tools": allow_tools})
        return self.replies.pop(0)


def run(replies, profile=None, country="IND", history=None):
    llm = FakeLLM(replies)
    profile = {} if profile is None else profile
    turn = agent.run_turn(llm, "system", history or [], "hi", profile, country)
    return turn, llm, profile


# ---------------------------------------------------------------- loop
def test_tool_result_is_compacted_and_country_is_forced():
    call = ToolCall("project_cover_decay", {"cover": 500000, "country": "USA", "years": 10}, id="c1")
    turn, llm, _ = run([LLMReply("", [call]), LLMReply("Your cover halves. <profile>{}</profile>")])
    assert turn.text == "Your cover halves." and turn.llm_calls == 2
    assert turn.tool_log[0]["args"]["country"] == "IND"           # sidebar wins over the model
    full = turn.tool_log[0]["result"]
    sent = llm.calls[1]["messages"][-1]["results"][0]
    assert sent["id"] == "c1" and "sources" in full and "sources" not in sent["result"]
    assert len(sent["result"]["series"]) == 3 < len(full["series"])  # years 0, 5, 10


def test_at_most_six_tool_calls_and_last_round_is_text_only():
    calls = [ToolCall("project_cover_decay", {"cover": 100000 * (k + 1)}) for k in range(8)]
    turn, llm, _ = run([LLMReply("", calls), LLMReply("Done.")])
    assert len(turn.tool_log) == agent.MAX_TOOL_CALLS
    results = llm.calls[1]["messages"][-1]["results"]
    assert sum("error" in r["result"] for r in results) == 2
    assert llm.calls[1]["allow_tools"] is False


def test_round_limit_bounds_llm_calls():
    loop = [LLMReply("", [ToolCall("project_cover_decay", {"cover": 1000})]) for _ in range(agent.MAX_ROUNDS - 1)]
    turn, llm, _ = run(loop + [LLMReply("Final.")])
    assert turn.text == "Final." and len(llm.calls) == agent.MAX_ROUNDS
    assert llm.calls[-1]["allow_tools"] is False


def test_empty_reply_raises_friendly_error():
    with pytest.raises(LLMError):
        run([LLMReply("<profile>{}</profile>")])


# ---------------------------------------------------------------- profile without extra calls
def test_profile_line_is_saved_and_hidden_in_one_call():
    reply = LLMReply('Thanks! How much do you save?\n<profile>{"age": "32", "dependants": 2, '
                     '"annual_income": 1800000, "smoker": "no", "nonsense": 1}</profile>')
    turn, llm, profile = run([reply])
    assert len(llm.calls) == 1 and turn.text == "Thanks! How much do you save?"
    assert profile == {"age": 32, "dependants": 2, "annual_income": 1800000.0, "smoker": False}


@pytest.mark.parametrize("text, facts", [
    ('Hi\n```\n<profile>{"age": 40}</profile>\n```', {"age": 40}),
    ('Hi <profile>{"age": 40}', {"age": 40}),                     # unclosed tag
    ("Hi <profile>{not json}</profile>", {}),
    ("Hi", {}),
])
def test_split_profile_edge_cases(text, facts):
    clean, found = agent.split_profile(text)
    assert clean == "Hi" and found == facts


def test_tool_inputs_fill_profile_without_overwriting_user_facts():
    call = ToolCall("predict_term_premium", {"age": 33, "sex": "female", "smoker": False, "cover": 1e7,
                                             "term_years": 28})
    _, _, profile = run([LLMReply("", [call]), LLMReply("Priced.")], profile={"age": 32})
    assert profile["age"] == 32 and profile["sex"] == "female" and profile["smoker"] is False


def test_context_goes_to_model_but_history_stays_compact():
    turn, llm, _ = run([LLMReply("", [ToolCall("project_cover_decay", {"cover": 1})]), LLMReply("Ok.")],
                       profile={"age": 30})
    sent = llm.calls[0]["messages"][-1]["text"]
    assert sent.startswith("[Context from Kavach") and '"age":30' in sent and "NOT complete" in sent
    assert sent.endswith("hi")
    assert [(m["role"], m["text"]) for m in turn.messages] == [("user", "hi"), ("assistant", "Ok.")]


def test_recommend_is_filled_from_profile_and_passes_the_gate():
    profile = {"age": 32, "dependants": 2, "annual_income": 1_800_000.0, "employment_type": "salaried",
               "has_health_cover": True, "has_life_cover": False, "debts": 4_000_000.0, "owns_vehicle": False}
    turn, _, _ = run([LLMReply("", [ToolCall("recommend_cover_types", {"profile": {"country": "KEN"}})]),
                      LLMReply("Plan.")], profile=profile)
    r = turn.tool_log[0]["result"]
    assert r["status"] == "ok" and r["country"] == "IND"


def test_protection_score_gets_covers_from_profile():
    profile = {"annual_income": 1_800_000.0, "life_cover": 5_000_000.0, "health_sum_insured": 500_000.0,
               "goals": "home"}
    turn, _, _ = run([LLMReply("", [ToolCall("protection_score", {"profile": {}, "current_covers": {}})]),
                      LLMReply("Score shown.")], profile)
    args = turn.tool_log[0]["args"]
    assert args["current_covers"]["life_cover"] == 5_000_000
    assert "goals" not in args["profile"]  # text goal does not fit the numeric field
    assert "score_today" in turn.tool_log[0]["result"]


def test_schemas_are_the_calculation_tools():
    assert sorted(s["name"] for s in agent.tool_schemas()) == sorted(tools.TOOLS)


# ---------------------------------------------------------------- prompts and charts
@pytest.mark.parametrize("code", ["USA", "IND", "KEN"])
def test_prompt_per_country(code):
    for lang in prompts.LANGUAGES[code]:
        p = prompts.build_system_prompt(code, lang, simple_mode=lang != "English")
        assert f"ONLY in {lang}" in p and prompts.DISCLAIMER in p and f'country="{code}"' in p
        assert "<profile>" in p and "annual_income" in p
        assert "{language}" not in p and "{fields}" not in p and "{disclaimer}" not in p
        assert lang in prompts.GREETINGS
    assert "SIMPLE MODE" not in prompts.build_system_prompt(code, "English", False)


def test_prompt_is_short():
    assert len(prompts.build_system_prompt("IND", "English", True)) < 6000  # ~1.5k tokens, sent every call


def test_charts_from_real_tool_output():
    assert charts.figure_for("project_cover_decay", tools.project_cover_decay(500000, "IND", 10)) is not None
    assert charts.figure_for("cost_of_delay", tools.cost_of_delay(32, 5, "IND", {"sex": "female"})) is not None
    score = tools.protection_score({"country": "KEN", "annual_income": 360000}, {})
    assert charts.figure_for("protection_score", score) is not None
    assert charts.figure_for("check_red_flags", {}) is None
    assert charts.figure_for("project_cover_decay", {"error": "x"}) is None
