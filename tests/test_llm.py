"""Provider layer with mocked SDK / HTTP responses (no network, no key)."""
import json

import httpx
import pytest

from src import llm as L
from src.llm import LLMReply, ToolCall


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(L.time, "sleep", lambda s: None)


# ---------------------------------------------------------------- fallback chain
class FakeModel:
    def __init__(self, label, outcomes):
        self.label, self.outcomes, self.calls = label, list(outcomes), 0

    def generate(self, *a, **k):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return LLMReply(out, model=self.label)


def test_daily_quota_moves_on_and_is_skipped_afterwards():
    a = FakeModel("flash", [L._DailyQuota(3600)])
    b = FakeModel("lite", ["one", "two"])
    chain = L.FallbackLLM([a, b])
    assert chain.generate("s", []).text == "one" and chain.active == "lite"
    assert chain.generate("s", []).text == "two"
    assert a.calls == 1  # not asked again while exhausted
    assert chain.available() == ["lite"]


def test_busy_model_is_not_marked_exhausted():
    a = FakeModel("flash", [L._Unavailable("503"), "back"])
    b = FakeModel("lite", ["one"])
    chain = L.FallbackLLM([a, b])
    assert chain.generate("s", []).text == "one"
    assert chain.generate("s", []).text == "back"


def test_all_exhausted_gives_daily_message_without_requests():
    a = FakeModel("flash", [L._DailyQuota(3600)])
    chain = L.FallbackLLM([a])
    for _ in range(2):
        with pytest.raises(L.RateLimitError) as e:
            chain.generate("s", [])
        assert str(e.value) == L.DAILY_LIMIT_MSG
    assert a.calls == 1


def test_all_busy_gives_wait_a_minute_message():
    chain = L.FallbackLLM([FakeModel("flash", [L._Unavailable("x")]), FakeModel("lite", [L._DailyQuota(60)])])
    with pytest.raises(L.RateLimitError, match="busy"):
        chain.generate("s", [])


def test_quota_comes_back_after_reset(monkeypatch):
    a = FakeModel("flash", [L._DailyQuota(100), "again"])
    chain = L.FallbackLLM([a])
    with pytest.raises(L.RateLimitError):
        chain.generate("s", [])
    now = L.time.time()
    monkeypatch.setattr(L.time, "time", lambda: now + 101)
    assert chain.generate("s", []).text == "again"


def test_get_llm_builds_chain_with_lite_and_optional_groq():
    names = [m.label for m in L.get_llm({"GEMINI_API_KEY": "k"}).models]
    assert names == list(L.GEMINI_MODELS + L.GEMINI_LITE_MODELS)
    with_groq = L.get_llm({"GEMINI_API_KEY": "k", "GROQ_API_KEY": "g", "GEMINI_MODEL": "gemini-3.6-flash"})
    labels = [m.label for m in with_groq.models]
    assert labels[0] == "gemini-3.6-flash" and labels.count("gemini-3.6-flash") == 1
    assert labels[-2:] == [f"groq/{m}" for m in L.GROQ_MODELS]
    assert [m.label for m in L.get_llm({"GEMINI_API_KEY": "k", "GROQ_API_KEY": "PASTE_HERE"}).models] == names
    with pytest.raises(L.LLMError):
        L.get_llm({"GEMINI_API_KEY": "PASTE_HERE"})


@pytest.mark.parametrize("msg, wait", [("Please retry in 12.5s.", 13.5), ("{'retryDelay': '7s'}", 8.0),
                                       ("Please try again in 1m2.5s", 63.5), ("quota", 2.0)])
def test_retry_delay_parsing(msg, wait):
    assert L._retry_delay(msg) == wait


# ---------------------------------------------------------------- Gemini
class FakeModels:
    def __init__(self, outcomes):
        self.outcomes, self.seen = list(outcomes), []

    def generate_content(self, model, contents, config):
        from google.genai import types
        self.seen.append(contents)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        if isinstance(out, types.Part):
            parts = [out]
        else:
            parts = [types.Part(text=out)]
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model",
                                                                                               parts=parts))])


def gemini(outcomes, name="gemini-3.8-flash"):
    fake = FakeModels(outcomes)
    return L.GeminiModel(type("C", (), {"models": fake})(), name), fake


def api_error(code, message="x"):
    from google.genai import errors
    return errors.APIError(code, {"error": {"code": code, "message": message, "status": "S"}})


def test_gemini_retries_overload_once_then_answers():
    m, fake = gemini([api_error(503), "OK"])
    reply = m.generate("s", [{"role": "user", "text": "hi"}])
    assert reply.text == "OK" and reply.model == "gemini-3.8-flash" and len(fake.seen) == 2


@pytest.mark.parametrize("err, kind", [
    (api_error(429, "quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier retryDelay': '500s'"), L._DailyQuota),
    (api_error(404), L._DailyQuota),
    (api_error(400), L._Unavailable),
    (httpx.ReadTimeout("slow"), L._Unavailable),
])
def test_gemini_error_kinds(err, kind):
    m, fake = gemini([err, err])
    with pytest.raises(kind):
        m.generate("s", [{"role": "user", "text": "hi"}])
    assert len(fake.seen) == 1  # none of these is retried on the same model


def test_gemini_function_calls_parsed():
    from google.genai import types
    part = types.Part(function_call=types.FunctionCall(name="project_cover_decay", args={"cover": 5}))
    m, _ = gemini([part])
    reply = m.generate("s", [{"role": "user", "text": "hi"}], tools=[{"name": "project_cover_decay",
                                                                     "description": "d", "parameters": {
                                                                         "type": "object", "properties": {}}}])
    assert reply.tool_calls[0].name == "project_cover_decay" and reply.tool_calls[0].args == {"cover": 5}
    assert reply.native is not None


def test_gemini_replays_own_turns_natively_and_flattens_other_models():
    from google.genai import types
    native = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="t", args={}))])
    turn = [{"role": "user", "text": "hi"},
            {"role": "assistant", "text": "", "tool_calls": [ToolCall("t", {"a": 1})], "native": native,
             "model": "gemini-3.8-flash"},
            {"role": "tool", "results": [{"id": None, "name": "t", "result": {"x": 1}}]}]
    same = L.GeminiModel(None, "gemini-3.8-flash")._contents(turn)
    assert same[1] is native and same[2].parts[0].function_response.name == "t"
    other = L.GeminiModel(None, "gemini-3.5-flash-lite")._contents(turn)
    assert not any(p.function_call or p.function_response for c in other for p in c.parts)
    assert "[Called tools: t" in other[1].parts[0].text and "[Tool results]" in other[2].parts[0].text


# ---------------------------------------------------------------- Groq
def groq(handler, model="openai/gpt-oss-120b"):
    return L.GroqModel("key", model, http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_groq_request_format_and_tool_call_parsing():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "x1", "type": "function", "function": {"name": "project_cover_decay",
                                                          "arguments": '{"cover": 500000}'}}]}}]})

    schema = {"name": "project_cover_decay", "description": "d", "parameters": {"type": "object", "properties": {}}}
    history = [{"role": "user", "text": "hi"},
               {"role": "assistant", "text": "", "tool_calls": [ToolCall("t", {"a": 1})], "native": object(),
                "model": "gemini-3.8-flash"},
               {"role": "tool", "results": [{"id": None, "name": "t", "result": {"x": 1}}]}]
    reply = groq(handler).generate("sys", history, [schema], allow_tools=False)
    assert reply.tool_calls == [ToolCall("project_cover_decay", {"cover": 500000}, "x1")]
    assert reply.model == "groq/openai/gpt-oss-120b" and seen["auth"] == "Bearer key"
    assert seen["tool_choice"] == "none" and seen["tools"][0]["function"]["name"] == "project_cover_decay"
    msgs = seen["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[2]["tool_calls"][0]["id"] == msgs[3]["tool_call_id"]  # generated id links call and result


def test_groq_daily_limit_and_bad_key_skip_model():
    day = groq(lambda r: httpx.Response(429, text="Rate limit reached on tokens per day (TPD). "
                                                  "Please try again in 7m12s."))
    with pytest.raises(L._DailyQuota) as e:
        day.generate("s", [{"role": "user", "text": "hi"}])
    assert e.value.retry_after == 433
    with pytest.raises(L._DailyQuota):
        groq(lambda r: httpx.Response(401, text="invalid key")).generate("s", [])


def test_groq_per_minute_limit_retries_once():
    hits = []

    def handler(request):
        hits.append(1)
        if len(hits) == 1:
            return httpx.Response(429, headers={"retry-after": "3"}, text="tokens per minute (TPM)")
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    assert groq(handler).generate("s", [{"role": "user", "text": "hi"}]).text == "OK" and len(hits) == 2
