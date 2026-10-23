"""The only module that talks to LLM providers.

The agent calls get_llm(secrets).generate(system, messages, tools, allow_tools)
and gets an LLMReply. Behind it, FallbackLLM tries models in order:

    Gemini Flash (newest first) -> Gemini Flash-Lite -> Groq (only if GROQ_API_KEY is set)

A model whose daily quota is used up is skipped (no request spent) until its
quota resets. To add a provider (e.g. the Claude API), write a class with the
same generate() contract as GeminiModel and add it in get_llm().

Neutral transcript messages:
    {"role": "user", "text": str}
    {"role": "assistant", "text": str, "tool_calls": [ToolCall], "native": object | None, "model": str | None}
    {"role": "tool", "results": [{"id": str | None, "name": str, "result": dict}]}
"""
import json
import re
import time
from dataclasses import dataclass, field

import httpx

# Free-tier quotas are per model (20 requests/day for each Flash model on this project, Oct 2026),
# so every extra model in the chain adds its own daily quota.
GEMINI_MODELS = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash")
GEMINI_LITE_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
GROQ_MODELS = ("openai/gpt-oss-120b", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TEMPERATURE = 0.3
TIMEOUT_S = 45
MAX_WAIT_S = 20  # longest we wait for a per-minute limit before trying the next model
GROQ_MAX_WAIT_S = 60  # Groq is the last resort and its free token-per-minute limits are small, so wait longer


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str | None = None


@dataclass
class LLMReply:
    text: str
    tool_calls: list = field(default_factory=list)
    native: object = None  # provider's own message, replayed as-is to the same model (Gemini thought signatures)
    model: str | None = None


class LLMError(RuntimeError):
    """Any provider failure, carrying a message that is safe to show the user."""


class RateLimitError(LLMError):
    pass


class _DailyQuota(Exception):
    def __init__(self, retry_after):
        super().__init__()
        self.retry_after = retry_after


class _Unavailable(Exception):
    """Busy, rate-limited after one retry, timed out, or rejected: try the next model for this call."""


RATE_LIMIT_MSG = ("Kavach's free AI service is busy for the moment. Please wait about a minute and send your "
                  "message again.")
DAILY_LIMIT_MSG = ("Kavach has used up today's free AI quota on every model it can use. Please try again later "
                   "(Gemini quotas reset at midnight Pacific time).")


def _retry_delay(message, attempt=0):
    """Server-suggested delay ('retry in 12.3s', retryDelay '12s', 'try again in 1m2.5s'), else backoff."""
    m = re.search(r"(?:retry in|try again in|retryDelay'?:?\s*'?)\s*(?:(\d+)m)?\s*([\d.]+)s", message)
    if m:
        return 60 * float(m.group(1) or 0) + float(m.group(2)) + 1
    return 2.0 * 2 ** attempt


# ---------------------------------------------------------------- Gemini
class GeminiModel:
    """One Gemini model via the official google-genai SDK."""

    def __init__(self, client, model):
        self._client, self.label = client, model

    @staticmethod
    def make_client(api_key):
        from google import genai
        from google.genai import types
        # The SDK's own retries are off so the fallback chain alone decides what happens on errors.
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1)))

    @staticmethod
    def _tools(schemas):
        from google.genai import types
        return [types.Tool(function_declarations=[
            types.FunctionDeclaration(name=s["name"], description=s["description"],
                                      parameters_json_schema=s["parameters"]) for s in schemas])]

    def _contents(self, messages):
        """Messages this model produced are replayed natively (keeps thought signatures); tool exchanges
        from another model are flattened to text, which every model accepts."""
        from google.genai import types
        out, native_ok = [], False
        for m in messages:
            if m["role"] == "user":
                out.append(types.Content(role="user", parts=[types.Part(text=m["text"])]))
            elif m["role"] == "assistant":
                native_ok = m.get("native") is not None and m.get("model") == self.label
                if native_ok:
                    out.append(m["native"])
                    continue
                text = m.get("text") or ""
                if m.get("tool_calls"):
                    text += "\n[Called tools: " + "; ".join(
                        f"{c.name}({json.dumps(c.args, ensure_ascii=False)})" for c in m["tool_calls"]) + "]"
                out.append(types.Content(role="model", parts=[types.Part(text=text.strip() or "(no text)")]))
            elif m["role"] == "tool":
                if native_ok:
                    out.append(types.Content(role="user", parts=[
                        types.Part(function_response=types.FunctionResponse(id=r.get("id"), name=r["name"],
                                                                            response=r["result"]))
                        for r in m["results"]]))
                else:
                    out.append(types.Content(role="user", parts=[types.Part(text="[Tool results] " + json.dumps(
                        [{"tool": r["name"], "result": r["result"]} for r in m["results"]], ensure_ascii=False))]))
        return out

    def generate(self, system, messages, tools=None, allow_tools=True):
        from google.genai import errors, types
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=TEMPERATURE,
            tools=self._tools(tools) if tools else None,
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                mode="AUTO" if allow_tools else "NONE")) if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        contents = self._contents(messages)
        for attempt in range(2):
            try:
                response = self._client.models.generate_content(model=self.label, contents=contents, config=config)
                break
            except errors.APIError as e:
                msg = str(e)
                if e.code == 429 and "PerDay" in msg:
                    raise _DailyQuota(min(_retry_delay(msg), 86_400)) from None
                if e.code == 404:  # retired or not offered to this key
                    raise _DailyQuota(86_400) from None
                wait = _retry_delay(msg, attempt)
                if e.code not in (429, 500, 502, 503) or attempt == 1 or wait > MAX_WAIT_S:
                    raise _Unavailable(f"{self.label}: HTTP {e.code}") from None
                time.sleep(wait)
            except httpx.HTTPError as e:  # timeout or dropped connection: waiting again rarely helps
                raise _Unavailable(f"{self.label}: {type(e).__name__}") from None
        cand = (response.candidates or [None])[0]
        content = cand.content if cand is not None else None
        parts = (content.parts if content is not None else None) or []
        text = "".join(p.text for p in parts if p.text and not p.thought)
        calls = [ToolCall(name=p.function_call.name, args=dict(p.function_call.args or {}), id=p.function_call.id)
                 for p in parts if p.function_call]
        return LLMReply(text=text.strip(), tool_calls=calls, native=content if parts else None, model=self.label)


# ---------------------------------------------------------------- Groq (OpenAI-compatible API)
class GroqModel:
    """One Groq model over its OpenAI-compatible REST API (httpx, no extra package)."""

    def __init__(self, api_key, model, http=None):
        self._key, self.label = api_key, f"groq/{model}"
        self._model = model
        self._http = http or httpx.Client(timeout=TIMEOUT_S)

    @staticmethod
    def _messages(system, messages):
        out, ids = [{"role": "system", "content": system}], []
        for n, m in enumerate(messages):
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                ids = [c.id or f"call_{n}_{k}" for k, c in enumerate(m.get("tool_calls", []))]
                msg = {"role": "assistant", "content": m.get("text") or ""}
                if ids:
                    msg["tool_calls"] = [{"id": i, "type": "function", "function": {
                        "name": c.name, "arguments": json.dumps(c.args, ensure_ascii=False)}}
                        for i, c in zip(ids, m["tool_calls"])]
                out.append(msg)
            elif m["role"] == "tool":
                out += [{"role": "tool", "tool_call_id": r.get("id") or ids[k],
                         "content": json.dumps(r["result"], ensure_ascii=False)} for k, r in enumerate(m["results"])]
        return out

    def generate(self, system, messages, tools=None, allow_tools=True):
        body = {"model": self._model, "messages": self._messages(system, messages), "temperature": TEMPERATURE,
                "max_completion_tokens": 2048}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": s["name"], "description": s["description"],
                                                               "parameters": s["parameters"]}} for s in tools]
            body["tool_choice"] = "auto" if allow_tools else "none"
        for attempt in range(2):
            try:
                r = self._http.post(GROQ_URL, json=body, headers={"Authorization": f"Bearer {self._key}"})
            except httpx.HTTPError as e:
                raise _Unavailable(f"{self.label}: {type(e).__name__}") from None
            if r.status_code == 200:
                break
            msg = r.text
            if r.status_code == 429 and "per day" in msg.lower():
                raise _DailyQuota(min(_retry_delay(msg), 86_400))
            if r.status_code in (401, 403, 404):  # bad key or model not offered: skip for a day
                raise _DailyQuota(86_400)
            wait = float(r.headers.get("retry-after") or _retry_delay(msg, attempt))
            retryable = r.status_code in (429, 500, 502, 503) or "tool_use_failed" in msg
            if not retryable or attempt == 1 or wait > GROQ_MAX_WAIT_S:
                raise _Unavailable(f"{self.label}: HTTP {r.status_code}")
            time.sleep(wait)
        msg = r.json()["choices"][0]["message"]
        try:
            calls = [ToolCall(name=c["function"]["name"], args=json.loads(c["function"].get("arguments") or "{}"),
                              id=c.get("id")) for c in msg.get("tool_calls") or []]
        except json.JSONDecodeError:
            raise _Unavailable(f"{self.label}: malformed tool arguments") from None
        return LLMReply(text=(msg.get("content") or "").strip(), tool_calls=calls, model=self.label)


# ---------------------------------------------------------------- fallback chain
class FallbackLLM:
    """Tries each model in order; remembers which ones are out of daily quota so no request is wasted."""

    def __init__(self, models):
        self.models = list(models)
        self.exhausted_until = {}  # label -> unix time when its quota should be back
        self.active = None         # label of the model that answered last

    def available(self):
        now = time.time()
        return [m.label for m in self.models if self.exhausted_until.get(m.label, 0) <= now]

    def generate(self, system, messages, tools=None, allow_tools=True):
        outcome = None
        for m in self.models:
            if self.exhausted_until.get(m.label, 0) > time.time():
                continue
            try:
                reply = m.generate(system, messages, tools, allow_tools)
            except _DailyQuota as e:
                self.exhausted_until[m.label] = time.time() + e.retry_after
                outcome = outcome or "daily"
            except _Unavailable:
                outcome = "busy"
            else:
                self.active = m.label
                return reply
        if outcome == "busy":
            raise RateLimitError(RATE_LIMIT_MSG)
        raise RateLimitError(DAILY_LIMIT_MSG)


def get_llm(secrets):
    """Build the fallback chain from a secrets mapping (st.secrets or a dict).

    GEMINI_API_KEY is required. Optional: GEMINI_MODEL (tried first), GROQ_API_KEY, GROQ_MODEL.
    """
    key = secrets.get("GEMINI_API_KEY", "")
    if not key or key == "PASTE_HERE":
        raise LLMError("No Gemini API key found. Add GEMINI_API_KEY to .streamlit/secrets.toml.")
    client = GeminiModel.make_client(key)
    names = list(dict.fromkeys([secrets.get("GEMINI_MODEL")] + list(GEMINI_MODELS + GEMINI_LITE_MODELS)))
    models = [GeminiModel(client, n) for n in names if n]
    groq_key = secrets.get("GROQ_API_KEY", "")
    if groq_key and groq_key != "PASTE_HERE":
        http = httpx.Client(timeout=TIMEOUT_S)
        groq = list(dict.fromkeys([secrets.get("GROQ_MODEL")] + list(GROQ_MODELS)))
        models += [GroqModel(groq_key, n, http) for n in groq if n]
    return FallbackLLM(models)
