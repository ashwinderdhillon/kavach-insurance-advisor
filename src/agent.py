"""Agent loop: the model requests tools, we run them, return the results, and the
model replies. Provider-neutral: it only uses src.llm.LLMReply / ToolCall and
the transcript format documented in src/llm.py.

Built to spend as few LLM calls as possible per user message:
- Profile facts come back in a <profile>{...}</profile> line at the end of the
  model's reply (stripped before display) and from the inputs it passes to
  tools, so recording them never needs an extra round trip.
- Completeness is computed here and sent with the message, so the model does not
  need a tool call to check the information gate.
- Tool results go back without sources and long series (the UI shows those),
  and earlier turns are kept as plain text (user message + final reply).
Profiles live in Streamlit session state only.
"""
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src import tools
from src.llm import LLMError

ROOT = Path(__file__).resolve().parents[1]
MAX_TOOL_CALLS = 6  # calculation-tool calls per user message
MAX_ROUNDS = 4      # LLM calls per user message; the last one is always text-only

PROFILE_FIELDS = {
    "age": int, "sex": str, "state": str, "rural": bool, "dependants": int, "children": int,
    "household_size": int, "parents_dependent": bool, "parents_age": int, "annual_income": float,
    "spouse_income": float, "employment_type": str, "occupation_risk": str, "annual_expenses": float,
    "liquid_assets": float, "debts": float, "goals": str, "has_health_cover": bool, "health_cover_source": str,
    "health_sum_insured": float, "has_life_cover": bool, "life_cover": float, "annual_premiums": float,
    "owns_vehicle": bool, "owns_home": bool, "smoker": bool, "bmi": float, "chronic_conditions": str,
}
HEALTH_FIELDS = {"smoker", "bmi", "chronic_conditions"}
PROFILE_RE = re.compile(r"`{0,3}\s*<profile>(.*?)(?:</profile>|$)\s*`{0,3}", re.DOTALL | re.IGNORECASE)
DROP_FOR_LLM = {"sources", "source", "url", "source_url", "disclaimer", "assumptions", "symbol", "tool", "kind",
                "fetched", "method"}


@lru_cache(maxsize=1)
def tool_schemas():
    return json.loads((ROOT / "data" / "tool_schemas.json").read_text(encoding="utf-8"))["tools"]


def completeness(profile):
    share, missing = tools._completeness(profile)
    return round(share, 2), missing


@dataclass
class Turn:
    text: str
    tool_log: list = field(default_factory=list)  # [{name, args, result}] for calculation tools
    messages: list = field(default_factory=list)  # compact history to append: user text + final reply
    llm_calls: int = 0


# ---------------------------------------------------------------- profile
def _coerce(kind, value):
    if value is None or isinstance(value, (dict, list)):
        return None
    if kind is bool:
        if isinstance(value, str):
            v = value.strip().lower()
            return True if v in ("true", "yes", "y", "1") else False if v in ("false", "no", "n", "0") else None
        return bool(value)
    try:
        return kind(value) if kind is not str else str(value).strip() or None
    except (TypeError, ValueError):
        return None


def merge_profile(profile, facts, overwrite=True):
    """Add known fields with the right types; returns the names that changed."""
    changed = []
    for k, v in (facts or {}).items():
        kind = PROFILE_FIELDS.get(k)
        val = _coerce(kind, v) if kind else None
        if val is None or (not overwrite and k in profile) or profile.get(k) == val:
            continue
        profile[k] = val
        changed.append(k)
    return changed


def split_profile(text):
    """Remove the <profile>{...}</profile> line from a reply; return (clean text, facts dict)."""
    facts = {}
    for m in PROFILE_RE.finditer(text or ""):
        try:
            found = json.loads(m.group(1).strip() or "{}")
            if isinstance(found, dict):
                facts.update(found)
        except json.JSONDecodeError:
            pass
    return PROFILE_RE.sub("", text or "").strip(), facts


def _facts_from_args(args):
    """Profile facts in a tool call's inputs (flat or under 'profile' / 'current_covers')."""
    facts = {k: v for k, v in args.items() if k in PROFILE_FIELDS}
    for section in ("profile", "current_covers"):
        if isinstance(args.get(section), dict):
            facts.update({k: v for k, v in args[section].items() if k in PROFILE_FIELDS})
    return facts


def context_block(profile):
    share, missing = completeness(profile)
    known = json.dumps(profile, ensure_ascii=False, separators=(",", ":"))
    gate = "complete enough to recommend" if not missing or share >= 0.7 else "NOT complete enough to recommend"
    return (f"[Context from Kavach, not typed by the user] Known profile: {known}. Completeness {share:.0%}, "
            f"{gate}" + (f"; missing: {', '.join(missing)}" if missing else "") + ".")


# ---------------------------------------------------------------- tools
def _force_country(name, args, code):
    """The sidebar decides the country, whatever the model passes."""
    args = dict(args or {})
    if name in ("recommend_cover_types", "protection_score"):
        args["profile"] = {**(args.get("profile") or {}), "country": code}
    else:
        args["country"] = code
    return args


def _fits(schema_type, value):
    if schema_type in ("integer", "number"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, {"boolean": bool, "string": str}.get(schema_type, object))


def _fill_from_profile(name, args, profile):
    """The profile-based tools (recommend_cover_types gates on completeness) get every fact the user has
    already told us; values the model passes win."""
    if name not in ("recommend_cover_types", "protection_score"):
        return args
    props = next(s for s in tool_schemas() if s["name"] == name)["parameters"]["properties"]
    for section in ("profile", "current_covers"):
        if section in props:
            allowed = props[section]["properties"]
            known = {k: v for k, v in profile.items() if k in allowed and _fits(allowed[k].get("type"), v)}
            args[section] = {**known, **(args.get(section) or {})}
    return args


def for_llm(obj):
    """Tool result as the model sees it: no citations or long series (they are shown in the UI)."""
    if isinstance(obj, dict):
        out = {k: for_llm(v) for k, v in obj.items() if k not in DROP_FOR_LLM}
        if isinstance(obj.get("series"), list):
            s = obj["series"]
            out["series"] = [p for i, p in enumerate(s) if i % 5 == 0 or i == len(s) - 1]
        return out
    if isinstance(obj, list):
        return [for_llm(v) for v in obj]
    return obj


# ---------------------------------------------------------------- loop
def run_turn(llm, system, history, user_text, profile, country):
    """Run one user message. history: prior compact transcript (not modified). profile: updated in place.

    Returns a Turn whose .messages should be appended to the history.
    """
    code, _ = tools._country(country)
    schemas = tool_schemas()
    msgs = list(history) + [{"role": "user", "text": f"{context_block(profile)}\n\n{user_text}"}]
    log, used, reply = [], 0, None
    for rnd in range(MAX_ROUNDS):
        allow = used < MAX_TOOL_CALLS and rnd < MAX_ROUNDS - 1
        reply = llm.generate(system, msgs, schemas, allow_tools=allow)
        if not reply.tool_calls or not allow:
            break
        msgs.append({"role": "assistant", "text": reply.text, "tool_calls": reply.tool_calls,
                     "native": reply.native, "model": reply.model})
        results = []
        for call in reply.tool_calls:
            if used >= MAX_TOOL_CALLS:
                result = {"error": f"Tool limit reached ({MAX_TOOL_CALLS} per message). Answer with the results "
                                   "you have."}
            else:
                used += 1
                args = _fill_from_profile(call.name, _force_country(call.name, call.args, code), profile)
                merge_profile(profile, _facts_from_args(args), overwrite=False)
                result = tools.call_tool(call.name, args)
                log.append({"name": call.name, "args": args, "result": result})
            results.append({"id": call.id, "name": call.name, "result": for_llm(result)})
        msgs.append({"role": "tool", "results": results})

    text, facts = split_profile(reply.text if reply else "")
    merge_profile(profile, facts)
    if not text:
        raise LLMError("Kavach could not put together a reply. Please rephrase or send your message again.")
    return Turn(text=text, tool_log=log, llm_calls=rnd + 1,
                messages=[{"role": "user", "text": user_text},
                          {"role": "assistant", "text": text, "tool_calls": [], "native": None, "model": None}])
