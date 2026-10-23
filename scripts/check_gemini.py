"""Check that the Gemini key in .streamlit/secrets.toml works. Prints only model
names and success/failure, never the key.

    python -m scripts.check_gemini
"""
import re
import sys
import tomllib
from pathlib import Path

from google import genai
from google.genai import errors

ROOT = Path(__file__).resolve().parents[1]
SKIP = ("lite", "image", "tts", "live", "audio", "embedding", "thinking", "exp", "preview")


def _version(name):
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def main():
    key = tomllib.loads((ROOT / ".streamlit" / "secrets.toml").read_text(encoding="utf-8")).get("GEMINI_API_KEY", "")
    if not key or key == "PASTE_HERE":
        sys.exit("FAILURE: GEMINI_API_KEY is missing or still the placeholder.")
    client = genai.Client(api_key=key)
    try:
        names = [m.name.removeprefix("models/") for m in client.models.list()
                 if "generateContent" in (m.supported_actions or [])]
    except errors.APIError as e:
        sys.exit(f"FAILURE: could not list models (HTTP {e.code}: {e.status}).")
    flash = sorted((n for n in names if "flash" in n and not any(s in n for s in SKIP)), key=_version, reverse=True)
    print("Stable Flash models visible to this key:", ", ".join(flash) or "none")
    if "gemini-flash-latest" in names:
        print("Alias available: gemini-flash-latest")
    for model in flash[:4]:
        try:
            r = client.models.generate_content(model=model, contents="Reply with the single word OK.")
            print(f"SUCCESS: {model} replied {r.text.strip()[:20]!r}")
            return
        except errors.APIError as e:
            print(f"  {model}: HTTP {e.code} {e.status}")
    sys.exit("FAILURE: no Flash model accepted a test call.")


if __name__ == "__main__":
    main()
