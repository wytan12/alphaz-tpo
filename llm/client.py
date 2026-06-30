"""
Thin OpenAI-compatible LLM client with a graceful offline fallback.

If LLM_API_KEY is unset, `chat_json` returns None so callers fall back to
deterministic rule-based logic — the whole repo stays runnable with no API key.
"""
import json
import re
import time
import config as C

try:
    from openai import OpenAI
    _client = OpenAI(base_url=C.LLM_BASE_URL, api_key=C.LLM_API_KEY) if C.LLM_API_KEY else None
except Exception:
    _client = None

MAX_RETRIES = 6        # tries before giving up on a single call
BASE_DELAY = 2.0       # seconds; exponential backoff: 2,4,8,... capped at 60


def available():
    return _client is not None


def _retry_after(err):
    """Extract a server-suggested wait (e.g. Gemini 'retry in 45.6s') if present."""
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", str(err), re.I)
    return float(m.group(1)) if m else None


def chat(prompt, system="You are a precise optimization assistant.", temperature=0.2, model=None):
    if not available():
        return None
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _client.chat.completions.create(
                model=model or C.LLM_MODEL, temperature=temperature,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            transient = ("429" in msg or "rate" in msg.lower()
                         or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "overload" in msg.lower())
            if not transient or attempt == MAX_RETRIES:
                print(f"[llm] giving up after {attempt} attempt(s): {msg[:120]}")
                return None
            wait = _retry_after(e) or delay
            print(f"[llm] rate-limited (attempt {attempt}/{MAX_RETRIES}); waiting {wait:.0f}s…")
            time.sleep(wait)
            delay = min(delay * 2, 60)      # exponential backoff, capped
    return None


def chat_json(prompt, system="Return ONLY valid JSON.", temperature=0.0, model=None):
    out = chat(prompt, system, temperature, model=model)
    if out is None:
        return None
    out = out.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None
