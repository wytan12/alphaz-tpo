"""
Central configuration: the nurse-scheduling instance + LLM settings.

NOTE ON THE INSTANCE
--------------------
The assessment refers to "the nurse scheduling instance defined in the appendix
(5 days, 3 shifts, 8 workers)", but the appendix was not provided. The instance
below matches the stated dimensions and is a reasonable, fully-specified stand-in.
If/when the official appendix arrives, edit ONLY this file to match it — every
other module reads the instance from here.
"""
import os

# Load a local .env file if present (so LLM_API_KEY etc. don't need setx/$env:).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not installed -> fall back to OS env vars only

# ---- dimensions ----
WORKERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
DAYS    = ["Mon", "Tue", "Wed", "Thu", "Fri"]
SHIFTS  = ["M", "E", "N"]          # Morning, Evening, Night

# ---- hard-constraint parameters ----
COVERAGE = {s: 2 for s in SHIFTS}  # workers required per (day, shift)
MAX_SHIFTS_PER_WORKER = 4          # per week
MIN_SHIFTS_PER_WORKER = 2

# ---- soft / preference data (higher = worker prefers it) ----
# preference score in [0,3]; missing pairs default to 1 (neutral)
PREFERENCES = {
    ("A", "N"): 0,   # A dislikes nights  -> drives the "Worker A too tired" cases
    ("B", "N"): 3,
    ("C", "M"): 3,
    ("D", "N"): 2,
    ("E", "E"): 3,
}

# objective weights
W_FAIRNESS = 5     # weight on workload spread (max-min)
W_PREF     = 1     # weight on preference satisfaction
W_NIGHTS   = 2     # penalty per consecutive-night pattern (tiredness)

# ---- LLM (hosted, OpenAI-compatible) ----
# Set via env. Works with DeepSeek / Qwen / OpenAI / local vLLM.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")          # empty -> rule-based fallback
LLM_MODEL    = os.getenv("LLM_MODEL", "deepseek-chat")

def pref(worker, shift):
    return PREFERENCES.get((worker, shift), 1)
