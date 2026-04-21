"""
Gita Guru — Content Guardrails
Uses the local `detoxify` BERT model (no API key needed) to classify
user prompts before they reach the main inference model.

Two-layer check:
  1. detoxify toxicity scores (≥ threshold → block)
  2. keyword blocklist for off-topic / harmful topics
"""

import logging
import re

logger = logging.getLogger("gita_guru")

# ---------------------------------------------------------------------------
# Thresholds — tune these if you get false positives / false negatives
# ---------------------------------------------------------------------------
TOXICITY_THRESHOLD = 0.6        # General toxicity
SEVERE_TOXICITY_THRESHOLD = 0.4 # Severe toxicity (lower = stricter)
THREAT_THRESHOLD = 0.5
IDENTITY_ATTACK_THRESHOLD = 0.5
SEXUAL_EXPLICIT_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Keyword blocklist — catches clearly off-topic harmful requests
# that might score low on the toxicity model (e.g. "how to make a bomb")
# ---------------------------------------------------------------------------
BLOCKED_PATTERNS = [
    # ── Weapons & Explosives ──────────────────────────────────────────────────
    r"\b(bomb|explosive|detonate|grenade|c4|dynamite)\b",
    r"\bhow to (make|build|create|synthesize|assemble) (a |an |)weapon",
    r"\b(gun|rifle|pistol|firearm|ammunition).*(buy|get|make|illegal|untraceable)\b",
    r"\b(illegal|untraceable|unregistered).*(gun|firearm|weapon|rifle|pistol)\b",
    r"\b(stab|strangle|poison|murder|assassinate)\b",
    r"\bhow to (kill|hurt|injure|attack) (someone|a person|people)\b",
    r"\b(shoot|shooting).*(school|crowd|people|mall|church)\b",

    # ── Hacking / Offensive Cybersecurity ────────────────────────────────────
    r"\b(hack|cracking|exploit|ddos|phish|brute.?force|sql.?inject|xss|malware|ransomware|keylogger|spyware|rootkit|botnet)\b",
    r"\bscript.*(hack|exploit|attack|break.?in|infiltrat)",
    r"\bwrite.*(hack|exploit|malware|virus|trojan|payload|worm|backdoor)",

    # ── Terrorism & Extremism ─────────────────────────────────────────────────
    r"\b(terroris[mt]|genocide|mass killing|ethnic cleansing)\b",
    r"\b(nazi|white supremac|neo.?nazi|extremis[mt])\b",
    r"\b(jihad|infidel|kafir).*(kill|destroy|eliminate|attack)\b",

    # ── Child Safety ──────────────────────────────────────────────────────────
    r"\b(child porn|csam|underage|minor).*(sex|nude|explicit|image)\b",

    # ── Drugs ─────────────────────────────────────────────────────────────────
    r"\b(meth|heroin|cocaine|fentanyl|crack|opium|lsd|ecstasy|mdma)\b",
    r"\b(weed|cannabis).*(buy|sell|grow|smuggle)\b",
    r"\bhow to (get high|overdose|spike a drink)\b",
    r"\b(cook|make|synthesize|buy|sell).*(meth|heroin|cocaine|fentanyl|crack)\b",

    # ── Self Harm ─────────────────────────────────────────────────────────────
    r"\bsuicide (method|how to|instructions|plan)\b",
    r"\bself[ -]harm (method|how to|instructions)\b",

    # ── Hate Speech ───────────────────────────────────────────────────────────
    r"\b(slur|racial abuse|hate speech).*(say|use|write)\b",

    # ── Scam / Financial Fraud ───────────────────────────────────────────────
    r"\b(scam|fraud|ponzi|pyramid scheme)\b",
    r"\b(money launder(ing)?|launder.{0,10}money)\b",
    r"\b(fake (invoice|id|passport|document|currency))\b",
    r"\bhow to (cheat|steal|defraud|con) (someone|people|customers)\b",

    # ── Privacy Attacks ───────────────────────────────────────────────────────
    r"\b(credit card|ssn|social security|aadhaar).*(steal|get|hack|clone)\b",
    r"\b(dox|doxx|doxxing|stalking?) (someone|a person|people)\b",
    r"\bhow to (dox|doxx|stalk) (someone|a person)\b",
    r"\bfind (someone.s |)(address|phone number|personal info).*(without|illegally|secretly)\b",

    # ── Jailbreak / Prompt Injection ──────────────────────────────────────────
    r"\b(ignore (all |previous |your |)instructions|forget your (rules|training|guidelines|persona))\b",
    r"\b(act as|pretend (you are|to be)|you are now) (an? )?(evil|unrestricted|unfiltered|jailbreak|uncensored)\b",
    r"\bdan (prompt|mode|jailbreak)\b",
    r"\b(developer|god|sudo|root|admin) mode\b",
    r"\byou have no (restrictions|rules|filters|limits)\b",
    r"\b(disregard|bypass|override|disable).*(filter|guardrail|restriction|safety|rule)\b",

    # ── Spam / Scam Promos ────────────────────────────────────────────────────
    r"\b(crypto|nft|investment).*(guaranteed|profit|scheme|get rich)\b",

    # ── Explicit Sexual Content ───────────────────────────────────────────────
    r"\b(sexual(ly)?|sex|nude|naked|porn|erotic|xxx|hentai|nsfw)\b",
    r"\bsomething (sexual|explicit|dirty|nsfw|adult)\b",
    r"\b(masturbat|intercourse|orgasm|genitals?|penis|vagina)\b",

]

_BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


# ---------------------------------------------------------------------------
# Load detoxify model at module level (once per cold-start)
# ---------------------------------------------------------------------------
_detoxify_model = None

def _load_detoxify():
    global _detoxify_model
    if _detoxify_model is None:
        try:
            from detoxify import Detoxify
            logger.info("Loading detoxify content guard model...")
            _detoxify_model = Detoxify("original")
            logger.info("Detoxify model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load detoxify model: %s", str(e))
            _detoxify_model = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_prompt(text: str) -> dict:
    """
    Check a user prompt for inappropriate content.

    Returns:
        {"safe": True}  — prompt is fine, pass to inference
        {"safe": False, "reason": "..."}  — prompt blocked
    """
    if not text or not text.strip():
        return {"safe": False, "reason": "Empty prompt."}

    # --- Layer 1: keyword blocklist (fast, no model needed) ---
    for pattern in _BLOCKED_RE:
        if pattern.search(text):
            logger.warning("Prompt blocked by keyword pattern: %s", pattern.pattern)
            return {
                "safe": False,
                "reason": (
                    "Your request contains content that cannot be addressed here. "
                    "Please ask a question related to the Bhagavad Gita, spirituality, or life philosophy."
                ),
            }

    # --- Layer 2: detoxify toxicity classifier ---
    _load_detoxify()

    if _detoxify_model is None:
        # If model failed to load, fail-open so real users aren't blocked
        logger.warning("Detoxify model unavailable — skipping toxicity check.")
        return {"safe": True}

    try:
        scores = _detoxify_model.predict(text)
        # scores is a dict: {toxicity, severe_toxicity, obscene,
        #                    identity_attack, insult, threat, sexual_explicit}

        checks = [
            ("toxicity",         scores.get("toxicity", 0),         TOXICITY_THRESHOLD),
            ("severe_toxicity",  scores.get("severe_toxicity", 0),  SEVERE_TOXICITY_THRESHOLD),
            ("threat",           scores.get("threat", 0),           THREAT_THRESHOLD),
            ("identity_attack",  scores.get("identity_attack", 0),  IDENTITY_ATTACK_THRESHOLD),
            ("sexual_explicit",  scores.get("sexual_explicit", 0),  SEXUAL_EXPLICIT_THRESHOLD),
        ]

        for label, score, threshold in checks:
            if score >= threshold:
                logger.warning(
                    "Prompt blocked by detoxify — %s score=%.3f (threshold=%.2f)",
                    label, score, threshold,
                )
                return {
                    "safe": False,
                    "reason": (
                        "Your message contains inappropriate content and cannot be processed. "
                        "Please keep your questions respectful and related to spiritual topics."
                    ),
                }

    except Exception as e:
        logger.error("Detoxify inference error: %s — failing open.", str(e))
        return {"safe": True}

    return {"safe": True}
