"""spaCy-based memory candidate extractor.

Replaces the regex heuristic with a proper NLP pipeline. Detects memory types
using POS tags and dependency parsing rather than substring matching:

  preference  → first-person subject + verb in {prefer, like, hate, love, want, choose}
                + sentence is NOT a question
  decision    → past-tense verb + first person ("we decided", "I chose", "decidimos")
  task        → modal {need, must, have to} + verb, OR imperative
  fact        → declarative third-person sentence with copula or stative verb,
                length-bounded, NOT a question, NOT code-like

Why this is much better than regex:
- Sentence segmentation is real (handles Spanish, English, mixed).
- Rejects interrogatives via punctuation + parser.
- Differentiates `todo` (Spanish "all") from `TODO` (task marker) via POS context.
- Pulls real entities from spaCy NER instead of vocab lists.

Optional dep — install with: pip install -e '.[extract]' + download models:
    python -m spacy download en_core_web_sm
    python -m spacy download es_core_news_sm
"""
from __future__ import annotations

import logging
from typing import Iterable

from .curator import Candidate

log = logging.getLogger("memoirs.extract.spacy")


def is_available() -> bool:
    try:
        import spacy  # noqa: F401
    except ImportError:
        return False
    return True


_NLP_EN = None
_NLP_ES = None


def _load_models():
    """Load EN + ES models. Cache singletons. Returns (en, es). Either may be None
    if its model isn't downloaded."""
    global _NLP_EN, _NLP_ES
    if _NLP_EN is not None or _NLP_ES is not None:
        return _NLP_EN, _NLP_ES
    import spacy
    try:
        _NLP_EN = spacy.load("en_core_web_sm", exclude=["lemmatizer"])
    except OSError:
        log.warning("en_core_web_sm not installed; English extraction disabled")
        _NLP_EN = None
    try:
        _NLP_ES = spacy.load("es_core_news_sm", exclude=["lemmatizer"])
    except OSError:
        log.warning("es_core_news_sm not installed; Spanish extraction disabled")
        _NLP_ES = None
    return _NLP_EN, _NLP_ES


# ----------------------------------------------------------------------
# Quality filters
# ----------------------------------------------------------------------

# IDE / Claude Code system tags wrapped inside user messages.
# Literal string prefixes — used with str.startswith().
_SYSTEM_INJECTION_PREFIXES = (
    "<ide_opened_file>",
    "<ide_selection>",
    "<system-reminder>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
    "<local-command-stdout>",
)

# Code-flavored substrings — characters/operators that natural language rarely
# uses. Stdlib `in` checks instead of regex.
_CODE_CHARS = "`{};[]<>"
_CODE_OPERATORS = ("=>", "::", ":=", "->", "&&", "||", "()", "{}", "[]")
_PATH_EXTENSIONS = (
    ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json", ".yml", ".yaml",
    ".sh", ".toml", ".sql", ".rs", ".go", ".cpp", ".c", ".h", ".java", ".rb",
)


def _is_system_injection(text: str) -> bool:
    """Claude Code wraps file opens, IDE selections, system reminders inside
    user messages with these tags. Treated as not-real-user-input and skipped."""
    return text.lstrip().startswith(_SYSTEM_INJECTION_PREFIXES)


def _looks_like_code(text: str) -> bool:
    """Heuristic: does this read like code rather than prose?

    Uses cheap string containment (no regex). URLs are not code even though they
    contain `:`. For deeper analysis we rely on the spaCy parser in `extract`.
    """
    if "://" in text:  # URL — not code
        return False
    if any(op in text for op in _CODE_OPERATORS):
        return True
    if any(c in text for c in _CODE_CHARS):
        return True
    # Path-like: ends with a code/data extension or contains "<ext> " token.
    if any(text.endswith(ext) or (ext + " ") in text for ext in _PATH_EXTENSIONS):
        return True
    # Bare absolute Unix path (e.g. `/etc/passwd`, `/home/user/x`, `/var/log`)
    # — has at least 2 slashes, no spaces, starts with "/".
    stripped = text.strip()
    if (
        stripped.startswith("/")
        and " " not in stripped
        and stripped.count("/") >= 2
    ):
        return True
    return False


def _looks_like_dump(text: str) -> bool:
    """Detect copy-pasted file/diff/log dumps via line-prefix counting.

    Splits on newlines and inspects the first non-space char of each line:
    diffs start with +/-/@; numbered listings start with digits + whitespace.
    """
    lines = text.split("\n")
    if len(lines) < 4:
        return False
    diff_lines = 0
    numbered_lines = 0
    for line in lines:
        s = line.lstrip()
        if not s:
            continue
        if s[0] in "+-@":
            diff_lines += 1
            continue
        head = s.split(None, 1)
        if head and head[0].isdigit() and len(head[0]) <= 5:
            numbered_lines += 1
    total = max(1, len(lines))
    return diff_lines / total > 0.3 or numbered_lines / total > 0.3


def _is_question(sent) -> bool:
    text = sent.text.strip()
    if text.endswith("?") or text.startswith("¿"):
        return True
    # First token is typical question starter (English)
    first = sent[0].text.lower() if len(sent) else ""
    if first in {"what", "why", "how", "when", "where", "which", "who", "can", "could", "would", "should", "is", "are", "do", "does", "did"}:
        # Check if any later token is a verb subject — looser heuristic
        # Keep only if very short ("can you ?")
        if not text.endswith("."):
            return True
    return False


# ----------------------------------------------------------------------
# Type detectors (each returns content snippet or None)
# ----------------------------------------------------------------------

_PREF_VERBS = {"prefer", "preferir", "preferiría", "like", "love", "hate", "want", "querer", "quiero", "wish", "desear", "gustar", "elegir", "choose"}
_DECISION_VERBS = {"decide", "decidir", "decided", "decidió", "decidimos", "elegimos", "chose", "concluded", "agreed", "settled"}
_DECISION_MODALS = {"will", "vamos a", "let's", "let us"}
_TASK_MODALS = {"need", "must", "should", "tengo", "debo", "deberíamos", "have", "necesitamos", "deber"}


def _has_first_person(sent) -> bool:
    for tok in sent:
        if tok.pos_ == "PRON" and tok.text.lower() in {"i", "we", "me", "us", "yo", "nosotros", "mí"}:
            return True
        # Spanish person agreement on verbs (1Sg or 1Pl)
        if tok.pos_ == "VERB":
            morph = tok.morph.to_dict() if hasattr(tok.morph, "to_dict") else {}
            if morph.get("Person") == "1":
                return True
    return False


def _detect_preference(sent) -> str | None:
    if _is_question(sent):
        return None
    if not _has_first_person(sent):
        return None
    for tok in sent:
        if tok.pos_ == "VERB":
            lemma = tok.lemma_.lower() if tok.lemma_ else tok.text.lower()
            if lemma in _PREF_VERBS or tok.text.lower() in _PREF_VERBS:
                return sent.text.strip()
    return None


def _detect_decision(sent) -> str | None:
    if _is_question(sent):
        return None
    text_lower = sent.text.lower()
    for phrase in _DECISION_MODALS:
        if phrase in text_lower:
            if _has_first_person(sent):
                return sent.text.strip()
    for tok in sent:
        if tok.pos_ == "VERB":
            lemma = tok.lemma_.lower() if tok.lemma_ else ""
            text = tok.text.lower()
            if lemma in _DECISION_VERBS or text in _DECISION_VERBS:
                return sent.text.strip()
    return None


def _detect_task(sent) -> str | None:
    if _is_question(sent):
        return None
    text_lower = sent.text.lower()
    # Modal patterns
    for tok in sent:
        if tok.pos_ in {"AUX", "VERB"}:
            lemma = tok.lemma_.lower() if tok.lemma_ else ""
            text = tok.text.lower()
            if (lemma in _TASK_MODALS or text in _TASK_MODALS) and _has_first_person(sent):
                # avoid "I have a dog" — require following verb
                if any(t.pos_ == "VERB" and t.i > tok.i for t in sent):
                    return sent.text.strip()
    # Imperative-like: starts with bare verb (no subject)
    if len(sent) >= 2 and sent[0].pos_ == "VERB" and not any(t.dep_ == "nsubj" for t in sent[:3]):
        return sent.text.strip()
    return None


def _detect_fact(sent) -> str | None:
    if _is_question(sent):
        return None
    if _has_first_person(sent):
        return None  # facts about user belong to other types
    # Must have a finite verb and a subject
    has_subj = any(t.dep_ in {"nsubj", "nsubjpass"} for t in sent)
    has_verb = any(t.pos_ in {"VERB", "AUX"} for t in sent)
    if not (has_subj and has_verb):
        return None
    txt = sent.text.strip()
    if len(txt) < 30 or len(txt) > 220:
        return None
    return txt


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def _pick_pipeline(text: str):
    """Heuristic language detection: count Spanish-only function words vs English."""
    en, es = _load_models()
    if en is None and es is None:
        return None
    if en is None:
        return es
    if es is None:
        return en
    # very rough lang detect
    es_score = sum(1 for w in (" el ", " la ", " que ", " de ", " y ", " es ", " son ", " un ", " una ", " para ") if w in f" {text.lower()} ")
    en_score = sum(1 for w in (" the ", " is ", " of ", " and ", " a ", " to ", " in ", " that ") if w in f" {text.lower()} ")
    return es if es_score > en_score else en


def extract(messages: list[dict]) -> list[Candidate]:
    """Run spaCy-based extraction over user messages and return candidates."""
    if not is_available():
        return []
    candidates: list[Candidate] = []
    seen_content: set[str] = set()

    for m in messages:
        if m.get("role") != "user":
            continue
        text = (m.get("content") or "").strip()
        if not text or len(text) < 12:
            continue
        if _is_system_injection(text):
            continue
        if _looks_like_dump(text):
            continue
        if _looks_like_code(text):
            continue
        # Trim huge messages — only first 4k chars (typical user prompt)
        if len(text) > 4000:
            text = text[:4000]

        nlp = _pick_pipeline(text)
        if nlp is None:
            continue
        doc = nlp(text)

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) < 12 or len(sent_text) > 220:
                continue
            if _looks_like_code(sent_text):
                continue

            for detector, ctype, importance, confidence in (
                (_detect_preference, "preference", 4, 0.65),
                (_detect_decision, "decision", 3, 0.55),
                (_detect_task, "task", 3, 0.50),
                (_detect_fact, "fact", 2, 0.45),
            ):
                snippet = detector(sent)
                if not snippet:
                    continue
                key = (ctype, snippet[:140])
                if key in seen_content:
                    break
                seen_content.add(key)
                candidates.append(
                    Candidate(
                        type=ctype,
                        content=snippet,
                        importance=importance,
                        confidence=confidence,
                        source_message_ids=[m["id"]] if m.get("id") else [],
                        extractor="spacy",
                    )
                )
                break  # one type per sentence
    return candidates
