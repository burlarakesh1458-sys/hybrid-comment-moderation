"""Stage 1: deterministic rules for the clear-cut cases.

The block lexicon is stored as salted SHA-256 hashes of normalised tokens, so the repository never
contains the plaintext slur/profanity list. Tokens are normalised (lower-case, common leetspeak
folded, repeated letters collapsed) before hashing, which also catches simple obfuscation such as
"1d10t" or "stuuupid".
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import RulesCfg

URL_RE = re.compile(r"(https?://|www\.)\S+", re.I)
WORD_RE = re.compile(r"[a-z0-9@$!*]+", re.I)
LETTER_RE = re.compile(r"[a-zA-Z]")
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a",
                      "$": "s", "!": "i", "*": ""})
REPEAT_RE = re.compile(r"([a-z])\1{2,}")

# Short comments that are clearly benign on a creator platform (appreciation / simple questions).
BENIGN_PATTERNS = [
    re.compile(p, re.I)
    for p in [
        r"^(thanks?( you)?|thank u|thx|ty)\b",
        r"^(great|good|nice|excellent|awesome|well[- ]written|interesting|informative) "
        r"(article|piece|post|video|story|point|read|job|work|comment)\b",
        r"^(well said|agreed|i agree|exactly|so true|very true|good point|great point)\b",
        r"^(love (this|it)|congrat(s|ulations))\b",
    ]
]
PROMO_RE = re.compile(
    r"(click here|work from home|earn \$?\d+|make \$\d+|per hour from home|buy followers|"
    r"free (iphone|gift ?card)|whatsapp me|dm me for)",
    re.I,
)


def normalise_token(tok: str) -> str:
    t = tok.lower().translate(LEET)
    t = re.sub(r"[^a-z]", "", t)
    return REPEAT_RE.sub(r"\1\1", t)


def hash_token(tok: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{tok}".encode()).hexdigest()


def tokens(text: str) -> list[str]:
    return [t for t in (normalise_token(w) for w in WORD_RE.findall(text)) if t]


@dataclass
class RuleResult:
    action: str  # "allow" | "block" | "spam" | "pass"
    rule: str | None = None
    detail: dict = field(default_factory=dict)


class RuleEngine:
    def __init__(self, cfg: RulesCfg, lexicon_hashes: set[str] | None = None):
        self.cfg = cfg
        self.lexicon = lexicon_hashes or set()

    @classmethod
    def from_file(cls, cfg: RulesCfg, path: Path) -> RuleEngine:
        hashes: set[str] = set()
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    hashes.add(line.split()[0])
        return cls(cfg, hashes)

    def lexicon_hits(self, text: str) -> int:
        salt = self.cfg.lexicon_salt
        toks = tokens(text)
        cands = set(toks) | {t[:-1] for t in toks if t.endswith("s") and len(t) > 3}
        return sum(1 for t in cands if hash_token(t, salt) in self.lexicon)

    def evaluate(self, text: str) -> RuleResult:
        if not self.cfg.enabled:
            return RuleResult("pass")
        stripped = text.strip()
        if not stripped:
            return RuleResult("allow", "empty")

        n_links = len(URL_RE.findall(stripped))
        if n_links >= self.cfg.spam_min_links:
            return RuleResult("spam", "many_links", {"links": n_links})
        if PROMO_RE.search(stripped):
            return RuleResult("spam", "promo_phrase")
        toks = tokens(stripped)
        if len(toks) >= 10:
            top = max(toks.count(t) for t in set(toks))
            if top / len(toks) > self.cfg.spam_max_repeat_ratio:
                return RuleResult("spam", "repetition", {"ratio": round(top / len(toks), 2)})

        hits = self.lexicon_hits(stripped)
        if hits:
            return RuleResult("block", "lexicon", {"hits": hits})

        if self.cfg.allow_no_letters and not LETTER_RE.search(stripped):
            return RuleResult("allow", "no_letters")
        words = stripped.split()
        if len(words) <= self.cfg.allow_short_benign_max_words and any(
            p.search(stripped) for p in BENIGN_PATTERNS
        ):
            return RuleResult("allow", "short_benign")
        return RuleResult("pass")
