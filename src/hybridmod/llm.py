"""Stage 3: LLM review for the low-confidence band.

* Structured output: the model must return JSON that validates against `CommentVerdict`
  (category enum, violates flag consistent with the category, confidence in [0, 1]).
* Validation + retry: invalid or missing items are re-asked (only those items), with the
  validation error fed back to the model, up to `max_retries` times. Items that still fail are
  returned as None and routed to the human queue by the pipeline.
* Rate limits: HTTP 429 honours `retry-after`; these waits do not consume validation retries.
* Disk cache (JSONL, append-only): keyed by model + prompt version + text, so experiment runs are
  resumable and never pay twice for the same comment.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError, model_validator

from .config import LLMCfg

PROMPT_VERSION = "v1"


class Category(StrEnum):
    toxicity = "toxicity"
    threat = "threat"
    insult = "insult"
    identity_attack = "identity_attack"
    sarcasm_critique_ok = "sarcasm_critique_ok"
    question = "question"
    appreciation = "appreciation"
    neutral_ok = "neutral_ok"


VIOLATING = {Category.toxicity, Category.threat, Category.insult, Category.identity_attack}


class CommentVerdict(BaseModel):
    id: int
    category: Category
    violates: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def _consistent(self) -> CommentVerdict:
        if self.violates != (self.category in VIOLATING):
            raise ValueError(
                f"violates={self.violates} is inconsistent with category={self.category.value}"
            )
        return self


SYSTEM_PROMPT = """You are a comment moderator for a creator platform. For each comment decide if it \
violates the community policy and give ONE category.

Violating categories (violates=true):
- toxicity: rude, disrespectful or unreasonable language likely to make people leave the \
discussion (profanity aimed at people, demeaning tone, crude obscenity).
- insult: insulting, inflammatory or negative remarks aimed at a person or group ("idiot", \
"moron", "you people are pathetic").
- threat: wishes or intends harm, violence or injury to someone.
- identity_attack: negative or hateful statements targeting people because of race, religion, \
gender, sexual orientation, disability or nationality.

Allowed categories (violates=false):
- sarcasm_critique_ok: criticism, sarcasm or strong disagreement about ideas, policies, \
institutions or public actions without personal insults. Harsh but civil is allowed.
- question: a genuine question.
- appreciation: thanks, praise, support.
- neutral_ok: any other civil comment, including ones that merely MENTION an identity group.

Mentioning an identity group is not an attack. Judge the comment, not the topic.
Return ONLY a JSON object: {"results": [{"id": <int>, "category": "<category>", \
"violates": <bool>, "confidence": <0..1>, "reason": "<max 15 words>"}]} with one entry per input \
id, in the same order."""


def build_user_prompt(items: list[tuple[int, str]], max_chars: int = 1500) -> str:
    lines = ["Classify these comments:"]
    for i, text in items:
        lines.append(json.dumps({"id": i, "comment": text[:max_chars]}, ensure_ascii=False))
    return "\n".join(lines)


def parse_batch(raw: str, expected_ids: list[int]) -> tuple[dict[int, CommentVerdict], list[str]]:
    """Parse a model response. Returns (valid verdicts by id, list of error strings)."""
    errors: list[str] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {}, [f"response is not valid JSON: {e.msg}"]
    if isinstance(data, list):
        data = {"results": data}
    items = data.get("results") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}, ['top-level object must have a "results" list']
    out: dict[int, CommentVerdict] = {}
    want = set(expected_ids)
    for it in items:
        try:
            v = CommentVerdict.model_validate(it)
        except ValidationError as e:
            errors.append(f"item {it.get('id') if isinstance(it, dict) else '?'}: "
                          f"{e.errors()[0]['msg']}")
            continue
        if v.id in want:
            out[v.id] = v
    missing = want - set(out)
    if missing:
        errors.append(f"missing or invalid ids: {sorted(missing)}")
    return out, errors


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{PROMPT_VERSION}|{text}".encode()).hexdigest()[:32]


@dataclass
class LLMStats:
    calls: int = 0
    retries: int = 0
    rate_limited: int = 0
    failures: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_latencies_ms: list[float] = field(default_factory=list)

    def cost_usd(self, cfg: LLMCfg) -> float:
        return (self.prompt_tokens * cfg.price_input_per_m
                + self.completion_tokens * cfg.price_output_per_m) / 1e6


@dataclass
class ReviewResult:
    verdict: CommentVerdict | None
    latency_ms: float  # wall time of the request that produced it (0 for cache hits at runtime)
    prompt_tokens: float
    completion_tokens: float
    cached: bool
    attempts: int


class ChatTransport(Protocol):
    def __call__(self, messages: list[dict]) -> tuple[str, dict]: ...


class RateLimited(Exception):
    def __init__(self, wait_s: float, message: str = ""):
        super().__init__(message or f"rate limited, retry after {wait_s}s")
        self.wait_s = wait_s


class OpenAICompatTransport:
    """POST /chat/completions on any OpenAI-compatible endpoint (Groq, Gemini)."""

    def __init__(self, cfg: LLMCfg, api_key: str | None = None):
        self.cfg = cfg
        key = api_key or os.environ.get(cfg.api_key_env, "")
        if not key:
            raise RuntimeError(f"{cfg.api_key_env} is not set")
        self.client = httpx.Client(
            base_url=cfg.base_url.rstrip("/"), timeout=cfg.timeout_s,
            headers={"Authorization": f"Bearer {key}"},
        )

    def __call__(self, messages: list[dict]) -> tuple[str, dict]:
        body = {
            "model": self.cfg.model, "messages": messages, "temperature": 0,
            "response_format": {"type": "json_object"}, "max_tokens": 1500,
        }
        if self.cfg.reasoning_effort:
            body["reasoning_effort"] = self.cfg.reasoning_effort
        r = self.client.post("/chat/completions", json=body)
        if r.status_code == 429:
            raise RateLimited(float(r.headers.get("retry-after", "10")), r.text[:300])
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"].get("content") or "", data.get("usage", {})


class JsonlCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.data: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path and path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, rec: dict) -> None:
        with self._lock:
            self.data[rec["key"]] = rec
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class LLMReviewer:
    def __init__(self, cfg: LLMCfg, transport: ChatTransport | None = None,
                 cache_path: Path | None = None, max_rate_wait_s: float = 900,
                 sleep=time.sleep):
        self.cfg = cfg
        self.transport = transport or OpenAICompatTransport(cfg)
        self.cache = JsonlCache(cache_path)
        self.stats = LLMStats()
        self.max_rate_wait_s = max_rate_wait_s
        self._sleep = sleep

    def _call(self, messages: list[dict]) -> tuple[str, dict, float]:
        waited = 0.0
        while True:
            t0 = time.perf_counter()
            try:
                raw, usage = self.transport(messages)
                return raw, usage, (time.perf_counter() - t0) * 1000
            except RateLimited as e:
                self.stats.rate_limited += 1
                if waited + e.wait_s > self.max_rate_wait_s:
                    raise
                waited += e.wait_s
                self._sleep(e.wait_s + 0.5)

    def _review_batch(self, items: list[tuple[int, str]]) -> dict[int, ReviewResult]:
        pending = dict(items)
        results: dict[int, ReviewResult] = {}
        feedback: list[str] = []
        attempts = 0
        while pending and attempts <= self.cfg.max_retries:
            attempts += 1
            batch = list(pending.items())
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(batch)}]
            if feedback:
                msgs.append({"role": "user", "content": "Your previous answer was invalid: "
                             + "; ".join(feedback[:5]) + ". Answer again for these ids only."})
            try:
                raw, usage, ms = self._call(msgs)
            except (httpx.HTTPError, RateLimited, KeyError) as e:
                feedback = [f"transport error {type(e).__name__}"]
                self.stats.retries += 1
                continue
            self.stats.calls += 1
            self.stats.call_latencies_ms.append(ms)
            pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            self.stats.prompt_tokens += pt
            self.stats.completion_tokens += ct
            valid, feedback = parse_batch(raw, [i for i, _ in batch])
            for i, v in valid.items():
                results[i] = ReviewResult(v, ms, pt / len(batch), ct / len(batch), False, attempts)
                del pending[i]
            if pending:
                self.stats.retries += 1
        for i in pending:
            self.stats.failures += 1
            results[i] = ReviewResult(None, 0.0, 0.0, 0.0, False, attempts)
        return results

    def review(self, texts: list[str]) -> list[ReviewResult]:
        """Review texts, using the cache, batching misses `batch_size` at a time."""
        out: list[ReviewResult | None] = [None] * len(texts)
        misses: list[int] = []
        for idx, t in enumerate(texts):
            rec = self.cache.get(cache_key(self.cfg.model, t))
            if rec and rec.get("verdict"):
                self.stats.cache_hits += 1
                out[idx] = ReviewResult(CommentVerdict.model_validate(rec["verdict"]),
                                        rec["latency_ms"], rec["prompt_tokens"],
                                        rec["completion_tokens"], True, rec.get("attempts", 1))
            else:
                misses.append(idx)
        bs = self.cfg.batch_size
        for start in range(0, len(misses), bs):
            chunk = misses[start:start + bs]
            res = self._review_batch([(k, texts[idx]) for k, idx in enumerate(chunk)])
            for k, idx in enumerate(chunk):
                r = res[k]
                if r.verdict is not None:
                    r.verdict = r.verdict.model_copy(update={"id": 0})
                    self.cache.put({
                        "key": cache_key(self.cfg.model, texts[idx]), "model": self.cfg.model,
                        "prompt_version": PROMPT_VERSION, "verdict": r.verdict.model_dump(mode="json"),
                        "latency_ms": round(r.latency_ms, 1), "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens, "attempts": r.attempts,
                        "batch_size": len(chunk),
                    })
                out[idx] = r
        return out  # type: ignore[return-value]
