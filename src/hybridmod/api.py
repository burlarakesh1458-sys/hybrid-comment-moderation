"""FastAPI service: single + batch moderation, human review queue, live config."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import Config, load_config
from .pipeline import HybridModerator
from .review_queue import ReviewQueue

log = logging.getLogger("hybridmod.api")


class ModerateIn(BaseModel):
    text: str = Field(min_length=0, max_length=10_000)
    comment_id: str | None = None


class BatchIn(BaseModel):
    comments: list[ModerateIn] = Field(min_length=1, max_length=500)


class ResolveIn(BaseModel):
    human_action: str = Field(pattern="^(allow|hide|penalize)$")
    reviewer: str = Field(min_length=1, max_length=100)
    note: str = ""


def build_moderator(cfg: Config) -> HybridModerator:
    """Wire the real components from config. The LLM stage is optional: without an API key the
    band falls back to the ML threshold and is routed to human review."""
    from .classifier import ToxicityClassifier
    from .llm import LLMReviewer
    from .rules import RuleEngine

    rules = RuleEngine.from_file(cfg.rules, cfg.path(cfg.rules.lexicon_path))
    clf = ToxicityClassifier.load(cfg.path(cfg.classifier.model_path))
    reviewer = None
    if cfg.llm.enabled and os.environ.get(cfg.llm.api_key_env):
        reviewer = LLMReviewer(cfg.llm, cache_path=cfg.path(cfg.llm.cache_path))
    else:
        log.warning("LLM stage disabled (no %s); band -> ML threshold + human review",
                    cfg.llm.api_key_env)
    queue = ReviewQueue(cfg.path(cfg.human_review.queue_path))
    return HybridModerator(cfg, rules, clf, reviewer, queue=queue,
                           band_to_human_without_llm=cfg.llm.enabled)


def create_app(moderator: HybridModerator | None = None, cfg: Config | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = cfg or load_config()
        state["cfg"] = c
        state["mod"] = moderator or build_moderator(c)
        yield

    app = FastAPI(title="Hybrid Comment Moderation", version="1.0.0", lifespan=lifespan)

    def mod() -> HybridModerator:
        return state["mod"]

    def queue(m: HybridModerator = Depends(mod)) -> ReviewQueue:
        if m.queue is None:
            raise HTTPException(503, "review queue not configured")
        return m.queue

    @app.get("/healthz")
    def healthz(m: HybridModerator = Depends(mod)):
        return {"status": "ok", "llm_enabled": m.use_llm, "rules_enabled": m.use_rules}

    @app.get("/v1/config")
    def config(m: HybridModerator = Depends(mod)):
        c = m.cfg.model_dump()
        c["llm"].pop("api_key_env", None)
        return c

    @app.post("/v1/moderate")
    def moderate(body: ModerateIn, m: HybridModerator = Depends(mod)):
        return m.moderate(body.text, body.comment_id).to_dict()

    @app.post("/v1/moderate/batch")
    def moderate_batch(body: BatchIn, m: HybridModerator = Depends(mod)):
        ds = m.moderate_many([c.text for c in body.comments],
                             [c.comment_id or "" for c in body.comments])
        out = [d.to_dict() for d in ds]
        summary = {
            "n": len(out),
            "hidden": sum(d["flagged"] for d in out),
            "human_review": sum(d["human_review"] for d in out),
            "llm_calls_routed": sum(d["llm_called"] for d in out),
        }
        return {"results": out, "summary": summary}

    @app.get("/v1/review-queue")
    def list_queue(status: str = Query("pending", pattern="^(pending|resolved)$"),
                   limit: int = Query(50, ge=1, le=500), q: ReviewQueue = Depends(queue)):
        return {"items": q.list(status, limit), "stats": q.stats()}

    @app.post("/v1/review-queue/{item_id}/resolve")
    def resolve(item_id: int, body: ResolveIn, q: ReviewQueue = Depends(queue)):
        try:
            return q.resolve(item_id, body.human_action, body.reviewer, body.note)
        except KeyError:
            raise HTTPException(404, "item not found") from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    return app


app = create_app()
