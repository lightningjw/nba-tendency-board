"""The ask box as a small web service (FastAPI). One endpoint does the work: POST /ask.

    python -m tendency serve --root data/synthetic            # local, on http://127.0.0.1:8000
    uvicorn tendency.ask_service:from_env --factory           # a host: configured by the environment, below

POST /ask  {"question": "..."}  ->  the query that was run, the rows it got and one sentence written
from those rows (contract/schemas/ask_response.schema.json). GET /health says what the service is
reading and who translates, so the app can show it and a host can probe it.

Environment, read by `from_env`:
    ASK_TABLE           path to ask_cells.parquet (required)
    ASK_TRANSLATOR      auto (default) | llm | rules.  auto = llm when a key is set, else rules
    ASK_MODEL           model for llm (default claude-opus-5-5)
    ASK_EFFORT          low (default) | medium | high
    ASK_ALLOW_ORIGINS   comma-separated origins allowed to call from a browser (default *)
    ASK_DAILY_LIMIT     questions sent to the model per UTC day before the service answers 429 (default 300)
    ANTHROPIC_API_KEY   needed for llm; read from the environment only
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import CONTRACT_VERSION, ask

DAILY_LIMIT = 300


class _Budget:
    """How many questions went to the model today (UTC). In memory: a restart starts the day again."""

    def __init__(self, limit: int):
        self.limit, self.day, self.used, self.lock = int(limit), None, 0, threading.Lock()

    def take(self) -> bool:
        today = datetime.now(timezone.utc).date()
        with self.lock:
            if today != self.day:
                self.day, self.used = today, 0
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


def has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def make_translator(table: ask.Table, translator: str = "auto", model: str | None = None, effort: str | None = None, client=None):
    if translator not in ("auto", "llm", "rules"):
        raise ask.AskError("the translator is one of auto, llm, rules")
    if translator == "auto":
        translator = "llm" if (client is not None or has_key()) else "rules"
    if translator == "rules":
        return ask.RulesTranslator(table)
    return ask.LLMTranslator(table, client, model, effort)             # raises AskError here, at start, when there is no key


def create_app(table_path: Path | str, translator: str = "auto", model: str | None = None, effort: str | None = None,
               origins: list[str] | None = None, daily_limit: int = DAILY_LIMIT, client=None):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    table = ask.Table(table_path)
    tr = make_translator(table, translator, model, effort, client)
    budget = _Budget(daily_limit)
    app = FastAPI(title="Tendency Board ask box", version=CONTRACT_VERSION, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=origins or ["*"], allow_methods=["GET", "POST"], allow_headers=["content-type"])

    class Question(BaseModel):
        question: str

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "contract_version": CONTRACT_VERSION, "data_kind": table.data_kind, "as_of_utc": table.as_of,
                "seasons": table.season_labels, "n_teams": len(table.teams), "max_question": ask.MAX_QUESTION,
                "translated_by": {"kind": tr.kind, "model": getattr(tr, "model", None)}}

    @app.post("/ask")
    def ask_endpoint(body: Question) -> dict:                           # a plain def: FastAPI runs it in its thread pool
        question = " ".join(body.question.split())
        if not question:
            raise HTTPException(422, "The question is empty.")
        if len(question) > ask.MAX_QUESTION:
            raise HTTPException(422, f"The question is longer than {ask.MAX_QUESTION} characters.")
        if tr.kind == "llm" and not budget.take():
            raise HTTPException(429, f"The service has passed its limit of {budget.limit} questions for today (UTC). Try again tomorrow.")
        return ask.ask(question, table, tr)

    return app


def from_env():
    """The app for a host: `uvicorn tendency.ask_service:from_env --factory`."""
    path = os.environ.get("ASK_TABLE")
    if not path:
        raise ask.AskError("set ASK_TABLE to the path of ask_cells.parquet")
    origins = [o.strip() for o in os.environ.get("ASK_ALLOW_ORIGINS", "*").split(",") if o.strip()]
    return create_app(path, os.environ.get("ASK_TRANSLATOR", "auto"), os.environ.get("ASK_MODEL"), os.environ.get("ASK_EFFORT"),
                      origins, int(os.environ.get("ASK_DAILY_LIMIT", DAILY_LIMIT)))
