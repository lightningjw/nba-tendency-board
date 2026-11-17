"""Phase 3: LLM-written team summaries and the ask box.

Rule from the brief: the LLM writes only from a tendency table passed to it and translates questions
into queries over those tables. It never answers from its own memory.
"""
from . import NotBuilt


def summarize_team_report(report: dict) -> str:
    raise NotBuilt("LLM-written summaries", "Phase 3")


def question_to_query(question: str) -> dict:
    raise NotBuilt("The ask box", "Phase 3")
