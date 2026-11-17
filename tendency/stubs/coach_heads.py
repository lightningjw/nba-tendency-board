"""Phase 2: coach-decision heads (next substitution, timeout timing, closing lineup, late fouls, challenges).

Needs new label columns in possessions.py (or a separate dead-ball event table) and new head ids in
contract/schemas/manifest.schema.json. Each head still needs a frequency baseline first.
"""
from . import NotBuilt


def label_coach_decisions(possessions):
    raise NotBuilt("Coach-decision heads", "Phase 2")
