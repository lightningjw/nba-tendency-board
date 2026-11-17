"""Phase 3: live mode. A small FastAPI service polls tonight's game from the CDN feed, runs
possessions.segment_game on the partial file, and serves the replay contract for the game in progress.

Open question from the brief: whether cdn.nba.com is reachable from a cloud host at all.
"""
from . import NotBuilt


def serve_live(game_id: str):
    raise NotBuilt("Live mode", "Phase 3")
