"""Shared fixtures: a hand-built mini game builder, and one full synthetic pipeline run per session."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tendency import cdn_fields as F  # noqa: E402

HOME, AWAY = 100, 200
H = [101, 102, 103, 104, 105, 106, 107]      # 101-105 start
A = [201, 202, 203, 204, 205, 206, 207]


def clock(seconds: float) -> str:
    return f"PT{int(seconds // 60):02d}M{seconds % 60:05.2f}S"


class Game:
    """Tiny play-by-play builder in the assumed CDN shape. Scores are tracked automatically."""

    def __init__(self):
        self.actions, self.score, self.period = [], {HOME: 0, AWAY: 0}, 0

    def ev(self, t, clk, sub="", team=None, pid=None, **kw):
        a = {F.ACTION_NUMBER: len(self.actions) + 1, F.ORDER_NUMBER: (len(self.actions) + 1) * 10000, F.PERIOD: self.period,
             F.CLOCK: clock(clk), F.ACTION_TYPE: t, F.SUB_TYPE: sub, F.DESCRIPTOR: kw.pop("descriptor", ""), F.PERSON_ID: pid or 0}
        if team:
            a[F.TEAM_ID] = team
        pts = kw.pop("pts", 0)
        if pts:
            self.score[team] += pts
        a[F.SCORE_HOME], a[F.SCORE_AWAY] = str(self.score[HOME]), str(self.score[AWAY])
        a.update(kw)
        self.actions.append(a)
        return self

    def start(self, period=1):
        self.period = period
        return self.ev("period", 720 if period <= 4 else 300, "start")

    def end(self):
        return self.ev("period", 0, "end")

    def shot(self, clk, team, pid, made, three=False, area="Mid-Range", dist=15, **kw):
        if three and area == "Mid-Range":
            area = "Above the Break 3"
        return self.ev("3pt" if three else "2pt", clk, "Jump Shot", team, pid, shotResult="Made" if made else "Missed",
                       pts=(3 if three else 2) if made else 0, **{F.AREA: area, F.SHOT_DISTANCE: dist}, **kw)

    def reb(self, clk, team, pid, offensive=False):
        return self.ev("rebound", clk, "offensive" if offensive else "defensive", team, pid)

    def tov(self, clk, team, pid):
        return self.ev("turnover", clk, "bad pass", team, pid)

    def foul(self, clk, team, pid, drawn=None, sub="personal", descriptor=""):
        return self.ev("foul", clk, sub, team, pid, descriptor=descriptor, **({F.FOUL_DRAWN_PERSON_ID: drawn} if drawn else {}))

    def ft(self, clk, team, pid, k, n, made, technical=False, **kw):
        return self.ev("freethrow", clk, f"{k} of {n}", team, pid, shotResult="Made" if made else "Missed", pts=1 if made else 0,
                       descriptor="technical" if technical else "", **kw)

    def sub(self, clk, team, out_pid, in_pid):
        return self.ev("substitution", clk, "out", team, out_pid).ev("substitution", clk, "in", team, in_pid)

    def timeout(self, clk, team):
        return self.ev("timeout", clk, "full", team)

    def pbp(self, game_id="0022500001"):
        return {F.GAME: {F.GAME_ID: game_id, F.ACTIONS: self.actions}}


def make_box(game_id="0022500001", tip="2025-10-21T23:30:00Z"):
    def side(tid, ids, name):
        return {F.BOX_TEAM_ID: tid, F.BOX_TEAM_TRICODE: name[:3].upper(), F.BOX_TEAM_NAME: name, F.BOX_TEAM_CITY: "Test",
                F.BOX_PLAYERS: [{F.BOX_PLAYER_ID: p, F.BOX_PLAYER_NAME: f"P{p}", F.BOX_PLAYER_STARTER: "1" if i < 5 else "0"} for i, p in enumerate(ids)]}
    return {F.GAME: {F.GAME_ID: game_id, F.BOX_GAME_TIME_UTC: tip, F.BOX_HOME: side(HOME, H, "Homers"), F.BOX_AWAY: side(AWAY, A, "Awayers")}}


@pytest.fixture
def game():
    return Game()


@pytest.fixture
def box():
    return make_box()


@pytest.fixture(scope="session")
def synth_run(tmp_path_factory):
    """Generate synthetic raw games and run the whole pipeline once."""
    from tendency.pipeline import run
    from tendency.synth import write_synthetic_raw
    root = tmp_path_factory.mktemp("synthetic")
    out = tmp_path_factory.mktemp("export")
    ids = write_synthetic_raw(root, seed=7)
    counts = run(root, out, max_replays=6, log=lambda *_: None, transformer_epochs=2)     # the transformer trains only if PyTorch is installed
    poss = pd.read_parquet(root / "processed" / "possessions.parquet")
    return {"root": root, "out": out, "ids": ids, "counts": counts, "poss": poss,
            "games": json.loads((root / "processed" / "games.json").read_text())}


@pytest.fixture(scope="session")
def small_poss(synth_run):
    """First 48 synthetic games: enough history for features, small enough for repeated rebuilds."""
    poss = synth_run["poss"]
    first = poss.drop_duplicates("game_id").sort_values("tipoff")["game_id"].head(48)
    return poss[poss["game_id"].isin(first)].reset_index(drop=True)
