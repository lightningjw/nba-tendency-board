"""The hot-hand report: after a player makes a shot, does he finish his team's next possession more often, and does it pay?

Kept apart from the team reports on purpose. It is one league-wide question, not a tendency of a team.

For every offensive possession, look at the same team's previous trip in the same period. If that
trip ended in a field-goal attempt by a player who is still on the floor, the possession is "after
his make" or "after his miss". Two outcomes on it:

* **Share.** Did he finish this possession? Compared with what his usage and the four teammates on
  the floor predict (the same choose-one strengths the tendency layer uses, fitted per team and
  season), so a star who makes and takes a lot of shots is not counted as hot for being a star.
* **Points.** Points the team scored on the possession, minus that team's season average.

A player who has just shot is in the flow of the offense whether or not the shot went in, so both
groups sit above expectation. The make-minus-miss difference is the hot-hand number.

Intervals: a bootstrap over whole games, like the scorecard. A null result is a result.

Play-by-play does not say who a play was run for. "Finishes the next possession more often" is all
this measures; it does not claim the ball was sent to him.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tendency_layer import _eligible, _on_floor, fit_usage

INTERVAL_LEVEL = 0.95
N_BOOT = 1000
SEED = 20261123
MIN_EVENTS = 200            # per group; below this the report says "not enough" instead of printing wide noise
DEFINITION = ("The same team's previous trip in the same period ended in a field-goal attempt by a player who is still on the floor. "
              "Share is how often he finishes this possession; expected is what his usage and the four teammates on the floor predict. "
              "Points are the team's on this possession, against its season average.")


def events(poss: pd.DataFrame) -> pd.DataFrame:
    """One row per possession that follows a field-goal attempt by a player still on the floor.

    Columns: game_id, made (the previous shot), y (he finished this one), e (expected share), pts_diff (points minus the team's season mean).
    """
    df = _eligible(poss).sort_values(["game_id", "poss_idx"]).reset_index(drop=True)
    g = df.groupby(["game_id", "off_team_id"], sort=False)
    prev_shot = (g["ending_type"].shift(1) == "shot") & g["finisher_id"].shift(1).notna() & g["shot_made"].shift(1).notna()
    same_trip = (g["period"].shift(1) == df["period"]) & (df["poss_idx"] - g["poss_idx"].shift(1) <= 2)     # really the trip before, not one from before a gap
    df["prev_ok"] = (prev_shot & same_trip).values
    df["prev_fin"] = g["finisher_id"].shift(1).astype("Int64").fillna(-1).astype(np.int64).values
    df["prev_made"] = g["shot_made"].shift(1).astype("boolean").fillna(False).astype(bool).values
    ok, lu_all, hit_all = _on_floor(df)
    out = []
    for (_, _), grp in ok.groupby(["season", "off_team_id"], sort=False):
        idx = grp.index.values
        players, lu = np.unique(lu_all[idx], return_inverse=True)
        lu = lu.reshape(-1, 5)
        hit = hit_all[idx]
        w = fit_usage(lu, hit.argmax(axis=1), len(players))
        e = w[lu] / w[lu].sum(axis=1, keepdims=True)
        is_prev = lu_all[idx] == grp["prev_fin"].values[:, None]
        rows = np.flatnonzero(grp["prev_ok"].values & is_prev.any(axis=1))            # he is still on the floor
        slot = is_prev[rows].argmax(axis=1)
        pts = grp["points"].values.astype(float)
        out.append(pd.DataFrame({"game_id": grp["game_id"].values[rows], "made": grp["prev_made"].values[rows],
                                 "y": hit[rows, slot].astype(float), "e": e[rows, slot], "pts": pts[rows], "pts_diff": pts[rows] - pts.mean()}))
    cols = ["game_id", "made", "y", "e", "pts", "pts_diff"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def _ci(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [(1 - INTERVAL_LEVEL) / 2 * 100, (1 + INTERVAL_LEVEL) / 2 * 100])
    return [round(float(lo), 4), round(float(hi), 4)]


def _verdict(interval: list[float], up: str, down: str) -> str:
    return up if interval[0] > 0 else down if interval[1] < 0 else "no_clear_difference"


def hot_hand_report(poss: pd.DataFrame) -> dict:
    """The contract's hot_hand.json body (without data_kind). `available` is false when there is too little to say anything."""
    base = {"interval_level": INTERVAL_LEVEL, "bootstrap": f"{N_BOOT:,} resamples of whole games", "definition": DEFINITION,
            "seasons": sorted(int(s) for s in poss["season"].unique()), "n_games": 0, "after": [], "make_minus_miss": None, "verdict": None}
    ev = events(poss)
    n_make, n_miss = int(ev["made"].sum()), int((~ev["made"].astype(bool)).sum())
    if min(n_make, n_miss) < MIN_EVENTS:
        return {"available": False, "reason": f"Fewer than {MIN_EVENTS} possessions follow a make or a miss by a player still on the floor", **base}
    ev["d"] = ev["y"] - ev["e"]
    ev["one"] = 1.0
    games = pd.Index(ev["game_id"].unique())
    stats = {}
    for made in (True, False):
        s = ev[ev["made"] == made].groupby("game_id")[["one", "y", "e", "d", "pts", "pts_diff"]].sum().reindex(games, fill_value=0.0)
        stats[made] = {c: s[c].values for c in s.columns}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(len(games), size=(N_BOOT, len(games)))

    def mean(made: bool, col: str, i=None) -> np.ndarray | float:
        s = stats[made]
        if i is None:
            return float(s[col].sum() / s["one"].sum())
        return s[col][i].sum(axis=1) / np.maximum(s["one"][i].sum(axis=1), 1.0)

    after = []
    for made in (True, False):
        after.append({"prior": "make" if made else "miss", "n": int(stats[made]["one"].sum()),
                      "share": round(mean(made, "y"), 4), "expected_share": round(mean(made, "e"), 4),
                      "share_vs_expected": {"value": round(mean(made, "d"), 4), "interval": _ci(mean(made, "d", idx))},
                      "points": round(mean(made, "pts"), 4),
                      "points_vs_team": {"value": round(mean(made, "pts_diff"), 4), "interval": _ci(mean(made, "pts_diff", idx))}})
    diff = {"share": {"value": round(mean(True, "d") - mean(False, "d"), 4), "interval": _ci(mean(True, "d", idx) - mean(False, "d", idx))},
            "points": {"value": round(mean(True, "pts_diff") - mean(False, "pts_diff"), 4),
                       "interval": _ci(mean(True, "pts_diff", idx) - mean(False, "pts_diff", idx))}}
    return {"available": True, "reason": None, **base, "n_games": int(len(games)), "after": after, "make_minus_miss": diff,
            "verdict": {"share": _verdict(diff["share"]["interval"], "rises", "falls"), "points": _verdict(diff["points"]["interval"], "more", "fewer")}}
