"""Leakage guards.

`test_no_pregame_leak` and `test_no_ingame_leak` are the real guards: they fail if any
registered feature for a game moves when the outcomes of that game or later ones change.
The two `test_checker_catches_*` tests prove the guard has teeth by feeding it a
deliberately leaky feature.
"""
from __future__ import annotations

import numpy as np

from tendency import features as ft
from tendency.features import Features, build_features
from tendency.leakcheck import ingame_leaks, pregame_leaks

PRE_POSS, PRE_CAND = ft.PREGAME_POSS, ft.PREGAME_CAND
IN_POSS, IN_CAND = ft.INGAME_POSS, ft.INGAME_CAND


def _games(poss):
    return poss.drop_duplicates("game_id").sort_values("tipoff")["game_id"].tolist()


def test_every_model_feature_is_registered_with_a_kind():
    for head, cols in ft.FEATURES.items():
        missing = [c for c in cols if c not in ft.KIND]
        assert not missing, f"{head}: features without a context/pregame/ingame kind: {missing}"
        assert len(cols) == len(set(cols))
    assert set(ft.KIND.values()) == {"context", "pregame", "ingame"}


def test_no_pregame_leak(small_poss):
    games = _games(small_poss)
    for gid in (games[5], games[30]):
        assert pregame_leaks(build_features, small_poss, gid, PRE_POSS, PRE_CAND) == []


def test_no_ingame_leak(small_poss):
    gid = _games(small_poss)[20]
    for k in (1, 60, 150):
        assert ingame_leaks(build_features, small_poss, gid, k, IN_POSS, IN_CAND) == []


def test_pregame_features_do_move_when_earlier_games_change(small_poss):
    """Sanity check in the other direction: the features are not constants."""
    from tendency.leakcheck import scramble_outcomes
    games = _games(small_poss)
    target = games[30]
    tip = small_poss.loc[small_poss["game_id"] == target, "tipoff"].iloc[0]
    a = build_features(small_poss).poss
    b = build_features(scramble_outcomes(small_poss, (small_poss["tipoff"] < tip).values)).poss
    col = "base_end_turnover"
    va, vb = a.loc[a["game_id"] == target, col].values, b.loc[b["game_id"] == target, col].values
    assert not np.allclose(va[: min(len(va), len(vb))], vb[: min(len(va), len(vb))])


def _leaky_pregame(poss):
    """Adds a same-game aggregate: the offense's turnover share in THIS game."""
    f = build_features(poss)
    share = poss.assign(t=(poss["ending_type"] == "turnover").astype(float)).groupby(["game_id", "off_team_id"])["t"].mean().rename("leaky")
    return Features(f.poss.merge(share, left_on=["game_id", "off_team_id"], right_index=True), f.cand)


def _leaky_ingame(poss):
    """Adds a running count that includes the current possession's own outcome."""
    f = build_features(poss)
    p = poss.sort_values(["game_id", "poss_idx"]).copy()
    p["leaky"] = (p["ending_type"] == "turnover").astype(int).groupby([p["game_id"], p["off_team_id"]]).cumsum()
    return Features(f.poss.merge(p[["game_id", "poss_idx", "leaky"]], on=["game_id", "poss_idx"]), f.cand)


def test_checker_catches_a_same_game_pregame_leak(small_poss):
    gid = _games(small_poss)[30]
    assert pregame_leaks(_leaky_pregame, small_poss, gid, PRE_POSS + ["leaky"], PRE_CAND) == ["leaky"]


def test_checker_catches_an_ingame_leak_of_the_current_possession(small_poss):
    gid = _games(small_poss)[20]
    assert ingame_leaks(_leaky_ingame, small_poss, gid, 60, IN_POSS + ["leaky"], IN_CAND) == ["leaky"]


def test_first_games_get_the_neutral_prior(small_poss):
    """With no earlier games, baselines are uniform and usage is one in five."""
    f = build_features(small_poss)
    first = _games(small_poss)[0]
    p = f.poss[f.poss["game_id"] == first]
    assert np.allclose(p[[f"base_end_{c}" for c in ("shot", "foul_drawn", "turnover")]].values, 1 / 3)
    c = f.cand[f.cand["game_id"].astype(object) == first]
    assert np.allclose(c["p_base"].values, 0.2) and np.allclose(c["p_eff_on"].values, 0.0)


def test_baselines_are_probability_vectors(small_poss):
    f = build_features(small_poss)
    assert np.allclose(f.poss[[c for c in f.poss if c.startswith("base_end_")]].sum(axis=1), 1.0)
    assert np.allclose(f.poss[[c for c in f.poss if c.startswith("base_fam_")]].sum(axis=1), 1.0)
    assert np.allclose(f.cand["p_base"].values.reshape(-1, 5).sum(axis=1), 1.0)
    assert not f.poss[ft.FEATURES["ending_type"]].isna().any().any()
    assert not f.cand[ft.FEATURES["finisher"]].isna().any().any()
