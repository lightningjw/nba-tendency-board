"""Leakage check used by tests/test_leakage.py.

Idea: if a feature for game G only uses earlier games, then rewriting the outcomes
of G and of every later game cannot change it. Same for in-game features and
later possessions of the same game.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

KEYS_POSS = ["game_id", "poss_idx"]
KEYS_CAND = ["game_id", "poss_idx", "player_id"]


def scramble_outcomes(poss: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Overwrite the outcome columns on the masked rows with different values."""
    out = poss.copy()
    idx = np.flatnonzero(mask)
    was_tov = out["ending_type"].values[idx] == "turnover"
    out.loc[out.index[idx], "ending_type"] = np.where(was_tov, "shot", "turnover")
    out.loc[out.index[idx], "shot_family"] = pd.Series(np.where(was_tov, "midrange", None), index=out.index[idx], dtype=object)
    old_fin = out["finisher_id"].astype("float64").fillna(-1).values[idx]
    new_fin = [None if len(l) == 0 else int(l[-1]) if f == l[0] else int(l[0])
               for l, f in zip(out["off_lineup"].values[idx], old_fin)]
    out.loc[out.index[idx], "finisher_id"] = pd.array(new_fin, dtype="Int64")
    # step 2.2 labels: a turnover becomes a floater, anything else loses its style; tempo flips early <-> late
    if "action_style" in out.columns:
        out.loc[out.index[idx], "action_style"] = pd.Series(np.where(was_tov, "floater", None), index=out.index[idx], dtype=object)
    if "tempo" in out.columns:
        old = out["tempo"].values[idx]
        out.loc[out.index[idx], "tempo"] = pd.Series(np.where(old == "early", "late", "early"), index=out.index[idx], dtype=object)
    return out


def _changed(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], cols: list[str]) -> list[str]:
    m = a[keys + cols].merge(b[keys + cols], on=keys, suffixes=("_a", "_b"))
    if len(m) != len(a):
        return ["<row set changed>"]
    return [c for c in cols if not np.allclose(m[f"{c}_a"].astype(float), m[f"{c}_b"].astype(float), equal_nan=True)]


def pregame_leaks(build: Callable, poss: pd.DataFrame, game_id: str, poss_cols: list[str], cand_cols: list[str]) -> list[str]:
    """Columns of game `game_id` that move when outcomes of that game and all later games are scrambled."""
    tip = poss.loc[poss["game_id"] == game_id, "tipoff"].iloc[0]
    base, alt = build(poss), build(scramble_outcomes(poss, (poss["tipoff"] >= tip).values))
    sel = lambda d: d[d["game_id"].astype(object) == game_id].assign(game_id=game_id)  # noqa: E731
    # eligibility can change when outcomes are scrambled, so compare on rows present in both
    return (_changed_common(sel(base.poss), sel(alt.poss), KEYS_POSS, poss_cols)
            + _changed_common(sel(base.cand), sel(alt.cand), KEYS_CAND, cand_cols))


def ingame_leaks(build: Callable, poss: pd.DataFrame, game_id: str, from_idx: int, poss_cols: list[str], cand_cols: list[str]) -> list[str]:
    """Columns of possessions <= from_idx that move when outcomes of possessions >= from_idx in the game are scrambled."""
    mask = ((poss["game_id"] == game_id) & (poss["poss_idx"] >= from_idx)).values
    base, alt = build(poss), build(scramble_outcomes(poss, mask))
    sel = lambda d: d[(d["game_id"].astype(object) == game_id) & (d["poss_idx"] <= from_idx)].assign(game_id=game_id)  # noqa: E731
    return (_changed_common(sel(base.poss), sel(alt.poss), KEYS_POSS, poss_cols)
            + _changed_common(sel(base.cand), sel(alt.cand), KEYS_CAND, cand_cols))


def _changed_common(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], cols: list[str]) -> list[str]:
    m = a[keys + cols].merge(b[keys + cols], on=keys, suffixes=("_a", "_b"))
    if len(m) == 0:
        return ["<no common rows to compare>"]
    return [c for c in cols if not np.allclose(m[f"{c}_a"].astype(float), m[f"{c}_b"].astype(float), equal_nan=True)]
