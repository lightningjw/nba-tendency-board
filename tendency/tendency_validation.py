"""Does the tendency layer earn its place? Fit on the first half of a season, predict the second half.

For every candidate cell (team or player, situation, outcome) that has enough data in both halves,
each method gives a rate and a posterior for it from first-half data only. Two questions:

* log-loss of the predicted rate on second-half outcomes (lower is better), and
* coverage: does the method's 90% predictive interval for the second-half count contain what
  happened about 90% of the time? The predictive interval includes binomial noise in the second
  half, so a calibrated method scores about 0.90. Rates also drift between halves (trades,
  injuries, coaching), which no method here models, so somewhat under 0.90 is expected.

Methods: raw first-half rates (Jeffreys interval), the old stand-in (`tendencies.py`), and the
hierarchical model (`tendency_layer.py`). A method ships only if it beats the stand-in here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from . import tendencies as standin
from . import tendency_layer as layer

N_DRAWS = 4000
SEED = 20261123
LEVEL = 0.90


def split_halves(poss: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First and second half of the games, by tip time."""
    games = poss.drop_duplicates("game_id").sort_values(["tipoff", "game_id"])["game_id"].values
    first = set(games[: len(games) // 2])
    m = poss["game_id"].isin(first)
    return poss[m], poss[~m]


def _key(df: pd.DataFrame) -> pd.Series:
    pid = df["player_id"].fillna(0).astype(int).astype(str)
    return df["team_id"].astype(str) + "|" + df["scope"] + "|" + pid + "|" + df["sid"] + "|" + df["head"] + "|" + df["cls"].astype(str)


def method_draws(first: pd.DataFrame, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """For each method: a frame indexed by cell key with the predicted rate `p` and `draws` (N_DRAWS posterior draws of the rate)."""
    cand = layer._candidates(first)
    est = layer.estimate(cand)
    est["key"] = _key(est)
    k, n = est["k"].values, est["n"].values
    out = {}
    out["raw"] = pd.DataFrame({"p": (k + 0.5) / (n + 1.0), "draws": list(rng.beta(k + 0.5, n - k + 0.5, size=(N_DRAWS, len(est))).T)}, index=est["key"])
    z = rng.standard_normal((N_DRAWS, len(est)))
    out["hierarchical"] = pd.DataFrame({"p": expit(est["post_logit"].values),
                                        "draws": list(expit(est["post_logit"].values + z * est["post_sd"].values).T)}, index=est["key"])
    rows = []
    for t, rep in standin.team_reports(first, {}, top_k=None).items():
        for r in rep["tendencies"]:
            m, b = r["prior_strength"], r["baseline_rate"]
            rows.append((f"{t}|{r['scope']}|{r['player_id'] or 0}|{r['situation']['id']}|{r['outcome']['head']}|{r['outcome']['class']}",
                         b * m + r["k"], (1 - b) * m + r["n"] - r["k"]))
    s = pd.DataFrame(rows, columns=["key", "a", "b"]).set_index("key")
    out["stand_in"] = pd.DataFrame({"p": (s["a"] / (s["a"] + s["b"])).values, "draws": list(rng.beta(s["a"].values, s["b"].values, size=(N_DRAWS, len(s))).T)},
                                   index=s.index)
    return out


def score(pred: pd.DataFrame, actual: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Log-loss per trial and predictive-interval coverage of one method on the cells in `actual` (index = key; columns k, n)."""
    p = np.clip(pred.loc[actual.index, "p"].values, 1e-6, 1 - 1e-6)
    k2, n2 = actual["k"].values, actual["n"].values
    ll = -(k2 * np.log(p) + (n2 - k2) * np.log(1 - p)).sum() / n2.sum()
    draws = np.stack(pred.loc[actual.index, "draws"].values)                     # (cells, draws)
    sim = rng.binomial(n2[:, None].astype(int), draws)
    lo, hi = np.percentile(sim, [(1 - LEVEL) / 2 * 100, (1 + LEVEL) / 2 * 100], axis=1)
    return {"log_loss": round(float(ll), 5), "coverage": round(float(((k2 >= lo) & (k2 <= hi)).mean()), 4),
            "mean_abs_error": round(float(np.abs(p - k2 / n2).mean()), 4),
            "interval_width": round(float(((hi - lo) / n2).mean()), 4)}


def validate(poss: pd.DataFrame, extra: dict[str, pd.DataFrame] | None = None) -> dict:
    """Half-season validation on one season's possessions. `extra` adds methods (e.g. a PyMC fit) in `method_draws` shape."""
    rng = np.random.default_rng(SEED)
    first, second = split_halves(poss)
    preds = method_draws(first, rng)
    preds.update(extra or {})
    act = layer._candidates(second)
    act["key"] = _key(act)
    act = act.set_index("key")[["scope", "k", "n"]]
    common = act.index
    for p in preds.values():
        common = common.intersection(p.index)
    act = act.loc[common]
    out = {"interval_level": LEVEL, "n_games_first": int(first["game_id"].nunique()), "n_games_second": int(second["game_id"].nunique()),
           "scopes": []}
    for scope in ("team", "player"):
        a = act[act["scope"] == scope]
        if a.empty:
            continue
        out["scopes"].append({"scope": scope, "n_cells": int(len(a)), "n_trials": int(a["n"].sum()),
                              "methods": [{"method": name, **score(p, a, rng)} for name, p in preds.items()]})
    return out
