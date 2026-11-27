"""Does a defensive fingerprint carry over? Fit on the first half of a season, predict each second-half game.

Everything a method knows comes from first-half games: the opponents' usual rates, the league's
start-type shifts and the defense's deviation. It is then asked for the probability of each outcome
on every second-half possession the defense (or the unit) played, and scored by log-loss.

Team cells, five methods:

* `league`: the first-half league rate. Neither side matters.
* `pooled_allowed`: what this defense allowed in the first half, pooled toward the league. No
  opponent information: the unadjusted number.
* `opponents`: what tonight's opponent usually does. The defense does not matter. THE REFERENCE:
  a fingerprint that cannot beat it says nothing about the defense.
* `opponents_raw`: the opponents' rate moved by the defense's first-half deviation, not pooled.
* `fingerprint`: the same with the pooled deviation. What the report shows.

Unit cells (one player's lineups, one five), three methods on the possessions the unit defended:
`team` (the team's fingerprint, as if the unit were any five; the reference), `unit_raw` and `unit`.

Differences come with a 95% interval from resampling whole second-half games. Coverage is checked
the way `tendency_validation` checks a deviation: does the first-half interval, widened by the second
half's own noise, contain the second-half deviation.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import defense
from . import tendency_layer as layer
from .tendency_validation import pick_seasons

N_BOOT = 1000
SEED = 20261127
LEVEL = defense.INTERVAL_LEVEL
GROUPS = [("shot_profile", "Where opponents' shots come from (five areas)"), ("foul_drawn", "Opponent possessions ending in a foul drawn"),
          ("turnover", "Opponent possessions ending in a turnover")]
TEAM_METHODS = {"league": "League rate", "pooled_allowed": "What this defense allowed, pooled toward the league (no opponent adjustment)",
                "opponents": "What the opponent usually does (no defense)", "opponents_raw": "Opponent's usual, moved by the defense's raw deviation",
                "fingerprint": "Opponent's usual, moved by the pooled fingerprint"}
UNIT_METHODS = {"team": "The team's fingerprint, whoever is on the floor", "unit_raw": "The unit's raw deviation", "unit": "The unit's pooled fingerprint"}
REFERENCE = {"defense_team": "opponents", "defense_player": "team", "defense_lineup": "team"}
SHIPPED = {"defense_team": "fingerprint", "defense_player": "unit", "defense_lineup": "unit"}


def _rng(*names: str) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32("|".join(names).encode())])


def _group(head: str, cls: str) -> str:
    return "shot_profile" if head == "shot_family" else cls


def _loss(p: np.ndarray, hit: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.where(hit, np.log(p), np.log(1 - p))


def _paired(a: np.ndarray, b: np.ndarray, n: np.ndarray, rng: np.random.Generator) -> dict:
    """Per-trial log-loss of a minus b from per-game sums, with a 95% interval from resampling games."""
    idx = rng.integers(len(n), size=(N_BOOT, len(n)))
    diffs = (a[idx].sum(axis=1) - b[idx].sum(axis=1)) / np.maximum(n[idx].sum(axis=1), 1.0)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(float((a.sum() - b.sum()) / n.sum()), 6) + 0.0, "interval": [round(float(lo), 6) + 0.0, round(float(hi), 6) + 0.0]}


def _verdict(d: dict) -> str:
    lo, hi = d["interval"]
    return "ahead" if hi < 0 else "behind" if lo > 0 else "level"


def validate(poss: pd.DataFrame, drift: dict | None = None, interleave: bool = False, by_start: bool = True) -> dict:
    """Half-season check on one season. `drift` (from `defense.estimate_drift` on an earlier season) only widens intervals."""
    first, second = defense.split_halves(poss, interleave)
    expect = defense.Expectation(first, by_start)
    cand = defense.candidates(first, expect)
    est = defense.estimate(cand)
    est["shown"] = defense.estimate(cand, drift)["prob_apart"].values >= defense.SHOW_PROB      # judged as a report would judge it
    out = {"interval_level": LEVEL, "n_games_first": int(first["game_id"].nunique()), "n_games_second": int(second["game_id"].nunique()),
           "split": "alternating games" if interleave else "by date", "slices": [], "coverage": []}
    if est.empty:
        return out
    df = layer._eligible(second)
    teams, T = expect.teams, len(expect.teams)
    df = df[np.isin(df["off_team_id"].values, teams) & np.isin(df["def_team_id"].values, teams)].reset_index(drop=True)
    games = pd.Index(sorted(df["game_id"].unique()))
    gi = games.get_indexer(df["game_id"])
    d = np.searchsorted(teams, df["def_team_id"].values)
    lu, lu_ok = defense._lineups(df)
    units2 = {(s, j, key): idx for s, j, key, idx in defense._units(d, lu, lu_ok, T, tuple(defense.SCOPES), 1)}
    ef = layer._eligible(first)
    d1 = np.searchsorted(teams, ef["def_team_id"].values)
    by_cell = est.set_index("key")
    # per (scope, group, method): per-game loss sums; per (scope, group): per-game trials and the cells scored
    loss: dict[tuple[str, str, str], np.ndarray] = {}
    trials: dict[tuple[str, str], np.ndarray] = {}
    n_cells: dict[tuple[str, str], int] = {}

    def add(scope: str, grp: str, method: str, rows: np.ndarray, p: np.ndarray, hit: np.ndarray) -> None:
        a = loss.setdefault((scope, grp, method), np.zeros(len(games)))
        np.add.at(a, gi[rows], _loss(p, hit))

    for head, cls, _ in defense.OUTCOMES:
        grp = _group(head, cls)
        hit, den = defense._masks(df, head, cls)
        hit1, den1 = defense._masks(ef, head, cls)
        p_opp, _ = expect.at(df, head, cls)
        eta = logit(np.clip(p_opp, 1e-6, 1 - 1e-6))
        # ---- team cells ----
        k1 = np.bincount(d1[den1 & hit1], minlength=T).astype(float)
        n1 = np.bincount(d1[den1], minlength=T).astype(float)
        y1, v1 = layer.emp_logit(k1, np.maximum(n1, 1e-9))
        mu, _, m, _ = layer.eb_normal(y1, v1, center=True)
        tc = est[(est["scope"] == "defense_team") & (est["head"] == head) & (est["cls"] == cls)]
        fp, raw = np.full(T, np.nan), np.full(T, np.nan)
        j = np.searchsorted(teams, tc["team_id"].values)
        fp[j] = tc["fp_logit"].values
        raw[j] = layer.emp_logit(tc["k"].values, tc["n"].values)[0] - tc["opp_logit"].values
        rows = np.flatnonzero(den & ~np.isnan(fp[d]))
        if len(rows):
            h, dd = hit[rows], d[rows]
            add("defense_team", grp, "league", rows, np.full(len(rows), k1.sum() / n1.sum()), h)
            add("defense_team", grp, "pooled_allowed", rows, expit(mu + m[dd]), h)
            add("defense_team", grp, "opponents", rows, p_opp[rows], h)
            add("defense_team", grp, "opponents_raw", rows, expit(eta[rows] + raw[dd]), h)
            add("defense_team", grp, "fingerprint", rows, expit(eta[rows] + fp[dd]), h)
            t = trials.setdefault(("defense_team", grp), np.zeros(len(games)))
            np.add.at(t, gi[rows], 1.0)
            n_cells[("defense_team", grp)] = n_cells.get(("defense_team", grp), 0) + int(len(np.unique(dd)))
        # ---- unit cells ----
        uc = est[(est["scope"] != "defense_team") & (est["head"] == head) & (est["cls"] == cls)]
        for c in uc.itertuples():
            jt = int(np.searchsorted(teams, c.team_id))
            idx = units2.get((c.scope, jt, c.unit))
            if idx is None or np.isnan(fp[jt]):
                continue
            rows = idx[den[idx]]
            if len(rows) < defense.MIN_N:
                continue
            h = hit[rows]
            add(c.scope, grp, "team", rows, expit(eta[rows] + fp[jt]), h)
            add(c.scope, grp, "unit_raw", rows, expit(eta[rows] + layer.emp_logit(c.k, c.n)[0] - c.opp_logit), h)
            add(c.scope, grp, "unit", rows, expit(eta[rows] + c.fp_logit), h)
            t = trials.setdefault((c.scope, grp), np.zeros(len(games)))
            np.add.at(t, gi[rows], 1.0)
            n_cells[(c.scope, grp)] = n_cells.get((c.scope, grp), 0) + 1
            if c.shown:                                       # the same comparison on the rows a report would pick out
                add(c.scope, grp, "shown:team", rows, expit(eta[rows] + fp[jt]), h)
                add(c.scope, grp, "shown:unit", rows, expit(eta[rows] + c.fp_logit), h)
                t = trials.setdefault((c.scope, grp, "shown"), np.zeros(len(games)))
                np.add.at(t, gi[rows], 1.0)
                n_cells[(c.scope, grp, "shown")] = n_cells.get((c.scope, grp, "shown"), 0) + 1

    for scope in defense.SCOPES:
        methods = TEAM_METHODS if scope == "defense_team" else UNIT_METHODS
        for grp, glabel in GROUPS:
            if (scope, grp) not in trials:
                continue
            n = trials[(scope, grp)]
            ref = loss[(scope, grp, REFERENCE[scope])]
            rows = []
            for mth, label in methods.items():
                a = loss[(scope, grp, mth)]
                vs = None if mth == REFERENCE[scope] else _paired(a, ref, n, _rng(scope, grp, "games"))
                row = {"method": mth, "label": label, "log_loss": round(float(a.sum() / n.sum()), 5), "vs_reference": vs,
                       "verdict": _verdict(vs) if vs else None, "vs_unadjusted": None}
                if scope == "defense_team" and mth == "fingerprint":
                    row["vs_unadjusted"] = _paired(a, loss[(scope, grp, "pooled_allowed")], n, _rng(scope, grp, "games"))
                rows.append(row)
            picked = None
            if scope != "defense_team":
                picked = {"min_probability": defense.SHOW_PROB, "n_cells": 0, "n_trials": 0, "vs_reference": None, "verdict": None}
                if (scope, grp, "shown") in trials:
                    ns = trials[(scope, grp, "shown")]
                    vs = _paired(loss[(scope, grp, "shown:unit")], loss[(scope, grp, "shown:team")], ns, _rng(scope, grp, "shown games"))
                    picked.update(n_cells=n_cells[(scope, grp, "shown")], n_trials=int(ns.sum()), vs_reference=vs, verdict=_verdict(vs))
            out["slices"].append({"scope": scope, "scope_label": defense.SCOPES[scope], "outcomes": grp, "outcomes_label": glabel,
                                  "n_cells": n_cells[(scope, grp)], "n_trials": int(n.sum()), "reference": REFERENCE[scope], "shipped": SHIPPED[scope],
                                  "methods": rows, "picked": picked})

    # ---- do the intervals the report shows cover the second half's deviation? ----
    c2 = defense.candidates(second)
    if len(c2):
        covs = {}
        for name, dr in (("coverage", None), ("coverage_with_drift", drift)):
            if name == "coverage_with_drift" and not drift:
                continue
            m = defense.estimate(cand, dr).merge(c2, on="key", suffixes=("", "_2"))
            y2, v2 = layer.emp_logit(m["k_2"].values, m["n_2"].values)
            zs = (y2 - m["opp_logit_2"].values - m["fp_logit"].values) / np.sqrt(m["fp_sd"].values ** 2 + v2 + m["opp_var_2"].values)
            covs[name] = pd.Series(np.abs(zs) < float(layer.Z)).groupby(m["scope"].values).agg(["mean", "size"])
        for scope, label in defense.SCOPES.items():
            if scope in covs["coverage"].index:
                out["coverage"].append({"scope": scope, "label": label, "n_cells": int(covs["coverage"].loc[scope, "size"]),
                                        "coverage": round(float(covs["coverage"].loc[scope, "mean"]), 4),
                                        "coverage_with_drift": round(float(covs["coverage_with_drift"].loc[scope, "mean"]), 4) if "coverage_with_drift" in covs else None})
    return out


def report(poss: pd.DataFrame) -> dict:
    """The `defense` block of tendency_validation.json: the latest complete season, drift from the one before."""
    season, ref = pick_seasons(poss)
    drift = defense.estimate_drift(poss[poss["season"] == ref]) if ref is not None else None
    v = validate(poss[poss["season"] == season], drift)
    base = {"season": season, "interval_level": LEVEL, "split": v["split"], "n_games_first": v["n_games_first"], "n_games_second": v["n_games_second"],
            "bootstrap": f"{N_BOOT:,} resamples of whole second-half games; 95% interval of the log-loss difference",
            "drift": {"available": drift is not None, "season": ref, "n_games": drift["n_games"] if drift else 0,
                      "rate": [{"scope": s, "label": lab, "sd": round(float(np.sqrt(drift["rate"][s])), 4), "n_cells": drift["n_cells"][s]}
                               for s, lab in defense.SCOPES.items()] if drift else []},
            "slices": v["slices"], "coverage": v["coverage"]}
    if not v["slices"]:
        return {"available": False, "reason": "No defensive cell has enough trials in both halves of the season", **base}
    return {"available": True, "reason": None, **base}
