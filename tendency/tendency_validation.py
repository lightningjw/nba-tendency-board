"""Does the tendency layer earn its place? Fit on the first half of a season, predict the second half.

Every method gives, from first-half games only, a rate and a posterior for each cell (team or player,
situation, outcome). All methods are scored on the SAME cells: those every method has an estimate
for and that have at least MIN_N trials in the second half. Three numbers per method:

* log-loss of the predicted rate on second-half outcomes (lower is better), with a paired interval
  against the stand-in from resampling whole second-half games;
* coverage: how often the method's 90% predictive interval for the second-half count contains what
  happened. The predictive interval includes second-half binomial noise, so a calibrated method
  scores about 0.90;
* the mean width of that interval, as a rate. Narrower is better only at equal coverage.

Methods: raw first-half rates (Jeffreys posterior), the old stand-in (`tendencies.py`), the
hierarchical model (`tendency_layer.py`), and anything passed in `extra` (the PyMC fit).

Each is scored twice: as is, and with the drift term (`tendency_layer.estimate_drift`, measured on an
EARLIER season) added to its posterior on the logit scale. `control()` shows why the term is needed:
halves made by alternating games, with no time between them, cover close to 0.90 without it.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import tendencies as standin
from . import tendency_layer as layer

N_DRAWS = 4000
N_BOOT = 1000
SEED = 20261123
LEVEL = 0.90
MIN_N = layer.MIN_N
LABELS = {"raw": "Raw first-half rate", "stand_in": "Stand-in (simple shrinkage)", "hierarchical": "Hierarchical (partial pooling)",
          "pymc": "Hierarchical, full Bayes (PyMC)"}


@dataclass
class Pred:
    """One method's predictions: cell keys, the predicted rate, and posterior draws of the rate (cells x N_DRAWS)."""
    keys: pd.Index
    p: np.ndarray
    draws: np.ndarray
    rank: dict[str, list[str]] | None = None      # team -> its row keys, best first (methods that rank rows)

    def take(self, keys) -> "Pred":
        i = self.keys.get_indexer(keys)
        if (i < 0).any():
            raise KeyError("prediction missing for a requested cell")
        return Pred(pd.Index(keys), self.p[i], self.draws[i])

    def with_drift(self, var: np.ndarray, rng: np.random.Generator) -> "Pred":
        """The same posterior with drift variance `var` (one value per cell) added on the logit scale."""
        d = np.clip(self.draws, 1e-9, 1 - 1e-9)
        d = expit(logit(d) + rng.standard_normal(d.shape) * np.sqrt(var)[:, None])
        return Pred(self.keys, d.mean(axis=1), d)


def _rng(*names: str) -> np.random.Generator:
    """One random stream per (method, purpose), so adding a method never changes another method's numbers."""
    return np.random.default_rng([SEED, zlib.crc32("|".join(names).encode())])


def _scope(keys) -> np.ndarray:
    return np.array([k.split("|")[1] for k in keys])


def method_preds(first: pd.DataFrame, drift: dict | None = None) -> dict[str, Pred]:
    """Raw, stand-in and hierarchical predictions from first-half possessions. None of them includes drift here."""
    cand = layer.candidates(first)
    est = layer.estimate(cand)
    keys = pd.Index(est["key"])
    k, n = est["k"].values, est["n"].values
    out = {"raw": Pred(keys, (k + 0.5) / (n + 1.0), _rng("raw", "draws").beta(k + 0.5, n - k + 0.5, size=(N_DRAWS, len(est))).T)}

    rows, rank = [], {}
    for t, rep in standin.team_reports(first, {}, top_k=None).items():
        for r in rep["tendencies"]:
            m, b = r["prior_strength"], r["baseline_rate"]
            key = layer.cell_key(t, r["scope"], r["player_id"], r["situation"]["id"], r["outcome"]["head"], r["outcome"]["class"])
            rows.append((key, b * m + r["k"], (1 - b) * m + r["n"] - r["k"]))
            rank.setdefault(str(t), []).append(key)                          # the stand-in returns its rows best first
    s = pd.DataFrame(rows, columns=["key", "a", "b"])
    out["stand_in"] = Pred(pd.Index(s["key"]), (s["a"] / (s["a"] + s["b"])).values, _rng("stand_in", "draws").beta(s["a"].values, s["b"].values, size=(N_DRAWS, len(s))).T, rank)

    d = expit(est["rate_logit"].values[:, None] + _rng("hierarchical", "draws").standard_normal((len(est), N_DRAWS)) * est["rate_sd"].values[:, None])
    shipped = layer.estimate(cand, drift) if drift else est      # rows are ranked the way the report ranks them
    order = pd.DataFrame({"team": shipped["team_id"].astype(str).values, "key": shipped["key"].values,
                          "score": [layer._to_row(c, {})["rank_score"] for c in shipped.itertuples()]}).sort_values(["score", "key"], ascending=[False, True])
    out["hierarchical"] = Pred(keys, d.mean(axis=1), d, {t: g["key"].tolist() for t, g in order.groupby("team")})
    return out


def _interval(pred: Pred, n2: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sim = rng.binomial(n2[:, None].astype(np.int64), pred.draws)
    return tuple(np.percentile(sim, [(1 - LEVEL) / 2 * 100, (1 + LEVEL) / 2 * 100], axis=1))


def _score(pred: Pred, k2: np.ndarray, n2: np.ndarray, rng: np.random.Generator) -> dict:
    p = np.clip(pred.p, 1e-6, 1 - 1e-6)
    lo, hi = _interval(pred, n2, rng)
    return {"log_loss": round(float(-(k2 * np.log(p) + (n2 - k2) * np.log(1 - p)).sum() / n2.sum()), 5),
            "coverage": round(float(((k2 >= lo) & (k2 <= hi)).mean()), 4),
            "interval_width": round(float(((hi - lo) / n2).mean()), 4),
            "mean_abs_error": round(float(np.abs(p - k2 / n2).mean()), 4)}


def _game_matrices(counts: pd.DataFrame, keys: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """(games x cells) arrays of k and n for the given cells."""
    c = counts[counts["key"].isin(keys)]
    games = pd.Index(sorted(counts["game_id"].unique()))
    K, N = np.zeros((len(games), len(keys))), np.zeros((len(games), len(keys)))
    gi, ci = games.get_indexer(c["game_id"]), keys.get_indexer(c["key"])
    np.add.at(K, (gi, ci), c["k"].values)
    np.add.at(N, (gi, ci), c["n"].values)
    return K, N


def _paired(p: np.ndarray, p_ref: np.ndarray, K: np.ndarray, N: np.ndarray, rng: np.random.Generator) -> dict:
    """Log-loss of p minus log-loss of p_ref, with a 95% interval from resampling whole games."""
    def per_game(q):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return -(K @ np.log(q) + (N - K) @ np.log(1 - q))
    a, b, n = per_game(p), per_game(p_ref), N.sum(axis=1)
    idx = rng.integers(len(n), size=(N_BOOT, len(n)))
    diffs = (a[idx].sum(axis=1) - b[idx].sum(axis=1)) / n[idx].sum(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": round(float((a.sum() - b.sum()) / n.sum()), 6) + 0.0, "interval": [round(float(lo), 6) + 0.0, round(float(hi), 6) + 0.0]}


def common_cells(preds: dict[str, Pred], tot: pd.DataFrame) -> dict[str, pd.Index]:
    """The cells every method is scored on, by scope: an estimate from EVERY method, and at least MIN_N second-half trials."""
    common = tot.index[tot["n"] >= MIN_N]
    for p in preds.values():
        common = common.intersection(p.keys)
    common = common.sort_values()
    return {scope: common[_scope(common) == scope] for scope in ("team", "player")}


def validate(poss: pd.DataFrame, drift: dict | None = None, extra: dict[str, Pred] | None = None, interleave: bool = False) -> dict:
    """Half-season validation on one season's possessions.

    `drift` is `tendency_layer.estimate_drift` of an earlier season; without it only the "as is" columns are filled.
    `extra` adds methods (the PyMC fit) as `Pred` objects. `interleave` alternates games instead of splitting by date.
    """
    first, second = layer.split_halves(poss, interleave)
    preds = method_preds(first, drift)
    preds.update(extra or {})
    counts = layer.cell_counts(second)
    tot = counts.groupby("key")[["k", "n"]].sum()
    cells = common_cells(preds, tot)
    out = {"interval_level": LEVEL, "n_games_first": int(first["game_id"].nunique()), "n_games_second": int(second["game_id"].nunique()),
           "split": "alternating games" if interleave else "by date", "min_n": MIN_N, "slices": [], "selection": []}

    def drift_var(keys) -> np.ndarray:
        return np.array([drift["rate"][s] for s in _scope(keys)])

    for scope, keys in cells.items():
        if not len(keys):
            continue
        k2, n2 = tot.loc[keys, "k"].values.astype(float), tot.loc[keys, "n"].values.astype(float)
        K, N = _game_matrices(counts, keys)
        ref = preds["stand_in"].take(keys)
        methods = []
        for name, pr in preds.items():
            pr = pr.take(keys)
            boot = _rng(scope, "games")                                       # the same resampled games for every method
            row = {"method": name, "label": LABELS.get(name, name), **_score(pr, k2, n2, _rng(name, scope, "sim")),
                   "vs_stand_in": None if name == "stand_in" else _paired(pr.p, ref.p, K, N, boot), "with_drift": None, "verdict": None}
            if drift:
                prd = pr.with_drift(drift_var(keys), _rng(name, scope, "drift"))
                row["with_drift"] = {**_score(prd, k2, n2, _rng(name, scope, "drift sim")), "vs_stand_in": _paired(prd.p, ref.p, K, N, _rng(scope, "games"))}
            if name != "stand_in":                                            # judged in the form that would ship: with the drift term when there is one
                lo, hi = (row["with_drift"] or row)["vs_stand_in"]["interval"]
                row["verdict"] = "ahead" if hi < 0 else "behind" if lo > 0 else "level"
            methods.append(row)
        out["slices"].append({"scope": scope, "n_cells": int(len(keys)), "n_trials": int(n2.sum()), "methods": methods})

    # selection: the rows each ranking method would have SHOWN (its top ten per team). Do their intervals still cover?
    for name, pr in preds.items():
        if not pr.rank:
            continue
        keys = pd.Index([k for rows in pr.rank.values() for k in rows[: layer.TOP_K]])
        keys = keys[keys.isin(tot.index[tot["n"] >= MIN_N])]
        if not len(keys):
            continue
        k2, n2 = tot.loc[keys, "k"].values.astype(float), tot.loc[keys, "n"].values.astype(float)
        sel = pr.take(keys)
        lo, hi = _interval(sel, n2, _rng(name, "selection"))
        row = {"method": name, "label": LABELS.get(name, name), "n_rows": int(len(keys)),
               "coverage": round(float(((k2 >= lo) & (k2 <= hi)).mean()), 4), "coverage_with_drift": None}
        if drift:
            lo, hi = _interval(sel.with_drift(drift_var(keys), _rng(name, "selection drift")), n2, _rng(name, "selection drift sim"))
            row["coverage_with_drift"] = round(float(((k2 >= lo) & (k2 <= hi)).mean()), 4)
        out["selection"].append(row)
    out["deviation"] = _deviation_coverage(first, second, drift)
    return out


def _deviation_coverage(first: pd.DataFrame, second: pd.DataFrame, drift: dict | None) -> list[dict]:
    """The interval the report SHOWS is on the deviation from baseline. Does it cover the second half's deviation?

    The second-half deviation is observed with noise (its own count's and its own baseline's), which is
    added to the interval before checking. Hierarchical model only: the other methods define the baseline differently.
    """
    c2 = layer.candidates(second)
    z = float(layer.Z)
    out = []
    covs = {}
    for name, dr in (("coverage", None), ("coverage_with_drift", drift)):
        if name == "coverage_with_drift" and not drift:
            continue
        m = layer.estimate(layer.candidates(first), dr).merge(c2, on="key", suffixes=("", "_2"))
        y2, v2 = layer.emp_logit(m["k_2"].values, m["n_2"].values)
        zs = (y2 - m["base_logit_2"].values - m["dev_logit"].values) / np.sqrt(m["dev_sd"].values ** 2 + v2 + m["base_var_2"].values)
        covs[name] = pd.Series(np.abs(zs) < z).groupby(m["kind"].values).agg(["mean", "size"])
    for kind, label in layer.KINDS.items():
        if kind in covs["coverage"].index:
            out.append({"kind": kind, "label": label, "n_cells": int(covs["coverage"].loc[kind, "size"]),
                        "coverage": round(float(covs["coverage"].loc[kind, "mean"]), 4),
                        "coverage_with_drift": round(float(covs["coverage_with_drift"].loc[kind, "mean"]), 4) if "coverage_with_drift" in covs else None})
    return out


def control(poss: pd.DataFrame) -> dict:
    """Raw-rate coverage when the halves alternate games. No time passes between them, so no drift: the interval arithmetic alone."""
    v = validate(poss, interleave=True)
    return {s["scope"]: {"coverage": next(m["coverage"] for m in s["methods"] if m["method"] == "raw"), "n_cells": s["n_cells"]} for s in v["slices"]}


def pick_seasons(poss: pd.DataFrame) -> tuple[int, int | None]:
    """(season to validate on, season to measure drift on). The latest COMPLETE season, and the one before it.

    A season counts as complete with at least 80% as many games as the fullest season on disk, so a
    season in progress is never split in half.
    """
    n = poss.groupby("season")["game_id"].nunique()
    full = n[n >= 0.8 * n.max()].index
    season = int(full.max())
    earlier = [int(x) for x in n.index if x < season]
    return season, (max(earlier) if earlier else None)


def report(poss: pd.DataFrame, extra: dict[str, Pred] | None = None) -> dict:
    """The contract's tendency_validation.json body (without data_kind)."""
    season, ref = pick_seasons(poss)
    one = poss[poss["season"] == season]
    drift = layer.estimate_drift(poss[poss["season"] == ref]) if ref is not None else None
    v = validate(one, drift, extra)
    base = {"season": season, "shipped": "hierarchical", "interval_level": LEVEL, "min_n": MIN_N, "split": v["split"],
            "n_games_first": v["n_games_first"], "n_games_second": v["n_games_second"],
            "bootstrap": f"{N_BOOT:,} resamples of whole second-half games; 95% interval of the log-loss difference",
            "drift": {"available": drift is not None, "season": ref, "n_games": drift["n_games"] if drift else 0,
                      "rate": [{"scope": s, "sd": round(float(np.sqrt(drift["rate"][s])), 4), "n_cells": drift["n_cells"][s]} for s in ("team", "player")] if drift else [],
                      "deviation": [{"kind": k, "label": lab, "sd": round(float(np.sqrt(drift["dev"][k])), 4), "n_cells": drift["n_cells"][k]}
                                    for k, lab in layer.KINDS.items()] if drift else []},
            "control": [], "slices": v["slices"], "selection": v["selection"], "deviation": v["deviation"]}
    if not v["slices"]:
        return {"available": False, "reason": f"No cell has {MIN_N} trials in both halves of the season", **base}
    base["control"] = [{"scope": s, **c} for s, c in control(one).items()]
    return {"available": True, "reason": None, **base}
