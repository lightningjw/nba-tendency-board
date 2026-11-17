"""Baselines and LightGBM models, one per head, with honest evaluation.

* Split by game and by season, never by possession. The last season is the
  held-out season; within earlier seasons the latest 20% of games are held out;
  the latest 15% of the remaining games are used only to pick the tree count.
* Each LightGBM model starts from its baseline's log-odds (init_score), so it
  learns corrections to the baseline rather than relearning it.
* Evaluation: log-loss with a game-clustered bootstrap interval for model,
  baseline and their difference, plus a reliability table for each.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURES, Features
from .possessions import ENDING_CLASSES, SHOT_FAMILIES

SEED = 7
HOLDOUT_GAME_FRACTION = 0.20
VALID_GAME_FRACTION = 0.15
MAX_ROUNDS = 400
EARLY_STOP = 30
INTERVAL_LEVEL = 0.95
N_BOOT = 1000
N_BINS = 10
MIN_TRAIN_GAMES = 8
EPS = 1e-9
LGB_PARAMS = dict(learning_rate=0.04, num_leaves=15, min_data_in_leaf=80, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=2, deterministic=True, force_col_wise=True)
HEADS = ["finisher", "ending_type", "shot_family"]


class NotEnoughData(RuntimeError):
    pass


def assign_splits(games: pd.DataFrame) -> pd.Series:
    """game_id -> train | valid | holdout_games | holdout_season. `games` needs game_id, season, tipoff."""
    g = games.drop_duplicates("game_id").sort_values(["tipoff", "game_id"])
    seasons = sorted(g["season"].unique())
    split = pd.Series("train", index=g["game_id"].values, dtype=object)
    if len(seasons) >= 2:
        split[g.loc[g["season"] == seasons[-1], "game_id"].values] = "holdout_season"
    rest = g[g["season"].isin(seasons[:-1] if len(seasons) >= 2 else seasons)]
    n_hold = int(round(len(rest) * HOLDOUT_GAME_FRACTION))
    if n_hold:
        split[rest["game_id"].values[-n_hold:]] = "holdout_games"
    train = rest["game_id"].values[: len(rest) - n_hold]
    n_valid = int(round(len(train) * VALID_GAME_FRACTION))
    if n_valid:
        split[train[-n_valid:]] = "valid"
    if (split == "train").sum() < MIN_TRAIN_GAMES:
        raise NotEnoughData(f"only {(split == 'train').sum()} training games; need at least {MIN_TRAIN_GAMES}. Ingest a longer date range.")
    return split


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_multiclass(X: pd.DataFrame, y: np.ndarray, base: np.ndarray, part: np.ndarray, k: int) -> tuple[lgb.Booster, int]:
    """Multiclass LightGBM on top of the baseline's log-probabilities."""
    init = np.log(np.clip(base, EPS, 1.0))
    tr, va = part == "train", part == "valid"
    dtr = lgb.Dataset(X[tr], y[tr], init_score=init[tr].ravel(order="F"), free_raw_data=False)
    params = {**LGB_PARAMS, "objective": "multiclass", "num_class": k}
    if va.sum() >= 200:
        dva = lgb.Dataset(X[va], y[va], init_score=init[va].ravel(order="F"), reference=dtr)
        bst = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        rounds = max(1, bst.best_iteration or MAX_ROUNDS)
    else:
        rounds = 60
        bst = lgb.train(params, dtr, rounds)
    return bst, rounds


def predict_multiclass(bst: lgb.Booster, rounds: int, X: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    raw = bst.predict(X, num_iteration=rounds, raw_score=True)
    return _softmax(np.asarray(raw).reshape(len(X), -1) + np.log(np.clip(base, EPS, 1.0)))


def fit_binary(X: pd.DataFrame, y: np.ndarray, base: np.ndarray, part: np.ndarray) -> tuple[lgb.Booster, int]:
    b = np.clip(base, 1e-4, 1 - 1e-4)
    init = np.log(b / (1 - b))
    tr, va = part == "train", part == "valid"
    dtr = lgb.Dataset(X[tr], y[tr], init_score=init[tr], free_raw_data=False)
    params = {**LGB_PARAMS, "objective": "binary"}
    if va.sum() >= 1000:
        dva = lgb.Dataset(X[va], y[va], init_score=init[va], reference=dtr)
        bst = lgb.train(params, dtr, MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        rounds = max(1, bst.best_iteration or MAX_ROUNDS)
    else:
        rounds = 60
        bst = lgb.train(params, dtr, rounds)
    return bst, rounds


def predict_finisher(bst: lgb.Booster, rounds: int, X: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    """Per-candidate scores renormalised across the five players. Returns shape (n_possessions, 5)."""
    b = np.clip(base, 1e-4, 1 - 1e-4)
    raw = bst.predict(X, num_iteration=rounds, raw_score=True) + np.log(b / (1 - b))
    p = (1.0 / (1.0 + np.exp(-raw))).reshape(-1, 5)
    return p / p.sum(axis=1, keepdims=True)


# ---- evaluation -------------------------------------------------------------------
def row_log_loss(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))


def reliability(P: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Pooled one-vs-rest reliability: every (probability, did-it-happen) pair across classes."""
    p = P.ravel()
    hit = (np.arange(P.shape[1])[None, :] == y[:, None]).ravel().astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        mp, ob = (float(p[m].mean()), float(hit[m].mean())) if n else (None, None)
        if n:
            ece += n / len(p) * abs(mp - ob)
        bins.append({"lo": round(float(edges[b]), 2), "hi": round(float(edges[b + 1]), 2), "n": n,
                     "mean_pred": None if mp is None else round(mp, 4), "obs_rate": None if ob is None else round(ob, 4)})
    return {"ece": round(float(ece), 4), "bins": bins}


def evaluate(P_model: np.ndarray, P_base: np.ndarray, y: np.ndarray, game_ids: np.ndarray, split: str) -> dict:
    empty = {"split": split, "available": False, "reason": None, "n_possessions": 0, "n_games": 0, "model": None,
             "baseline": None, "delta": None, "verdict": None, "top1": None, "reliability": None}
    if len(y) == 0:
        return {**empty, "reason": "no held-out possessions in this split (the data covers a single season or too few games)"}
    lm, lb = row_log_loss(P_model, y), row_log_loss(P_base, y)
    codes, _ = pd.factorize(game_ids)
    n_games = int(codes.max()) + 1
    sm, sb, cnt = (np.bincount(codes, weights=v, minlength=n_games) for v in (lm, lb, np.ones(len(y))))
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_games, size=(N_BOOT, n_games))
    n = cnt[draws].sum(axis=1)
    bm, bb = sm[draws].sum(axis=1) / n, sb[draws].sum(axis=1) / n
    lo, hi = (1 - INTERVAL_LEVEL) / 2 * 100, (1 + INTERVAL_LEVEL) / 2 * 100

    def metric(point: float, boots: np.ndarray) -> dict:
        return {"log_loss": round(float(point), 5), "interval": [round(float(np.percentile(boots, lo)), 5), round(float(np.percentile(boots, hi)), 5)]}

    delta = metric(lm.mean() - lb.mean(), bm - bb)
    verdict = "model_better" if delta["interval"][1] < 0 else ("baseline_better" if delta["interval"][0] > 0 else "inconclusive")
    return {"split": split, "available": True, "reason": None, "n_possessions": int(len(y)), "n_games": n_games,
            "model": metric(lm.mean(), bm), "baseline": metric(lb.mean(), bb), "delta": delta, "verdict": verdict,
            "top1": {"model": round(float((P_model.argmax(1) == y).mean()), 4), "baseline": round(float((P_base.argmax(1) == y).mean()), 4), "n": int(len(y))},
            "reliability": {"model": reliability(P_model, y), "baseline": reliability(P_base, y)}}


@dataclass
class ModelRun:
    preds: pd.DataFrame          # per eligible held-out possession: probability lists for each head
    scorecard_heads: list[dict]
    splits: pd.Series
    rounds: dict


def run_models(feat: Features, log=print) -> ModelRun:
    poss, cand = feat.poss, feat.cand
    if len(cand) != 5 * len(poss) or not (cand["game_id"].astype(object).values[::5] == poss["game_id"].values).all() \
            or not (cand["poss_idx"].values[::5] == poss["poss_idx"].values).all():
        raise RuntimeError("candidate rows are not aligned with possession rows")
    splits = assign_splits(poss[["game_id", "season", "tipoff"]])
    part = poss["game_id"].map(splits).values
    heads_out, rounds, cols = [], {}, {}

    # ---- ending type and shot family: multiclass ------------------------------------
    for head, classes, base_prefix in (("ending_type", ENDING_CLASSES, "base_end_"), ("shot_family", SHOT_FAMILIES, "base_fam_")):
        X = poss[FEATURES[head]].astype(np.float32)
        base = poss[[base_prefix + c for c in classes]].values
        base = base / base.sum(axis=1, keepdims=True)
        label = poss[head].map({c: i for i, c in enumerate(classes)})
        has = label.notna().values                     # shot family exists only when the possession ended in a shot
        y = label.fillna(0).astype(int).values
        fit_part = np.where(has, part, "unused")
        bst, r = fit_multiclass(X, y, base, fit_part, len(classes))
        P = predict_multiclass(bst, r, X, base)
        rounds[head] = r
        cols[head] = (P, base)
        heads_out.append({"head": head, "splits": [
            evaluate(P[m], base[m], y[m], poss["game_id"].values[m], s)
            for s in ("holdout_games", "holdout_season") for m in [has & (part == s)]]})
        log(f"models: {head}: {r} rounds, {int((fit_part == 'train').sum())} training possessions")

    # ---- finisher: per-candidate binary, renormalised over the five -----------------
    cpart = np.repeat(part, 5)
    has_c = cand["has_fin"].values                       # team turnovers have no finisher
    Xc = cand[FEATURES["finisher"]]
    bst, r = fit_binary(Xc, cand["is_fin"].values, cand["p_base"].values, np.where(has_c, cpart, "unused"))
    P = predict_finisher(bst, r, Xc, cand["p_base"].values)
    base = cand["p_base"].values.reshape(-1, 5)
    rounds["finisher"] = r
    yslot = cand["is_fin"].values.reshape(-1, 5).argmax(axis=1)
    has = poss["finisher_id"].notna().values
    heads_out.insert(0, {"head": "finisher", "splits": [
        evaluate(P[m], base[m], yslot[m], poss["game_id"].values[m], s)
        for s in ("holdout_games", "holdout_season") for m in [has & (part == s)]]})
    log(f"models: finisher: {r} rounds, {int((has & (part == 'train')).sum())} training possessions")
    cols["finisher"] = (P, base)

    held = np.isin(part, ["holdout_games", "holdout_season"])
    preds = poss.loc[held, ["game_id", "poss_idx"]].copy()
    for head in HEADS:
        Pm, Pb = cols[head]
        preds[f"{head}_model"] = [np.round(v, 4).tolist() for v in Pm[held]]
        preds[f"{head}_baseline"] = [np.round(v, 4).tolist() for v in Pb[held]]
    return ModelRun(preds.reset_index(drop=True), heads_out, splits, rounds)
