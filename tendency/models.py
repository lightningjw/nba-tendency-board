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


def _bin_sums(P: np.ndarray, y: np.ndarray, codes: np.ndarray, n_games: int, n_bins: int = N_BINS) -> np.ndarray:
    """Per game and probability band: [count, sum of predicted, sum of hits]. Shape (n_games, n_bins, 3).

    Pooled one-vs-rest: every (probability, did-it-happen) pair across classes counts once.
    """
    k = P.shape[1]
    p = P.ravel()
    hit = (np.arange(k)[None, :] == y[:, None]).ravel().astype(float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    flat = np.repeat(codes, k) * n_bins + idx
    size = n_games * n_bins
    return np.stack([np.bincount(flat, minlength=size), np.bincount(flat, weights=p, minlength=size),
                     np.bincount(flat, weights=hit, minlength=size)], axis=1).reshape(n_games, n_bins, 3)


def game_sums(P_model: np.ndarray, P_base: np.ndarray, y: np.ndarray, game_ids: np.ndarray) -> list[dict]:
    """Everything the scorecard needs from one head, reduced to one small record per game.

    These records are what the prospective log accumulates (no possession rows), and what
    `evaluate_sums` turns into log-loss, a game-bootstrap interval, top-pick accuracy and reliability.
    """
    if len(y) == 0:
        return []
    lm, lb = row_log_loss(P_model, y), row_log_loss(P_base, y)
    codes, uniq = pd.factorize(game_ids)
    n_games = len(uniq)
    cnt = np.bincount(codes, minlength=n_games)
    sm, sb = np.bincount(codes, weights=lm, minlength=n_games), np.bincount(codes, weights=lb, minlength=n_games)
    tm = np.bincount(codes, weights=(P_model.argmax(1) == y).astype(float), minlength=n_games)
    tb = np.bincount(codes, weights=(P_base.argmax(1) == y).astype(float), minlength=n_games)
    bm, bb = _bin_sums(P_model, y, codes, n_games), _bin_sums(P_base, y, codes, n_games)
    return [{"game_id": str(uniq[g]), "n": int(cnt[g]), "ll_model": float(sm[g]), "ll_base": float(sb[g]),
             "top1_model": int(tm[g]), "top1_base": int(tb[g]),
             "bins_model": [[int(b[0]), float(b[1]), float(b[2])] for b in bm[g]],
             "bins_base": [[int(b[0]), float(b[1]), float(b[2])] for b in bb[g]]} for g in range(n_games)]


def _reliability_from_bins(bins: np.ndarray) -> dict:
    """bins: (n_bins, 3) of [count, sum predicted, sum hits], already summed over games."""
    n_bins = len(bins)
    edges = np.linspace(0, 1, n_bins + 1)
    total = bins[:, 0].sum()
    out, ece = [], 0.0
    for b in range(n_bins):
        n = int(bins[b, 0])
        mp, ob = (float(bins[b, 1] / n), float(bins[b, 2] / n)) if n else (None, None)
        if n:
            ece += n / total * abs(mp - ob)
        out.append({"lo": round(float(edges[b]), 2), "hi": round(float(edges[b + 1]), 2), "n": n,
                    "mean_pred": None if mp is None else round(mp, 4), "obs_rate": None if ob is None else round(ob, 4)})
    return {"ece": round(float(ece), 4), "bins": out}


def reliability(P: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Pooled one-vs-rest reliability: every (probability, did-it-happen) pair across classes."""
    return _reliability_from_bins(_bin_sums(P, y, np.zeros(len(y), dtype=np.int64), 1, n_bins)[0])


def evaluate_sums(records: list[dict], split: str, empty_reason: str | None = None) -> dict:
    """One scorecard entry from per-game sums (see `game_sums`). The bootstrap resamples whole games."""
    empty = {"split": split, "available": False, "reason": None, "n_possessions": 0, "n_games": 0, "model": None,
             "baseline": None, "delta": None, "verdict": None, "top1": None, "reliability": None}
    records = [r for r in records if r["n"] > 0]
    if not records:
        return {**empty, "reason": empty_reason or "no held-out possessions in this split (the data covers a single season or too few games)"}
    cnt = np.array([r["n"] for r in records], dtype=float)
    sm, sb = np.array([r["ll_model"] for r in records]), np.array([r["ll_base"] for r in records])
    n_games, n_total = len(records), int(cnt.sum())
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_games, size=(N_BOOT, n_games))
    n = cnt[draws].sum(axis=1)
    bm, bb = sm[draws].sum(axis=1) / n, sb[draws].sum(axis=1) / n
    lo, hi = (1 - INTERVAL_LEVEL) / 2 * 100, (1 + INTERVAL_LEVEL) / 2 * 100

    def metric(point: float, boots: np.ndarray) -> dict:
        return {"log_loss": round(float(point), 5), "interval": [round(float(np.percentile(boots, lo)), 5), round(float(np.percentile(boots, hi)), 5)]}

    mean_m, mean_b = sm.sum() / n_total, sb.sum() / n_total
    delta = metric(mean_m - mean_b, bm - bb)
    verdict = "model_better" if delta["interval"][1] < 0 else ("baseline_better" if delta["interval"][0] > 0 else "inconclusive")
    top_m, top_b = sum(r["top1_model"] for r in records), sum(r["top1_base"] for r in records)
    return {"split": split, "available": True, "reason": None, "n_possessions": n_total, "n_games": n_games,
            "model": metric(mean_m, bm), "baseline": metric(mean_b, bb), "delta": delta, "verdict": verdict,
            "top1": {"model": round(top_m / n_total, 4), "baseline": round(top_b / n_total, 4), "n": n_total},
            "reliability": {"model": _reliability_from_bins(np.sum([r["bins_model"] for r in records], axis=0)),
                            "baseline": _reliability_from_bins(np.sum([r["bins_base"] for r in records], axis=0))}}


def evaluate(P_model: np.ndarray, P_base: np.ndarray, y: np.ndarray, game_ids: np.ndarray, split: str) -> dict:
    return evaluate_sums(game_sums(P_model, P_base, y, game_ids), split)


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


# ---- frozen models for the prospective log (train once, save, predict later) ---------------
def frozen_parts(games: pd.DataFrame) -> pd.Series:
    """game_id -> train | valid for a model that is frozen and scored only on games played later.

    No held-out split: the hold-out is the future. The latest 15% of games only pick the tree count.
    """
    g = games.drop_duplicates("game_id").sort_values(["tipoff", "game_id"])
    part = pd.Series("train", index=g["game_id"].values, dtype=object)
    n_valid = int(round(len(g) * VALID_GAME_FRACTION))
    if n_valid:
        part[g["game_id"].values[-n_valid:]] = "valid"
    if (part == "train").sum() < MIN_TRAIN_GAMES:
        raise NotEnoughData(f"only {(part == 'train').sum()} training games before the cut-off; need at least {MIN_TRAIN_GAMES}.")
    return part


def _head_inputs(feat: Features, head: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """(X, baseline probabilities, label index, has-label mask) for one head. Finisher is per candidate row."""
    poss, cand = feat.poss, feat.cand
    if head == "finisher":
        y = cand["is_fin"].values.reshape(-1, 5).argmax(axis=1)
        return cand[FEATURES["finisher"]], cand["p_base"].values, y, poss["finisher_id"].notna().values
    classes, prefix = (ENDING_CLASSES, "base_end_") if head == "ending_type" else (SHOT_FAMILIES, "base_fam_")
    base = poss[[prefix + c for c in classes]].values
    base = base / base.sum(axis=1, keepdims=True)
    label = poss[head].map({c: i for i, c in enumerate(classes)})
    return poss[FEATURES[head]].astype(np.float32), base, label.fillna(0).astype(int).values, label.notna().values


def fit_frozen(feat: Features, log=print) -> dict[str, tuple[lgb.Booster, int]]:
    """Fit the three heads on every game in `feat` (same parameters and baseline offsets as `run_models`)."""
    poss, cand = feat.poss, feat.cand
    if len(cand) != 5 * len(poss):
        raise RuntimeError("candidate rows are not aligned with possession rows")
    part = poss["game_id"].map(frozen_parts(poss[["game_id", "season", "tipoff"]])).values
    out = {}
    for head in ("ending_type", "shot_family"):
        X, base, y, has = _head_inputs(feat, head)
        out[head] = fit_multiclass(X, y, base, np.where(has, part, "unused"), base.shape[1])
        log(f"train: {head}: {out[head][1]} rounds, {int((has & (part == 'train')).sum()):,} training possessions")
    X, base, _, has = _head_inputs(feat, "finisher")
    out["finisher"] = fit_binary(X, cand["is_fin"].values, base, np.where(cand["has_fin"].values, np.repeat(part, 5), "unused"))
    log(f"train: finisher: {out['finisher'][1]} rounds, {int((has & (part == 'train')).sum()):,} training possessions")
    return out


def predict_heads(models: dict[str, tuple[lgb.Booster, int]], feat: Features) -> dict[str, dict]:
    """Model and baseline probabilities for every row of `feat`, per head, plus labels where they exist."""
    out = {}
    for head in HEADS:
        X, base, y, has = _head_inputs(feat, head)
        bst, rounds = models[head]
        if head == "finisher":
            P, B = predict_finisher(bst, rounds, X, base), base.reshape(-1, 5)
        else:
            P, B = predict_multiclass(bst, rounds, X, base), base
        out[head] = {"model": P, "baseline": B, "y": y, "has": has}
    return out
