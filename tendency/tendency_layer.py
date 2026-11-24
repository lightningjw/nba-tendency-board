"""The tendency layer: partial pooling across teams, situations and players (empirical Bayes, hierarchical logit).

A tendency is a deviation from a baseline in a defined situation, with an interval and a sample size.
Everything is estimated on the logit scale, where effects add:

* **Team, all possessions.** logit(rate) = league + team effect. Team effects for one outcome share
  a normal prior whose spread is estimated from the 30 teams, so a team with few possessions is
  pulled toward the league and the league's real spread decides how hard.
* **Team, in a situation.** The deviation is the team-by-situation interaction: how the team's rate
  in the situation differs from its rate in its OTHER possessions, beyond the same contrast for the
  league. Using the other possessions (not all of them) keeps the two sides independent, so the
  baseline's own uncertainty enters the interval exactly. Interactions for one (situation, outcome)
  share a prior across teams.
* **Player, in a situation.** The baseline is what his usage and the four teammates on the floor
  predict: strengths w are fitted so that P(i finishes | five on the floor) = w_i / sum of the five
  (a choose-one Bradley-Terry model, fitted on his team's possessions OUTSIDE the situation), then
  averaged over the lineups he actually played with in the situation. Deviations for one situation
  share a prior across all players in the league. Nothing here is a fixed constant.

Rows are ranked by the posterior probability that the deviation is larger than a practical threshold
(one percentage point) in its own direction. That builds selection in: a row gets to the top by being
probably real and big enough to matter, not by having one end of its interval far from zero.

`validate()` is what decides whether this ships: fit on the first half of a season, predict the
second half, and compare with raw rates and with the old stand-in (`tendencies.py`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm

from .possessions import ENDING_CLASSES, SHOT_FAMILIES
from .tendencies import END_LABEL, FAM_LABEL, SITUATIONS, situation_masks

METHOD = {"id": "eb_hierarchical_logit", "label": "Partial pooling across teams, situations and players (empirical Bayes, hierarchical logit)",
          "final": True}
INTERVAL_LEVEL = 0.90
THRESHOLD = 0.01                 # a deviation smaller than one percentage point is not worth a scout's time
MIN_N = 25
MIN_N_PLAYER_TOTAL = 100
TOP_K = 10
USAGE_PRIOR_POSS = 20.0          # each player starts as an average finisher over this many possessions
TAU2_BOUNDS = (1e-6, 4.0)
Z = float(norm.ppf((1 + INTERVAL_LEVEL) / 2))


# ---- building blocks ----------------------------------------------------------------------
def emp_logit(k, n) -> tuple[np.ndarray, np.ndarray]:
    """Empirical logit and its sampling variance (the +0.5 keeps 0 and n finite)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    return np.log((k + 0.5) / (n - k + 0.5)), 1.0 / (k + 0.5) + 1.0 / (n - k + 0.5)


def eb_normal(y: np.ndarray, v: np.ndarray, center: bool) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Normal-normal empirical Bayes: y_i ~ N(theta_i, v_i), theta_i ~ N(mu, tau2).

    tau2 by marginal maximum likelihood; mu is the precision-weighted mean when `center`, else 0.
    Returns (mu, tau2, posterior mean of theta_i - mu, posterior variance).
    """
    y, v = np.asarray(y, float), np.asarray(v, float)
    if len(y) < 3:
        return 0.0, TAU2_BOUNDS[0], np.zeros_like(y), np.full_like(y, TAU2_BOUNDS[0])

    def mu_at(t2: float) -> float:
        w = 1.0 / (t2 + v)
        return float((w * y).sum() / w.sum()) if center else 0.0

    def nll(log_t2: float) -> float:
        t2 = np.exp(log_t2)
        return float(np.sum(np.log(t2 + v) + (y - mu_at(t2)) ** 2 / (t2 + v)))

    res = minimize_scalar(nll, bounds=(np.log(TAU2_BOUNDS[0]), np.log(TAU2_BOUNDS[1])), method="bounded")
    tau2 = float(np.exp(res.x))
    mu = mu_at(tau2)
    shrink = tau2 / (tau2 + v)
    return mu, tau2, shrink * (y - mu), shrink * v


def fit_usage(lineups: np.ndarray, finisher_slot: np.ndarray, n_players: int, iters: int = 60) -> np.ndarray:
    """Strengths w with P(player in slot j finishes | lineup) = w_j / sum(w over the five). MM algorithm.

    `lineups` is (n_possessions, 5) of player indices; `finisher_slot` the slot (0-4) that finished.
    Every player also gets USAGE_PRIOR_POSS possessions as an average finisher, which keeps a
    ten-possession player from looking like a star or a ghost.
    """
    k = np.bincount(lineups[np.arange(len(lineups)), finisher_slot], minlength=n_players).astype(float)
    exposure = np.bincount(lineups.ravel(), minlength=n_players).astype(float)
    w = np.full(n_players, 0.2)
    for _ in range(iters):
        inv = 1.0 / w[lineups].sum(axis=1)
        denom = np.bincount(lineups.ravel(), weights=np.repeat(inv, 5), minlength=n_players)
        w_new = (k + 0.2 * USAGE_PRIOR_POSS) / (denom + USAGE_PRIOR_POSS)      # the prior possessions sit in an average lineup (sum = 1)
        w_new *= 0.2 * exposure.sum() / max((w_new * exposure).sum(), 1e-12)   # scale is free; pin the exposure-weighted mean at 0.2
        done = np.max(np.abs(w_new - w)) < 1e-7
        w = w_new
        if done:
            break
    return w


# ---- candidates: one record per (scope, team/player, situation, outcome) ---------------------
def _candidates(poss: pd.DataFrame) -> pd.DataFrame:
    """Every candidate row with what the estimator needs: counts in the situation, the baseline logit and its variance."""
    df = poss[poss["ending_type"].isin(ENDING_CLASSES) & ~poss["is_heave"]].reset_index(drop=True)
    masks = situation_masks(df)
    team = df["off_team_id"].values
    teams = np.array(sorted(np.unique(team)))
    t_idx = np.searchsorted(teams, team)
    is_shot = df["ending_type"].values == "shot"
    specs = [("ending_type", c, df["ending_type"].values == c, np.ones(len(df), bool), END_LABEL[c]) for c in ENDING_CLASSES]
    specs += [("shot_family", c, df["shot_family"].values == c, is_shot, FAM_LABEL[c]) for c in SHOT_FAMILIES]

    def by_team(m: np.ndarray) -> np.ndarray:
        return np.bincount(t_idx[m], minlength=len(teams)).astype(float)

    rows = []
    for sid, slabel in SITUATIONS:
        sm = masks[sid]
        for head, cls, hit, denom, label in specs:
            d_in, d_out = sm & denom, ~sm & denom
            n_in, k_in = by_team(d_in), by_team(d_in & hit)
            if sid == "all":
                N, K = n_in.sum(), k_in.sum()
                if N < MIN_N or K in (0, N):
                    continue
                for j, t in enumerate(teams):
                    if n_in[j] >= MIN_N:
                        rows.append(dict(scope="team", team_id=int(t), player_id=None, sid=sid, slabel=slabel, head=head, cls=cls, label=label,
                                         kind="league_same_situation", k=k_in[j], n=n_in[j], base_logit=logit(K / N), base_var=0.0,
                                         group=f"team|{sid}|{head}|{cls}", center=True))
                continue
            n_out, k_out = by_team(d_out), by_team(d_out & hit)
            if min(n_in.sum(), n_out.sum()) < MIN_N or k_in.sum() in (0, n_in.sum()) or k_out.sum() in (0, n_out.sum()):
                continue
            shift = logit(k_in.sum() / n_in.sum()) - logit(k_out.sum() / n_out.sum())     # the league's own contrast
            y_out, v_out = emp_logit(k_out, n_out)
            for j, t in enumerate(teams):
                if n_in[j] >= MIN_N and n_out[j] >= MIN_N:
                    rows.append(dict(scope="team", team_id=int(t), player_id=None, sid=sid, slabel=slabel, head=head, cls=cls, label=label,
                                     kind="team_elsewhere_plus_league_shift", k=k_in[j], n=n_in[j], base_logit=y_out[j] + shift,
                                     base_var=v_out[j], group=f"team|{sid}|{head}|{cls}", center=False))

    # player scope: finishes while on the floor in the situation, against the lineup-adjusted expectation
    ok = df[df["lineup_ok"] & df["finisher_id"].notna()].reset_index(drop=True)
    if len(ok):
        pm = situation_masks(ok)
        for t, g in ok.groupby("off_team_id", sort=True):
            idx = g.index.values
            lu_ids = np.array(g["off_lineup"].tolist(), dtype=np.int64).reshape(-1, 5)
            fin = g["finisher_id"].values.astype(np.int64)
            hit = lu_ids == fin[:, None]
            keep = hit.any(axis=1)                                   # the finisher must be one of the five
            idx, lu_ids, hit = idx[keep], lu_ids[keep], hit[keep]
            if len(idx) < MIN_N_PLAYER_TOTAL:
                continue
            players, lu = np.unique(lu_ids, return_inverse=True)
            lu = lu.reshape(-1, 5)
            slot = hit.argmax(axis=1)
            total = np.bincount(lu.ravel(), minlength=len(players))
            for sid, slabel in SITUATIONS[1:]:
                sm = pm[sid][idx]
                if sm.sum() < MIN_N or (~sm).sum() < MIN_N_PLAYER_TOTAL:
                    continue
                w = fit_usage(lu[~sm], slot[~sm], len(players))       # usage from his OTHER possessions
                e = w[lu[sm]] / w[lu[sm]].sum(axis=1, keepdims=True)  # expected share given the five on the floor
                n_s = np.bincount(lu[sm].ravel(), minlength=len(players)).astype(float)
                k_s = np.bincount(lu[sm][np.arange(sm.sum()), slot[sm]], minlength=len(players)).astype(float)
                e_s = np.bincount(lu[sm].ravel(), weights=e.ravel(), minlength=len(players))
                n_o = np.bincount(lu[~sm].ravel(), minlength=len(players)).astype(float)
                k_o = np.bincount(lu[~sm][np.arange((~sm).sum()), slot[~sm]], minlength=len(players)).astype(float)
                _, v_o = emp_logit(k_o, np.maximum(n_o, 1))
                for j, pid in enumerate(players):
                    if n_s[j] >= MIN_N and total[j] >= MIN_N_PLAYER_TOTAL and n_o[j] >= MIN_N:
                        r = float(np.clip(e_s[j] / n_s[j], 1e-4, 1 - 1e-4))
                        rows.append(dict(scope="player", team_id=int(t), player_id=int(pid), sid=sid, slabel=slabel, head="finisher", cls=None,
                                         label=None, kind="player_lineup_adjusted", k=k_s[j], n=n_s[j], base_logit=logit(r), base_var=v_o[j],
                                         group=f"player|{sid}", center=False))
    return pd.DataFrame(rows)


def estimate(cand: pd.DataFrame) -> pd.DataFrame:
    """Add the posterior for each candidate: logit-scale mean and sd of its rate, and the prior spread used."""
    if cand.empty:
        return cand.assign(post_logit=[], post_sd=[], tau2=[])
    out = cand.copy()
    out["post_logit"], out["post_sd"], out["tau2"] = np.nan, np.nan, np.nan
    for _, g in cand.groupby("group", sort=False):
        y, v = emp_logit(g["k"].values, g["n"].values)
        d, var = y - g["base_logit"].values, v + g["base_var"].values
        mu, tau2, m, pv = eb_normal(d, var, bool(g["center"].iloc[0]))
        # with `center`, the group mean is part of the effect being reported (team vs league rate)
        out.loc[g.index, "post_logit"] = g["base_logit"].values + mu + m
        out.loc[g.index, "post_sd"] = np.sqrt(pv)
        out.loc[g.index, "tau2"] = tau2
    return out


def _to_row(c, player_names: dict[int, str]) -> dict:
    r = float(expit(c.base_logit))
    p = float(expit(c.post_logit))
    lo, hi = float(expit(c.post_logit - Z * c.post_sd)) - r, float(expit(c.post_logit + Z * c.post_sd)) - r
    dev = p - r
    if dev >= 0:
        prob = float(norm.sf((logit(min(r + THRESHOLD, 1 - 1e-6)) - c.post_logit) / c.post_sd))
    else:
        prob = float(norm.cdf((logit(max(r - THRESHOLD, 1e-6)) - c.post_logit) / c.post_sd))
    label = c.label if c.scope == "team" else f"{player_names.get(int(c.player_id), str(int(c.player_id)))}: share of finishes while on the floor"
    return {
        "id": f"{c.scope}:{int(c.player_id) if c.scope == 'player' else 0}:{c.sid}:{c.head}:{c.cls}", "scope": c.scope,
        "player_id": int(c.player_id) if c.scope == "player" else None, "situation": {"id": c.sid, "label": c.slabel},
        "outcome": {"head": c.head, "class": c.cls, "label": label}, "baseline_kind": c.kind,
        "baseline_rate": round(r, 4), "raw_rate": round(float(c.k / c.n), 4), "shrunk_rate": round(p, 4), "deviation": round(dev, 4),
        "interval": [round(lo, 4), round(hi, 4)], "n": int(c.n), "k": int(c.k),
        "prior_strength": round(float(1.0 / (c.tau2 * r * (1 - r))), 1),     # the prior's weight, in possessions at this baseline
        "rank_score": round(prob, 4), "prob_beyond": round(prob, 4), "excludes_zero": bool(lo > 0 or hi < 0),
    }


def team_reports(poss: pd.DataFrame, player_names: dict[int, str], top_k: int | None = TOP_K) -> dict[int, dict]:
    """Return {team_id: report dict matching team_report.schema.json}."""
    df = poss[poss["ending_type"].isin(ENDING_CLASSES) & ~poss["is_heave"]]
    est = estimate(_candidates(poss))
    n_poss, n_games = df.groupby("off_team_id").size(), df.groupby("off_team_id")["game_id"].nunique()
    reports = {}
    for t in sorted(df["off_team_id"].unique()):
        mine = est[est["team_id"] == t] if len(est) else est
        rows = sorted((_to_row(c, player_names) for c in mine.itertuples()), key=lambda x: -x["rank_score"])
        shown = rows[:top_k]
        reports[int(t)] = {"team_id": int(t), "n_games": int(n_games[t]), "n_possessions": int(n_poss[t]), "n_candidates": len(rows),
                           "method": METHOD, "interval_level": INTERVAL_LEVEL, "threshold": THRESHOLD, "sort": "rank_score_desc",
                           "tendencies": shown}
    return reports
