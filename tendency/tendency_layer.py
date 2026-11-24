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

Tendencies also move. `estimate_drift` measures how much, from the two halves of an earlier season
(the variance of the change that binomial noise does not explain, logit scale, per scope), and
`estimate(..., drift=...)` adds it, so an interval is about the stretch of games to come and not only
the ones already played. Without it 90% intervals covered about 80% of the next half-season.

`tendency_validation.validate()` decides whether this ships: fit on the first half of a season,
predict the second half, and compare with raw rates and with the old stand-in (`tendencies.py`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm

from .possessions import ENDING_CLASSES
from .tendencies import SITUATIONS, allowed, outcome_specs, situation_masks

METHOD = {"id": "eb_hierarchical_logit", "label": "Partial pooling across teams, situations and players (empirical Bayes, hierarchical logit)",
          "final": True}                 # final because the half-season validation supports it; see DECISIONS.md 86-87
INTERVAL_LEVEL = 0.90
THRESHOLD = 0.01                 # a deviation smaller than one percentage point is not worth a scout's time
MIN_N = 25
MIN_N_PLAYER_TOTAL = 100
TOP_K = 10
USAGE_PRIOR_POSS = 20.0          # each player starts as an average finisher over this many possessions
TAU2_BOUNDS = (1e-6, 4.0)
DRIFT_BOUNDS = (1e-8, 1.0)
KINDS = {"league_same_situation": "Team, all possessions (against the league)",
         "team_elsewhere_plus_league_shift": "Team, in a situation (against its other possessions)",
         "player_lineup_adjusted": "Player, in a situation (against his lineup-adjusted usage)"}
NO_DRIFT = {"rate": {"team": 0.0, "player": 0.0}, "dev": {k: 0.0 for k in KINDS}}
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


# ---- cells: one per (scope, team/player, situation, outcome) ----------------------------------
def cell_key(team_id, scope, player_id, sid, head, cls) -> str:
    """One string per cell, built the same way for every method, so cells can be matched across methods and halves."""
    return f"{int(team_id)}|{scope}|{int(player_id) if scope == 'player' else 0}|{sid}|{head}|{cls if isinstance(cls, str) else '-'}"


def keys_of(df: pd.DataFrame) -> list[str]:
    return [cell_key(*r) for r in zip(df["team_id"], df["scope"], df["player_id"], df["sid"], df["head"], df["cls"])]


def split_halves(poss: pd.DataFrame, interleave: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First and second half of the games by tip time. `interleave` alternates games instead (a no-drift control)."""
    games = poss.drop_duplicates("game_id").sort_values(["tipoff", "game_id"])["game_id"].values
    first = set(games[::2] if interleave else games[: len(games) // 2])
    m = poss["game_id"].isin(first)
    return poss[m], poss[~m]


def _eligible(poss: pd.DataFrame) -> pd.DataFrame:
    return poss[poss["ending_type"].isin(ENDING_CLASSES) & ~poss["is_heave"]].reset_index(drop=True)


def _on_floor(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Possessions usable for player rows (five known players, the finisher one of them), their lineups and who finished."""
    ok = df[df["lineup_ok"] & df["finisher_id"].notna()].reset_index(drop=True)
    if ok.empty:
        return ok, np.zeros((0, 5), np.int64), np.zeros((0, 5), bool)
    lu = np.array(ok["off_lineup"].tolist(), dtype=np.int64).reshape(-1, 5)
    hit = lu == ok["finisher_id"].values.astype(np.int64)[:, None]
    keep = hit.any(axis=1)
    return ok[keep].reset_index(drop=True), lu[keep], hit[keep]


def cell_counts(poss: pd.DataFrame) -> pd.DataFrame:
    """k and n of every cell in every game: columns game_id, key, scope, k, n. Same cell definitions as `candidates`.

    The validation needs counts by game (to resample games) and for cells too thin for `candidates` to keep.
    """
    df = _eligible(poss)
    masks = situation_masks(df)
    game, team = df["game_id"].values, df["off_team_id"].values
    frames = []
    for sid, _ in SITUATIONS:
        for head, cls, hit, denom, _ in outcome_specs(df):
            if not allowed(sid, head):
                continue
            d = masks[sid] & denom
            g = pd.DataFrame({"game_id": game[d], "team_id": team[d], "k": hit[d].astype(np.int64)}).groupby(["game_id", "team_id"], sort=False)["k"].agg(["sum", "size"]).reset_index()
            frames.append(pd.DataFrame({"game_id": g["game_id"].values, "key": [cell_key(t, "team", None, sid, head, cls) for t in g["team_id"]],
                                        "scope": "team", "k": g["sum"].values, "n": g["size"].values}))
    ok, lu, hit = _on_floor(df)
    if len(ok):
        pm = situation_masks(ok)
        long = pd.DataFrame({"row": np.repeat(np.arange(len(ok)), 5), "game_id": np.repeat(ok["game_id"].values, 5),
                             "team_id": np.repeat(ok["off_team_id"].values, 5), "player_id": lu.ravel(), "k": hit.ravel().astype(np.int64)})
        for sid, _ in SITUATIONS[1:]:
            g = long[pm[sid][long["row"].values]].groupby(["game_id", "team_id", "player_id"], sort=False)["k"].agg(["sum", "size"]).reset_index()
            frames.append(pd.DataFrame({"game_id": g["game_id"].values,
                                        "key": [cell_key(t, "player", p, sid, "finisher", None) for t, p in zip(g["team_id"], g["player_id"])],
                                        "scope": "player", "k": g["sum"].values, "n": g["size"].values}))
    return pd.concat(frames, ignore_index=True)


def candidates(poss: pd.DataFrame) -> pd.DataFrame:
    """Every candidate row with what the estimator needs: counts in the situation, the baseline logit and its variance.

    `k_out`, `n_out` and `out_logit` describe the same team's or player's OTHER possessions (the baseline's
    raw material). The empirical-Bayes fit uses them only through `base_logit` and `base_var`; the PyMC fit models them.
    """
    df = _eligible(poss)
    masks = situation_masks(df)
    team = df["off_team_id"].values
    teams = np.array(sorted(np.unique(team)))
    t_idx = np.searchsorted(teams, team)
    specs = outcome_specs(df)

    def by_team(m: np.ndarray) -> np.ndarray:
        return np.bincount(t_idx[m], minlength=len(teams)).astype(float)

    rows = []
    for sid, slabel in SITUATIONS:
        sm = masks[sid]
        for head, cls, hit, denom, label in specs:
            if not allowed(sid, head):
                continue
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
                                         group=f"team|{sid}|{head}|{cls}", center=True, k_out=0.0, n_out=0.0, out_logit=np.nan))
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
                                     base_var=v_out[j], group=f"team|{sid}|{head}|{cls}", center=False,
                                     k_out=k_out[j], n_out=n_out[j], out_logit=y_out[j]))

    # player scope: finishes while on the floor in the situation, against the lineup-adjusted expectation
    ok, lu_all, hit_all = _on_floor(df)
    if len(ok):
        pm = situation_masks(ok)
        for t, g in ok.groupby("off_team_id", sort=True):
            idx = g.index.values
            if len(idx) < MIN_N_PLAYER_TOTAL:
                continue
            players, lu = np.unique(lu_all[idx], return_inverse=True)
            lu = lu.reshape(-1, 5)
            slot = hit_all[idx].argmax(axis=1)
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
                e_o = np.bincount(lu[~sm].ravel(), weights=(w[lu[~sm]] / w[lu[~sm]].sum(axis=1, keepdims=True)).ravel(), minlength=len(players))
                n_o = np.bincount(lu[~sm].ravel(), minlength=len(players)).astype(float)
                k_o = np.bincount(lu[~sm][np.arange((~sm).sum()), slot[~sm]], minlength=len(players)).astype(float)
                _, v_o = emp_logit(k_o, np.maximum(n_o, 1))
                for j, pid in enumerate(players):
                    if n_s[j] >= MIN_N and total[j] >= MIN_N_PLAYER_TOTAL and n_o[j] >= MIN_N:
                        r = float(np.clip(e_s[j] / n_s[j], 1e-4, 1 - 1e-4))
                        rows.append(dict(scope="player", team_id=int(t), player_id=int(pid), sid=sid, slabel=slabel, head="finisher", cls=None,
                                         label=None, kind="player_lineup_adjusted", k=k_s[j], n=n_s[j], base_logit=logit(r), base_var=v_o[j],
                                         group=f"player|{sid}", center=False, k_out=k_o[j], n_out=n_o[j],
                                         out_logit=logit(float(np.clip(e_o[j] / n_o[j], 1e-4, 1 - 1e-4)))))
    out = pd.DataFrame(rows)
    if len(out):
        out["key"] = keys_of(out)
    return out


def estimate(cand: pd.DataFrame, drift: dict | None = None) -> pd.DataFrame:
    """Add each candidate's posterior, on the logit scale.

    Two different things are estimated, and they are not the same number:

    * the DEVIATION from the baseline (`dev_logit`, `dev_sd`): what the report shows. Its noise is the
      cell's own plus the baseline's, so an uncertain baseline widens the interval.
    * the cell's own RATE (`rate_logit`, `rate_sd`): what the validation scores. The count in the cell
      and "baseline plus a typical deviation" are two views of the same rate, weighted by precision.

    `drift` (from `estimate_drift`) adds the between-half variance: per scope for the rate, per kind of row for the deviation.
    """
    cols = ["dev_logit", "dev_sd", "rate_logit", "rate_sd", "tau2"]
    if cand.empty:
        return cand.assign(**{c: [] for c in cols})
    drift = drift or NO_DRIFT
    out = cand.copy()
    for c in cols:
        out[c] = np.nan
    for _, g in cand.groupby("group", sort=False):
        y, v = emp_logit(g["k"].values, g["n"].values)
        base, bv = g["base_logit"].values, g["base_var"].values
        d = y - base
        mu, tau2, m, pv = eb_normal(d, v + bv, bool(g["center"].iloc[0]))
        dr_rate, dr_dev = drift["rate"][g["scope"].iloc[0]], drift["dev"][g["kind"].iloc[0]]
        w = (tau2 + bv) / (tau2 + bv + v)                    # weight on the cell's own count
        # with `center`, the group mean is part of the effect being reported (team vs league rate)
        out.loc[g.index, "dev_logit"] = mu + m
        out.loc[g.index, "dev_sd"] = np.sqrt(pv + dr_dev)
        out.loc[g.index, "rate_logit"] = base + mu + w * (d - mu)
        out.loc[g.index, "rate_sd"] = np.sqrt(w * v + dr_rate)
        out.loc[g.index, "tau2"] = tau2
    return out


def _excess_var(d: np.ndarray, v: np.ndarray) -> float:
    """Variance of d that its sampling variance v does not explain: d_i ~ N(0, v_i + x), x by maximum likelihood."""
    if len(d) < 10:
        return 0.0
    res = minimize_scalar(lambda lx: float(np.sum(np.log(np.exp(lx) + v) + d ** 2 / (np.exp(lx) + v))),
                          bounds=(np.log(DRIFT_BOUNDS[0]), np.log(DRIFT_BOUNDS[1])), method="bounded")
    x = float(np.exp(res.x))
    return 0.0 if x < 10 * DRIFT_BOUNDS[0] else x


def estimate_drift(poss: pd.DataFrame, interleave: bool = False) -> dict:
    """How much cells move between the two halves of one season, beyond binomial noise (variance, logit scale).

    `rate`, per scope, is the drift of a cell's own rate. `dev`, per kind of row, is the drift of its
    deviation from the baseline. They differ a lot: a team's overall rates move (trades, injuries, a
    new rotation), and its situational rates move with them, so the situational DEVIATION barely moves.
    Both are everything binomial sampling does not explain: real change plus the little extra spread
    from possessions in one game not being independent. Measured on a season OTHER than the one
    being reported, so the report never tunes its own intervals.
    """
    a, b = split_halves(poss, interleave)
    c1, c2 = candidates(a), candidates(b)
    out = {"rate": {"team": 0.0, "player": 0.0}, "dev": {k: 0.0 for k in KINDS}, "n_cells": {**{k: 0 for k in KINDS}, "team": 0, "player": 0},
           "n_games": int(poss["game_id"].nunique())}
    if c1.empty or c2.empty:
        return out
    m = c1.merge(c2, on="key", suffixes=("_1", "_2"))
    y1, v1 = emp_logit(m["k_1"].values, m["n_1"].values)
    y2, v2 = emp_logit(m["k_2"].values, m["n_2"].values)
    d_rate, d_dev = y2 - y1, (y2 - m["base_logit_2"].values) - (y1 - m["base_logit_1"].values)
    v_dev = v1 + v2 + m["base_var_1"].values + m["base_var_2"].values
    for scope in ("team", "player"):
        i = (m["scope_1"] == scope).values
        out["rate"][scope], out["n_cells"][scope] = _excess_var(d_rate[i], (v1 + v2)[i]), int(i.sum())
    for kind in KINDS:
        i = (m["kind_1"] == kind).values
        out["dev"][kind], out["n_cells"][kind] = _excess_var(d_dev[i], v_dev[i]), int(i.sum())
    return out


def _to_row(c, player_names: dict[int, str]) -> dict:
    return _row_and_z(c, player_names)[0]


def _row_and_z(c, player_names: dict[int, str]) -> tuple[dict, float]:
    """The report row, and how many posterior sds the deviation clears the threshold by.

    The second number orders rows exactly as the probability does, but does not run out of digits:
    after a full season dozens of rows have a probability that rounds to 1.0000.
    """
    r = float(expit(c.base_logit))
    at = c.base_logit + c.dev_logit
    p = float(expit(at))
    lo, hi = float(expit(at - Z * c.dev_sd)) - r, float(expit(at + Z * c.dev_sd)) - r
    dev = p - r
    if dev >= 0:
        z = float((at - logit(min(r + THRESHOLD, 1 - 1e-6))) / c.dev_sd)
    else:
        z = float((logit(max(r - THRESHOLD, 1e-6)) - at) / c.dev_sd)
    prob = float(norm.cdf(z))
    label = c.label if c.scope == "team" else f"{player_names.get(int(c.player_id), str(int(c.player_id)))}: share of finishes while on the floor"
    return {
        "id": f"{c.scope}:{int(c.player_id) if c.scope == 'player' else 0}:{c.sid}:{c.head}:{c.cls if isinstance(c.cls, str) else None}", "scope": c.scope,
        "player_id": int(c.player_id) if c.scope == "player" else None, "situation": {"id": c.sid, "label": c.slabel},
        "outcome": {"head": c.head, "class": c.cls if isinstance(c.cls, str) else None, "label": label}, "baseline_kind": c.kind,
        "baseline_rate": round(r, 4), "raw_rate": round(float(c.k / c.n), 4), "shrunk_rate": round(p, 4), "deviation": round(dev, 4),
        "interval": [round(lo, 4), round(hi, 4)], "n": int(c.n), "k": int(c.k),
        "prior_strength": round(float(1.0 / (c.tau2 * r * (1 - r))), 1),     # the prior's weight, in possessions at this baseline
        "rank_score": round(prob, 4), "prob_beyond": round(prob, 4), "excludes_zero": bool(lo > 0 or hi < 0),
    }, z


def all_rows(poss: pd.DataFrame, player_names: dict[int, str], drift: dict | None = None) -> dict[int, list[dict]]:
    """{team_id: every candidate row, best first}."""
    est = estimate(candidates(poss), drift)
    out: dict[int, list[tuple]] = {}
    for c in est.itertuples():
        out.setdefault(int(c.team_id), []).append(_row_and_z(c, player_names))
    return {t: [row for row, _ in sorted(rows, key=lambda x: (-x[1], x[0]["id"]))] for t, rows in out.items()}


def team_reports(poss: pd.DataFrame, player_names: dict[int, str], top_k: int | None = TOP_K, drift: dict | None = None) -> dict[int, dict]:
    """Return {team_id: report dict matching team_report.schema.json}. `drift` comes from `estimate_drift` on an earlier season."""
    df = poss[poss["ending_type"].isin(ENDING_CLASSES) & ~poss["is_heave"]]
    rows_by_team = all_rows(poss, player_names, drift)
    n_poss, n_games = df.groupby("off_team_id").size(), df.groupby("off_team_id")["game_id"].nunique()
    has_drift = bool(drift) and any(v > 0 for v in drift["dev"].values())
    reports = {}
    for t in sorted(df["off_team_id"].unique()):
        rows = rows_by_team.get(int(t), [])
        reports[int(t)] = {"team_id": int(t), "n_games": int(n_games[t]), "n_possessions": int(n_poss[t]), "n_candidates": len(rows),
                           "method": METHOD, "interval_level": INTERVAL_LEVEL, "threshold": THRESHOLD, "drift_included": has_drift,
                           "sort": "rank_score_desc", "tendencies": rows[:top_k]}
    return reports
