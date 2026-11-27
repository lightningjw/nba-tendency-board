"""Defense proxy fingerprints: what opponents did against a defense, against what those opponents usually do.

A PROXY FROM OUTCOMES. Play-by-play records how a possession ended and who was on the floor. It does
not record where defenders stood, who guarded whom or what the defense was trying to do, so nothing
here identifies a pick-and-roll coverage or any other scheme, and nothing here names one.

Seven outcomes of an opponent possession: where its shot came from (five areas), whether it ended in
a foul drawn, whether it ended in a turnover. For each one:

* **What the opponents usually do** (`Expectation`). Each opponent's own rate on offense against
  every OTHER defense (so the defense being described never sets its own bar), pooled across the 30
  offenses the way the tendency layer pools teams, then moved by how the league's rate changes with
  the way a possession starts. A team that turns the ball over hands its opponents fast breaks; that
  is its offense's doing, and the start-type step keeps it out of the defense's fingerprint.
* **Team row.** The defense's deviation from that expectation on the logit scale, pooled across the
  30 defenses (`tendency_layer.estimate`). The opponents' own uncertainty is in the interval. This
  is what a team report shows.
* **Unit rows** (one five-man lineup; the lineups that include one player). The unit's rate is
  pooled toward "the same opponents, moved by this team's pooled deviation in its OTHER possessions"
  and read against the opponents' expectation. BUILT, CHECKED AND NOT SHOWN: on held-out halves of
  2024-25 and 2025-26 a unit's fingerprint predicted no better than its team's (DECISIONS.md).
  `candidates` still makes the rows so `defense_validation` repeats that check on every export.

Intervals include how much a fingerprint moved between the halves of an earlier season
(`estimate_drift`). `defense_validation.validate()` is the check: fit on the first half of a season,
predict each second-half game.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from . import tendency_layer as layer
from .possessions import SHOT_FAMILIES

LIMIT = ("A proxy from outcomes. Play-by-play shows how opponents' possessions ended and which five were on the floor; it does not show "
         "where defenders stood or who guarded whom. It cannot identify a pick-and-roll coverage or any other scheme. That needs tracking or video.")
BASELINE = ("What the same opponents did against every other defense this season, weighted by how often each was faced, "
            "and moved by how the league's rates change with the way a possession starts")
INTERVAL_LEVEL = layer.INTERVAL_LEVEL
THRESHOLD = layer.THRESHOLD
MIN_N = layer.MIN_N                       # trials of one outcome, inside the unit and outside it
MIN_UNIT_POSS = 100                       # defensive possessions before a unit gets cells at all
SHOW_PROB = 0.90                          # a unit row would be worth showing if it is this likely to sit apart from its team
STRATA = ["made_basket", "def_rebound", "turnover"]      # how the opponent's possession started; everything else is one more stratum
SCOPES = {"defense_team": "Team defense, every opponent possession",
          "defense_player": "Lineups that include one player",
          "defense_lineup": "One five-man lineup"}
KINDS = {"defense_vs_opponents": "Against what the same opponents do against every other defense",
         "defense_unit_vs_team_elsewhere": "Against the same opponents and this team's defense in its other possessions"}
NO_DRIFT = {"rate": {s: 0.0 for s in SCOPES}, "dev": {k: 0.0 for k in KINDS}}
AREA_LABEL = {"rim": "Rim", "paint": "Paint (non-rim)", "midrange": "Midrange", "corner3": "Corner three", "above_break3": "Above-the-break three"}
OUTCOMES = [("shot_family", c, f"{AREA_LABEL[c]} share of opponents' shots") for c in SHOT_FAMILIES] + [
    ("ending_type", "foul_drawn", "Opponent possessions ending in a foul drawn"),
    ("ending_type", "turnover", "Opponent possessions ending in a turnover")]
Z = layer.Z


def _masks(df: pd.DataFrame, head: str, cls: str) -> tuple[np.ndarray, np.ndarray]:
    """(hit, denominator) for one outcome."""
    if head == "shot_family":
        return (df["shot_family"].values == cls), (df["ending_type"].values == "shot")
    return (df["ending_type"].values == cls), np.ones(len(df), bool)


def _stratum(df: pd.DataFrame) -> np.ndarray:
    s = df["start_type"].values
    out = np.full(len(df), len(STRATA), np.int64)
    for i, name in enumerate(STRATA):
        out[s == name] = i
    return out


class Expectation:
    """What each offense usually does, learned from `ref`, for any possession it plays against a given defense.

    For offense O against defense D: O's rate in `ref` against every defense but D, pooled across
    offenses, with the league's start-type shift (`by_start`). The pooling and the shifts are learned
    without D's possessions too, so nothing D did on defense reaches its own expectation. `at()`
    returns the probability for target possessions and the variance of its logit (the offense's own
    sampling noise).
    """

    def __init__(self, ref: pd.DataFrame, by_start: bool = True):
        ref = layer._eligible(ref)
        self.teams = np.array(sorted(set(ref["off_team_id"].unique()) | set(ref["def_team_id"].unique())))
        T, S = len(self.teams), len(STRATA) + 1
        o, d, s = np.searchsorted(self.teams, ref["off_team_id"].values), np.searchsorted(self.teams, ref["def_team_id"].values), _stratum(ref)
        self.by_start = by_start
        self.fit: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for head, cls, _ in OUTCOMES:
            hit, den = _masks(ref, head, cls)
            N = np.zeros((T, T, S))                          # trials by offense, defense, start type
            K = np.zeros((T, T, S))
            np.add.at(N, (o[den], d[den], s[den]), 1.0)
            np.add.at(K, (o[den & hit], d[den & hit], s[den & hit]), 1.0)
            n_od, k_od = N.sum(axis=2), K.sum(axis=2)
            n_l, k_l = n_od.sum(axis=1)[:, None] - n_od, k_od.sum(axis=1)[:, None] - k_od      # offense O against everyone but D
            y, v = layer.emp_logit(k_l, n_l)
            eta, var = np.zeros((T, T)), np.zeros((T, T))
            for j in range(T):                               # how much offenses differ, judged without defense j's games
                seen = n_l[:, j] > 0
                mu, tau2, _, _ = layer.eb_normal(y[seen, j], v[seen, j], center=True)
                w = tau2 / (tau2 + v[:, j])
                eta[:, j], var[:, j] = mu + w * (y[:, j] - mu), w * v[:, j]
            shift = np.zeros((T, S))                         # by defense: the league's start-type contrast without that defense
            if by_start:
                ks, ns = K.sum(axis=(0, 1))[None, :] - K.sum(axis=0), N.sum(axis=(0, 1))[None, :] - N.sum(axis=0)
                tot = ks.sum(axis=1) / np.maximum(ns.sum(axis=1), 1.0)
                ok = (ns >= layer.MIN_N) & (ks > 0) & (ks < ns) & ((tot > 0) & (tot < 1))[:, None]
                rate = np.clip(ks / np.maximum(ns, 1.0), 1e-6, 1 - 1e-6)
                shift = np.where(ok, logit(rate) - logit(np.clip(tot, 1e-6, 1 - 1e-6))[:, None], 0.0)
                # an intercept per (O, D) so that O's own mix of start types gives back its pooled rate
                n_s = N.sum(axis=1)[:, None, :] - N
                a, target = eta.copy(), n_l * expit(eta)
                for _ in range(25):
                    p = expit(a[:, :, None] + shift[None, :, :])
                    f, g = (n_s * p).sum(axis=2) - target, (n_s * p * (1 - p)).sum(axis=2)
                    a = a - np.where(g > 1e-9, f / np.maximum(g, 1e-9), 0.0)
                eta = a
            self.fit[(head, cls)] = (eta, var, shift)

    def at(self, target: pd.DataFrame, head: str, cls: str) -> tuple[np.ndarray, np.ndarray]:
        eta, var, shift = self.fit[(head, cls)]
        o, d = np.searchsorted(self.teams, target["off_team_id"].values), np.searchsorted(self.teams, target["def_team_id"].values)
        return expit(eta[o, d] + shift[d, _stratum(target)]), var[o, d]


def unit_key(players) -> str:
    return "-".join(str(int(p)) for p in sorted(players))


def cell_key(team_id, scope, unit, head, cls) -> str:
    return f"{int(team_id)}|{scope}|{unit or 0}|{head}|{cls}"


def _sums(g: np.ndarray, n_g: int, o: np.ndarray, T: int, hit: np.ndarray, p: np.ndarray, pv: np.ndarray):
    """Per group: trials, hits, expected hits, and the variance of the expected hits that comes from the offenses."""
    n = np.bincount(g, minlength=n_g).astype(float)
    k = np.bincount(g, weights=hit.astype(float), minlength=n_g)
    e = np.bincount(g, weights=p, minlength=n_g)
    G = np.bincount(g * T + o, weights=p * (1 - p), minlength=n_g * T).reshape(n_g, T)       # sensitivity to each offense's logit
    V = np.zeros((n_g, T))
    np.maximum.at(V, (g, o), pv)                                                               # one variance per (defense, offense)
    return n, k, e, G, V


def _base(n, e, G, V) -> tuple[np.ndarray, np.ndarray]:
    """Logit of the expected rate and the variance of that logit."""
    r = np.clip(e / np.maximum(n, 1e-9), 1e-4, 1 - 1e-4)
    return logit(r), (G ** 2 * V).sum(axis=-1) / np.maximum(n * r * (1 - r), 1e-9) ** 2


def _lineups(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(the defending five of each possession, sorted; whether it is known). An unknown five is a row of -1."""
    ok = df["lineup_ok"].values.astype(bool) & (df["def_lineup"].map(len).values == 5) if len(df) else np.zeros(0, bool)
    lu = np.full((len(df), 5), -1, np.int64)
    if ok.any():
        lu[ok] = np.sort(np.array(df["def_lineup"][ok].tolist(), dtype=np.int64).reshape(-1, 5), axis=1)
    return lu, ok


def _units(d: np.ndarray, lu: np.ndarray, lu_ok: np.ndarray, T: int, scopes, min_poss: int) -> list[tuple[str, int, str, np.ndarray]]:
    """(scope, team index, unit key, rows it defended) for every five and every player with `min_poss` defensive possessions."""
    units = []
    for j in range(T):
        idx = np.flatnonzero((d == j) & lu_ok)
        if not len(idx):
            continue
        if "defense_lineup" in scopes:
            fives, inv, cnt = np.unique(lu[idx], axis=0, return_inverse=True, return_counts=True)
            inv = inv.ravel()
            for u in np.flatnonzero(cnt >= min_poss):
                units.append(("defense_lineup", j, unit_key(fives[u]), idx[inv == u]))
        if "defense_player" in scopes:
            pl, cnt = np.unique(lu[idx], return_counts=True)
            for p_id in pl[cnt >= min_poss]:
                units.append(("defense_player", j, unit_key([p_id]), idx[(lu[idx] == p_id).any(axis=1)]))
    return units


def candidates(poss: pd.DataFrame, expect: Expectation | None = None, scopes=tuple(SCOPES)) -> pd.DataFrame:
    """Every defensive cell with what `tendency_layer.estimate` needs.

    `expect` is where the opponents' usual rates come from; by default the same possessions (each
    offense against every other defense). `opp_logit` is the expectation from the opponents alone;
    for team rows it is also the baseline, for unit rows the baseline adds the team's deviation elsewhere.
    """
    df = layer._eligible(poss)
    expect = expect or Expectation(df)
    teams = expect.teams
    T = len(teams)
    known = np.isin(df["off_team_id"].values, teams) & np.isin(df["def_team_id"].values, teams)
    df = df[known].reset_index(drop=True)
    o, d = np.searchsorted(teams, df["off_team_id"].values), np.searchsorted(teams, df["def_team_id"].values)
    lu, lu_ok = _lineups(df)
    rows = []
    units = _units(d, lu, lu_ok, T, scopes, MIN_UNIT_POSS)
    for head, cls, label in OUTCOMES:
        hit, den = _masks(df, head, cls)
        p, pv = expect.at(df, head, cls)
        m = den
        n, k, e, G, V = _sums(d[m], T, o[m], T, hit[m], p[m], pv[m])
        b, bv = _base(n, e, G, V)
        y, v = layer.emp_logit(k, np.maximum(n, 1e-9))
        big = n >= MIN_N
        tau2 = layer.eb_normal((y - b)[big], (v + bv)[big], center=False)[1]            # how much the 30 defenses differ on this outcome
        if "defense_team" in scopes:
            for j in np.flatnonzero(n >= MIN_N):
                rows.append(dict(scope="defense_team", team_id=int(teams[j]), unit=None, head=head, cls=cls, label=label, kind="defense_vs_opponents",
                                 k=k[j], n=n[j], base_logit=b[j], base_var=bv[j], opp_logit=b[j], opp_var=bv[j],
                                 group=f"defense_team|{head}|{cls}", center=False, k_out=0.0, n_out=0.0))
        if not units:
            continue
        row_unit = np.concatenate([np.full(len(ix), ui) for ui, (_, _, _, ix) in enumerate(units)])
        row_idx = np.concatenate([ix for _, _, _, ix in units])
        keep = m[row_idx]
        ru, ri = row_unit[keep], row_idx[keep]
        n_i, k_i, e_i, G_i, V_i = _sums(ru, len(units), o[ri], T, hit[ri], p[ri], pv[ri])
        tj = np.array([j for _, j, _, _ in units])
        n_o, k_o, e_o, G_o = n[tj] - n_i, k[tj] - k_i, e[tj] - e_i, G[tj] - G_i
        V_u = V[tj]
        b_i, bv_i = _base(n_i, e_i, G_i, V_u)
        b_o, bv_o = _base(n_o, e_o, G_o, V_u)
        y_o, v_o = layer.emp_logit(k_o, np.maximum(n_o, 1e-9))
        w_o = tau2 / (tau2 + v_o + bv_o)                     # the team's deviation elsewhere is itself pooled across the 30 defenses
        d_o, dv_o = w_o * (y_o - b_o), w_o * (v_o + bv_o)
        for ui, (scope, j, key, _) in enumerate(units):
            if n_i[ui] >= MIN_N and n_o[ui] >= MIN_N:
                rows.append(dict(scope=scope, team_id=int(teams[j]), unit=key, head=head, cls=cls, label=label, kind="defense_unit_vs_team_elsewhere",
                                 k=k_i[ui], n=n_i[ui], base_logit=b_i[ui] + d_o[ui], base_var=dv_o[ui] + bv_i[ui],
                                 opp_logit=b_i[ui], opp_var=bv_i[ui], group=f"{scope}|{head}|{cls}", center=False, k_out=k_o[ui], n_out=n_o[ui]))
    out = pd.DataFrame(rows)
    if len(out):
        out["key"] = [cell_key(t, s, u, h, c) for t, s, u, h, c in zip(out["team_id"], out["scope"], out["unit"], out["head"], out["cls"])]
    return out


def estimate(cand: pd.DataFrame, drift: dict | None = None) -> pd.DataFrame:
    """The layer's posterior for every cell, plus the two numbers a row shows.

    `fp_logit` and `fp_sd`: the cell's deviation from the opponents' expectation (the fingerprint).
    For a team row that is the pooled deviation from its baseline. For a unit row it is the unit's
    pooled rate minus the opponents' expectation, so it carries the team's defense with it.
    """
    est = layer.estimate(cand, drift or NO_DRIFT)
    if est.empty:
        return est.assign(fp_logit=[], fp_sd=[], prob_apart=[])
    team = (est["scope"] == "defense_team").values
    est["fp_logit"] = np.where(team, est["dev_logit"], est["rate_logit"] - est["opp_logit"])
    est["fp_sd"] = np.where(team, est["dev_sd"], np.sqrt(est["rate_sd"] ** 2 + est["opp_var"]))
    # how likely the cell sits more than THRESHOLD from its own baseline (for a unit: from its team's other possessions)
    r = expit(est["base_logit"].values)
    at, up = est["base_logit"].values + est["dev_logit"].values, est["dev_logit"].values >= 0
    edge = np.where(up, logit(np.minimum(r + THRESHOLD, 1 - 1e-6)), logit(np.maximum(r - THRESHOLD, 1e-6)))
    est["prob_apart"] = norm.cdf(np.where(up, at - edge, edge - at) / est["dev_sd"].values)
    return est


def split_halves(poss: pd.DataFrame, interleave: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    return layer.split_halves(poss, interleave)


def estimate_drift(poss: pd.DataFrame, interleave: bool = False) -> dict:
    """How much a fingerprint moves between the two halves of one season, beyond sampling noise (variance, logit scale).

    `rate`, per scope: the deviation from the opponents' expectation. `dev`, per kind of row: the
    deviation from the row's own baseline. Each half sets its own expectations from its own games.
    Measured on a season other than the one being reported.
    """
    a, b = split_halves(poss, interleave)
    c1, c2 = candidates(a), candidates(b)
    out = {"rate": {s: 0.0 for s in SCOPES}, "dev": {k: 0.0 for k in KINDS}, "n_cells": {**{s: 0 for s in SCOPES}, **{k: 0 for k in KINDS}},
           "n_games": int(poss["game_id"].nunique())}
    if c1.empty or c2.empty:
        return out
    m = c1.merge(c2, on="key", suffixes=("_1", "_2"))
    y1, v1 = layer.emp_logit(m["k_1"].values, m["n_1"].values)
    y2, v2 = layer.emp_logit(m["k_2"].values, m["n_2"].values)
    d_fp = (y2 - m["opp_logit_2"].values) - (y1 - m["opp_logit_1"].values)
    v_fp = v1 + v2 + m["opp_var_1"].values + m["opp_var_2"].values
    d_dev = (y2 - m["base_logit_2"].values) - (y1 - m["base_logit_1"].values)
    v_dev = v1 + v2 + m["base_var_1"].values + m["base_var_2"].values
    for scope in SCOPES:
        i = (m["scope_1"] == scope).values
        out["rate"][scope], out["n_cells"][scope] = layer._excess_var(d_fp[i], v_fp[i]), int(i.sum())
    for kind in KINDS:
        i = (m["kind_1"] == kind).values
        out["dev"][kind], out["n_cells"][kind] = layer._excess_var(d_dev[i], v_dev[i]), int(i.sum())
    return out


def _row(c) -> dict:
    """One fingerprint row: expected from the opponents, allowed, the pooled difference and its interval."""
    r = float(expit(c.opp_logit))
    at = c.opp_logit + c.fp_logit
    p = float(expit(at))
    lo, hi = float(expit(at - Z * c.fp_sd)) - r, float(expit(at + Z * c.fp_sd)) - r
    dev = p - r
    z = float((at - logit(min(r + THRESHOLD, 1 - 1e-6))) / c.fp_sd) if dev >= 0 else float((logit(max(r - THRESHOLD, 1e-6)) - at) / c.fp_sd)
    return {"outcome": {"head": c.head, "class": c.cls, "label": c.label}, "expected_rate": round(r, 4), "raw_rate": round(float(c.k / c.n), 4),
            "shrunk_rate": round(p, 4), "deviation": round(dev, 4) + 0.0, "interval": [round(lo, 4) + 0.0, round(hi, 4) + 0.0], "n": int(c.n), "k": int(c.k),
            "prob_beyond": round(float(norm.cdf(z)), 4), "excludes_zero": bool(lo > 0 or hi < 0)}


def team_sections(poss: pd.DataFrame, drift: dict | None = None) -> dict[int, dict]:
    """{team_id: the "Defense" section of its report}. `poss` is one season; `drift` comes from an earlier one.

    Team rows only. Unit rows did not beat the team's fingerprint on held-out games, so no report shows them.
    """
    df = layer._eligible(poss)
    est = estimate(candidates(df, scopes=("defense_team",)), drift)
    n_poss, n_games = df.groupby("def_team_id").size(), df.groupby("def_team_id")["game_id"].nunique()
    order = {(h, c): i for i, (h, c, _) in enumerate(OUTCOMES)}
    has_drift = bool(drift) and any(v > 0 for v in drift["rate"].values())
    out = {}
    for t in sorted(df["def_team_id"].unique()):
        mine = est[est["team_id"] == t] if len(est) else est
        out[int(t)] = {"note": LIMIT, "interval_level": INTERVAL_LEVEL, "threshold": THRESHOLD, "drift_included": has_drift,
                       "season": int(df["season"].max()), "n_games": int(n_games[t]), "n_possessions": int(n_poss[t]),
                       "baseline": BASELINE, "rows": [_row(c) for c in sorted(mine.itertuples(), key=lambda c: order[(c.head, c.cls)])]}
    return out
