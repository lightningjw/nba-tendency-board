"""Team-report tendencies: a SIMPLE SHRINKAGE ESTIMATE, not the final tendency layer.

Each tendency is a rate in a defined situation compared with a baseline rate:

* team scope, all possessions: team's rate vs the league's rate (a main effect)
* team scope, a situation:     team's rate in the situation vs what its overall rate
                               plus the league-wide situation shift would predict
                               (added on the logit scale). The deviation is the
                               team-by-situation interaction, so a team that simply
                               shoots a lot of threes everywhere does not show up
                               again under every situation.
* player scope: player's share of his team's finishes while on the floor in the
                situation vs his own share across all situations

Estimate: beta-binomial empirical Bayes. Prior = Beta centred on the baseline with
strength m (pseudo-possessions). For team rows m is a method-of-moments estimate
from the spread across teams, clipped to [20, 2000]; for player rows m is fixed.
Interval: equal-tailed posterior interval minus the baseline (the baseline is
treated as fixed, which makes intervals slightly too narrow - see DECISIONS.md).

Ranking: by the interval bound nearest zero, signed so that bigger = stronger
evidence of a deviation in either direction. Point estimates never set the order.

The partial-pooling model across team, player and situation (Phase 2) replaces this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

from . import cdn_fields as F
from .possessions import ACTION_STYLES, ENDING_CLASSES, SHOT_FAMILIES

METHOD = {"id": "eb_beta_binomial_shrinkage", "label": "Simple shrinkage estimate (empirical Bayes, beta-binomial)", "final": False}
INTERVAL_LEVEL = 0.90
M_PLAYER_FIXED = 60.0
M_CLIP = (20.0, 2000.0)
MIN_N = 25
TOP_K = 10

SITUATIONS = [
    ("all", "All possessions"),
    ("after_timeout", "After a timeout"),
    ("clutch", "Last 5:00 of Q4/OT, within 5"),
    ("start_def_rebound", "Off a defensive rebound"),
    ("start_turnover", "Off a turnover"),
    ("start_made_basket", "After an opponent score"),
    ("def_in_penalty", "Defense in the penalty"),
    ("q4", "Fourth quarter"),
    ("two_for_one", "Two-for-one window (30 to 45 s left in the period)"),
]
END_LABEL = {"shot": "Possessions ending in a shot", "foul_drawn": "Possessions ending in a foul drawn", "turnover": "Possessions ending in a turnover"}
FAM_LABEL = {"rim": "Rim share of shots", "paint": "Paint (non-rim) share of shots", "midrange": "Midrange share of shots",
             "corner3": "Corner-three share of shots", "above_break3": "Above-the-break-three share of shots"}
STYLE_LABEL = {c: f"{lab} share of shots" for c, lab in zip(F.ACTION_STYLES, F.ACTION_STYLE_LABELS)}
TEMPO_LABEL = {"early": f"Early share of possessions (first action within {F.TEMPO_EARLY_MAX_S:.0f} s)",
               "late": f"Late share of possessions (first action after {F.TEMPO_LATE_MIN_S:.0f} s or more)"}


# These two situations are read just before the deciding event, so a timeout or foul in the MIDDLE of a
# possession sets them, and a possession interrupted that way is late by construction. No tempo rows there.
MID_POSSESSION_SITUATIONS = {"after_timeout", "def_in_penalty"}


def allowed(sid: str, head: str) -> bool:
    """Is (situation, outcome family) a cell worth estimating? Shared by the stand-in, the layer and the validation."""
    return not (head == "tempo" and sid in MID_POSSESSION_SITUATIONS)


def _flag(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """(is true, is defined) for a true / false / not-defined column."""
    s = df[col]
    return s.fillna(False).astype(bool).values, s.notna().values


def outcome_specs(df: pd.DataFrame) -> list[tuple]:
    """Every team-scope outcome a tendency can be about: (head, class, hit mask, denominator mask, label).

    Shared by the stand-in and the tendency layer so both estimate the same cells. The step 2.2
    outcomes are added only when the table carries their columns.
    """
    n = len(df)
    is_shot = df["ending_type"].values == "shot"
    specs = [("ending_type", c, df["ending_type"].values == c, np.ones(n, bool), END_LABEL[c]) for c in ENDING_CLASSES]
    specs += [("shot_family", c, df["shot_family"].values == c, is_shot, FAM_LABEL[c]) for c in SHOT_FAMILIES]
    if "action_style" in df.columns:
        styled = df["action_style"].notna().values
        specs += [("action_style", c, df["action_style"].values == c, styled, STYLE_LABEL[c]) for c in ACTION_STYLES]
    if "assisted" in df.columns:
        hit, defined = _flag(df, "assisted")
        specs.append(("creation", "assisted", hit, defined, "Assisted share of made baskets"))
    if "tempo" in df.columns:
        timed = df["tempo"].notna().values
        specs += [("tempo", c, df["tempo"].values == c, timed, TEMPO_LABEL[c]) for c in ("early", "late")]
    if "fast_break" in df.columns:
        hit, defined = _flag(df, "fast_break")
        specs.append(("fast_break", "fast_break", hit, defined, "Fast-break share of possessions ending in a shot or at the line"))
    return specs


def situation_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    window = df["two_for_one_window"].values.astype(bool) if "two_for_one_window" in df.columns else np.zeros(len(df), bool)
    return {
        "two_for_one": window,
        "all": np.ones(len(df), dtype=bool),
        "after_timeout": df["after_timeout"].values.astype(bool),
        "clutch": ((df["period"] >= 4) & (df["start_clock_s"] <= 300) & (df["margin_start"].abs() <= 5)).values,
        "start_def_rebound": (df["start_type"] == "def_rebound").values,
        "start_turnover": (df["start_type"] == "turnover").values,
        "start_made_basket": (df["start_type"] == "made_basket").values,
        "def_in_penalty": df["def_in_penalty"].values.astype(bool),
        "q4": (df["period"] == 4).values,
    }


def _logit(p):
    return np.log(p / (1 - p))


def _mom_strength(k: np.ndarray, n: np.ndarray, r: np.ndarray) -> float:
    """Method-of-moments prior strength from the spread of team rates around their baselines."""
    ok = (n >= 10) & (r > 0) & (r < 1)
    if ok.sum() < 3:
        return M_CLIP[1]
    p, rr = k[ok] / n[ok], r[ok]
    between = np.var(p - rr, ddof=1) - np.mean(rr * (1 - rr) / n[ok])
    if between <= 0:
        return M_CLIP[1]
    return float(np.clip(np.mean(rr * (1 - rr)) / between - 1, *M_CLIP))


def _row(scope, player_id, sit, head, cls, label, kind, r, k, n, m) -> dict:
    a, b = r * m + k, (1 - r) * m + (n - k)
    lo, hi = beta.ppf([(1 - INTERVAL_LEVEL) / 2, (1 + INTERVAL_LEVEL) / 2], a, b) - r
    dev = a / (a + b) - r
    rank = lo if dev >= 0 else -hi
    return {
        "id": f"{scope}:{player_id or 0}:{sit[0]}:{head}:{cls}", "scope": scope, "player_id": player_id,
        "situation": {"id": sit[0], "label": sit[1]}, "outcome": {"head": head, "class": cls, "label": label},
        "baseline_kind": kind, "baseline_rate": round(float(r), 4), "raw_rate": round(k / n, 4),
        "shrunk_rate": round(float(a / (a + b)), 4), "deviation": round(float(dev), 4),
        "interval": [round(float(lo), 4), round(float(hi), 4)], "n": int(n), "k": int(k),
        "prior_strength": round(float(m), 1), "rank_score": round(float(rank), 4), "excludes_zero": bool(lo > 0 or hi < 0),
    }


def team_reports(poss: pd.DataFrame, player_names: dict[int, str], top_k: int | None = TOP_K) -> dict[int, dict]:
    """Return {team_id: report dict matching team_report.schema.json}."""
    df = poss[poss["ending_type"].isin(ENDING_CLASSES) & ~poss["is_heave"]].reset_index(drop=True)
    masks = situation_masks(df)
    teams = sorted(df["off_team_id"].unique())
    team_arr = df["off_team_id"].values
    out: dict[int, list[dict]] = {t: [] for t in teams}

    specs = outcome_specs(df)

    all_mask = masks["all"]
    for sit in SITUATIONS:
        sm = masks[sit[0]]
        for head, cls, hit, denom, label in specs:
            if not allowed(sit[0], head):
                continue
            d = sm & denom
            n_all, k_all = int(d.sum()), int((d & hit).sum())
            if n_all < MIN_N or k_all in (0, n_all):
                continue
            r_sit = k_all / n_all
            n_t = np.array([int((d & (team_arr == t)).sum()) for t in teams])
            k_t = np.array([int((d & hit & (team_arr == t)).sum()) for t in teams])
            if sit[0] == "all":
                kind, r_t = "league_same_situation", np.full(len(teams), r_sit)
            else:
                # expected rate = team's overall rate shifted by the league-wide situation effect (logit scale)
                kind = "team_overall_plus_league_shift"
                da = all_mask & denom
                r_league_all = (da & hit).sum() / da.sum()
                tn = np.array([max(1, int((da & (team_arr == t)).sum())) for t in teams])
                tk = np.array([int((da & hit & (team_arr == t)).sum()) for t in teams])
                r_team = np.clip((tk + 0.5) / (tn + 1.0), 1e-4, 1 - 1e-4)
                r_t = 1 / (1 + np.exp(-(_logit(r_team) + _logit(r_sit) - _logit(r_league_all))))
            m = _mom_strength(k_t, n_t, r_t)
            for t, k, n, r in zip(teams, k_t, n_t, r_t):
                if n >= MIN_N:
                    out[t].append(_row("team", None, sit, head, cls, label, kind, float(r), int(k), int(n), m))

    # player scope: share of finishes while on the floor, situation vs the player's own overall share
    ok = df[df["lineup_ok"] & df["finisher_id"].notna()].reset_index(drop=True)
    pm = situation_masks(ok)
    ex = ok[["off_team_id", "off_lineup", "finisher_id"]].copy()
    ex["row"] = np.arange(len(ok))
    ex = ex.explode("off_lineup").rename(columns={"off_lineup": "player_id"})
    ex["player_id"] = ex["player_id"].astype("int64")
    ex["fin"] = (ex["finisher_id"] == ex["player_id"]).fillna(False).astype(int)
    overall = ex.groupby(["off_team_id", "player_id"])["fin"].agg(["sum", "count"])
    for sit in SITUATIONS[1:]:
        sub = ex[pm[sit[0]][ex["row"].values]]
        g = sub.groupby(["off_team_id", "player_id"])["fin"].agg(["sum", "count"])
        for (t, pid), row in g.iterrows():
            k, n = int(row["sum"]), int(row["count"])
            tot_k, tot_n = overall.loc[(t, pid)]
            r = tot_k / tot_n
            if n >= MIN_N and tot_n >= 4 * MIN_N and 0 < r < 1:
                name = player_names.get(int(pid), str(pid))
                out[int(t)].append(_row("player", int(pid), sit, "finisher", None, f"{name}: share of finishes while on the floor",
                                        "player_overall", r, k, n, M_PLAYER_FIXED))

    n_poss = df.groupby("off_team_id").size()
    n_games = df.groupby("off_team_id")["game_id"].nunique()
    reports = {}
    for t in teams:
        rows = sorted(out[t], key=lambda x: -x["rank_score"])[:top_k]
        reports[int(t)] = {"team_id": int(t), "n_games": int(n_games[t]), "n_possessions": int(n_poss[t]),
                           "n_candidates": len(out[t]),
                           "method": METHOD, "interval_level": INTERVAL_LEVEL, "sort": "rank_score_desc", "tendencies": rows}
    return reports
