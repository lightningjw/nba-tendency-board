"""Coach-decision heads (step 2.4): three decisions read off the dead-ball table.

Everything here describes what coaches DID at a stoppage. Nothing says what they should have done,
and nothing reads intent: a timeout is a timeout the feed charged, a substitution is a substitution.

* `timeout`: at a stoppage in regulation where no mandatory timeout is due and the team has one left,
  does it call a full timeout? One row per stoppage and team (after a made basket or free throw, only
  the team scored on). Baseline: the league's rate for that kind of stoppage, part of the game, and
  how many points the opponent has scored unanswered.
* `foul_sub`: a player on the floor picks up his first to fifth personal foul; is he taken out at
  that whistle? Baseline: the league's rate for that foul number in that period.
* `closing_five`: in a game within five points with five minutes left, is this player one of his
  team's five who play most of those minutes? Predicted as the third quarter ends, for everyone who
  has played. Baseline: how often he has closed for this team before, pulled toward the league's rate
  for starters or for bench players.

Every baseline and every "prior" feature is counted from games that tipped off strictly earlier
(earlier seasons at half weight), so a row never sees its own game or a later one. Each model is
LightGBM on top of its baseline's log-odds, on the same game splits as the possession heads.

Not built, and why (the profile numbers are in the manifest): late deliberate fouls (the scorer's
"take" word marks them, but it is one narrow situation and the step allows three heads), challenges
(the feed logs the timeout only: not what was challenged or whether it worked), and the next
substitution in general (who replaces whom is a much larger label space than one whistle supports).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from . import cdn_fields as F
from .models import evaluate, evaluate_sums, fit_binary
from .stoppages import CLOSING_WINDOW_S, RUN_POSSESSIONS, SOFT_KINDS, STOPPAGE_KINDS

DECISION_HEADS = ["timeout", "foul_sub", "closing_five"]
MODEL_ID = "lightgbm"
PAST_SEASON_WEIGHT = 0.5             # the same discount the possession features give earlier seasons
M_CELL = 50.0                        # pseudo-rows pulling a baseline cell toward the overall rate
M_TEAM = 100.0                       # pseudo-rows pulling a team's own lean toward none
M_PLAYER_CLOSE = 3.0                 # pseudo-games pulling a player's closing rate toward his role's
M_START = 20.0                       # pseudo-rows behind the starting guess below
START_RATE = {"timeout": 0.05, "foul_sub": 0.2, "closing_five": 0.5}   # only matters on the first nights of the first season
CLOSE_MARGIN = 5                     # a close game: within this many points when five minutes are left
MAX_FOUL = 5                         # the sixth foul is a disqualification, not a decision
MIN_TRAIN_ROWS = 200
REPORT_LEVEL = 0.90
PRIOR_MIN, PRIOR_MAX = 2.0, 200.0   # pseudo-rows the league's rate may weigh in a team's report row
TIMEOUT_KINDS = ["foul", "turnover", "violation", "jump_ball", "team_rebound", "made_basket", "made_free_throw"]
RUN_EDGES = [-1, 3, 6, 999]          # opponent's unanswered points: 0 to 3, 4 to 6, 7 or more
RUN_LABELS = ["0 to 3", "4 to 6", "7 or more"]

TIMEOUT_FEATURES = ["period", "clock_s", "game_secs_left", "margin", "timeouts_left", "opp_timeouts_left", "timeouts_this_period",
                    "secs_to_mandatory", "secs_since_timeout", "kind_code", "own_trigger", "opp_unanswered", "own_unanswered",
                    "opp_run_points", "own_run_points", "is_home", "team_lean"]
FOUL_FEATURES = ["foul_n", "period", "clock_s", "game_secs_left", "margin", "starter", "stint_s", "secs_played", "is_home",
                 "prior_secs", "team_lean"]
CLOSING_FEATURES = ["starter", "secs_pre", "secs_q3", "on_floor_q3", "fouls_q3", "margin_q3", "is_home", "pre_rank",
                    "prior_close_k", "prior_close_n", "prior_secs", "team_lean"]
FEATURES = {"timeout": TIMEOUT_FEATURES, "foul_sub": FOUL_FEATURES, "closing_five": CLOSING_FEATURES}
UNITS = {"timeout": "stoppages", "foul_sub": "fouls", "closing_five": "players in close games"}


# ---- counts from earlier games only ------------------------------------------------------
def asof_counts(df: pd.DataFrame, keys: list[str], ycol: str) -> tuple[np.ndarray, np.ndarray]:
    """For each row, (sum of `ycol`, number of rows) over rows with the same keys in games that tipped off strictly earlier.

    This season counts in full and earlier seasons at PAST_SEASON_WEIGHT. Games with the same tip time
    never count for each other.
    """
    d = pd.DataFrame({k: df[k].values for k in keys})
    d["_all"], d["season"], d["tipoff"], d["_y"] = 0, df["season"].values, df["tipoff"].values, df[ycol].astype(float).values
    ks = ["_all"] + keys
    g = d.groupby(ks + ["season", "tipoff"], sort=False, dropna=False)["_y"].agg(["sum", "size"]).reset_index().sort_values("tipoff", kind="stable")
    grp = g.groupby(ks + ["season"], sort=False, dropna=False)
    g["k_s"], g["n_s"] = grp["sum"].cumsum() - g["sum"], grp["size"].cumsum() - g["size"]
    tot = g.groupby(ks + ["season"], sort=False, dropna=False)[["sum", "size"]].sum().reset_index().sort_values("season", kind="stable")
    grp2 = tot.groupby(ks, sort=False, dropna=False)
    tot["k_p"], tot["n_p"] = grp2["sum"].cumsum() - tot["sum"], grp2["size"].cumsum() - tot["size"]
    g = g.merge(tot[ks + ["season", "k_p", "n_p"]], on=ks + ["season"], how="left")
    out = d.merge(g[ks + ["season", "tipoff", "k_s", "n_s", "k_p", "n_p"]], on=ks + ["season", "tipoff"], how="left")
    return (out["k_s"] + PAST_SEASON_WEIGHT * out["k_p"]).values, (out["n_s"] + PAST_SEASON_WEIGHT * out["n_p"]).values


def _cell_baseline(df: pd.DataFrame, cell_keys: list[str], head: str) -> np.ndarray:
    """The league's rate in the row's cell from earlier games, pulled toward the league's overall rate from earlier games."""
    k0, n0 = asof_counts(df, [], "y")
    overall = (k0 + M_START * START_RATE[head]) / (n0 + M_START)
    k, n = asof_counts(df, cell_keys, "y")
    return np.clip((k + M_CELL * overall) / (n + M_CELL), 1e-4, 1 - 1e-4)


def _team_lean(df: pd.DataFrame) -> np.ndarray:
    """How much more often than its baseline this team has done it in earlier games (0 = like the league)."""
    tmp = df[["team_id", "season", "tipoff"]].copy()
    tmp["r"] = df["y"].astype(float).values - df["p_base"].values
    k, n = asof_counts(tmp, ["team_id"], "r")
    return k / (n + M_TEAM)


def _tip(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["tipoff_utc"], utc=True, format="ISO8601").dt.tz_localize(None)


def _game_secs_left(period: pd.Series, clock_s: pd.Series) -> pd.Series:
    return clock_s + (4 - period.clip(upper=4)) * F.REGULATION_PERIOD_SECONDS


# ---- decision 1: a timeout at a stoppage ------------------------------------------------------
def timeout_rows(stops: pd.DataFrame) -> pd.DataFrame:
    """One row per stoppage and team that could choose to call a full timeout there."""
    s = stops[stops["kind"].isin(TIMEOUT_KINDS) & ~stops["mandatory_due"] & (stops["period"] <= 4)]
    frames = []
    for side, opp in (("home", "away"), ("away", "home")):
        team = s[f"{side}_team_id"]
        d = pd.DataFrame({
            "game_id": s["game_id"], "season": s["season"], "tipoff_utc": s["tipoff_utc"], "stop_idx": s["stop_idx"], "team_id": team,
            "opp_team_id": s[f"{opp}_team_id"], "kind": s["kind"], "period": s["period"], "clock_s": s["clock_s"],
            "game_secs_left": _game_secs_left(s["period"], s["clock_s"]),
            "margin": s[f"{side}_score"] - s[f"{opp}_score"],
            "timeouts_left": (F.TIMEOUTS_PER_GAME - s[f"{side}_timeouts_used"]).clip(lower=0),
            "opp_timeouts_left": (F.TIMEOUTS_PER_GAME - s[f"{opp}_timeouts_used"]).clip(lower=0),
            "timeouts_this_period": s["timeouts_this_period"],
            "secs_to_mandatory": np.where(s["timeouts_this_period"] < 1, s["clock_s"] - F.MANDATORY_FIRST_S,
                                          np.where(s["timeouts_this_period"] < 2, s["clock_s"] - F.MANDATORY_SECOND_S, 999.0)),
            "secs_since_timeout": s["secs_since_timeout"].astype(float).fillna(-1.0),
            "kind_code": s["kind"].map({k: i for i, k in enumerate(STOPPAGE_KINDS)}),
            "own_trigger": (s["trigger_team_id"] == team).astype(int),
            "opp_unanswered": s[f"{opp}_unanswered"], "own_unanswered": s[f"{side}_unanswered"],
            "opp_run_points": s[f"{opp}_run_points"], "own_run_points": s[f"{side}_run_points"],
            "is_home": int(side == "home"), "y": s[f"{side}_timeout"].astype(int)})
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    scored = d["kind"].isin(SOFT_KINDS) & (d["own_trigger"] == 1)        # the team that just scored does not inbound
    d = d[(d["timeouts_left"] > 0) & ~scored].copy()
    d["tipoff"] = _tip(d)
    d = d.sort_values(["tipoff", "game_id", "stop_idx", "is_home"], kind="stable").reset_index(drop=True)
    d["run_bucket"] = pd.cut(d["opp_unanswered"], RUN_EDGES, labels=False).astype(int)
    d["kind_group"] = np.where(d["kind"].isin(SOFT_KINDS), 0, np.where(d["kind"] == "foul", 1, 2))
    d["phase"] = np.where((d["period"] == 4) & (d["clock_s"] <= 180), 2, np.where(d["period"] == 4, 1, 0))
    d["p_base"] = _cell_baseline(d, ["kind_group", "phase", "run_bucket"], "timeout")
    d["team_lean"] = _team_lean(d)
    return d


# ---- decision 2: a player sits after a foul --------------------------------------------------
def foul_rows(stops: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """One row per personal foul (first to fifth) by a player on the floor; y = taken out before the clock ran again."""
    s = stops.sort_values(["game_id", "stop_idx"], kind="stable").reset_index(drop=True)
    gid, per, clk = s["game_id"].values, s["period"].values, s["clock_s"].values
    out = []
    has = np.flatnonzero(s["foul_player_ids"].str.len().values > 0)
    cols = {c: s[c].values for c in ("foul_player_ids", "foul_team_ids", "foul_counts", "foul_technical", "home_team_id", "away_team_id",
                                     "home_lineup", "away_lineup", "home_stint_s", "away_stint_s", "home_secs_played", "away_secs_played",
                                     "home_subs_out", "away_subs_out", "home_score", "away_score", "season", "tipoff_utc", "stop_idx", "kind")}
    for i in has:
        if cols["kind"][i] == "period_break":
            continue
        gone: set[int] = set()
        j = i
        while j < len(s) and gid[j] == gid[i] and per[j] == per[i] and clk[j] == clk[i]:   # the free throws of this whistle split the row
            gone |= set(cols["home_subs_out"][j]) | set(cols["away_subs_out"][j])
            j += 1
        for p, tm, n, tech in zip(cols["foul_player_ids"][i], cols["foul_team_ids"][i], cols["foul_counts"][i], cols["foul_technical"][i]):
            side = "home" if tm == cols["home_team_id"][i] else "away"
            five = list(cols[f"{side}_lineup"][i])
            if tech or n > MAX_FOUL or p not in five:
                continue
            k = five.index(p)
            opp = "away" if side == "home" else "home"
            out.append((gid[i], cols["season"][i], cols["tipoff_utc"][i], cols["stop_idx"][i], int(tm), int(p), int(n), int(min(per[i], 5)),
                        float(clk[i]), int(cols[f"{side}_score"][i] - cols[f"{opp}_score"][i]), float(cols[f"{side}_stint_s"][i][k]),
                        float(cols[f"{side}_secs_played"][i][k]), int(side == "home"), int(p in gone)))
    d = pd.DataFrame(out, columns=["game_id", "season", "tipoff_utc", "stop_idx", "team_id", "player_id", "foul_n", "period", "clock_s", "margin",
                                   "stint_s", "secs_played", "is_home", "y"])
    d["tipoff"] = _tip(d)
    d["game_secs_left"] = _game_secs_left(d["period"], d["clock_s"])
    d = d.merge(_player_priors(players), on=["game_id", "player_id"], how="left")
    d["starter"] = d["starter"].fillna(False).astype(int)
    d["prior_secs"] = d["prior_secs"].fillna(0.0)
    d = d.sort_values(["tipoff", "game_id", "stop_idx", "player_id"], kind="stable").reset_index(drop=True)
    d["p_base"] = _cell_baseline(d, ["foul_n", "period"], "foul_sub")
    d["team_lean"] = _team_lean(d)
    return d


def _player_priors(players: pd.DataFrame) -> pd.DataFrame:
    """Per player and game: starter tonight, and his average seconds a game in earlier games (one 20-minute game of padding)."""
    p = players[["game_id", "player_id", "season", "tipoff_utc", "starter", "secs_total"]].copy()
    p["tipoff"] = _tip(p)
    played = p[p["secs_total"] > 0]
    k, n = asof_counts(played, ["player_id"], "secs_total")
    prior = pd.DataFrame({"game_id": played["game_id"].values, "player_id": played["player_id"].values, "prior_secs": (k + 1200.0) / (n + 1.0)})
    return p[["game_id", "player_id", "starter"]].merge(prior, on=["game_id", "player_id"], how="left")


# ---- decision 3: who closes a close game ------------------------------------------------------
def closing_rows(players: pd.DataFrame) -> pd.DataFrame:
    """One row per player who had played before the fourth quarter of a game that was close with five minutes left."""
    p = players[players["margin_close"].notna() & (players["margin_close"].abs() <= CLOSE_MARGIN)].copy()
    p = p.sort_values(["game_id", "team_id", "secs_close", "secs_total", "player_id"], ascending=[True, True, False, False, True], kind="stable")
    p["y"] = ((p.groupby(["game_id", "team_id"]).cumcount() < 5) & (p["secs_close"] > 0)).astype(int)
    p["secs_pre"] = p["secs_q1"] + p["secs_q2"] + p["secs_q3"]
    d = p[(p["secs_pre"] > 0) & (p["fouls_q3"].fillna(0) < 6)].copy()
    d["tipoff"] = _tip(d)
    d = d.merge(_player_priors(players)[["game_id", "player_id", "prior_secs"]], on=["game_id", "player_id"], how="left")
    d["prior_secs"] = d["prior_secs"].fillna(0.0)
    d = d.sort_values(["tipoff", "game_id", "team_id", "player_id"], kind="stable").reset_index(drop=True)
    d["pre_rank"] = d.groupby(["game_id", "team_id"])["secs_pre"].rank(method="first", ascending=False)
    for c in ("starter", "on_floor_q3", "is_home"):
        d[c] = d[c].fillna(False).astype(int)
    d["fouls_q3"] = d["fouls_q3"].fillna(0).astype(int)
    d["margin_q3"] = d["margin_q3"].fillna(0).astype(int)
    k0, n0 = asof_counts(d, [], "y")
    kr, nr = asof_counts(d, ["starter"], "y")
    role = (kr + M_START * (k0 + M_START * START_RATE["closing_five"]) / (n0 + M_START)) / (nr + M_START)
    kp, npl = asof_counts(d, ["team_id", "player_id"], "y")
    d["prior_close_k"], d["prior_close_n"] = kp, npl
    d["p_base"] = np.clip((kp + M_PLAYER_CLOSE * role) / (npl + M_PLAYER_CLOSE), 1e-4, 1 - 1e-4)
    d["team_lean"] = _team_lean(d)
    return d


def decision_tables(stops: pd.DataFrame, players: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"timeout": timeout_rows(stops), "foul_sub": foul_rows(stops, players), "closing_five": closing_rows(players)}


# ---- models ----------------------------------------------------------------------------------
@dataclass
class CoachRun:
    heads: list[dict]                      # scorecard entries, one per decision head
    tables: dict[str, pd.DataFrame]        # the decision rows, with p_base and (where a model was fitted) p_model
    rounds: dict = field(default_factory=dict)


def run_coach(stops: pd.DataFrame, players: pd.DataFrame, splits: pd.Series, log=print) -> CoachRun:
    """Fit and score the three heads on the same game splits as the possession heads."""
    tables = decision_tables(stops, players)
    heads, rounds = [], {}
    for head in DECISION_HEADS:
        d = tables[head]
        part = d["game_id"].map(splits).fillna("unused").values
        y, base = d["y"].values.astype(int), d["p_base"].values
        tr = part == "train"
        reason = None
        if tr.sum() < MIN_TRAIN_ROWS or len(np.unique(y[tr])) < 2:
            reason = f"too few training rows for this decision ({int(tr.sum())}; it needs {MIN_TRAIN_ROWS} with both outcomes)"
            heads.append({"head": head, "model_id": MODEL_ID, "splits": [evaluate_sums([], s, reason) for s in ("holdout_games", "holdout_season")]})
            log(f"coach: {head}: not fitted, {reason}")
            continue
        X = d[FEATURES[head]].astype(np.float32)
        bst, r = fit_binary(X, y, base, part)
        b = np.clip(base, 1e-4, 1 - 1e-4)
        p = 1.0 / (1.0 + np.exp(-(bst.predict(X, num_iteration=r, raw_score=True) + np.log(b / (1 - b)))))
        d["p_model"] = p
        rounds[head] = r
        P, B = np.column_stack([1 - p, p]), np.column_stack([1 - b, b])
        heads.append({"head": head, "model_id": MODEL_ID, "splits": [
            evaluate(P[m], B[m], y[m], d["game_id"].values[m], s) for s in ("holdout_games", "holdout_season") for m in [part == s]]})
        log(f"coach: {head}: {r} rounds, {int(tr.sum()):,} training rows, {y[tr].mean():.1%} yes")
    return CoachRun(heads, tables, rounds)


# ---- what the manifest says about the table ---------------------------------------------------------
KIND_LABELS = {"foul": "Foul", "turnover": "Turnover with no steal", "violation": "Violation", "jump_ball": "Jump ball",
               "team_rebound": "Missed shot nobody secured", "made_basket": "Made basket", "made_free_throw": "Made last free throw",
               "timeout_live": "Timeout called in live play", "period_break": "Between periods", "unlogged": "Substitution with no whistle logged"}


def manifest_heads() -> list[dict]:
    return [
        {"id": "timeout", "label": "Timeout at a stoppage", "classes": ["no", "yes"], "class_labels": ["No timeout", "Timeout"],
         "conditional_on": "play has stopped in regulation, no mandatory timeout is due, and the team has a timeout left (after a made basket or "
                           "free throw, only the team scored on)",
         "baseline": "The league's rate in earlier games for that kind of stoppage, part of the game, and opponent's unanswered points "
                     "(0 to 3, 4 to 6, 7 or more)",
         "kind": "decision", "unit": UNITS["timeout"], "logged_before_tipoff": False},
        {"id": "foul_sub", "label": "A player sits after a foul", "classes": ["no", "yes"], "class_labels": ["Stays in", "Taken out"],
         "conditional_on": "a player on the floor commits his first to fifth personal foul; taken out means before the clock runs again",
         "baseline": "The league's rate in earlier games for that foul number in that period",
         "kind": "decision", "unit": UNITS["foul_sub"], "logged_before_tipoff": False},
        {"id": "closing_five", "label": "Who closes a close game", "classes": ["no", "yes"], "class_labels": ["Does not close", "Closes"],
         "conditional_on": f"the game is within {CLOSE_MARGIN} points with five minutes left in the fourth quarter; predicted as the third quarter "
                           "ends, for each player who has played; closing means being one of the team's five with the most of those minutes",
         "baseline": "How often the player has closed for this team in earlier close games, pulled toward the league's rate for starters or for bench players",
         "kind": "decision", "unit": UNITS["closing_five"], "logged_before_tipoff": False},
    ]


def stoppage_profile(stops: pd.DataFrame, checks: dict, tables: dict[str, pd.DataFrame]) -> dict:
    """What the dead-ball table holds, the one check the feed allows, and the decisions that were not modelled."""
    n_games = int(stops["game_id"].nunique())
    counts = stops["kind"].value_counts()
    n_take = int(stops["take_foul_team_id"].notna().sum())
    late = stops[stops["take_foul_team_id"].notna() & (stops["period"] >= 4) & (stops["clock_s"] <= 60)]
    n_chal = int((stops["home_challenge"] | stops["away_challenge"]).sum())
    n_subs = int(stops["home_subs_out"].str.len().sum() + stops["away_subs_out"].str.len().sum())
    n_to = int(stops["home_timeout"].sum() + stops["away_timeout"].sum())
    n_due = int(((stops["home_timeout"] | stops["away_timeout"]) & stops["mandatory_due"]).sum())
    n_live = int(((stops["home_timeout"] | stops["away_timeout"]) & (stops["kind"] == "timeout_live")).sum())
    return {
        "n_stoppages": int(len(stops)), "n_games": n_games, "run_possessions": RUN_POSSESSIONS,
        "kinds": [{"kind": k, "label": KIND_LABELS[k], "n": int(counts.get(k, 0))} for k in STOPPAGE_KINDS],
        "personal_foul_check": {"n": int(checks.get("pf_checks", 0)), "agree": int(checks.get("pf_agree", 0))},
        "timeouts": {"n_full": n_to, "n_when_mandatory_due": n_due, "n_live_ball": n_live},
        "decision_rows": [{"head": h, "unit": UNITS[h], "n": int(len(tables[h])), "n_yes": int(tables[h]["y"].sum())} for h in DECISION_HEADS],
        "not_modelled": [
            {"id": "late_fouls", "label": "Late deliberate fouls", "n": n_take,
             "why": f"The scorer's \"take\" word marks them cleanly ({len(late):,} of {n_take:,} here came in the last minute of the fourth quarter or "
                    "overtime), but it is one narrow situation that the score and the clock nearly decide, and the step allows three heads."},
            {"id": "challenges", "label": "Challenges", "n": n_chal,
             "why": "The feed logs a challenge as a timeout and nothing else: not which call was challenged and not whether it succeeded. "
                    f"There are {n_chal / max(1, n_games):.2f} a game here, too few to model without knowing what was at stake."},
            {"id": "next_substitution", "label": "The next substitution in general", "n": n_subs,
             "why": "Who comes in for whom is a choice among every pairing on the roster. The foul-trouble head is the one slice of it with a "
                    "yes-or-no label at a single whistle."},
        ],
    }


# ---- the team report section ------------------------------------------------------------------------
def _eb_rows(cells: list[tuple[str, str, pd.Series]], d: pd.DataFrame, team_id: int, level: float) -> list[dict]:
    """Team rate against the league's in each cell, with a beta posterior that borrows from the league.

    The prior's strength comes from how much teams differ in that cell (method of moments), so a cell
    where every team behaves alike pulls a thin team sample hard, and one where teams differ pulls less.
    It is capped at PRIOR_MAX pseudo-rows and at half the league's own sample: thirty teams with nine
    fouls each cannot show that teams are identical, so a thin sample is always left some room.
    """
    out = []
    lo, hi = (1 - level) / 2, (1 + level) / 2
    for cid, label, mask in cells:
        c = d[mask.values]
        n_l, k_l = len(c), int(c["y"].sum())
        t = c[c["team_id"] == team_id]
        n, k = len(t), int(t["y"].sum())
        if n_l == 0:
            continue
        p = k_l / n_l
        g = c.groupby("team_id")["y"].agg(["sum", "size"])
        cap = max(PRIOR_MIN, min(PRIOR_MAX, n_l / 2))         # the league's rate is itself a count: never lean on it harder than that
        m = min(50.0, cap)
        if len(g) > 2 and 0 < p < 1:
            ni, pi = g["size"].values.astype(float), g["sum"].values / g["size"].values
            denom = ni.sum() - (ni ** 2).sum() / ni.sum()
            tau2 = ((ni * (pi - p) ** 2).sum() - (len(g) - 1) * p * (1 - p)) / denom if denom > 0 else 0.0
            m = float(np.clip(p * (1 - p) / tau2 - 1, PRIOR_MIN, cap)) if tau2 > 1e-9 else cap
        a, b = k + m * p, n - k + m * (1 - p)
        out.append({"id": cid, "label": label, "n": n, "k": k, "raw_rate": round(k / n, 4) if n else None,
                    "rate": round(a / (a + b), 4), "interval": [round(float(beta_dist.ppf(lo, a, b)), 4), round(float(beta_dist.ppf(hi, a, b)), 4)],
                    "league_rate": round(p, 4), "league_n": n_l, "prior_strength": round(m, 1)})
    return out


FOUL_CELLS = [("foul_2_q1", "Second foul in the first quarter", 2, (1,)), ("foul_2_q2", "Second foul in the second quarter", 2, (2,)),
              ("foul_3_h1", "Third foul in the first half", 3, (1, 2)), ("foul_3_q3", "Third foul in the third quarter", 3, (3,)),
              ("foul_4_q3", "Fourth foul in the third quarter", 4, (3,)), ("foul_4_q4", "Fourth foul in the fourth quarter", 4, (4,)),
              ("foul_5_q4", "Fifth foul in the fourth quarter", 5, (4,))]
COACH_NOTE = ("What this team's bench did at stoppages, counted from play-by-play. These are descriptions, not grades: nothing here says a "
              "timeout or a substitution was right or wrong. Each rate is pulled toward the league's by how much teams differ, and its interval says "
              "how far the count can be trusted.")


def team_sections(tables: dict[str, pd.DataFrame], season: int, names: dict[int, str] | None = None, level: float = REPORT_LEVEL) -> dict[int, dict]:
    """team_id -> the `coach_decisions` section of its report, for one season."""
    t = tables["timeout"][tables["timeout"]["season"] == season]
    f = tables["foul_sub"][tables["foul_sub"]["season"] == season]
    c = tables["closing_five"][tables["closing_five"]["season"] == season]
    t_cells = [(f"timeout_run_{i}", f"Opponent has scored {lab} unanswered", t["run_bucket"] == i) for i, lab in enumerate(RUN_LABELS)]
    f_cells = [(cid, lab, (f["foul_n"] == n) & f["period"].isin(per)) for cid, lab, n, per in FOUL_CELLS]
    lo, hi = (1 - level) / 2, (1 + level) / 2
    out = {}
    for team in sorted(set(t["team_id"]) | set(f["team_id"]) | set(c["team_id"])):
        ct = c[c["team_id"] == team]
        g = ct.groupby("player_id")["y"].agg(["sum", "size"]).reset_index().sort_values(["sum", "size", "player_id"], ascending=[False, False, True])
        closers = [{"player_id": int(r.player_id), "n": int(r.size), "k": int(r.sum), "rate": round(r.sum / r.size, 4),
                    "interval": [round(float(beta_dist.ppf(lo, r.sum + 0.5, r.size - r.sum + 0.5)), 4),
                                 round(float(beta_dist.ppf(hi, r.sum + 0.5, r.size - r.sum + 0.5)), 4)]}
                   for r in g.head(8).itertuples(index=False)]
        out[int(team)] = {"note": COACH_NOTE, "interval_level": level, "season": int(season),
                          "timeouts": _eb_rows(t_cells, t, team, level), "foul_trouble": _eb_rows(f_cells, f, team, level),
                          "closing": {"n_close_games": int(ct["game_id"].nunique()), "close_margin": CLOSE_MARGIN,
                                      "window_s": CLOSING_WINDOW_S, "players": closers}}
    return out
