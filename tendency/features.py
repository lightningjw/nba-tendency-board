"""Features and baselines.

Three kinds of column, and every model feature must be registered as exactly one:

* context  - known from the play-by-play before the possession's deciding event
             (clock, score, fouls, timeouts, how the possession started).
* pregame  - aggregates over games that tipped off STRICTLY BEFORE this game.
             Built in DuckDB with `b.tipoff < a.tipoff` joins, so nothing from the
             same game or a later one can enter.
* ingame   - counts over EARLIER possessions of the same game only.

tests/test_leakage.py scrambles labels from a game onward (and from a possession
onward) and fails if any pregame (or ingame) column moves.

No feature describes a play call or a defensive coverage; play-by-play has neither.

Step 2.2 added two heads (action style, tempo). Their labels describe how a possession went, so they
enter features only as pregame aggregates over earlier games or in-game counts over earlier
possessions, exactly like the older labels. The three older heads keep their feature lists unchanged:
a model frozen before step 2.2 still gets the columns it was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from .possessions import ACTION_STYLES, ENDING_CLASSES, SHOT_FAMILIES, START_TYPES, TEMPO_CLASSES

M_TEAM = 200.0        # pseudo-possessions pulling a team's shares toward the league's
M_TEAM_TEMPO = 100.0  # same, for a team's tempo mix within one start type (a smaller pool than all possessions)
M_PLAYER = 100.0      # pseudo-possessions pulling a player's usage toward 1/5
M_PLAYER_MIX = 30.0   # pseudo-events pulling a player's mixes toward the league's
PAST_SEASON_WEIGHT = 0.5

E = [f"e_{c}" for c in ENDING_CLASSES]
FA = [f"f_{c}" for c in SHOT_FAMILIES]
SY = [f"s_{c}" for c in ACTION_STYLES]
TM = [f"t_{st}_{c}" for st in START_TYPES for c in TEMPO_CLASSES]      # tempo counts by how the possession started
US = [f"u_{st}_{c}" for st in START_TYPES for c in ACTION_STYLES]      # style counts by start type (league level only)

CONTEXT = ["period", "start_clock_s", "game_secs_left", "margin_start", "off_is_home", "def_in_penalty",
           "def_team_fouls_period", "off_team_fouls_period", "off_timeouts_left", "def_timeouts_left",
           "after_timeout", "start_type_code", "clutch"]
PREGAME_POSS = ([f"base_end_{c}" for c in ENDING_CLASSES] + [f"base_fam_{c}" for c in SHOT_FAMILIES]
                + ["team_eff_n", "lineup_usage_sum"]
                + [f"lineup_end_{c}" for c in ENDING_CLASSES] + [f"lineup_fam_{c}" for c in SHOT_FAMILIES])
INGAME_POSS = ["ig_team_poss"] + [f"ig_end_{c}" for c in ENDING_CLASSES] + [f"ig_fam_{c}" for c in SHOT_FAMILIES]
PREGAME_CAND = ["p_usage", "p_base", "p_eff_on", "p_usage_rank"] + [f"p_end_{c}" for c in ENDING_CLASSES]
INGAME_CAND = ["ig_p_fin", "ig_p_on", "ig_p_last"]

# ---- step 2.2 heads: their own lists, so the three above are untouched ----
CONTEXT_22 = ["two_for_one_window"]                                    # known when the possession starts
PREGAME_STYLE = ([f"base_sty_{c}" for c in ACTION_STYLES] + [f"lineup_sty_{c}" for c in ACTION_STYLES]
                 + [f"start_sty_{c}" for c in ACTION_STYLES])            # start_sty_* is the action-style baseline
PREGAME_TEMPO = [f"base_tmp_{c}" for c in TEMPO_CLASSES]
INGAME_STYLE = [f"ig_sty_{c}" for c in ACTION_STYLES]
INGAME_TEMPO = [f"ig_tmp_{c}" for c in TEMPO_CLASSES]
# Most context is read just before the deciding event (possessions.py), which is right for "what happens next".
# Tempo asks WHEN the first action comes, so anything that changed mid-possession (a timeout, a foul, a
# substitution) is a clue about elapsed time. The tempo head may only see what was known at the start:
START_KNOWN = ["period", "start_clock_s", "game_secs_left", "margin_start", "off_is_home", "start_type_code", "clutch", "two_for_one_window"]
PREGAME_TEAM = [c for c in PREGAME_POSS if not c.startswith("lineup_")]      # the five on the floor are read at the deciding event too
# What the tempo head cannot read at its own deciding event it can still read one possession back: the
# previous possession's fouls, timeouts and penalty state are the best view of where things stood when this
# one began, and how the previous one ended is known by then too. Added in step 2.3, after the transformer
# beat the trees on tempo for exactly this reason (it gets the previous possession as a token).
LAG_SNAPSHOT = ["def_in_penalty", "def_team_fouls_period", "off_team_fouls_period", "off_timeouts_left", "def_timeouts_left",
                "after_timeout", "off_is_home"]
LAG_TEMPO = [f"lag_{c}" for c in LAG_SNAPSHOT] + ["lag_ending", "lag_tempo"]
PREGAME_POSS_ALL = PREGAME_POSS + PREGAME_STYLE + PREGAME_TEMPO
INGAME_POSS_ALL = INGAME_POSS + INGAME_STYLE + INGAME_TEMPO + LAG_TEMPO
LABELS_22 = ["action_style", "tempo"]

FEATURES = {
    "ending_type": CONTEXT + PREGAME_POSS + INGAME_POSS,
    "shot_family": CONTEXT + PREGAME_POSS + INGAME_POSS,
    "finisher": CONTEXT + PREGAME_CAND + INGAME_CAND,
    "action_style": CONTEXT + CONTEXT_22 + PREGAME_POSS + PREGAME_STYLE + INGAME_POSS + INGAME_STYLE,
    "tempo": START_KNOWN + PREGAME_TEAM + PREGAME_TEMPO + INGAME_POSS + INGAME_TEMPO + LAG_TEMPO,
}
KIND = {**{c: "context" for c in CONTEXT + CONTEXT_22},
        **{c: "pregame" for c in PREGAME_POSS_ALL + PREGAME_CAND},
        **{c: "ingame" for c in INGAME_POSS_ALL + INGAME_CAND}}


def with_label_columns(poss: pd.DataFrame) -> pd.DataFrame:
    """Add the step 2.2 columns when a caller's table lacks them (the pregame log's reference rows do)."""
    missing = {c: None for c in LABELS_22 if c not in poss.columns}
    if "two_for_one_window" not in poss.columns:
        missing["two_for_one_window"] = False
    return poss.assign(**missing) if missing else poss


@dataclass
class Features:
    poss: pd.DataFrame      # one row per model-eligible possession
    cand: pd.DataFrame      # five rows per model-eligible possession (one per offensive player)


def _sums(alias: str, cols: list[str], prefix: str, where: str = "") -> str:
    flt = f" FILTER (WHERE {where})" if where else ""
    return ", ".join(f"COALESCE(SUM({alias}.{c}){flt}, 0) AS {prefix}{c}" for c in cols)


def pregame_tables(poss: pd.DataFrame, only_games: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Team, league and player aggregates over strictly earlier games.

    `only_games` limits the games the aggregates are computed FOR (the left side of each join); the
    history they are computed FROM is always every game in `poss`.
    """
    cols = ["game_id", "season", "tipoff", "off_team_id", "off_lineup", "finisher_id", "ending_type",
            "shot_family", "is_heave", "lineup_ok", "action_style", "tempo", "start_type"]
    con = duckdb.connect()
    try:
        con.register("p", pa.Table.from_pandas(poss[cols], preserve_index=False))
        if only_games is None:
            con.execute("CREATE TEMP VIEW target AS SELECT * FROM p")
        else:
            con.register("want", pa.table({"game_id": pa.array(sorted(set(only_games)), type=pa.string())}))
            con.execute("CREATE TEMP VIEW target AS SELECT p.* FROM p JOIN want USING (game_id)")
        con.execute("CREATE TEMP TABLE e AS SELECT * FROM p WHERE ending_type IN ('shot','foul_drawn','turnover') AND NOT is_heave")
        e_cases = ", ".join(f"SUM(CASE WHEN ending_type='{c}' THEN 1 ELSE 0 END) AS e_{c}" for c in ENDING_CLASSES)
        f_cases = ", ".join(f"SUM(CASE WHEN shot_family='{c}' THEN 1 ELSE 0 END) AS f_{c}" for c in SHOT_FAMILIES)
        s_cases = ", ".join(f"SUM(CASE WHEN action_style='{c}' THEN 1 ELSE 0 END) AS s_{c}" for c in ACTION_STYLES)
        t_cases = ", ".join(f"SUM(CASE WHEN start_type='{st}' AND tempo='{c}' THEN 1 ELSE 0 END) AS t_{st}_{c}"
                            for st in START_TYPES for c in TEMPO_CLASSES)
        u_cases = ", ".join(f"SUM(CASE WHEN start_type='{st}' AND action_style='{c}' THEN 1 ELSE 0 END) AS u_{st}_{c}"
                            for st in START_TYPES for c in ACTION_STYLES)
        con.execute(f"""CREATE TEMP TABLE tg AS
            SELECT game_id, season, tipoff, off_team_id AS team_id, COUNT(*) AS n, {e_cases}, {f_cases}, {s_cases}, {t_cases}, {u_cases}
            FROM e GROUP BY game_id, season, tipoff, off_team_id""")
        tcols = ["n"] + E + FA + SY + TM
        team = con.execute(f"""
            SELECT a.game_id, a.team_id, {_sums('b', tcols, 'tp_')}, {_sums('b', tcols, 'ts_', 'b.season = a.season')}
            FROM (SELECT DISTINCT game_id, season, tipoff, off_team_id AS team_id FROM target) a
            LEFT JOIN tg b ON b.team_id = a.team_id AND b.tipoff < a.tipoff
            GROUP BY a.game_id, a.team_id""").df()
        league = con.execute(f"""
            SELECT a.game_id, {_sums('b', tcols + US, 'lp_')}
            FROM (SELECT DISTINCT game_id, tipoff FROM target) a
            LEFT JOIN tg b ON b.tipoff < a.tipoff
            GROUP BY a.game_id""").df()
        fe = ", ".join(f"SUM(CASE WHEN finisher_id = player_id AND ending_type='{c}' THEN 1 ELSE 0 END) AS e_{c}" for c in ENDING_CLASSES)
        ff = ", ".join(f"SUM(CASE WHEN finisher_id = player_id AND shot_family='{c}' THEN 1 ELSE 0 END) AS f_{c}" for c in SHOT_FAMILIES)
        fs = ", ".join(f"SUM(CASE WHEN finisher_id = player_id AND action_style='{c}' THEN 1 ELSE 0 END) AS s_{c}" for c in ACTION_STYLES)
        con.execute(f"""CREATE TEMP TABLE pg AS
            WITH x AS (SELECT game_id, season, tipoff, finisher_id, ending_type, shot_family, action_style, UNNEST(off_lineup) AS player_id
                       FROM e WHERE lineup_ok)
            SELECT game_id, season, tipoff, player_id, COUNT(*) AS on_floor,
                   SUM(CASE WHEN finisher_id = player_id THEN 1 ELSE 0 END) AS fin, {fe}, {ff}, {fs}
            FROM x GROUP BY game_id, season, tipoff, player_id""")
        pcols = ["on_floor", "fin"] + E + FA + SY
        player = con.execute(f"""
            WITH a AS (SELECT DISTINCT game_id, season, tipoff, UNNEST(off_lineup) AS player_id FROM target)
            SELECT a.game_id, a.player_id, {_sums('b', pcols, 'pp_')}, {_sums('b', pcols, 'ps_', 'b.season = a.season')}
            FROM a LEFT JOIN pg b ON b.player_id = a.player_id AND b.tipoff < a.tipoff
            GROUP BY a.game_id, a.player_id""").df()
    finally:
        con.close()
    return {"team": team, "league": league, "player": player}


def _eff(df: pd.DataFrame, season_prefix: str, all_prefix: str, col: str) -> pd.Series:
    """Season-to-date count plus down-weighted earlier seasons."""
    s, a = df[season_prefix + col].astype(float), df[all_prefix + col].astype(float)
    return s + PAST_SEASON_WEIGHT * (a - s)


def build_features(poss: pd.DataFrame, only_games: list[str] | None = None) -> Features:
    """Features for every game in `poss`, or only for `only_games` (with all of `poss` as their history)."""
    poss = with_label_columns(poss).sort_values(["tipoff", "game_id", "poss_idx"]).reset_index(drop=True)
    pt = pregame_tables(poss, only_games)
    if only_games is not None:
        poss = poss[poss["game_id"].isin(set(only_games))].reset_index(drop=True)
        if poss.empty:
            raise ValueError("none of the requested games are in the possession table")

    # ---- league shares before each game (Laplace-smoothed) -----------------------
    lg = pt["league"].copy()
    lg_shots = lg[[f"lp_{c}" for c in FA]].sum(axis=1)
    for c in ENDING_CLASSES:
        lg[f"L_end_{c}"] = (lg[f"lp_e_{c}"] + 1.0) / (lg["lp_n"] + len(ENDING_CLASSES))
    for c in SHOT_FAMILIES:
        lg[f"L_fam_{c}"] = (lg[f"lp_f_{c}"] + 1.0) / (lg_shots + len(SHOT_FAMILIES))
    lg_styled = lg[[f"lp_{c}" for c in SY]].sum(axis=1)
    for c in ACTION_STYLES:
        lg[f"L_sty_{c}"] = (lg[f"lp_s_{c}"] + 1.0) / (lg_styled + len(ACTION_STYLES))
    for st in START_TYPES:
        tot = lg[[f"lp_t_{st}_{c}" for c in TEMPO_CLASSES]].sum(axis=1)
        for c in TEMPO_CLASSES:
            lg[f"L_tmp_{st}_{c}"] = (lg[f"lp_t_{st}_{c}"] + 1.0) / (tot + len(TEMPO_CLASSES))
    # how the league's style mix shifts with the way a possession started: a ratio per (start type, style), before each game
    shift = np.empty((len(lg), len(START_TYPES), len(ACTION_STYLES)))
    for i, st in enumerate(START_TYPES):
        tot = lg[[f"lp_u_{st}_{c}" for c in ACTION_STYLES]].sum(axis=1).values
        for j, c in enumerate(ACTION_STYLES):
            shift[:, i, j] = ((lg[f"lp_u_{st}_{c}"].values + 1.0) / (tot + len(ACTION_STYLES))) / lg[f"L_sty_{c}"].values
    shift_row = pd.Series(np.arange(len(lg)), index=lg["game_id"].values)
    lg = lg[["game_id"] + [c for c in lg.columns if c.startswith("L_")]]

    # ---- team baselines: shrunk season-to-date shares -----------------------------
    tm = pt["team"].merge(lg, on="game_id")
    n_eff = _eff(tm, "ts_", "tp_", "n")
    shots_eff = sum(_eff(tm, "ts_", "tp_", c) for c in FA)
    for c in ENDING_CLASSES:
        tm[f"base_end_{c}"] = (_eff(tm, "ts_", "tp_", f"e_{c}") + M_TEAM * tm[f"L_end_{c}"]) / (n_eff + M_TEAM)
    for c in SHOT_FAMILIES:
        tm[f"base_fam_{c}"] = (_eff(tm, "ts_", "tp_", f"f_{c}") + M_TEAM * tm[f"L_fam_{c}"]) / (shots_eff + M_TEAM)
    styled_eff = sum(_eff(tm, "ts_", "tp_", c) for c in SY)
    for c in ACTION_STYLES:
        tm[f"base_sty_{c}"] = (_eff(tm, "ts_", "tp_", f"s_{c}") + M_TEAM * tm[f"L_sty_{c}"]) / (styled_eff + M_TEAM)
    for st in START_TYPES:                                   # a team's tempo mix within each start type; the row's own start type picks one below
        tot = sum(_eff(tm, "ts_", "tp_", f"t_{st}_{c}") for c in TEMPO_CLASSES)
        for c in TEMPO_CLASSES:
            tm[f"bt_{st}_{c}"] = (_eff(tm, "ts_", "tp_", f"t_{st}_{c}") + M_TEAM_TEMPO * tm[f"L_tmp_{st}_{c}"]) / (tot + M_TEAM_TEMPO)
    tm["team_eff_n"] = n_eff
    tm = tm[["game_id", "team_id", "team_eff_n"] + [c for c in tm.columns if c.startswith(("base_", "bt_"))]]

    # ---- player pregame: usage and mixes -------------------------------------------
    pl = pt["player"].merge(lg, on="game_id")
    on_eff, fin_eff = _eff(pl, "ps_", "pp_", "on_floor"), _eff(pl, "ps_", "pp_", "fin")
    pshots = sum(_eff(pl, "ps_", "pp_", c) for c in FA)
    pl["p_usage"] = (fin_eff + M_PLAYER * 0.2) / (on_eff + M_PLAYER)
    pl["p_eff_on"] = on_eff
    for c in ENDING_CLASSES:
        pl[f"p_end_{c}"] = (_eff(pl, "ps_", "pp_", f"e_{c}") + M_PLAYER_MIX * pl[f"L_end_{c}"]) / (fin_eff + M_PLAYER_MIX)
    for c in SHOT_FAMILIES:
        pl[f"p_fam_{c}"] = (_eff(pl, "ps_", "pp_", f"f_{c}") + M_PLAYER_MIX * pl[f"L_fam_{c}"]) / (pshots + M_PLAYER_MIX)
    pstyled = sum(_eff(pl, "ps_", "pp_", c) for c in SY)
    for c in ACTION_STYLES:
        pl[f"p_sty_{c}"] = (_eff(pl, "ps_", "pp_", f"s_{c}") + M_PLAYER_MIX * pl[f"L_sty_{c}"]) / (pstyled + M_PLAYER_MIX)
    pl = pl[["game_id", "player_id", "p_usage", "p_eff_on"] + [f"p_end_{c}" for c in ENDING_CLASSES] + [f"p_fam_{c}" for c in SHOT_FAMILIES]
            + [f"p_sty_{c}" for c in ACTION_STYLES]]

    # ---- context -------------------------------------------------------------------
    df = poss.copy()
    df["game_secs_left"] = np.where(df["period"] <= 4, (4 - df["period"]) * 720.0 + df["start_clock_s"], df["start_clock_s"])
    df["start_type_code"] = df["start_type"].map({s: i for i, s in enumerate(START_TYPES)}).fillna(len(START_TYPES)).astype(int)
    df["clutch"] = (df["period"] >= 4) & (df["start_clock_s"] <= 300) & (df["margin_start"].abs() <= 5)
    for c in ("off_is_home", "def_in_penalty", "after_timeout", "clutch", "two_for_one_window"):
        df[c] = df[c].astype(int)

    # ---- in-game running counts over EARLIER possessions (cumsum minus self) --------
    counted = df["ending_type"].isin(ENDING_CLASSES) & ~df["is_heave"]
    grp = [df["game_id"], df["off_team_id"]]
    df["ig_team_poss"] = counted.astype(int).groupby(grp).cumsum() - counted.astype(int)
    for c in ENDING_CLASSES:
        ind = (counted & (df["ending_type"] == c)).astype(int)
        df[f"ig_end_{c}"] = ind.groupby(grp).cumsum() - ind
    for c in SHOT_FAMILIES:
        ind = (counted & (df["shot_family"] == c)).astype(int)
        df[f"ig_fam_{c}"] = ind.groupby(grp).cumsum() - ind
    for col, classes, prefix in (("action_style", ACTION_STYLES, "ig_sty_"), ("tempo", TEMPO_CLASSES, "ig_tmp_")):
        for c in classes:
            ind = (counted & (df[col] == c)).astype(int)
            df[prefix + c] = ind.groupby(grp).cumsum() - ind

    # ---- the possession before this one, by either team (-1 on a game's first possession) ----
    by_game = df.groupby("game_id", sort=False)
    for c in LAG_SNAPSHOT:
        df[f"lag_{c}"] = by_game[c].shift(1).fillna(-1).astype(float)
    df["lag_ending"] = by_game["ending_type"].shift(1).map({c: i for i, c in enumerate(ENDING_CLASSES)}).fillna(len(ENDING_CLASSES)).where(
        by_game["ending_type"].shift(1).notna(), -1).astype(float)
    df["lag_tempo"] = by_game["tempo"].shift(1).map({c: i for i, c in enumerate(TEMPO_CLASSES)}).fillna(-1).astype(float)

    df = df.merge(tm, left_on=["game_id", "off_team_id"], right_on=["game_id", "team_id"], how="left").drop(columns="team_id")
    st_code = df["start_type"].map({s: i for i, s in enumerate(START_TYPES)}).fillna(START_TYPES.index("other")).astype(int).values
    rows = np.arange(len(df))
    for c in TEMPO_CLASSES:                                  # the team's tempo mix for THIS row's start type
        df[f"base_tmp_{c}"] = df[[f"bt_{st}_{c}" for st in START_TYPES]].values[rows, st_code]
    df = df.drop(columns=[c for c in df.columns if c.startswith("bt_")])

    # ---- candidates: one row per (possession, offensive player) ----------------------
    # Built with integer keys and float32 columns: three seasons is ~3.7M candidate rows.
    codes, uniq = pd.factorize(df["game_id"], sort=True)
    df["gcode"] = codes.astype(np.int32)
    ok = df[df["lineup_ok"]]
    n = len(ok)
    lineups = np.array(ok["off_lineup"].tolist(), dtype=np.int64).reshape(n, 5)
    fin = ok["finisher_id"].astype("float64").fillna(-1).values.astype(np.int64)
    ex = pd.DataFrame({
        "gcode": np.repeat(ok["gcode"].values, 5), "poss_idx": np.repeat(ok["poss_idx"].values.astype(np.int32), 5),
        "slot": np.tile(np.arange(5, dtype=np.int8), n), "player_id": lineups.ravel(),
        "is_fin": (lineups == fin[:, None]).ravel().astype(np.int8),
        "has_fin": np.repeat(fin >= 0, 5), "eligible": np.repeat(ok["model_eligible"].values, 5),
    })
    pl["gcode"] = pd.Categorical(pl["game_id"], categories=uniq).codes.astype(np.int32)
    pcols = (["p_usage", "p_eff_on"] + [f"p_end_{c}" for c in ENDING_CLASSES] + [f"p_fam_{c}" for c in SHOT_FAMILIES]
             + [f"p_sty_{c}" for c in ACTION_STYLES])
    pl32 = pl[["gcode", "player_id"] + pcols].astype({c: np.float32 for c in pcols})
    ex = ex.merge(pl32, on=["gcode", "player_id"], how="left")
    del pl32, lineups
    key = [ex["gcode"], ex["poss_idx"]]
    ex["p_base"] = (ex["p_usage"] / ex["p_usage"].groupby(key).transform("sum")).astype(np.float32)
    ex["p_usage_rank"] = ex["p_usage"].groupby(key).rank(ascending=False, method="first").astype(np.float32)
    ex = ex.sort_values(["gcode", "player_id", "poss_idx"], kind="stable")
    pg = [ex["gcode"], ex["player_id"]]
    ex["ig_p_on"] = ex.groupby(["gcode", "player_id"]).cumcount().astype(np.float32)
    ex["ig_p_fin"] = (ex["is_fin"].astype(np.int32).groupby(pg).cumsum() - ex["is_fin"]).astype(np.float32)
    ex["ig_p_last"] = ex["is_fin"].groupby(pg).shift(1).fillna(0).astype(np.float32)
    ex = ex.sort_values(["gcode", "poss_idx", "slot"], kind="stable").reset_index(drop=True)

    # ---- lineup aggregates (usage-weighted), one value per possession -------------------
    first = np.arange(0, len(ex), 5)
    la = pd.DataFrame({"gcode": ex["gcode"].values[first], "poss_idx": ex["poss_idx"].values[first],
                       "lineup_usage_sum": ex["p_usage"].values.reshape(-1, 5).sum(axis=1)})
    w = ex["p_base"].values.reshape(-1, 5)
    for c in ENDING_CLASSES:
        la[f"lineup_end_{c}"] = (w * ex[f"p_end_{c}"].values.reshape(-1, 5)).sum(axis=1)
    for c in SHOT_FAMILIES:
        la[f"lineup_fam_{c}"] = (w * ex[f"p_fam_{c}"].values.reshape(-1, 5)).sum(axis=1)
    for c in ACTION_STYLES:
        la[f"lineup_sty_{c}"] = (w * ex[f"p_sty_{c}"].values.reshape(-1, 5)).sum(axis=1)
    ex = ex.drop(columns=[f"p_sty_{c}" for c in ACTION_STYLES])
    df = df.merge(la, on=["gcode", "poss_idx"], how="left")
    del la, w
    # action-style baseline: the lineup's mix, shifted by how the league's mix moves with this row's start type
    code = df["start_type"].map({s: i for i, s in enumerate(START_TYPES)}).fillna(START_TYPES.index("other")).astype(int).values
    adj = df[[f"lineup_sty_{c}" for c in ACTION_STYLES]].values * shift[shift_row.reindex(df["game_id"].values).values, code]
    adj = adj / adj.sum(axis=1, keepdims=True)
    df = pd.concat([df, pd.DataFrame(adj, columns=[f"start_sty_{c}" for c in ACTION_STYLES], index=df.index)], axis=1)   # one concat, not ten inserts
    del adj

    elig = df[df["model_eligible"]].sort_values(["gcode", "poss_idx"]).reset_index(drop=True)
    keep = ["game_id", "poss_idx", "season", "tipoff", "off_team_id", "off_lineup", "finisher_id", "ending_type", "shot_family"] + LABELS_22
    poss_out = elig[keep + CONTEXT + CONTEXT_22 + PREGAME_POSS_ALL + INGAME_POSS_ALL].copy()

    cand = ex[ex["eligible"].values].reset_index(drop=True)
    del ex
    if len(cand) != 5 * len(poss_out):
        raise RuntimeError(f"candidate table has {len(cand)} rows for {len(poss_out)} possessions; expected five each")
    cand_out = pd.DataFrame({"game_id": pd.Categorical.from_codes(cand["gcode"].values, categories=uniq),
                             "poss_idx": cand["poss_idx"].values, "slot": cand["slot"].values, "player_id": cand["player_id"].values,
                             "is_fin": cand["is_fin"].values, "has_fin": cand["has_fin"].values})
    for c in CONTEXT:
        cand_out[c] = np.repeat(elig[c].values.astype(np.float32), 5)
    for c in PREGAME_CAND + INGAME_CAND:
        cand_out[c] = cand[c].values
    return Features(poss_out, cand_out)
