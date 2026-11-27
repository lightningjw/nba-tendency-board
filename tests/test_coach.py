"""Step 2.4: the dead-ball table, the three coach-decision heads on it, and their leakage guards."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from conftest import AWAY, HOME, Game, make_box
from tendency import cdn_fields as F
from tendency import coach
from tendency.contract import validate
from tendency.possessions import segment_game
from tendency.stoppages import build_stoppages, mandatory_due


def table(g: Game, box=None):
    box = box or make_box()
    rows, _ = segment_game(g.pbp(), box)
    stops, players, checks = build_stoppages(g.pbp(), box, rows)
    return pd.DataFrame(stops), pd.DataFrame(players), checks


def pf(g: Game, clk, team, pid, n, **kw):
    """A personal foul with the running count printed the way the feed prints it."""
    return g.foul(clk, team, pid, **kw, **{}) if n is None else g.ev("foul", clk, "personal", team, pid, descriptor=kw.get("descriptor", ""),
                                                                    **{F.DESCRIPTION: f"P{pid} personal FOUL ({n} PF)"})


# ---- the dead-ball table -------------------------------------------------------------------------------------
def test_one_row_per_whistle_and_a_live_action_ends_it(game):
    g = game.start().shot(700, HOME, 101, True)                       # made basket: a row, with the new score
    g.timeout(697, AWAY)                                              # three seconds later: the team scored on calls time at that stoppage
    g.sub(697, AWAY, 201, 206)
    g.shot(680, AWAY, 202, False).reb(678, HOME, 102)                 # live again
    pf(g, 660, AWAY, 203, 1)
    g.sub(660, AWAY, 203, 207)
    g.ft(660, HOME, 101, 1, 2, True).ft(660, HOME, 101, 2, 2, True)   # the made last free throw opens its own row
    g.shot(640, AWAY, 202, False).reb(638, HOME, 102)
    g.timeout(630, HOME)                                              # nothing has stopped play: called with the ball live
    g.shot(620, HOME, 101, False).end()
    s, _, _ = table(g)
    assert s["kind"].tolist() == ["made_basket", "foul", "made_free_throw", "timeout_live"]
    mb, foul, ft, live = (s.iloc[i] for i in range(4))
    assert (mb.home_score, mb.away_score, mb.clock_s) == (2, 0, 700.0) and not mb.subs_allowed
    assert mb.away_timeout and not mb.home_timeout and mb.timeout_gap_s == 3.0 and list(mb.away_subs_out) == [201]
    assert list(foul.foul_player_ids) == [203] and list(foul.foul_counts) == [1] and list(foul.away_subs_out) == [203]
    assert foul.n_free_throws == 2 and (foul.home_score, ft.home_score) == (2, 4)       # the foul row is the score before the free throws
    assert ft.trigger_team_id == HOME and not ft.home_timeout and not ft.away_timeout
    assert live.home_timeout and list(live.home_lineup) == [101, 102, 103, 104, 105] and list(live.away_lineup) == [202, 204, 205, 206, 207]


def test_a_timeout_long_after_the_basket_is_a_live_ball_timeout(game):
    g = game.start().shot(700, HOME, 101, True).timeout(685, AWAY).shot(670, AWAY, 201, False).end()
    s, _, _ = table(g)
    assert s["kind"].tolist() == ["made_basket", "timeout_live"] and not s.iloc[0].away_timeout and s.iloc[1].away_timeout


def test_a_new_whistle_after_the_clock_runs_is_a_new_row_and_the_same_whistle_is_not(game):
    g = game.start()
    pf(g, 700, AWAY, 201, 1)
    g.ev("foul", 700, "technical", AWAY, 202)                         # same whistle
    pf(g, 695, AWAY, 201, 2, descriptor="take")                       # inbounded, fouled again: no live action between
    g.shot(680, HOME, 101, False).end()
    s, _, checks = table(g)
    assert s["kind"].tolist() == ["foul", "foul"]
    assert list(s.iloc[0].foul_player_ids) == [201, 202] and list(s.iloc[0].foul_technical) == [False, True]
    assert list(s.iloc[0].foul_counts) == [1, 0]                      # a technical is not a personal foul
    assert s.iloc[1].take_foul_team_id == AWAY and pd.isna(s.iloc[0].take_foul_team_id)
    assert checks == {"pf_checks": 2, "pf_agree": 2}


def test_an_and_one_is_a_foul_stoppage_and_a_turnover_with_a_steal_is_none(game):
    g = game.start().shot(700, HOME, 101, True)
    pf(g, 700, AWAY, 201, 1)
    g.ft(700, HOME, 101, 1, 1, False).reb(699, AWAY, 202)
    g.ev("turnover", 690, "bad pass", AWAY, 202, **{F.STEAL_PERSON_ID: 103}).ev("steal", 690, "", HOME, 103)
    g.tov(680, HOME, 103).shot(670, AWAY, 201, False).end()
    s, _, _ = table(g)
    assert s["kind"].tolist() == ["foul", "turnover"] and s.iloc[0].subs_allowed and s.iloc[1].clock_s == 680.0


def test_stints_fouls_and_seconds_played_are_tracked_for_both_fives(game):
    g = game.start()
    pf(g, 600, HOME, 101, 1)
    g.sub(600, HOME, 102, 106)
    g.shot(590, AWAY, 201, False).reb(588, HOME, 103).tov(500, HOME, 103).shot(490, AWAY, 201, False).end()
    s, _, _ = table(g)
    first, second = s.iloc[0], s.iloc[1]
    assert list(first.home_stint_s) == [120.0] * 5 and list(first.home_fouls) == [1, 0, 0, 0, 0]
    assert list(second.home_lineup) == [101, 103, 104, 105, 106]
    assert list(second.home_stint_s) == [220.0, 220.0, 220.0, 220.0, 100.0] and list(second.home_secs_played) == [220.0, 220.0, 220.0, 220.0, 100.0]
    assert list(second.home_fouls) == [1, 0, 0, 0, 0] and second.game_secs == 220.0


def test_the_run_in_progress_counts_unanswered_points_and_recent_possessions(game):
    g = game.start().shot(700, AWAY, 201, True).shot(690, HOME, 101, False).reb(688, AWAY, 202)
    g.shot(680, AWAY, 201, True, three=True).shot(670, HOME, 101, False).reb(668, AWAY, 202).shot(660, AWAY, 201, True)
    g.shot(650, HOME, 101, True).shot(640, AWAY, 201, False).end()
    s, _, _ = table(g)
    third = s[s["kind"] == "made_basket"].iloc[2]
    assert (third.away_unanswered, third.home_unanswered) == (7, 0) and (third.away_run_points, third.home_run_points) == (7, 0)
    after = s[s["kind"] == "made_basket"].iloc[3]                     # the home basket ends the run
    assert (after.away_unanswered, after.home_unanswered) == (0, 2) and (after.away_run_points, after.home_run_points) == (7, 2)


def test_mandatory_timeout_rule():
    assert not mandatory_due(1, 430.0, 0) and mandatory_due(1, 419.0, 0) and not mandatory_due(1, 419.0, 1)
    assert mandatory_due(3, 179.0, 1) and not mandatory_due(3, 179.0, 2) and not mandatory_due(5, 100.0, 0)


def test_period_break_row_holds_who_actually_changed(game):
    g = game.start().shot(700, HOME, 101, False).reb(698, AWAY, 201).end()
    g.start(2).shot(700, HOME, 106, False).reb(698, AWAY, 201).shot(690, AWAY, 206, False).end()   # nobody logged 106 or 206 coming in
    s, p, _ = table(g)
    brk = s[s["kind"] == "period_break"].iloc[0]
    assert list(brk.home_subs_in) == [106] and len(brk.home_subs_out) == 1 and list(brk.away_subs_in) == [206]
    assert len(s[s["kind"] == "period_break"]) == 1                   # the break after the last period is the end of the game
    q = p.set_index("player_id")
    assert q.loc[106, "secs_q2"] == 720.0 and q.loc[106, "secs_q1"] == 0.0 and q.loc[101, "secs_q1"] == 720.0


def _four_quarters(close: bool = True) -> Game:
    g = Game()
    for period in (1, 2, 3):
        g.start(period).shot(600, HOME, 101, True).shot(500, AWAY, 201, True).end()
    g.start(4).shot(600, HOME, 101, True, three=not close)
    pf(g, 400, HOME, 102, 1)
    g.sub(400, HOME, 102, 106)                                        # 106 plays all of the last five minutes; 102 none of them
    g.shot(350, AWAY, 201, True)
    pf(g, 100, AWAY, 202, 1)
    g.sub(100, AWAY, 202, 207)                                        # 207 plays 100 of the last 300 seconds, 202 plays 200
    g.shot(50, AWAY, 201, False).end()
    return g


def test_player_game_table_has_the_last_five_minutes_and_the_score_when_they_began():
    _, p, _ = table(_four_quarters())
    q = p.set_index("player_id")
    assert q.loc[106, "secs_close"] == 300.0 and q.loc[102, "secs_close"] == 0.0 and q.loc[202, "secs_close"] == 200.0 and q.loc[207, "secs_close"] == 100.0
    assert q.loc[101, "margin_close"] == 0 and q.loc[201, "margin_close"] == 0 and q.loc[101, "margin_q3"] == 0
    assert q.loc[102, "fouls_q3"] == 0 and q.loc[102, "fouls_final"] == 1 and bool(q.loc[102, "on_floor_q3"]) and not bool(q.loc[106, "on_floor_q3"])
    assert q.loc[101, "secs_total"] == 2880.0 and q.loc[106, "secs_q4"] == 400.0


# ---- the three decision tables -----------------------------------------------------------------------
def test_foul_rows_count_a_substitution_at_the_whistle_and_not_a_later_one(game):
    g = game.start()
    pf(g, 700, AWAY, 201, 1)
    g.ft(700, HOME, 101, 1, 2, True).ft(700, HOME, 101, 2, 2, True).sub(700, AWAY, 201, 206)    # logged after the trip, same clock: still this whistle
    g.shot(690, AWAY, 202, False).reb(688, HOME, 102)
    pf(g, 650, HOME, 103, 1)
    g.shot(640, AWAY, 202, False).reb(638, HOME, 102).tov(600, HOME, 102).sub(600, HOME, 103, 106)   # taken out a stoppage later
    g.ev("foul", 590, "technical", AWAY, 202).shot(580, AWAY, 202, False).end()
    s, p, _ = table(g)
    d = coach.foul_rows(s, p).set_index("player_id")
    assert d.index.tolist() == [201, 103]                             # the technical is not a row
    assert d.loc[201, "y"] == 1 and d.loc[103, "y"] == 0
    assert (d.loc[201, "foul_n"], d.loc[201, "period"], d.loc[201, "stint_s"], d.loc[201, "starter"], d.loc[201, "margin"]) == (1, 1, 20.0, 1, 0)


def test_the_sixth_foul_is_not_a_decision(game):
    g = game.start()
    for n in range(1, 7):
        pf(g, 700 - 10 * n, AWAY, 201, n)
        g.shot(695 - 10 * n, HOME, 101, False).reb(694 - 10 * n, AWAY, 202)
    g.end()
    s, p, checks = table(g)
    assert coach.foul_rows(s, p)["foul_n"].tolist() == [1, 2, 3, 4, 5] and checks["pf_agree"] == 6


def test_timeout_rows_are_the_stoppages_where_calling_one_is_a_choice(game):
    g = game.start().shot(700, HOME, 101, True).timeout(699, AWAY)    # choice: more than 6:59 left
    g.shot(680, AWAY, 201, False).reb(678, HOME, 102).timeout(670, HOME)          # live ball: never a decision row
    g.tov(600, HOME, 102).shot(590, AWAY, 201, True)
    g.tov(300, HOME, 102)                                             # 5:00 left and both timeouts so far count: no mandatory due
    g.shot(290, AWAY, 201, False).reb(288, HOME, 102).tov(170, HOME, 102)         # 2:50 with two taken: none due either
    g.shot(160, AWAY, 201, False).end()
    g.start(2).shot(700, HOME, 101, False).reb(698, AWAY, 201).tov(400, AWAY, 201).shot(390, HOME, 101, False).end()   # 6:40, none taken: due
    s, _, _ = table(g)
    assert s["mandatory_due"].tolist() == [False, False, False, False, False, False, False, True]
    d = coach.timeout_rows(s)
    assert "timeout_live" not in set(d["kind"]) and d["period"].max() == 1
    mb = d[d["kind"] == "made_basket"]
    assert mb["team_id"].tolist() == [AWAY, HOME] and mb["y"].tolist() == [1, 0]          # only the team scored on, each time
    tv = d[d["kind"] == "turnover"]
    assert len(tv) == 6 and tv["y"].sum() == 0 and set(tv["team_id"]) == {HOME, AWAY}
    first = mb.iloc[0]
    assert (first.margin, first.opp_unanswered, first.timeouts_left, first.own_trigger, first.run_bucket) == (-2, 2, 7, 0, 0)
    assert first.secs_to_mandatory == pytest.approx(700 - F.MANDATORY_FIRST_S) and first.secs_since_timeout == -1.0
    assert d.loc[d["stop_idx"] == 2, "secs_since_timeout"].min() > 0 and not d[coach.TIMEOUT_FEATURES].isna().any().any()


def test_closing_rows_take_the_five_with_most_of_the_last_five_minutes():
    s, p, _ = table(_four_quarters())
    d = coach.closing_rows(p)
    h = d[d["team_id"] == HOME].set_index("player_id")
    assert h.index.tolist() == [101, 102, 103, 104, 105]              # 106 closed but had not played before the fourth: not a candidate
    assert h["y"].tolist() == [1, 0, 1, 1, 1]
    a = d[d["team_id"] == AWAY].set_index("player_id")
    assert a["y"].tolist() == [1, 1, 1, 1, 1] and 207 not in a.index  # 202 played 200 of the 300 seconds, more than the man who replaced him
    assert h.loc[101, "secs_pre"] == 2160.0 and h.loc[101, "on_floor_q3"] == 1 and (d["p_base"] == 0.5).all()
    _, wide, _ = table(_four_quarters(close=False))                   # not within five... it is: 3 to 2 after the three. Make it a blowout:
    wide["margin_close"] = wide["margin_close"] * 10
    assert len(coach.closing_rows(wide[wide["margin_close"].abs() > 5])) == 0


# ---- leakage -------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth_tables(synth_run):
    proc = synth_run["root"] / "processed"
    return pd.read_parquet(proc / "stoppages.parquet"), pd.read_parquet(proc / "player_games.parquet")


AS_OF = {"timeout": ["p_base", "team_lean"], "foul_sub": ["p_base", "team_lean", "prior_secs"],
         "closing_five": ["p_base", "team_lean", "prior_secs", "prior_close_k", "prior_close_n"]}
LABEL_COLS = ["home_timeout", "away_timeout", "home_challenge", "away_challenge", "home_subs_out", "home_subs_in", "away_subs_out",
              "away_subs_in", "timeout_gap_s", "n_free_throws", "take_foul_team_id"]


def _rewrite_what_coaches_did(stops: pd.DataFrame, players: pd.DataFrame, game_ids: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """In the given games: every team calls time everywhere, every fouler is taken out, and the bench closes."""
    s, p = stops.copy(), players.copy()
    m = s["game_id"].isin(game_ids).values
    s.loc[m, "home_timeout"] = True
    s.loc[m, "away_timeout"] = True
    for side in ("home", "away"):
        s[f"{side}_subs_out"] = [list(l) if hit else list(o) for l, o, hit in zip(s[f"{side}_lineup"], s[f"{side}_subs_out"], m)]
    mp = p["game_id"].isin(game_ids).values
    p.loc[mp, "secs_close"] = np.where(p.loc[mp, "starter"], 0.0, 300.0)
    p.loc[mp, "secs_total"] = 1.0
    return s, p


def test_every_decision_feature_is_a_snapshot_or_an_as_of_count():
    """The whistle snapshot is covered by the truncation test and the as-of columns by the rewrite test; nothing else may be a feature."""
    snapshot = {"timeout": {"period", "clock_s", "game_secs_left", "margin", "timeouts_left", "opp_timeouts_left", "timeouts_this_period",
                            "secs_to_mandatory", "secs_since_timeout", "kind_code", "own_trigger", "opp_unanswered", "own_unanswered",
                            "opp_run_points", "own_run_points", "is_home"},
                "foul_sub": {"foul_n", "period", "clock_s", "game_secs_left", "margin", "starter", "stint_s", "secs_played", "is_home"},
                "closing_five": {"starter", "secs_pre", "secs_q3", "on_floor_q3", "fouls_q3", "margin_q3", "is_home", "pre_rank"}}
    for head in coach.DECISION_HEADS:
        assert set(coach.FEATURES[head]) == snapshot[head] | (set(AS_OF[head]) - {"p_base"}), head
        assert len(coach.FEATURES[head]) == len(set(coach.FEATURES[head]))


def test_a_stoppage_row_does_not_change_when_everything_after_it_is_cut_off(synth_run):
    root = synth_run["root"]
    gid = synth_run["ids"][20]
    pbp, box = json.loads((root / "raw" / "pbp" / f"{gid}.json").read_text()), json.loads((root / "raw" / "box" / f"{gid}.json").read_text())
    rows, _ = segment_game(pbp, box)
    full, _, _ = build_stoppages(pbp, box, rows)
    for cut in (150, 400):
        short = copy.deepcopy(pbp)
        short[F.GAME][F.ACTIONS] = pbp[F.GAME][F.ACTIONS][:cut]
        rows_s, _ = segment_game(short, box)
        part, _, _ = build_stoppages(short, box, rows_s)
        assert len(part) > 5
        for a, b in zip(full, part[:-1]):                             # the last one was still open when the feed was cut
            assert a == b
        a, b = full[len(part) - 1], part[-1]                          # ... and even it had its snapshot: only what happened next can differ
        assert {k: v for k, v in a.items() if k not in LABEL_COLS + ["kind", "subs_allowed", "trigger_team_id", "foul_player_ids", "foul_team_ids",
                                                                     "foul_counts", "foul_technical"]} == \
               {k: v for k, v in b.items() if k not in LABEL_COLS + ["kind", "subs_allowed", "trigger_team_id", "foul_player_ids", "foul_team_ids",
                                                                     "foul_counts", "foul_technical"]}


def test_closing_features_stop_at_the_end_of_the_third_quarter(synth_run):
    root = synth_run["root"]
    gid = synth_run["ids"][20]
    pbp, box = json.loads((root / "raw" / "pbp" / f"{gid}.json").read_text()), json.loads((root / "raw" / "box" / f"{gid}.json").read_text())
    acts = pbp[F.GAME][F.ACTIONS]
    q4 = next(i for i, a in enumerate(acts) if a[F.PERIOD] == 4)
    short = copy.deepcopy(pbp)
    short[F.GAME][F.ACTIONS] = acts[:q4]
    _, full, _ = build_stoppages(pbp, box, segment_game(pbp, box)[0])
    _, part, _ = build_stoppages(short, box, segment_game(short, box)[0])
    cols = ["player_id", "starter", "is_home", "secs_q1", "secs_q2", "secs_q3", "fouls_q3", "on_floor_q3", "margin_q3"]
    assert pd.DataFrame(full)[cols].equals(pd.DataFrame(part)[cols])


def _target(order: pd.DataFrame, d: pd.DataFrame) -> tuple[str, str]:
    """The first game from the 60th on that has rows for this decision (not every game is close late), and its tip time."""
    have = set(d["game_id"])
    row = order.iloc[60:][order.iloc[60:]["game_id"].isin(have)].iloc[0]
    return row["game_id"], row["tipoff_utc"]


def test_no_decision_baseline_or_prior_sees_its_own_game_or_a_later_one(synth_tables):
    stops, players = synth_tables
    order = stops.drop_duplicates("game_id").sort_values("tipoff_utc")
    base = coach.decision_tables(stops, players)
    for head, cols in AS_OF.items():
        target, tip = _target(order, base[head])
        later = set(order.loc[order["tipoff_utc"] >= tip, "game_id"])
        alt = coach.decision_tables(*_rewrite_what_coaches_did(stops, players, later))
        a, b = base[head][base[head]["game_id"] == target], alt[head][alt[head]["game_id"] == target]
        if head == "closing_five":                                    # who is a candidate does not depend on who closed
            assert a["player_id"].tolist() == b["player_id"].tolist()
        assert len(a) == len(b) and len(a) > 0, head
        assert not (a["y"].values == b["y"].values).all(), head       # the rewrite did change what happened in this game
        for c in cols:
            assert np.allclose(a[c].values, b[c].values), (head, c)


def test_the_guard_has_teeth_as_of_counts_move_when_earlier_games_change(synth_tables):
    stops, players = synth_tables
    order = stops.drop_duplicates("game_id").sort_values("tipoff_utc")
    base = coach.decision_tables(stops, players)
    for head in coach.DECISION_HEADS:
        target, tip = _target(order, base[head])
        earlier = set(order.loc[order["tipoff_utc"] < tip, "game_id"])
        alt = coach.decision_tables(*_rewrite_what_coaches_did(stops, players, earlier))
        a, b = base[head][base[head]["game_id"] == target], alt[head][alt[head]["game_id"] == target]
        assert len(a) == len(b) > 0 and not np.allclose(a["p_base"].values, b["p_base"].values), head


def test_first_night_gets_the_starting_guess_and_baselines_are_probabilities(synth_tables):
    t = coach.decision_tables(*synth_tables)
    for head, d in t.items():
        assert d["p_base"].between(1e-4, 1 - 1e-4).all() and not d[coach.FEATURES[head]].isna().any().any(), head
        first = d[d["tipoff"] == d["tipoff"].min()]
        assert np.allclose(first["p_base"], coach.START_RATE[head]) and np.allclose(first["team_lean"], 0.0), head


# ---- what the simulator plants, and the export -------------------------------------------------------------
def test_planted_behaviour_shows_up_in_the_tables(synth_tables):
    t = coach.decision_tables(*synth_tables)
    f = t["foul_sub"]
    over = f["foul_n"] - f["period"].clip(upper=4)
    assert f.loc[over >= 1, "y"].mean() > 0.5 > 0.2 > f.loc[over <= -1, "y"].mean()
    hook = f["team_id"] == 9900001 + __import__("tendency.synth", fromlist=["x"]).QUICK_HOOK_TEAM
    assert f.loc[hook & (over == 0), "y"].mean() > f.loc[~hook & (over == 0), "y"].mean() + 0.15
    to = t["timeout"].groupby("run_bucket")["y"].mean()
    assert to[2] > to[1] > to[0]


def test_export_carries_the_decision_heads_and_the_report_section(synth_run):
    out = synth_run["out"]
    man = json.loads((out / "manifest.json").read_text())
    heads = man["heads"][5:]
    assert [h["id"] for h in heads] == coach.DECISION_HEADS and all(h["kind"] == "decision" and h["unit"] and not h["logged_before_tipoff"] for h in heads)
    assert "coach-decision heads" not in man["not_built"]
    st = man["stoppages"]
    assert st["n_stoppages"] == sum(k["n"] for k in st["kinds"]) and st["personal_foul_check"]["n"] == st["personal_foul_check"]["agree"] > 1000
    assert [r["head"] for r in st["decision_rows"]] == coach.DECISION_HEADS and all(0 < r["n_yes"] < r["n"] for r in st["decision_rows"])
    assert {r["id"] for r in st["not_modelled"]} == {"late_fouls", "challenges", "next_substitution"} and all(r["n"] > 0 for r in st["not_modelled"])
    sc = json.loads((out / "scorecard.json").read_text())
    for h in sc["heads"][5:]:
        assert h["model_id"] == "lightgbm" and all(s["available"] for s in h["splits"])
    team = man and json.loads((out / "teams.json").read_text())["teams"][0]["team_id"]
    rep = json.loads((out / "team_reports" / f"{team}.json").read_text())
    cd = rep["coach_decisions"]
    assert len(cd["timeouts"]) == 3 and len(cd["foul_trouble"]) == 7 and cd["closing"]["n_close_games"] > 0
    for r in cd["timeouts"] + cd["foul_trouble"]:
        assert r["interval"][0] <= r["rate"] <= r["interval"][1] and r["k"] <= r["n"] and r["league_n"] >= r["n"]
    for pl in cd["closing"]["players"]:
        assert pl["k"] <= pl["n"] <= cd["closing"]["n_close_games"] and pl["interval"][0] <= pl["interval"][1]
    assert "right or wrong" in cd["note"]                              # descriptive framing travels with the numbers


def test_an_export_without_the_decision_heads_is_still_valid(synth_run):
    out = synth_run["out"]
    man = json.loads((out / "manifest.json").read_text())
    old = {**man, "heads": man["heads"][:5]}
    old.pop("stoppages")
    validate(old, "manifest")
    sc = json.loads((out / "scorecard.json").read_text())
    validate({**sc, "heads": sc["heads"][:5]}, "scorecard")
    team = json.loads((out / "teams.json").read_text())["teams"][0]["team_id"]
    rep = json.loads((out / "team_reports" / f"{team}.json").read_text())
    rep.pop("coach_decisions")
    validate(rep, "team_report")


def test_the_prospective_path_builds_no_stoppage_table(tmp_path, synth_run):
    """`train` and `predict` call build_possessions without the flag: the frozen model's inputs are untouched."""
    import shutil
    from tendency.pipeline import build_possessions
    for kind in ("pbp", "box"):
        (tmp_path / "raw" / kind).mkdir(parents=True)
        for gid in synth_run["ids"][:3]:
            shutil.copy(synth_run["root"] / "raw" / kind / f"{gid}.json", tmp_path / "raw" / kind / f"{gid}.json")
    poss, games = build_possessions(tmp_path, log=lambda *_: None)
    assert not (tmp_path / "processed" / "stoppages.parquet").exists() and "pf_checks" not in games[0]
    assert poss.columns.tolist() == synth_run["poss"].columns.tolist()


def test_a_game_the_dead_ball_table_cannot_read_keeps_its_possessions(tmp_path, synth_run, monkeypatch):
    import shutil
    from tendency import pipeline
    for kind in ("pbp", "box"):
        (tmp_path / "raw" / kind).mkdir(parents=True)
        for gid in synth_run["ids"][:3]:
            shutil.copy(synth_run["root"] / "raw" / kind / f"{gid}.json", tmp_path / "raw" / kind / f"{gid}.json")
    real, bad = pipeline.build_stoppages, synth_run["ids"][1]

    def flaky(pbp, box, rows):
        if pbp[F.GAME][F.GAME_ID] == bad:
            raise ValueError("unreadable")
        return real(pbp, box, rows)

    monkeypatch.setattr(pipeline, "build_stoppages", flaky)
    lines = []
    poss, games = pipeline.build_possessions(tmp_path, log=lines.append, stoppages=True)
    assert poss["game_id"].nunique() == 3 and len(games) == 3
    stops = pd.read_parquet(tmp_path / "processed" / "stoppages.parquet")
    assert stops["game_id"].nunique() == 2 and bad not in set(stops["game_id"])
    assert any("1 games could not be read" in l for l in lines)
