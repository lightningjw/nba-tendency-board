"""Possession segmentation, one test per rule in possessions.py's docstring."""
from __future__ import annotations

import json

import pytest

from tendency import cdn_fields as F
from tendency.possessions import FeedShapeError, parse_clock, segment_game, shot_family

from conftest import AWAY, HOME, make_box


def seg(game, box):
    rows, info = segment_game(game.pbp(), box)
    return rows, info


def test_clock_parsing():
    assert parse_clock("PT11M42.50S") == pytest.approx(702.5)
    assert parse_clock("PT00M00.00S") == 0.0
    assert parse_clock("PT5M") == 300.0
    with pytest.raises(FeedShapeError):
        parse_clock("11:42")


def test_basic_endings_alternate_possession(game, box):
    game.start().shot(700, HOME, 101, made=True).shot(685, AWAY, 201, made=False).reb(683, HOME, 102) \
        .tov(670, HOME, 103).shot(660, AWAY, 202, made=True, three=True).end()
    rows, info = seg(game, box)
    assert [(r["off_team_id"], r["ending_type"], r["finisher_id"]) for r in rows[:4]] == [
        (HOME, "shot", 101), (AWAY, "shot", 201), (HOME, "turnover", 103), (AWAY, "shot", 202)]
    assert [r["start_type"] for r in rows[:4]] == ["period_start", "made_basket", "def_rebound", "turnover"]
    assert rows[1]["shot_made"] is False and rows[3]["points"] == 3
    assert rows[1]["margin_start"] == -2          # away trails by the opening basket
    assert rows[4]["ending_type"] == "period_end" and rows[4]["off_team_id"] == HOME and not rows[4]["model_eligible"]
    assert info["anomalies"] == 0


def test_offensive_rebound_continues_the_possession(game, box):
    game.start().shot(700, HOME, 101, made=False).reb(698, HOME, 105, offensive=True) \
        .shot(690, HOME, 102, made=False, area="Restricted Area", dist=1).reb(688, HOME, 0, offensive=True) \
        .shot(684, HOME, 103, made=True, area="Restricted Area", dist=1).tov(670, AWAY, 201).end()
    rows, _ = seg(game, box)
    first = rows[0]
    assert first["off_team_id"] == HOME and first["n_off_reb"] == 2
    assert (first["ending_type"], first["finisher_id"], first["shot_family"], first["points"]) == ("shot", 103, "rim", 2)
    assert rows[1]["off_team_id"] == AWAY          # exactly one HOME possession before the turnover
    assert first["start_clock_s"] == 720.0 and first["end_clock_s"] == 684.0


def test_and_one_is_one_possession_ending_in_a_shot(game, box):
    game.start().shot(700, HOME, 101, made=True, area="Restricted Area", dist=1) \
        .foul(700, AWAY, 204, drawn=101, descriptor="shooting").ft(700, HOME, 101, 1, 1, made=True) \
        .shot(680, AWAY, 201, made=True).end()
    rows, _ = seg(game, box)
    p = rows[0]
    assert (p["ending_type"], p["finisher_id"], p["and_one"], p["points"], p["shot_made"]) == ("shot", 101, True, 3, True)
    assert rows[1]["off_team_id"] == AWAY and rows[1]["start_type"] == "free_throw"
    assert rows[1]["home_score_start"] == 3
    assert rows[1]["off_team_fouls_period"] == 1    # the and-one foul counts against AWAY


def test_and_one_missed_free_throw_then_offensive_rebound(game, box):
    game.start().shot(700, HOME, 101, made=True).foul(700, AWAY, 204, drawn=101, descriptor="shooting") \
        .ft(700, HOME, 101, 1, 1, made=False).reb(699, HOME, 105, offensive=True).tov(690, HOME, 102).end()
    rows, _ = seg(game, box)
    assert len([r for r in rows if r["off_team_id"] == HOME and r["period"] == 1 and r["ending_type"] != "period_end"]) == 1
    p = rows[0]
    assert (p["ending_type"], p["finisher_id"], p["points"], p["n_off_reb"], p["and_one"]) == ("turnover", 102, 2, 1, True)


def test_shooting_foul_trip_is_foul_drawn(game, box):
    game.start().foul(700, AWAY, 203, drawn=102, descriptor="shooting") \
        .ft(700, HOME, 102, 1, 2, made=True).reb(700, HOME, 0, offensive=True) \
        .ft(700, HOME, 102, 2, 2, made=False).reb(698, AWAY, 205).shot(680, AWAY, 201, made=True).end()
    rows, _ = seg(game, box)
    p = rows[0]
    assert (p["ending_type"], p["finisher_id"], p["points"], p["shot_family"]) == ("foul_drawn", 102, 1, None)
    assert p["n_off_reb"] == 0                      # the dead-ball rebound between free throws is ignored
    assert p["def_team_fouls_period"] == 0          # context is taken before the foul
    assert rows[1]["start_type"] == "def_rebound"


def test_technical_free_throw_never_ends_or_relabels(game, box):
    game.start().shot(700, HOME, 101, made=False).reb(698, HOME, 104, offensive=True) \
        .foul(695, AWAY, 202, sub="technical").ft(695, HOME, 101, 1, 1, made=True, technical=True) \
        .shot(690, HOME, 103, made=True, three=True) \
        .foul(688, HOME, 105, sub="technical").ft(688, AWAY, 201, 1, 1, made=True, technical=True) \
        .tov(675, AWAY, 202).end()
    rows, info = seg(game, box)
    p = rows[0]
    assert (p["off_team_id"], p["ending_type"], p["finisher_id"], p["points"]) == (HOME, "shot", 103, 3)
    assert p["tech_ft_points_off"] == 1 and p["n_off_reb"] == 1
    q = rows[1]
    assert (q["off_team_id"], q["ending_type"], q["points"], q["tech_ft_points_off"]) == (AWAY, "turnover", 0, 1)
    assert q["off_team_fouls_period"] == 0 and q["def_team_fouls_period"] == 0   # technicals are not team fouls
    total = sum(r["points"] + r["tech_ft_points_off"] + r["tech_ft_points_def"] for r in rows)
    assert total == info["final_score"]["home"] + info["final_score"]["away"] == 5


def test_end_of_period_heave_is_flagged_and_excluded(game, box):
    game.start().shot(20, HOME, 101, made=True, three=True, dist=26) \
        .shot(2.0, AWAY, 201, made=False, three=True, dist=30).reb(1.2, HOME, 102) \
        .shot(0.3, HOME, 103, made=False, three=True, area="Backcourt", dist=62).end()
    rows, _ = seg(game, box)
    normal, late30, heave = rows
    assert not normal["is_heave"] and normal["shot_family"] == "above_break3" and normal["model_eligible"]
    assert not late30["is_heave"]                   # late but not from heave range
    assert heave["is_heave"] and heave["shot_family"] == "heave" and heave["ending_type"] == "shot"
    assert not heave["model_eligible"] and heave["finisher_id"] == 103


def test_period_end_without_a_deciding_event(game, box):
    game.start().shot(30, HOME, 101, made=True).end()
    rows, _ = seg(game, box)
    assert [(r["off_team_id"], r["ending_type"]) for r in rows] == [(HOME, "shot"), (AWAY, "period_end")]
    assert rows[1]["finisher_id"] is None and not rows[1]["model_eligible"]


def test_bonus_free_throws_and_penalty_flag(game, box):
    g = game.start()
    for i in range(4):                              # four non-shooting fouls by AWAY: side out each time
        g.foul(700 - i, AWAY, 201, drawn=101)
    g.shot(690, HOME, 101, made=False).reb(689, AWAY, 202).tov(680, AWAY, 203) \
        .foul(670, AWAY, 204, drawn=105).ft(670, HOME, 105, 1, 2, made=True).ft(670, HOME, 105, 2, 2, made=True).end()
    rows, _ = seg(g, box)
    assert rows[0]["def_team_fouls_period"] == 4 and rows[0]["def_in_penalty"]
    bonus = rows[2]
    assert (bonus["ending_type"], bonus["finisher_id"], bonus["points"], bonus["def_in_penalty"]) == ("foul_drawn", 105, 2, True)


def test_lineups_follow_substitutions_and_timeouts(game, box):
    game.start().shot(700, HOME, 101, made=True).timeout(700, AWAY).sub(700, AWAY, 201, 206) \
        .shot(690, AWAY, 206, made=True).end()
    rows, _ = seg(game, box)
    assert rows[0]["off_lineup"] == [101, 102, 103, 104, 105] and not rows[0]["after_timeout"]
    assert rows[1]["off_lineup"] == [202, 203, 204, 205, 206] and rows[1]["after_timeout"] and rows[1]["lineup_ok"]
    assert rows[1]["off_timeouts_used"] == 1 and rows[1]["off_timeouts_left"] == F.TIMEOUTS_PER_GAME - 1


def test_missing_between_period_subs_are_repaired(game, box):
    game.start().shot(700, HOME, 101, made=True).end()
    game.start(2).shot(700, AWAY, 207, made=True).shot(680, HOME, 106, made=False).reb(678, AWAY, 201).end()
    rows, info = seg(game, box)
    q2 = [r for r in rows if r["period"] == 2]
    assert 207 in q2[0]["off_lineup"] and len(q2[0]["off_lineup"]) == 5 and q2[0]["lineup_ok"]
    assert 106 in q2[1]["off_lineup"] and q2[1]["lineup_ok"]
    assert info["lineup_repairs"] >= 2


def test_unexplained_change_of_possession_is_counted_not_hidden(game, box):
    game.start().foul(700, AWAY, 201, drawn=101).shot(690, AWAY, 202, made=True).end()
    rows, info = seg(game, box)
    assert rows[0]["ending_type"] == "unknown" and not rows[0]["model_eligible"] and info["anomalies"] == 1


def test_shot_family_falls_back_to_coordinates():
    base = {F.ACTION_TYPE: "2pt"}
    assert shot_family({**base, F.AREA: "In The Paint (Non-RA)"}) == "paint"
    assert shot_family({**base, F.X_LEGACY: 10, F.Y_LEGACY: 20}) == "rim"
    assert shot_family({**base, F.X_LEGACY: 40, F.Y_LEGACY: 100}) == "paint"
    assert shot_family({**base, F.X_LEGACY: 150, F.Y_LEGACY: 100}) == "midrange"
    assert shot_family({F.ACTION_TYPE: "3pt", F.X_LEGACY: 230, F.Y_LEGACY: 20}) == "corner3"
    assert shot_family({F.ACTION_TYPE: "3pt", F.X_LEGACY: 100, F.Y_LEGACY: 240}) == "above_break3"
    assert shot_family({**base}) is None


def test_bad_files_raise_a_clear_error(box):
    with pytest.raises(FeedShapeError, match="no actions"):
        segment_game({F.GAME: {F.GAME_ID: "0022500001", F.ACTIONS: []}}, box)
    with pytest.raises(FeedShapeError, match="missing"):
        segment_game({"wrong": {}}, box)
    with pytest.raises(FeedShapeError):
        segment_game({F.GAME: {F.GAME_ID: "x", F.ACTIONS: [{}]}}, {"game": {}})


def test_synthetic_games_reconcile_and_alternate(synth_run):
    """Across every synthetic game: points add up, possession counts are sane, nothing unexplained."""
    root = synth_run["root"]
    for gid in synth_run["ids"][::12]:
        pbp = json.loads((root / "raw" / "pbp" / f"{gid}.json").read_text())
        bx = json.loads((root / "raw" / "box" / f"{gid}.json").read_text())
        rows, info = segment_game(pbp, bx)
        total = sum(r["points"] + r["tech_ft_points_off"] + r["tech_ft_points_def"] for r in rows)
        assert total == info["final_score"]["home"] + info["final_score"]["away"]
        assert 150 <= len(rows) <= 280 and info["anomalies"] == 0 and info["synthetic"]
        assert all(r["lineup_ok"] for r in rows)
    poss = synth_run["poss"]
    for col, flag in (("n_off_reb", poss["n_off_reb"] > 0), ("and_one", poss["and_one"]), ("heave", poss["is_heave"]),
                      ("tech", (poss["tech_ft_points_off"] + poss["tech_ft_points_def"]) > 0)):
        assert flag.sum() > 50, f"synthetic data should exercise {col}"


def test_season_from_game_id():
    assert F.season_from_game_id("0022500061") == 2025
    assert make_box()[F.GAME][F.GAME_ID][3:5] == "25"


def test_between_period_subs_logged_before_the_period_start_are_applied(game, box):
    """Real feed: next-period substitutions arrive before that period's 'start' event."""
    game.start().shot(700, HOME, 101, made=True).end()
    game.period = 2
    game.sub(720, AWAY, 201, 207)
    game.ev("period", 720, "start").shot(700, AWAY, 207, made=True).end()
    rows, info = seg(game, box)
    q2 = [r for r in rows if r["period"] == 2][0]
    assert q2["off_lineup"] == [202, 203, 204, 205, 207] and q2["lineup_ok"] and info["lineup_repairs"] == 0


def test_free_throws_where_the_team_keeps_the_ball_do_not_end_the_possession(game, box):
    """Real feed: transition-take, flagrant, clear-path and away-from-play free throws."""
    game.start().shot(700, HOME, 101, made=True) \
        .foul(698, HOME, 102, drawn=201, descriptor="transition take") \
        .ev("freethrow", 698, "1 of 1", AWAY, 201, shotResult="Made", pts=1, descriptor="transition take") \
        .shot(685, AWAY, 202, made=True).tov(670, HOME, 103).end()
    rows, _ = seg(game, box)
    assert [(r["off_team_id"], r["ending_type"], r["points"]) for r in rows[:3]] == [(HOME, "shot", 2), (AWAY, "shot", 3), (HOME, "turnover", 0)]
    assert rows[1]["finisher_id"] == 202 and rows[1]["retained_ft"]


def test_heave_action_is_a_team_miss_excluded_from_models(game, box):
    """2025-26 feed: a missed end-of-period heave is its own action with no shooter."""
    game.start().shot(20, HOME, 101, made=True) \
        .ev("heave", 0.4, "", AWAY, None, **{F.SHOT_DISTANCE: 67.7, F.AREA: "Mid-Range"}).reb(0.1, AWAY, None, offensive=True).end()
    rows, _ = seg(game, box)
    h = rows[1]
    assert (h["off_team_id"], h["ending_type"], h["shot_family"], h["is_heave"], h["finisher_id"]) == (AWAY, "shot", "heave", True, None)
    assert not h["model_eligible"] and h["shot_made"] is False


def test_buzzer_shot_then_offensive_rebound_keeps_the_shot_label(game, box):
    game.start().shot(1.5, HOME, 101, made=False, area="Restricted Area", dist=2).reb(0.2, HOME, 104, offensive=True).end()
    rows, _ = seg(game, box)
    assert (rows[0]["ending_type"], rows[0]["finisher_id"], rows[0]["n_off_reb"]) == ("shot", 101, 1)


def test_last_two_minutes_penalty_rule_and_double_fouls(game, box):
    game.start().foul(600, AWAY, 201, drawn=101, descriptor="double").foul(100, AWAY, 202, drawn=101) \
        .shot(90, HOME, 101, made=True).end()
    rows, _ = seg(game, box)
    assert rows[0]["def_team_fouls_period"] == 1 and rows[0]["def_in_penalty"]   # one foul inside 2:00 puts AWAY in the penalty


def test_technical_between_periods_is_still_counted(game, box):
    game.start().shot(700, HOME, 101, made=True).end()
    game.ft(0, AWAY, 201, 1, 1, made=True, technical=True)
    game.start(2).shot(700, AWAY, 202, made=True).end()
    rows, info = seg(game, box)
    total = sum(r["points"] + r["tech_ft_points_off"] + r["tech_ft_points_def"] for r in rows)
    assert total == info["final_score"]["home"] + info["final_score"]["away"] == 5


def test_timeout_after_the_deciding_foul_does_not_mark_the_possession(game, box):
    """A timeout between the foul and the free throws happens after the outcome is decided."""
    game.start().foul(700, AWAY, 203, drawn=102, descriptor="shooting").timeout(700, AWAY) \
        .ft(700, HOME, 102, 1, 2, made=True).ft(700, HOME, 102, 2, 2, made=True).shot(680, AWAY, 201, made=True).end()
    rows, _ = seg(game, box)
    assert rows[0]["ending_type"] == "foul_drawn" and not rows[0]["after_timeout"]
    assert not rows[1]["after_timeout"]             # the timeout belonged to the dead ball of the previous possession


# ---- found by reading real 2025-26 games with `python -m tendency inspect` (step 1.2) ----
def test_technical_free_throw_between_basket_and_and_one_does_not_split_it(game, box):
    """Game 0022500166: made layup, shooting foul, a technical free throw missed (dead-ball team rebound), then the and-one."""
    game.start().shot(700, HOME, 101, made=True).foul(700, AWAY, 201, drawn=101, descriptor="shooting") \
        .foul(700, AWAY, 201, sub="technical") \
        .ft(700, HOME, 102, 1, 1, made=False, technical=True).ev("rebound", 700, "offensive", HOME) \
        .ft(700, HOME, 101, 1, 1, made=True).tov(680, AWAY, 202).end()
    rows, _ = seg(game, box)
    assert [(r["off_team_id"], r["ending_type"], r["points"], r["and_one"]) for r in rows[:2]] == [(HOME, "shot", 3, True), (AWAY, "turnover", 0, False)]
    assert rows[0]["finisher_id"] == 101


def test_away_from_play_free_throw_after_a_made_basket_ends_the_possession(game, box):
    """Game 0022500017: basket, away-from-play foul, one free throw by a teammate, then the OTHER team has the ball."""
    game.start().shot(700, HOME, 101, made=True).foul(700, AWAY, 201, drawn=102, descriptor="away-from-play") \
        .ev("freethrow", 700, "1 of 1", HOME, 102, shotResult="Made", pts=1, descriptor="away-from-play") \
        .foul(690, HOME, 103, drawn=201, descriptor="shooting").ft(690, AWAY, 201, 1, 2, made=True).ft(690, AWAY, 201, 2, 2, made=True).end()
    rows, info = seg(game, box)
    assert info["anomalies"] == 0
    assert [(r["off_team_id"], r["ending_type"], r["points"]) for r in rows[:2]] == [(HOME, "shot", 3), (AWAY, "foul_drawn", 2)]
    assert rows[0]["finisher_id"] == 101 and rows[0]["and_one"] and not rows[0]["retained_ft"]


def test_missed_away_from_play_free_throw_after_a_basket_is_live(game, box):
    """Game 0022500847: the free throw is missed and the defense rebounds it."""
    game.start().shot(700, HOME, 101, made=True, three=True).foul(700, AWAY, 201, drawn=102, descriptor="away-from-play") \
        .ev("freethrow", 700, "1 of 1", HOME, 102, shotResult="Missed", descriptor="away-from-play") \
        .reb(698, AWAY, 203).tov(690, AWAY, 204).end()
    rows, info = seg(game, box)
    assert info["anomalies"] == 0
    assert [(r["off_team_id"], r["ending_type"], r["points"]) for r in rows[:2]] == [(HOME, "shot", 3), (AWAY, "turnover", 0)]


def test_two_shot_away_from_play_trip_is_an_ordinary_trip(game, box):
    """Game 0022501070: an off-ball foul in the penalty, two free throws, then the other team's ball."""
    game.start().foul(700, AWAY, 201, drawn=102, descriptor="away-from-play") \
        .ev("freethrow", 700, "1 of 2", HOME, 102, shotResult="Made", pts=1, descriptor="away-from-play") \
        .ev("freethrow", 700, "2 of 2", HOME, 102, shotResult="Made", pts=1, descriptor="away-from-play") \
        .shot(685, AWAY, 202, made=False).reb(683, HOME, 103).end()
    rows, info = seg(game, box)
    assert info["anomalies"] == 0
    assert [(r["off_team_id"], r["ending_type"], r["points"]) for r in rows[:2]] == [(HOME, "foul_drawn", 2), (AWAY, "shot", 0)]


def test_lone_away_from_play_free_throw_still_keeps_the_ball(game, box):
    game.start().foul(700, AWAY, 201, drawn=102, descriptor="away-from-play") \
        .ev("freethrow", 700, "1 of 1", HOME, 102, shotResult="Made", pts=1, descriptor="away-from-play") \
        .shot(685, HOME, 103, made=True).end()
    rows, _ = seg(game, box)
    assert [(r["off_team_id"], r["ending_type"], r["points"], r["retained_ft"]) for r in rows[:1]] == [(HOME, "shot", 3, True)]


def test_basket_plus_flagrant_free_throw_is_two_possessions_for_the_same_team(game, box):
    """The basket ends its possession; the flagrant free throw and the ball that follows are the next one."""
    game.start().shot(700, HOME, 101, made=True).foul(700, AWAY, 201, drawn=101, descriptor="flagrant-type-1") \
        .ev("freethrow", 700, "1 of 1", HOME, 101, shotResult="Made", pts=1, descriptor="flagrant") \
        .foul(690, AWAY, 202, drawn=103, descriptor="shooting").ft(690, HOME, 103, 1, 2, made=True).ft(690, HOME, 103, 2, 2, made=True).end()
    rows, info = seg(game, box)
    assert info["anomalies"] == 0
    assert [(r["off_team_id"], r["ending_type"], r["points"], r["and_one"], r["retained_ft"]) for r in rows[:2]] == \
        [(HOME, "shot", 2, False, False), (HOME, "foul_drawn", 3, False, True)]
    assert rows[1]["finisher_id"] == 103


# ---- found on the first real nba_api fetch (game 0022600001, 2026-11-21): the live feed has no `area` ----
def test_a_shot_at_the_hoop_is_a_rim_shot_whatever_area_says():
    """Tips and dunks are logged at x = y = 0, distance 0, and the feed calls 44% of them "Mid-Range"."""
    tip = {F.ACTION_TYPE: "2pt", F.X_LEGACY: 0.0, F.Y_LEGACY: 0.0, F.SHOT_DISTANCE: 0.0}
    assert shot_family({**tip, F.AREA: "Mid-Range"}) == "rim"
    assert shot_family({**tip, F.AREA: "Restricted Area"}) == "rim"
    assert shot_family(tip) == "rim"
    assert shot_family({**tip, F.SHOT_DISTANCE: 14.0, F.AREA: "Mid-Range"}) == "midrange"        # coordinates missing, distance known: trust area
    assert shot_family({F.ACTION_TYPE: "2pt", F.X_LEGACY: 120.0, F.Y_LEGACY: 90.0, F.SHOT_DISTANCE: 15.0, F.AREA: "Mid-Range"}) == "midrange"


def test_coordinate_rule_matches_the_feeds_own_boundaries():
    two, three = {F.ACTION_TYPE: "2pt"}, {F.ACTION_TYPE: "3pt"}
    assert shot_family({**two, F.X_LEGACY: 0, F.Y_LEGACY: 43}) == "rim"                # 4.3 ft: still restricted area in the feed
    assert shot_family({**two, F.X_LEGACY: 0, F.Y_LEGACY: 46}) == "paint"
    assert shot_family({**two, F.X_LEGACY: 81, F.Y_LEGACY: 20}) == "midrange"          # outside the lane
    assert shot_family({**two, F.X_LEGACY: 10, F.Y_LEGACY: 140}) == "midrange"         # beyond the free-throw line
    assert shot_family({**three, F.X_LEGACY: 225, F.Y_LEGACY: 89}) == "corner3"
    assert shot_family({**three, F.X_LEGACY: 225, F.Y_LEGACY: 90}) == "above_break3"
    assert shot_family({**two, F.SHOT_DISTANCE: 4.3}) == "rim" and shot_family({**two, F.SHOT_DISTANCE: 6}) == "paint"
