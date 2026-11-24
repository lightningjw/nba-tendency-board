"""Step 2.2 labels: action style, creation, tempo, fast break, two-for-one. And the two heads built on them."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tendency import cdn_fields as F
from tendency import features as ft
from tendency import tendency_layer as layer
from tendency.features import Features, build_features
from tendency.leakcheck import ingame_leaks, pregame_leaks, scramble_outcomes
from tendency.possessions import ACTION_STYLES, action_style, segment_game, tempo_class, untimed_rebounds

from conftest import A, AWAY, H, HOME

# Every (subType, descriptor) pair seen on 657,380 real shots, 2023-24 to 2025-26, and the style it must get.
SEEN = {
    "plain_jumper": [("Jump Shot", ""), ("Jump Shot", "bank")],
    "drive": [("Layup", "driving"), ("Layup", "driving finger roll"), ("Layup", "driving reverse"), ("DUNK", "driving"), ("Hook", "driving"),
              ("Jump Shot", "driving bank"), ("Hook", "driving bank"), ("DUNK", "driving reverse")],
    "pullup": [("Jump Shot", "pullup"), ("Jump Shot", "running pullup"), ("Jump Shot", "pullup bank")],
    "running": [("Layup", "running"), ("Jump Shot", "running"), ("DUNK", "running"), ("Layup", "running finger roll"), ("Layup", "running reverse"),
                ("DUNK", "running reverse")],
    "floater": [("Jump Shot", "driving floating"), ("Jump Shot", "floating"), ("Jump Shot", "driving floating bank")],
    "turnaround_hook": [("Jump Shot", "fadeaway"), ("Jump Shot", "turnaround fadeaway"), ("Jump Shot", "turnaround"), ("Hook", "turnaround"), ("Hook", ""),
                        ("Jump Shot", "turnaround bank"), ("Hook", "turnaround bank"), ("Jump Shot", "fadeaway bank"),
                        ("Jump Shot", "turnaround fadeaway bank"), ("Hook", "bank")],
    "cut_lob": [("Layup", "cutting"), ("DUNK", "cutting"), ("DUNK", "alley-oop"), ("Layup", "alley-oop"), ("Layup", "cutting finger roll"),
                ("DUNK", "running alley-oop"), ("Layup", "running alley-oop")],
    "step_back": [("Jump Shot", "step back"), ("Jump Shot", "step back bank")],
    "putback": [("Layup", "tip"), ("Layup", "putback"), ("DUNK", "tip"), ("DUNK", "putback")],
    "plain_finish": [("Layup", ""), ("DUNK", ""), ("Layup", "reverse"), ("Layup", "finger roll"), ("DUNK", "reverse")],
}


def test_every_shot_seen_in_the_feed_maps_to_the_style_the_table_says():
    assert set(SEEN) == set(ACTION_STYLES) and len(ACTION_STYLES) == len(F.ACTION_STYLE_LABELS) == 10
    for style, pairs in SEEN.items():
        for sub, desc in pairs:
            assert action_style({F.SUB_TYPE: sub, F.DESCRIPTOR: desc}) == style, (sub, desc)
    assert sum(len(v) for v in SEEN.values()) == 50
    # a descriptor the feed has never sent still gets a style, from the subType alone
    assert action_style({F.SUB_TYPE: "Jump Shot", F.DESCRIPTOR: "something new"}) == "plain_jumper"
    assert action_style({F.SUB_TYPE: "shot"}) == "plain_finish" and action_style({}) == "plain_finish"


def test_tempo_classes_and_the_full_shot_clock_rule():
    assert tempo_class(600, 592) == "early" and tempo_class(600, 591.9) == "middle"          # 8 s is still early
    assert tempo_class(600, 584) == "late" and tempo_class(600, 584.1) == "middle"           # 16 s is already late
    assert tempo_class(23.9, 10) is None and tempo_class(24, 10) == "middle"                 # less than a shot clock left: not defined
    assert tempo_class(600, None) is None


# ---- the segmenter ------------------------------------------------------------------------------------
def rows_of(game, box):
    rows, info = segment_game(game.pbp(), box)
    return rows, info


def test_style_assist_and_fast_break_come_from_the_deciding_shot(game, box):
    g = game.start()
    g.shot(700, HOME, H[0], True, descriptor="cutting", assistPersonId=H[1], qualifiers=["fastbreak", "pointsinthepaint"])     # 0: assisted cut, fast break
    g.shot(690, AWAY, A[0], True, three=True, descriptor="pullup")                                                              # 1: unassisted pull-up three
    g.shot(680, HOME, H[2], False, descriptor="step back").reb(678, AWAY, A[1])                                                 # 2: a miss has no assist to credit
    g.tov(670, AWAY, A[2])                                                                                                      # 3: turnover
    g.end()
    rows, _ = rows_of(g, box)
    a, b, c, d = rows[:4]
    assert (a["action_style"], a["assisted"], a["assister_id"], a["fast_break"]) == ("cut_lob", True, H[1], True)
    assert (b["action_style"], b["assisted"], b["assister_id"], b["fast_break"]) == ("pullup", False, None, False)
    assert (c["action_style"], c["assisted"], c["assister_id"]) == ("step_back", None, None)
    assert (d["action_style"], d["assisted"], d["fast_break"]) == (None, None, None)       # turnovers carry no style and no scorer tag
    assert d["secs_to_first_action"] == 8.0 and d["tempo"] == "early"                      # 678 -> 670


def test_fast_break_on_a_trip_to_the_line_is_read_from_the_free_throws(game, box):
    g = game.start()
    g.tov(700, AWAY, A[0])
    g.foul(695, AWAY, A[1], drawn=H[0], descriptor="shooting")
    g.ft(695, HOME, H[0], 1, 2, True, qualifiers=["fastbreak", "fromturnover"]).ft(695, HOME, H[0], 2, 2, True, qualifiers=["fastbreak", "fromturnover"])
    g.end()
    rows, info = rows_of(g, box)
    trip = rows[1]
    assert trip["ending_type"] == "foul_drawn" and trip["fast_break"] is True and trip["action_style"] is None
    assert trip["secs_to_first_action"] == 5.0                                             # the foul decided it, at 695
    assert info["tag_checks"] == 1 and info["from_turnover_agree"] == 1 and info["second_chance_agree"] == 1


def test_an_offensive_rebound_does_not_restart_the_clock_on_tempo(game, box):
    g = game.start()
    g.shot(700, AWAY, A[0], True)                                                          # home ball from 700
    g.shot(694, HOME, H[0], False).reb(692, HOME, H[1], offensive=True)
    g.shot(670, HOME, H[1], True, descriptor="putback")
    g.end()
    rows, _ = rows_of(g, box)
    r = rows[1]
    assert r["n_off_reb"] == 1 and r["action_style"] == "putback"
    assert r["first_action_clock_s"] == 694.0 and r["secs_to_first_action"] == 6.0 and r["tempo"] == "early"


def test_two_for_one_window_is_a_situation_and_taken_is_its_label(game, box):
    g = game.start()
    g.shot(40, AWAY, A[0], True)                    # home gets it with 40 s left: in the window
    g.shot(31, HOME, H[0], True)                    # shoots with 31 left: took it. Away then starts at 31: also in the window
    g.shot(10, AWAY, A[1], True)                    # holds until 10: passed it up. Home starts at 10: outside
    g.shot(2, HOME, H[1], False)
    g.end()
    rows, _ = rows_of(g, box)
    assert [r["two_for_one_window"] for r in rows[:4]] == [False, True, True, False]
    assert [r["two_for_one_taken"] for r in rows[:4]] == [None, True, False, None]
    assert rows[3]["tempo"] is None                 # 10 seconds left at the start: no full shot clock, so no tempo


def _rebound_game(game, timed: bool):
    g = game.start()
    clk = 700
    for i in range(12):
        off, deff = (HOME, AWAY) if i % 2 == 0 else (AWAY, HOME)
        shooter, rebounder = (H[0], A[0]) if off == HOME else (A[0], H[0])
        g.shot(clk, off, shooter, False).reb(clk - 3 if timed else clk, deff, rebounder)
        clk -= 12
    return g.end()


def test_untimed_rebounds_are_detected_per_game_and_only_move_the_tempo_labels(game, box):
    """The live feed stamps a defensive rebound with the missed shot's clock; the archives stamp it about 3 s later."""
    from conftest import Game
    timed, live = _rebound_game(Game(), True), _rebound_game(Game(), False)
    assert untimed_rebounds(live.actions) and not untimed_rebounds(timed.actions)
    rt, it = segment_game(timed.pbp(), box)
    rl, il = segment_game(live.pbp(), box)
    assert il["untimed_rebounds"] and not it["untimed_rebounds"]
    # same basketball, two feeds: 9 seconds from securing the rebound to the next shot in both
    assert [r["secs_to_first_action"] for r in rt[1:11]] == [9.0] * 10
    assert [r["secs_to_first_action"] for r in rl[1:11]] == [9.0] * 10
    assert [r["start_clock_s"] for r in rl[1:4]] == [700.0, 688.0, 676.0]                  # the start clock itself is left as the feed gave it
    assert [r["start_clock_s"] for r in rt[1:4]] == [697.0, 685.0, 673.0]


# ---- leakage: the new labels are outcomes ---------------------------------------------------------------
def _games(poss):
    return poss.drop_duplicates("game_id").sort_values("tipoff")["game_id"].tolist()


def test_the_three_logged_heads_keep_exactly_the_features_a_frozen_model_was_trained_on():
    """A model frozen before step 2.2 is still scored every night. Its columns must not move."""
    poss25 = (["period", "start_clock_s", "game_secs_left", "margin_start", "off_is_home", "def_in_penalty", "def_team_fouls_period",
               "off_team_fouls_period", "off_timeouts_left", "def_timeouts_left", "after_timeout", "start_type_code", "clutch"]
              + ["base_end_shot", "base_end_foul_drawn", "base_end_turnover", "base_fam_rim", "base_fam_paint", "base_fam_midrange", "base_fam_corner3",
                 "base_fam_above_break3", "team_eff_n", "lineup_usage_sum", "lineup_end_shot", "lineup_end_foul_drawn", "lineup_end_turnover",
                 "lineup_fam_rim", "lineup_fam_paint", "lineup_fam_midrange", "lineup_fam_corner3", "lineup_fam_above_break3"]
              + ["ig_team_poss", "ig_end_shot", "ig_end_foul_drawn", "ig_end_turnover", "ig_fam_rim", "ig_fam_paint", "ig_fam_midrange", "ig_fam_corner3",
                 "ig_fam_above_break3"])
    assert ft.FEATURES["ending_type"] == poss25 and ft.FEATURES["shot_family"] == poss25
    assert ft.FEATURES["finisher"] == poss25[:13] + ["p_usage", "p_base", "p_eff_on", "p_usage_rank", "p_end_shot", "p_end_foul_drawn", "p_end_turnover",
                                                     "ig_p_fin", "ig_p_on", "ig_p_last"]


def test_new_head_features_are_registered_and_none_is_a_label_of_the_same_possession():
    labels = {"action_style", "assisted", "assister_id", "first_action_clock_s", "secs_to_first_action", "tempo", "fast_break", "two_for_one_taken",
              "end_clock_s", "points", "shot_made", "n_off_reb", "and_one"}
    for head in ("action_style", "tempo"):
        cols = ft.FEATURES[head]
        assert len(cols) == len(set(cols)) and not labels & set(cols)
        assert all(c in ft.KIND for c in cols)
    assert ft.KIND["two_for_one_window"] == "context"                  # known at the start: a situation, not an outcome
    # tempo asks when the first action comes, so it may not see anything read mid-possession
    mid = {"after_timeout", "def_in_penalty", "def_team_fouls_period", "off_team_fouls_period", "off_timeouts_left", "def_timeouts_left"}
    assert not mid & set(ft.FEATURES["tempo"]) and not [c for c in ft.FEATURES["tempo"] if c.startswith("lineup_")]
    assert mid <= set(ft.FEATURES["action_style"])                     # the other heads predict what happens next and may
    assert all(ft.KIND[c] == "pregame" for c in ft.PREGAME_STYLE + ft.PREGAME_TEMPO)
    assert all(ft.KIND[c] == "ingame" for c in ft.INGAME_STYLE + ft.INGAME_TEMPO)


def test_scrambling_rewrites_the_new_labels_so_the_guard_has_something_to_catch(small_poss):
    mask = np.ones(len(small_poss), bool)
    alt = scramble_outcomes(small_poss, mask)
    both = small_poss["tempo"].notna().values
    assert (alt["tempo"].values[both] != small_poss["tempo"].values[both]).all()
    assert (alt["action_style"].fillna("-").values != small_poss["action_style"].fillna("-").values).mean() > 0.6


def test_no_pregame_or_ingame_leak_in_the_new_features(small_poss):
    games = _games(small_poss)
    assert pregame_leaks(build_features, small_poss, games[30], ft.PREGAME_STYLE + ft.PREGAME_TEMPO, []) == []
    for k in (1, 60, 150):
        assert ingame_leaks(build_features, small_poss, games[20], k, ft.INGAME_STYLE + ft.INGAME_TEMPO, []) == []


def test_the_new_pregame_features_do_move_when_earlier_games_change(small_poss):
    games = _games(small_poss)
    tip = small_poss.loc[small_poss["game_id"] == games[30], "tipoff"].iloc[0]
    a = build_features(small_poss).poss
    b = build_features(scramble_outcomes(small_poss, (small_poss["tipoff"] < tip).values)).poss
    for col in ("base_sty_floater", "lineup_sty_floater", "start_sty_floater", "base_tmp_early"):
        va, vb = a.loc[a["game_id"] == games[30], col].values, b.loc[b["game_id"] == games[30], col].values
        n = min(len(va), len(vb))
        assert not np.allclose(va[:n], vb[:n]), col


def _leaky(col: str, value: str):
    """A feature that reads the current possession's own label: exactly what the rule forbids."""
    def build(poss):
        f = build_features(poss)
        p = poss.sort_values(["game_id", "poss_idx"]).copy()
        p["leaky"] = (p[col] == value).astype(int).groupby([p["game_id"], p["off_team_id"]]).cumsum()
        return Features(f.poss.merge(p[["game_id", "poss_idx", "leaky"]], on=["game_id", "poss_idx"]), f.cand)
    return build


@pytest.mark.parametrize("col,value", [("tempo", "early"), ("action_style", "floater")])
def test_the_guard_catches_a_feature_built_from_the_possessions_own_label(small_poss, col, value):
    gid = _games(small_poss)[20]
    g = small_poss[(small_poss["game_id"] == gid) & small_poss["model_eligible"]]
    # scrambling turns a turnover into a floater and flips every tempo, so start at a turnover that has a tempo
    k = int(g.loc[(g["poss_idx"] >= 40) & (g["ending_type"] == "turnover") & g["tempo"].notna(), "poss_idx"].iloc[0])
    assert ingame_leaks(_leaky(col, value), small_poss, gid, k, ["leaky"], []) == ["leaky"]


def test_features_build_on_a_table_from_before_step_2_2(small_poss):
    """The pregame log builds reference rows without the new columns; the three logged heads must not notice."""
    old = small_poss.drop(columns=["action_style", "tempo", "two_for_one_window"])
    a, b = build_features(small_poss).poss, build_features(old).poss
    cols = ft.FEATURES["ending_type"]
    assert np.allclose(a[cols].astype(float).values, b[cols].astype(float).values)


# ---- baselines ------------------------------------------------------------------------------------------
def test_new_baselines_are_probabilities_and_tempo_knows_how_the_possession_started(synth_run):
    f = build_features(synth_run["poss"]).poss
    sty = f[[f"start_sty_{c}" for c in ACTION_STYLES]].values
    tmp = f[[f"base_tmp_{c}" for c in F.TEMPO_CLASSES]].values
    assert np.allclose(sty.sum(axis=1), 1, atol=1e-6) and np.allclose(tmp.sum(axis=1), 1, atol=1e-6) and sty.min() > 0 and tmp.min() > 0
    late = f[f["tipoff"] > f["tipoff"].quantile(0.5)]
    codes = {s: i for i, s in enumerate(ft.START_TYPES)}
    quick = late.loc[late["start_type_code"] == codes["turnover"], "base_tmp_early"].mean()
    slow = late.loc[late["start_type_code"] == codes["made_basket"], "base_tmp_early"].mean()
    assert quick > slow + 0.1                       # the simulator runs after turnovers; a baseline blind to start type would not see it


# ---- export -----------------------------------------------------------------------------------------------
def test_export_carries_five_heads_label_coverage_and_the_new_fields(synth_run):
    out = synth_run["out"]
    man = json.loads((out / "manifest.json").read_text())
    assert man["contract_version"] == "1.3"
    assert [h["id"] for h in man["heads"]] == ["finisher", "ending_type", "shot_family", "action_style", "tempo"]
    assert [h["logged_before_tipoff"] for h in man["heads"]] == [True, True, True, False, False]
    assert man["heads"][3]["classes"] == ACTION_STYLES and len(man["heads"][3]["class_labels"]) == 10
    labels = {l["id"]: l for l in man["labels"]}
    assert set(labels) == {"action_style", "creation", "secs_to_first_action", "tempo", "fast_break", "two_for_one"}
    for l in labels.values():
        assert 0 < l["coverage"] <= 1 and l["n_defined"] <= l["n_possessions"] and l["coverage"] == pytest.approx(l["n_defined"] / l["n_possessions"], abs=1e-4)
    assert labels["creation"]["coverage"] < labels["action_style"]["coverage"] < labels["tempo"]["coverage"]
    assert "second-chance agrees" in labels["fast_break"]["note"]
    sc = json.loads((out / "scorecard.json").read_text())
    assert [h["head"] for h in sc["heads"]] == ["finisher", "ending_type", "shot_family", "action_style", "tempo"]
    for h in sc["heads"][3:]:
        for s in h["splits"]:
            assert s["available"] and s["n_possessions"] > 1000 and s["verdict"] in ("model_better", "baseline_better", "inconclusive")
    games = json.loads((out / "games.json").read_text())["games"]
    rep = json.loads((out / "replays" / f"{games[0]['game_id']}.json").read_text())
    seen = set()
    for p in rep["possessions"]:
        a = p["actual"]
        assert (a["action_style"] is not None) == (a["ending_type"] == "shot" and a["shot_family"] != "heave")
        assert (a["assisted"] is not None) == (a["action_style"] is not None and a["shot_made"] is True)
        seen.add(a["action_style"])
        if p["pred"]:
            assert len(p["pred"]["action_style"]["model"]) == 10 and len(p["pred"]["tempo"]["model"]) == 3
            assert abs(sum(p["pred"]["action_style"]["model"]) - 1) < 3e-3 and abs(sum(p["pred"]["tempo"]["baseline"]) - 1) < 3e-3
    assert len(seen - {None}) >= 7


def test_team_reports_can_show_the_new_outcomes_and_a_planted_style_quirk_is_found(synth_run):
    """The simulator makes team CED take three times its usual share of pull-ups in the fourth quarter."""
    rows = layer.all_rows(synth_run["poss"], {})
    heads = {r["outcome"]["head"] for rs in rows.values() for r in rs}
    assert {"action_style", "creation", "tempo", "fast_break"} <= heads
    assert any(r["situation"]["id"] == "two_for_one" for rs in rows.values() for r in rs)
    assert not [r for rs in rows.values() for r in rs if r["outcome"]["head"] == "tempo" and r["situation"]["id"] in ("after_timeout", "def_in_penalty")]
    assert [r for rs in rows.values() for r in rs if r["outcome"]["head"] == "tempo" and r["situation"]["id"] == "start_turnover"]
    hit = [r for r in rows[9900003] if r["situation"]["id"] == "q4" and r["outcome"] == {"head": "action_style", "class": "pullup", "label": "Pull-up share of shots"}][0]
    assert hit["deviation"] > 0.02 and hit["excludes_zero"] and hit["prob_beyond"] > 0.9
    control = [r for r in rows[9900008] if r["situation"]["id"] == "q4" and r["outcome"]["class"] == "pullup"][0]
    assert not control["excludes_zero"]
