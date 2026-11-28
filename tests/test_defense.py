"""Step 3.1: defense proxy fingerprints, the expectation they are measured against, and the half-season check."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tendency import defense, defense_validation
from tendency.contract import validate
from tendency.possessions import SHOT_FAMILIES
from tendency.synth import DEF_QUIRKS, RIM_PROTECTOR, _player_id, _team_id

STARTS = ["made_basket", "def_rebound", "turnover", "free_throw"]
SCHEME_WORDS = ["drop", "switch", "hedge", "blitz", "trap", "zone", "man-to-man"]


def world(n_teams=8, meetings=4, poss=90, seed=0, rim=None, def_rim=None, def_tov=None, start_p=None, rim_by_start=None, weight=None,
          protector=None) -> pd.DataFrame:
    """A hand-built league: one row per possession with just the columns the fingerprint reads.

    `rim[o]` is offense o's rim share of shots; `def_rim[d]` and `def_tov[d]` multiply the odds of a rim
    shot and of a turnover against defense d; `start_p[d]` is the mix of how defense d's opponents start
    possessions; `rim_by_start` multiplies rim odds by start type; `weight[o, d]` scales how many
    possessions o plays against d; `protector` = {team: odds multiplier}: with player 4 of that team on the
    floor (half of its possessions) opponents get to the rim less.
    """
    rng = np.random.default_rng(seed)
    rim = np.full(n_teams, 0.30) if rim is None else np.asarray(rim, float)
    odds = lambda p, m: p * m / (p * m + 1 - p)
    rows, g = [], 0
    for m in range(meetings):
        for a in range(n_teams):
            for b in range(n_teams):
                if a == b:
                    continue
                g += 1
                n = int(round(poss * (1.0 if weight is None else weight[a][b])))
                if n == 0:
                    continue
                sp = np.array(start_p[b] if start_p is not None else [0.4, 0.35, 0.15, 0.10])
                st = rng.choice(4, size=n, p=sp / sp.sum())
                on = rng.random(n) < 0.5                              # is the defense's player 4 on the floor
                p_tov = odds(0.14, 1.0 if def_tov is None else def_tov.get(b, 1.0))
                p_foul = 0.10
                u = rng.random(n)
                end = np.where(u < p_tov, "turnover", np.where(u < p_tov + p_foul, "foul_drawn", "shot"))
                mult = np.ones(n) * (1.0 if def_rim is None else def_rim.get(b, 1.0))
                if rim_by_start is not None:
                    mult = mult * np.asarray(rim_by_start)[st]
                if protector is not None and b in protector:
                    mult = mult * np.where(on, protector[b], 1.0)
                p_rim = odds(rim[a], mult)
                v = rng.random(n)
                rest = (v - p_rim) / np.maximum(1 - p_rim, 1e-9)
                fam = np.where(v < p_rim, "rim", np.array(SHOT_FAMILIES[1:])[np.clip((rest * 4).astype(int), 0, 3)])
                base = [1000 * b + i for i in range(1, 9)]
                for i in range(n):
                    five = [base[3] if not on[i] else base[4]] + base[:3] + [base[5 + (i % 3)]]
                    rows.append((f"{m:02d}{g:06d}", 2025, pd.Timestamp("2025-10-21") + pd.Timedelta(hours=g), a, b, end[i],
                                 fam[i] if end[i] == "shot" else None, STARTS[st[i]], sorted(five)))
    df = pd.DataFrame(rows, columns=["game_id", "season", "tipoff", "off_team_id", "def_team_id", "ending_type", "shot_family", "start_type", "def_lineup"])
    df["is_heave"], df["lineup_ok"] = False, True
    return df


def team_row(est: pd.DataFrame, team: int, cls: str):
    r = est[(est["scope"] == "defense_team") & (est["team_id"] == team) & (est["cls"] == cls)]
    assert len(r) == 1
    return r.iloc[0]


# ---- the expectation ---------------------------------------------------------------------------
def test_a_defense_never_sets_its_own_bar():
    """Rewrite everything that happened against one defense: what was expected of its opponents does not move at all."""
    df = world(seed=1, def_rim={3: 0.5})
    other = df.copy()
    mine = (other["def_team_id"] == 3).values
    other.loc[mine, "ending_type"] = "turnover"
    other.loc[mine, "shot_family"] = None
    a, b = defense.Expectation(df), defense.Expectation(other)
    target = df[df["def_team_id"] == 3]
    for head, cls, _ in defense.OUTCOMES:
        pa, va = a.at(target, head, cls)
        pb, vb = b.at(target, head, cls)
        assert np.array_equal(pa, pb) and np.array_equal(va, vb)
    # ... and the guard has teeth: the same rewrite does move what is expected against every OTHER defense
    elsewhere = df[df["def_team_id"] == 0]
    assert not np.allclose(a.at(elsewhere, "ending_type", "turnover")[0], b.at(elsewhere, "ending_type", "turnover")[0])


def test_no_credit_for_a_schedule_of_poor_offenses():
    """Nobody defends differently. Team 0 just plays the offenses that never get to the rim three times as often."""
    rim = [0.30, 0.18, 0.18, 0.18, 0.42, 0.42, 0.42, 0.30]
    weight = np.ones((8, 8))
    weight[1:4, 0], weight[4:7, 0] = 2.0, 0.4
    df = world(seed=2, rim=rim, weight=weight, poss=120)
    est = defense.estimate(defense.candidates(df, scopes=("defense_team",)))
    r = team_row(est, 0, "rim")
    league = (df["shot_family"] == "rim").sum() / (df["ending_type"] == "shot").sum()
    assert r.k / r.n < league - 0.04                                   # the raw number flatters this defense
    assert abs(defense._row(r)["deviation"]) < 0.012                   # the fingerprint does not
    assert defense._row(r)["interval"][0] < 0 < defense._row(r)["interval"][1]


def test_no_blame_for_fast_breaks_its_own_offense_gave_away():
    """Rim shots are easier off a turnover. Team 0's opponents start that way three times as often; its defense is ordinary."""
    start_p = {d: [0.40, 0.35, 0.15, 0.10] for d in range(8)}
    start_p[0] = [0.20, 0.20, 0.50, 0.10]
    df = world(seed=3, start_p=start_p, rim_by_start=[1.0, 1.2, 2.2, 1.0], poss=120)
    with_step = defense._row(team_row(defense.estimate(defense.candidates(df, scopes=("defense_team",))), 0, "rim"))
    flat = defense.Expectation(df, by_start=False)
    without = defense._row(team_row(defense.estimate(defense.candidates(df, flat, scopes=("defense_team",))), 0, "rim"))
    assert without["deviation"] > 0.02 and without["excludes_zero"]    # blamed for its offense's turnovers
    assert abs(with_step["deviation"]) < without["deviation"] / 2 and with_step["expected_rate"] > without["expected_rate"] + 0.02


# ---- team rows ------------------------------------------------------------------------------------
def test_a_real_difference_is_found_with_the_right_sign_and_the_rest_stay_near_zero():
    df = world(seed=4, n_teams=12, def_rim={5: 0.6}, def_tov={9: 1.6}, poss=100)
    est = defense.estimate(defense.candidates(df, scopes=("defense_team",)))
    rim, tov = defense._row(team_row(est, 5, "rim")), defense._row(team_row(est, 9, "turnover"))
    assert rim["deviation"] < -0.05 and rim["interval"][1] < 0 and rim["prob_beyond"] > 0.99
    assert tov["deviation"] > 0.03 and tov["interval"][0] > 0
    for t in range(12):
        if t != 5:
            assert abs(defense._row(team_row(est, t, "rim"))["deviation"]) < abs(rim["deviation"]) / 3
        if t != 9:
            assert abs(defense._row(team_row(est, t, "turnover"))["deviation"]) < tov["deviation"] / 3
        assert abs(defense._row(team_row(est, t, "foul_drawn"))["deviation"]) < 0.01      # nothing planted there


def test_a_row_adds_up_and_a_small_sample_is_pulled_toward_zero():
    big = world(seed=5, def_rim={2: 0.6}, poss=100)
    small = big[big["game_id"].isin(big.drop_duplicates("game_id")["game_id"].iloc[::6])]
    rows = {}
    for name, df in (("big", big), ("small", small)):
        r = defense._row(team_row(defense.estimate(defense.candidates(df, scopes=("defense_team",))), 2, "rim"))
        assert abs(r["shrunk_rate"] - r["expected_rate"] - r["deviation"]) < 2e-4
        assert r["interval"][0] <= r["deviation"] <= r["interval"][1] and r["k"] <= r["n"] and 0 <= r["prob_beyond"] <= 1
        rows[name] = r
    raw_gap = lambda r: r["raw_rate"] - r["expected_rate"]
    assert rows["small"]["n"] < rows["big"]["n"] / 4
    assert abs(rows["small"]["deviation"]) < abs(raw_gap(rows["small"]))                 # pooled toward the opponents' expectation
    assert rows["small"]["interval"][1] - rows["small"]["interval"][0] > rows["big"]["interval"][1] - rows["big"]["interval"][0]


def test_drift_widens_the_interval_and_none_is_found_where_nothing_changes():
    df = world(seed=6, def_rim={2: 0.6}, n_teams=12, meetings=6)
    cand = defense.candidates(df, scopes=("defense_team",))
    as_is = defense._row(team_row(defense.estimate(cand), 2, "rim"))
    drift = {"rate": {s: 0.01 for s in defense.SCOPES}, "dev": {k: 0.01 for k in defense.KINDS}}
    wide = defense._row(team_row(defense.estimate(cand, drift), 2, "rim"))
    assert wide["interval"][1] - wide["interval"][0] > as_is["interval"][1] - as_is["interval"][0] and wide["deviation"] == as_is["deviation"]
    measured = defense.estimate_drift(df)                              # the same defenses in both halves
    assert measured["n_cells"]["defense_team"] == 12 * 7 and measured["rate"]["defense_team"] < 0.02 ** 2


# ---- unit rows: built and checked, not shown --------------------------------------------------------
def test_a_planted_rim_protector_shows_in_his_lineups_and_a_unit_is_pooled_toward_its_team():
    df = world(seed=7, protector={3: 0.45}, def_rim={6: 0.6}, poss=100)
    est = defense.estimate(defense.candidates(df))
    units = est[(est["scope"] == "defense_player") & (est["cls"] == "rim")]
    him = units[(units["team_id"] == 3) & (units["unit"] == "3005")].iloc[0]          # player 4 of team 3: on for half its possessions
    out = units[(units["team_id"] == 3) & (units["unit"] == "3004")].iloc[0]          # the man who plays when he sits
    assert him.dev_logit / him.dev_sd < -3 and out.dev_logit / out.dev_sd > 3
    assert him.fp_logit < team_row(est, 3, "rim").fp_logit < out.fp_logit
    # team 6 defends the rim the same way whoever plays: its players' rows sit at the team's number, not at zero
    team6 = team_row(est, 6, "rim").fp_logit
    for r in units[units["team_id"] == 6].itertuples():
        assert abs(r.dev_logit) < abs(him.dev_logit) / 3 and abs(r.fp_logit - team6) < 0.15 and r.fp_logit < -0.25
    assert set(est["scope"]) == set(defense.SCOPES) and (est[est["scope"] == "defense_lineup"]["unit"].str.count("-") == 4).all()


def test_possessions_with_an_unknown_five_count_for_the_team_and_for_no_unit():
    df = world(seed=8, poss=60)
    df.loc[df.index[::2], "lineup_ok"] = False
    df["def_lineup"] = [five if ok else five[:3] for five, ok in zip(df["def_lineup"], df["lineup_ok"])]
    cand = defense.candidates(df)
    team = cand[(cand["scope"] == "defense_team") & (cand["cls"] == "turnover")]
    assert team["n"].sum() == len(df)
    units = cand[(cand["scope"] == "defense_player") & (cand["cls"] == "turnover")]
    assert (units["n"] + units["n_out"]).max() <= len(df) / 8 + 1e-9 and (units["n"] <= df["lineup_ok"].groupby(df["def_team_id"]).sum().max()).all()


# ---- the half-season check ----------------------------------------------------------------------------
def test_the_check_credits_a_real_defense_and_not_an_imagined_one():
    protectors = {t: 0.45 for t in (0, 2, 3, 5, 8)}
    real = defense_validation.validate(world(seed=9, n_teams=10, meetings=6, def_rim={1: 0.55, 4: 1.6}, def_tov={7: 1.7}, protector=protectors))
    none = defense_validation.validate(world(seed=9, n_teams=10, meetings=6))
    pick = lambda v, scope, grp: next(s for s in v["slices"] if s["scope"] == scope and s["outcomes"] == grp)
    ship = lambda s: next(m for m in s["methods"] if m["method"] == s["shipped"])
    for grp in ("shot_profile", "turnover"):
        s = pick(real, "defense_team", grp)
        assert ship(s)["verdict"] == "ahead" and len(ship(s)["vs_unadjusted"]["interval"]) == 2 and s["picked"] is None
        assert next(m for m in s["methods"] if m["method"] == s["reference"])["vs_reference"] is None
        assert ship(pick(none, "defense_team", grp))["verdict"] != "ahead"
    assert ship(pick(real, "defense_team", "foul_drawn"))["verdict"] != "ahead"             # nothing planted there
    # units: where half the teams have a rim protector, a player's own row beats his team's; where none has, it does not
    units = pick(real, "defense_player", "shot_profile")
    assert ship(units)["verdict"] == "ahead"
    assert units["picked"]["verdict"] == "ahead" and 10 <= units["picked"]["n_cells"] < units["n_cells"] / 2      # five protectors, five stand-ins, rim and the rest
    flat = pick(none, "defense_player", "shot_profile")
    assert ship(flat)["verdict"] != "ahead" and flat["picked"]["verdict"] != "ahead" and flat["picked"]["n_cells"] <= 3
    assert {c["scope"] for c in real["coverage"]} == set(defense.SCOPES)
    assert real["n_games_first"] == real["n_games_second"] and real["split"] == "by date"


def test_the_check_fits_on_the_first_half_only():
    """Scramble what happened in the second half: every method's fit is the same, so the reference's rate for a game is the same."""
    df = world(seed=10, n_teams=8, meetings=4, def_rim={1: 0.55})
    first, second = defense.split_halves(df)
    other = second.copy()
    other["ending_type"] = np.random.default_rng(0).permutation(other["ending_type"].values)
    other.loc[other["ending_type"] != "shot", "shot_family"] = None
    other.loc[(other["ending_type"] == "shot") & other["shot_family"].isna(), "shot_family"] = "rim"
    exp_a = defense.Expectation(defense.split_halves(pd.concat([first, second]))[0])
    exp_b = defense.Expectation(defense.split_halves(pd.concat([first, other]))[0])
    for key in exp_a.fit:
        assert all(np.array_equal(x, y) for x, y in zip(exp_a.fit[key], exp_b.fit[key]))
    a, b = defense_validation.validate(pd.concat([first, second])), defense_validation.validate(pd.concat([first, other]))
    assert a["slices"][0]["methods"][0]["log_loss"] != b["slices"][0]["methods"][0]["log_loss"]      # the scores do move; the fits cannot


# ---- the synthetic league, the export and the contract ---------------------------------------------------
def test_the_simulators_planted_defenses_reach_the_team_reports(synth_run):
    out = synth_run["out"]
    rep = lambda i: json.loads((out / "team_reports" / f"{_team_id(i)}.json").read_text())["defense"]
    row = lambda d, cls: next(r for r in d["rows"] if r["outcome"]["class"] == cls)
    (tov_team, (_, tov_x)), (rim_team, (_, rim_x)) = [(i, q) for i, q in DEF_QUIRKS.items() if q[0] == "turnover"][0], [(i, q) for i, q in DEF_QUIRKS.items() if q[0] == "rim"][0]
    assert tov_x > 1 > rim_x
    forced, denied, protected = row(rep(tov_team), "turnover"), row(rep(rim_team), "rim"), row(rep(RIM_PROTECTOR[0]), "rim")
    assert forced["deviation"] > 0.02 and forced["interval"][0] > 0
    assert denied["deviation"] < -0.04 and denied["interval"][1] < 0
    assert protected["deviation"] < -0.02 and protected["interval"][1] < 0
    for i in range(8):
        d = rep(i)
        assert [r["outcome"]["class"] for r in d["rows"]] == [c for _, c, _ in defense.OUTCOMES] and d["n_possessions"] > 1000
        assert abs(row(d, "foul_drawn")["deviation"]) < 0.01                               # the simulator plants nothing on fouls
        if i != tov_team:
            assert row(d, "turnover")["deviation"] < forced["deviation"] / 2
        if i not in (rim_team, RIM_PROTECTOR[0]):
            assert row(d, "rim")["deviation"] > denied["deviation"] / 2


def test_the_simulators_rim_protector_is_the_most_negative_player_on_his_team(synth_run):
    poss = synth_run["poss"]
    est = defense.estimate(defense.candidates(poss[poss["season"] == poss["season"].max()], scopes=("defense_team", "defense_player")))
    team = est[(est["scope"] == "defense_player") & (est["cls"] == "rim") & (est["team_id"] == _team_id(RIM_PROTECTOR[0]))]
    z = (team["dev_logit"] / team["dev_sd"]).values
    assert team["unit"].iloc[int(np.argmin(z))] == str(_player_id(RIM_PROTECTOR[0], RIM_PROTECTOR[1])) and z.min() < -3


def test_export_has_a_defense_section_that_names_no_scheme(synth_run):
    out = synth_run["out"]
    man = json.loads((out / "manifest.json").read_text())
    assert man["contract_version"] == "1.7" and "defense proxies" not in man["not_built"]
    for t in json.loads((out / "teams.json").read_text())["teams"]:
        d = json.loads((out / "team_reports" / f"{t['team_id']}.json").read_text())["defense"]
        assert "proxy from outcomes" in d["note"].lower() and "cannot identify" in d["note"] and d["season"] == max(man["coverage"]["seasons"])
        assert len(d["rows"]) == 7 and all(r["n"] >= defense.MIN_N and r["interval"][0] <= r["deviation"] <= r["interval"][1] for r in d["rows"])
        text = json.dumps({k: v for k, v in d.items() if k != "note"}).lower()
        assert not [w for w in SCHEME_WORDS if w in text]
        assert not [w for w in SCHEME_WORDS if w in d["note"].lower()]                    # the limit is stated without a list of names
    for text in (defense.LIMIT, defense.BASELINE, *defense.SCOPES.values(), *defense.KINDS.values(), *(lab for _, _, lab in defense.OUTCOMES),
                 *defense_validation.TEAM_METHODS.values(), *defense_validation.UNIT_METHODS.values(), *(lab for _, lab in defense_validation.GROUPS)):
        assert not [w for w in SCHEME_WORDS if w in text.lower()], text
    dv = json.loads((out / "tendency_validation.json").read_text())["defense"]
    assert dv["available"] and {s["scope"] for s in dv["slices"]} == set(defense.SCOPES) and dv["season"] == max(man["coverage"]["seasons"])
    team = [s for s in dv["slices"] if s["scope"] == "defense_team"]
    assert [s["outcomes"] for s in team] == ["shot_profile", "foul_drawn", "turnover"]
    assert all(len(s["methods"]) == 5 and s["reference"] == "opponents" and s["shipped"] == "fingerprint" for s in team)
    verdict = {s["outcomes"]: next(m for m in s["methods"] if m["method"] == "fingerprint")["verdict"] for s in team}
    assert verdict["shot_profile"] == verdict["turnover"] == "ahead" and verdict["foul_drawn"] != "ahead"
    assert dv["drift"]["available"] and dv["drift"]["season"] == min(man["coverage"]["seasons"])


def test_an_export_without_the_defense_blocks_is_still_valid(synth_run):
    out = synth_run["out"]
    team = json.loads((out / "teams.json").read_text())["teams"][0]["team_id"]
    rep = json.loads((out / "team_reports" / f"{team}.json").read_text())
    validate(rep, "team_report")
    rep.pop("defense")
    validate(rep, "team_report")
    tv = json.loads((out / "tendency_validation.json").read_text())
    tv.pop("defense")
    validate(tv, "tendency_validation")
    man = json.loads((out / "manifest.json").read_text())
    validate({**man, "contract_version": "1.5", "not_built": man["not_built"] + ["defense proxies"]}, "manifest")


def test_a_report_section_needs_no_earlier_season_and_says_so():
    df = world(seed=11, poss=60)
    sec = defense.team_sections(df)
    assert set(sec) == set(range(8)) and all(not s["drift_included"] and len(s["rows"]) == 7 for s in sec.values())
    with_drift = defense.team_sections(df, {"rate": {s: 0.01 for s in defense.SCOPES}, "dev": {k: 0.01 for k in defense.KINDS}})
    assert all(s["drift_included"] for s in with_drift.values())
    validate({"team_id": 0, "data_kind": "synthetic", "as_of_utc": "2025-10-21T00:00:00Z", "n_games": 1, "n_possessions": 1, "n_candidates": 0,
              "method": {"id": "x", "label": "x", "final": True}, "interval_level": 0.9, "sort": "rank_score_desc", "tendencies": [], "defense": sec[0]}, "team_report")


def test_the_pregame_path_never_reaches_the_defense_code():
    """`train`, `predict` and `score` freeze and score the possession heads. A defense section is an export-time description."""
    from tendency import features, models, prospective
    for mod in (prospective, models, features):
        assert not [n for n, v in vars(mod).items() if getattr(v, "__name__", "").startswith("tendency.defense")], mod.__name__
