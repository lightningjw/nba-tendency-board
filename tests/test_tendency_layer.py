"""The partial-pooling tendency layer, its half-season validation, and the hot-hand report."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from tendency import hot_hand, tendencies as standin
from tendency import tendency_layer as layer
from tendency import tendency_validation as tv
from tendency.contract import validate, validate_export_dir


# ---- hand-built possessions: enough columns for the tendency code, nothing else ---------------------
def make_poss(rows: list[dict]) -> pd.DataFrame:
    base = {"season": 2090, "period": 1, "start_clock_s": 600.0, "margin_start": 0, "start_type": "made_basket", "after_timeout": False,
            "def_in_penalty": False, "is_heave": False, "ending_type": "shot", "shot_family": "rim", "shot_made": False, "points": 0,
            "lineup_ok": True, "off_team_id": 1}
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["poss_idx"] = df.groupby("game_id").cumcount()
    df["tipoff"] = pd.Timestamp("2090-10-01") + pd.to_timedelta(df["game_id"].astype(int), unit="D")
    df["finisher_id"] = df["finisher_id"].astype("Int64")
    df["shot_made"] = df["shot_made"].astype("boolean")
    return df


def weak_bench_poss(seed: int = 0) -> pd.DataFrame:
    """Player 10 (strength 3) closes games with four weak teammates. His share rises in the fourth quarter; his behaviour does not.

    Three units: the starters (10 with 11-14), a second unit (one starter in turn with the bench, so the
    bench's strengths can be learned away from the fourth quarter), and the closing unit (10 with the bench).
    """
    rng = np.random.default_rng(seed)
    w = {10: 3.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 0.5, 16: 0.5, 17: 0.5, 18: 0.5}
    rows = []
    for g in range(100):
        for i in range(60):
            five = [10, 11, 12, 13, 14] if i < 30 else [11 + (g + i) % 4, 15, 16, 17, 18] if i < 45 else [10, 15, 16, 17, 18]
            p = np.array([w[x] for x in five])
            rows.append({"game_id": f"{g:04d}", "period": 4 if i >= 45 else 1 + i // 15, "off_lineup": five,
                         "finisher_id": int(rng.choice(five, p=p / p.sum()))})
    return make_poss(rows)


# ---- building blocks --------------------------------------------------------------------------------
def test_eb_normal_finds_the_spread_and_shrinks_noisy_rows_harder():
    rng = np.random.default_rng(1)
    theta = rng.normal(0, 0.3, 400)
    v = np.where(np.arange(400) % 2 == 0, 0.01, 0.25)
    y = theta + rng.normal(0, np.sqrt(v))
    mu, tau2, m, pv = layer.eb_normal(y, v, center=False)
    assert mu == 0 and 0.06 < tau2 < 0.13                           # true value 0.09
    ratio = np.abs(m) / np.abs(y)
    assert ratio[v == 0.01].mean() > 0.8 and ratio[v == 0.25].mean() < 0.4
    assert np.all(pv < v) and np.all(pv < tau2 + 1e-12)
    assert layer.eb_normal(rng.normal(0, np.sqrt(v)), v, center=False)[1] < 0.01      # no real spread: nearly full shrinkage


def test_usage_strengths_recover_who_finishes_given_the_five_on_the_floor():
    rng = np.random.default_rng(2)
    true = np.array([0.40, 0.25, 0.20, 0.15, 0.10, 0.30, 0.12, 0.08])
    lineups = np.array([rng.choice(8, 5, replace=False) for _ in range(6000)])
    slot = np.array([rng.choice(5, p=true[lu] / true[lu].sum()) for lu in lineups])
    w = layer.fit_usage(lineups, slot, 8)
    assert np.corrcoef(w, true)[0, 1] > 0.99
    assert np.allclose(w / w.sum(), true / true.sum(), atol=0.02)


def test_excess_variance_is_what_binomial_noise_does_not_explain():
    rng = np.random.default_rng(3)
    v = rng.uniform(0.005, 0.05, 3000)
    assert layer._excess_var(rng.normal(0, np.sqrt(v)), v) < 0.002
    assert 0.008 < layer._excess_var(rng.normal(0, np.sqrt(v + 0.01)), v) < 0.012


# ---- the estimator ----------------------------------------------------------------------------------
def test_an_uncertain_baseline_takes_its_noise_out_of_the_deviation_and_drift_widens_the_interval():
    def cand(base_var):
        return pd.DataFrame({"scope": "team", "kind": "team_elsewhere_plus_league_shift", "group": "g", "center": False,
                             "k": [30.0, 12, 20, 25, 18, 22, 15, 28], "n": 60.0, "base_logit": -0.6, "base_var": base_var})
    exact, unsure = layer.estimate(cand(0.0)), layer.estimate(cand(0.05))
    # the same spread of counts, but part of it is now the baseline's own noise: less is left to be a real deviation
    assert np.all(unsure["tau2"] < exact["tau2"]) and np.all(np.abs(unsure["dev_logit"]) < np.abs(exact["dev_logit"]))
    lone = pd.concat([cand(0.0), cand(0.0).iloc[[0]].assign(base_var=0.2)], ignore_index=True)      # one row with a shaky baseline
    e = layer.estimate(lone)
    assert abs(e["dev_logit"].iloc[8]) < abs(e["dev_logit"].iloc[0]) and e["dev_sd"].iloc[8] > e["dev_sd"].iloc[0]
    assert e["rate_sd"].iloc[8] > e["rate_sd"].iloc[0]              # and its own rate leans less on that baseline
    drift = {"rate": {"team": 0.01, "player": 0.0}, "dev": {**layer.NO_DRIFT["dev"], "team_elsewhere_plus_league_shift": 0.004}}
    moved = layer.estimate(cand(0.05), drift)
    assert np.allclose(moved["dev_sd"] ** 2, unsure["dev_sd"] ** 2 + 0.004) and np.allclose(moved["rate_sd"] ** 2, unsure["rate_sd"] ** 2 + 0.01)
    assert np.allclose(moved["dev_logit"], unsure["dev_logit"])     # drift changes how sure, not what


def test_the_predicted_rate_sits_between_the_count_and_the_baseline():
    c = pd.DataFrame({"scope": "team", "kind": "team_elsewhere_plus_league_shift", "group": "g", "center": False,
                      "k": [30.0, 12, 20, 25, 18, 22, 15, 28], "n": 60.0, "base_logit": -0.6, "base_var": 0.02})
    e = layer.estimate(c)
    raw, _ = layer.emp_logit(c["k"], c["n"])
    lo, hi = np.minimum(raw, -0.6), np.maximum(raw, -0.6)
    assert np.all((e["rate_logit"] >= lo - 1e-9) & (e["rate_logit"] <= hi + 1e-9))
    # the rate uses the count more than the deviation does: the baseline is itself uncertain, so it cannot pull as hard
    assert np.all(np.abs(e["rate_logit"] + 0.6) >= np.abs(e["dev_logit"]) - 1e-9)


def test_player_rows_adjust_for_who_else_is_on_the_floor():
    """The stand-in's stated weakness: a star with a weak bench around him looks like a tendency. The layer says it is the lineup."""
    poss = weak_bench_poss()
    old = [t for t in standin.team_reports(poss, {}, top_k=None)[1]["tendencies"] if t["player_id"] == 10 and t["situation"]["id"] == "q4"][0]
    new = [t for t in layer.all_rows(poss, {})[1] if t["player_id"] == 10 and t["situation"]["id"] == "q4"][0]
    assert old["deviation"] > 0.08 and old["excludes_zero"]         # fooled: 60% in the fourth against 49% overall
    assert abs(new["deviation"]) < 0.04 and not new["excludes_zero"] and new["baseline_kind"] == "player_lineup_adjusted"
    assert new["baseline_rate"] > old["baseline_rate"] + 0.08       # the expectation rose with the weak lineup
    assert abs(new["baseline_rate"] - 0.6) < 0.04 and abs(new["raw_rate"] - 0.6) < 0.04      # 3 / (3 + 4 x 0.5)


def test_report_rows_are_ranked_by_probability_and_keep_their_sample_sizes(synth_run):
    for f in (synth_run["out"] / "team_reports").glob("*.json"):
        rep = json.loads(f.read_text())
        assert rep["method"]["id"] == "eb_hierarchical_logit" and rep["threshold"] == 0.01 and rep["drift_included"] in (True, False)
        scores = [t["rank_score"] for t in rep["tendencies"]]
        assert scores == sorted(scores, reverse=True) and 0 < len(scores) <= 10
        for t in rep["tendencies"]:
            lo, hi = t["interval"]
            assert lo <= t["deviation"] <= hi and t["n"] >= 25 and 0 <= t["k"] <= t["n"]
            assert t["rank_score"] == t["prob_beyond"] and 0 <= t["prob_beyond"] <= 1
            assert t["excludes_zero"] == (lo > 0 or hi < 0)
            assert t["baseline_kind"] in layer.KINDS


def test_planted_team_and_player_quirks_are_recovered_and_shrunk(synth_run):
    """The simulator makes ALD take more corner threes after timeouts, and BIR's first player finish more in the clutch."""
    rows = layer.all_rows(synth_run["poss"], {})
    team = [t for t in rows[9900001] if t["scope"] == "team" and t["situation"]["id"] == "after_timeout" and t["outcome"]["class"] == "corner3"][0]
    assert team["deviation"] > 0 and team["excludes_zero"] and team["prob_beyond"] > 0.9
    assert team["deviation"] < team["raw_rate"] - team["baseline_rate"]
    # How many close games the draw holds, and how often he is on the floor for them, decides how SURE this row can be
    # (199 possessions in one draw, 80 in the next). What must hold in any draw: the quirk is in the data, the row
    # points the right way, it is shrunk, and it is the strongest clutch row on his team.
    clutch = [t for t in rows[9900002] if t["scope"] == "player" and t["situation"]["id"] == "clutch"]
    player = [t for t in clutch if t["player_id"] == 9910101][0]
    assert player["raw_rate"] > player["baseline_rate"] + 0.05 and 0 < player["deviation"] < player["raw_rate"] - player["baseline_rate"]
    assert max(clutch, key=lambda t: t["deviation"])["player_id"] == 9910101
    control = [t for t in rows[9900008] if t["scope"] == "team" and t["situation"]["id"] != "all"]      # the team with nothing planted
    assert np.mean([t["excludes_zero"] for t in control]) < 0.25


# ---- validation -------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def season(synth_run):
    return synth_run["poss"][synth_run["poss"]["season"] == 2091]


def test_cell_counts_agree_with_the_candidates_the_model_is_fitted_on(season):
    cand = layer.candidates(season)
    tot = layer.cell_counts(season).groupby("key")[["k", "n"]].sum()
    assert set(cand["key"]) <= set(tot.index) and cand["key"].is_unique
    assert (cand["scope"] == "player").sum() > 50 and (cand["scope"] == "team").sum() > 100
    got = tot.loc[cand["key"]]
    assert np.array_equal(got["k"].values, cand["k"].values) and np.array_equal(got["n"].values, cand["n"].values)


def test_every_method_is_scored_on_the_same_cells_players_included(season):
    """The first version of this validation silently dropped every player cell: the methods built their keys differently."""
    first, second = layer.split_halves(season)
    preds = tv.method_preds(first)
    assert set(preds) == {"raw", "stand_in", "hierarchical"}
    for p in preds.values():
        assert all(isinstance(k, str) for k in p.keys) and p.keys.is_unique
        assert sum("|player|" in k for k in p.keys) > 50 and sum("|team|" in k for k in p.keys) > 100
    tot = layer.cell_counts(second).groupby("key")[["k", "n"]].sum()
    cells = tv.common_cells(preds, tot)
    assert len(cells["team"]) > 100 and len(cells["player"]) > 50
    for scope, keys in cells.items():
        assert all(f"|{scope}|" in k for k in keys) and (tot.loc[keys, "n"] >= tv.MIN_N).all()
        for p in preds.values():
            assert len(p.take(keys).p) == len(keys)                  # raises if any method lacks any cell
    out = tv.validate(season)
    assert [s["scope"] for s in out["slices"]] == ["team", "player"]
    for s in out["slices"]:
        assert s["n_cells"] == len(cells[s["scope"]]) and [m["method"] for m in s["methods"]] == ["raw", "stand_in", "hierarchical"]
        assert s["n_trials"] == int(tot.loc[cells[s["scope"]], "n"].sum())


def test_validation_numbers_carry_intervals_and_do_not_move_when_a_method_is_added(season, synth_run):
    drift = layer.estimate_drift(synth_run["poss"][synth_run["poss"]["season"] == 2090])
    a = tv.validate(season, drift)
    for s in a["slices"]:
        for m in s["methods"]:
            assert 0 < m["coverage"] <= 1 and 0 < m["interval_width"] < 1 and m["with_drift"]["interval_width"] >= m["interval_width"] - 0.002
            if m["method"] == "stand_in":
                assert m["vs_stand_in"] is None and m["verdict"] is None
            else:
                lo, hi = m["with_drift"]["vs_stand_in"]["interval"]
                assert lo <= m["with_drift"]["vs_stand_in"]["diff"] <= hi
                assert m["verdict"] == ("ahead" if hi < 0 else "behind" if lo > 0 else "level")
    first, _ = layer.split_halves(season)
    copycat = tv.method_preds(first)["raw"]
    b = tv.validate(season, drift, extra={"copy_of_raw": copycat})
    for sa, sb in zip(a["slices"], b["slices"]):
        assert sa["methods"] == sb["methods"][:3]                    # each method has its own random stream
        assert sb["methods"][3]["log_loss"] == sb["methods"][0]["log_loss"]
    assert {r["method"] for r in a["selection"]} == {"stand_in", "hierarchical"} and all(r["n_rows"] > 0 for r in a["selection"])
    assert {r["kind"] for r in a["deviation"]} <= set(layer.KINDS)


def test_a_season_in_progress_is_never_the_one_split_in_half():
    def games(season, n):
        return pd.DataFrame({"season": season, "game_id": [f"{season}-{i}" for i in range(n)]})
    assert tv.pick_seasons(pd.concat([games(2023, 100), games(2024, 100), games(2025, 100), games(2026, 20)])) == (2025, 2024)
    assert tv.pick_seasons(pd.concat([games(2090, 112), games(2091, 112)])) == (2091, 2090)
    assert tv.pick_seasons(games(2025, 50)) == (2025, None)


# ---- hot hand ---------------------------------------------------------------------------------------
def hot_poss(boost: float, seed: int = 0, n_games: int = 80) -> pd.DataFrame:
    """One team, five players. After his own make a player's strength is multiplied by `boost` on the next trip."""
    rng = np.random.default_rng(seed)
    five, w0 = [1, 2, 3, 4, 5], np.array([2.0, 1.5, 1.0, 0.8, 0.7])
    rows = []
    for g in range(n_games):
        last = None
        for _ in range(90):
            w = w0.copy()
            if last is not None and last[1]:
                w[last[0]] *= boost
            i = int(rng.choice(5, p=w / w.sum()))
            made = bool(rng.random() < 0.47)
            rows.append({"game_id": f"{g:04d}", "off_lineup": five, "finisher_id": five[i], "shot_made": made, "points": 2 * int(made)})
            last = (i, made)
    return make_poss(rows)


def test_hot_hand_report_finds_a_planted_effect_and_is_quiet_without_one():
    hot = hot_hand.hot_hand_report(hot_poss(1.6))
    assert hot["available"] and hot["verdict"]["share"] == "rises" and hot["make_minus_miss"]["share"]["interval"][0] > 0.03
    null = hot_hand.hot_hand_report(hot_poss(1.0, seed=5))
    lo, hi = null["make_minus_miss"]["share"]["interval"]
    assert lo < 0 < hi and null["verdict"]["share"] == "no_clear_difference"
    for rep in (hot, null):
        for a in rep["after"]:
            assert a["n"] > 1000 and a["share_vs_expected"]["interval"][0] <= a["share_vs_expected"]["value"] <= a["share_vs_expected"]["interval"][1]
            assert abs(a["share"] - a["expected_share"] - a["share_vs_expected"]["value"]) < 2e-4
        plo, phi = rep["make_minus_miss"]["points"]["interval"]
        assert plo < 0 < phi                                         # nothing planted on points


def test_hot_hand_only_counts_the_trip_before_in_the_same_period_with_him_still_on_the_floor():
    rows = [{"game_id": "0001", "off_lineup": [1, 2, 3, 4, 5], "finisher_id": 1, "shot_made": True},
            {"game_id": "0001", "off_lineup": [1, 2, 3, 4, 5], "finisher_id": 2, "shot_made": False},         # after 1's make, 1 on the floor
            {"game_id": "0001", "off_lineup": [1, 6, 3, 4, 5], "finisher_id": 3, "shot_made": True},          # after 2's miss, but 2 has left
            {"game_id": "0001", "off_lineup": [1, 6, 3, 4, 5], "finisher_id": 3, "shot_made": True, "period": 2},   # new period
            {"game_id": "0001", "off_lineup": [1, 6, 3, 4, 5], "finisher_id": 1, "ending_type": "turnover", "shot_made": None, "period": 2},
            {"game_id": "0001", "off_lineup": [1, 6, 3, 4, 5], "finisher_id": 4, "shot_made": False, "period": 2}]  # after a turnover: not a shot
    ev = hot_hand.events(make_poss(rows))
    assert len(ev) == 2 and ev["made"].tolist() == [True, True] and ev["y"].tolist() == [0.0, 0.0]
    assert hot_hand.hot_hand_report(make_poss(rows))["available"] is False


# ---- contract 1.2 -----------------------------------------------------------------------------------
def test_export_carries_the_validation_and_the_hot_hand_report(synth_run):
    out = synth_run["out"]
    man = json.loads((out / "manifest.json").read_text())
    assert man["contract_version"] == "1.7" and man["tendency_method"]["final"] is True
    assert "partial-pooling tendency layer" not in man["not_built"]
    val = json.loads((out / man["files"]["tendency_validation"]).read_text())
    assert val["available"] and val["season"] == 2091 and val["drift"]["season"] == 2090 and val["data_kind"] == "synthetic"
    assert [s["scope"] for s in val["slices"]] == ["team", "player"] and val["control"]
    hot = json.loads((out / man["files"]["hot_hand"]).read_text())
    assert hot["available"] and hot["data_kind"] == "synthetic" and {a["prior"] for a in hot["after"]} == {"make", "miss"}
    counts = validate_export_dir(out)
    assert counts["tendency_validation"] == 1 and counts["hot_hand"] == 1


def test_an_export_without_the_new_fields_is_still_valid(synth_run, tmp_path):
    """A 1.1 export: no validation file, no hot-hand file, team reports in the stand-in's shape."""
    import shutil
    old = tmp_path / "old"
    shutil.copytree(synth_run["out"], old)
    man = json.loads((old / "manifest.json").read_text())
    for name in ("tendency_validation", "hot_hand"):
        (old / man["files"].pop(name)).unlink()
    man["contract_version"] = "1.1"
    (old / "manifest.json").write_text(json.dumps(man))
    reports = standin.team_reports(synth_run["poss"][synth_run["poss"]["season"] == 2091], {})
    for f in (old / "team_reports").glob("*.json"):
        rep = json.loads(f.read_text())
        rep = {**{k: v for k, v in rep.items() if k not in ("threshold", "drift_included")}, **{k: v for k, v in reports[rep["team_id"]].items() if k != "team_id"}}
        f.write_text(json.dumps(rep))
    counts = validate_export_dir(old)
    assert "tendency_validation" not in counts and "hot_hand" not in counts and counts["team_report"] == 8
    bad = copy.deepcopy(json.loads((synth_run["out"] / "hot_hand.json").read_text()))
    bad["after"][0]["prior"] = "heat check"
    with pytest.raises(Exception):
        validate(bad, "hot_hand")


# ---- the optional PyMC fit ----------------------------------------------------------------------------
def test_pymc_fit_runs_and_lands_near_the_fast_fit(season):
    pytest.importorskip("pymc")
    from tendency import tendency_pymc
    first, _ = layer.split_halves(season)
    cand = layer.candidates(first)
    p, info = tendency_pymc.pred(cand, draws=150, tune=300, chains=2)
    assert set(p.keys) == set(cand["key"]) and p.draws.shape == (len(cand), 300) and info["seconds"] > 0
    fast = layer.estimate(cand).set_index("key")["rate_logit"]
    from scipy.special import expit
    assert np.mean(np.abs(p.take(fast.index).p - expit(fast.values))) < 0.02
