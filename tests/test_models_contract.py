"""Splits, evaluation, tendencies, and the exported contract."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from tendency import contract
from tendency.contract import ContractError, validate, validate_export_dir
from tendency.models import NotEnoughData, assign_splits, evaluate, reliability


# ---- splits ---------------------------------------------------------------------------
def _games(n1=40, n2=20):
    rows = [{"game_id": f"A{i:03d}", "season": 2090, "tipoff": pd.Timestamp("2090-10-01") + pd.Timedelta(days=i)} for i in range(n1)]
    rows += [{"game_id": f"B{i:03d}", "season": 2091, "tipoff": pd.Timestamp("2091-10-01") + pd.Timedelta(days=i)} for i in range(n2)]
    return pd.DataFrame(rows)


def test_splits_are_by_game_and_by_season():
    g = _games()
    s = assign_splits(pd.concat([g, g, g]))            # many possessions per game
    assert s.index.is_unique and len(s) == 60           # one label per game: a game can never straddle splits
    assert set(s[s == "holdout_season"].index) == set(g.loc[g["season"] == 2091, "game_id"])
    tips = g.set_index("game_id")["tipoff"]
    assert tips[s[s == "train"].index].max() < tips[s[s == "valid"].index].min()
    assert tips[s[s == "valid"].index].max() < tips[s[s == "holdout_games"].index].min()
    assert (s == "holdout_games").sum() == 8


def test_single_season_has_no_holdout_season_and_tiny_data_is_refused():
    s = assign_splits(_games(40, 0))
    assert (s == "holdout_season").sum() == 0 and (s == "holdout_games").sum() == 8
    with pytest.raises(NotEnoughData):
        assign_splits(_games(6, 0))


def test_pipeline_splits_never_share_a_game(synth_run):
    man = json.loads((synth_run["out"] / "manifest.json").read_text())
    sp = man["splits"]
    assert sp["train"]["n_games"] + sp["holdout_games"]["n_games"] + sp["holdout_season"]["n_games"] == man["coverage"]["n_games"]
    assert sp["holdout_season"]["seasons"] == [2091] and sp["train"]["seasons"] == [2090]


# ---- evaluation -----------------------------------------------------------------------
def test_evaluate_reports_intervals_and_a_cautious_verdict():
    rng = np.random.default_rng(0)
    n, k = 4000, 3
    true = rng.dirichlet([4, 2, 1], size=n)
    y = np.array([rng.choice(k, p=p) for p in true])
    games = np.repeat(np.arange(40), n // 40)
    base = np.tile(true.mean(axis=0), (n, 1))
    good = evaluate(true, base, y, games, "holdout_games")
    assert good["verdict"] == "model_better" and good["delta"]["interval"][1] < 0
    assert good["model"]["interval"][0] <= good["model"]["log_loss"] <= good["model"]["interval"][1]
    same = evaluate(base, base, y, games, "holdout_games")
    assert same["verdict"] == "inconclusive" and same["delta"]["log_loss"] == 0
    bad = evaluate(np.tile([0.1, 0.1, 0.8], (n, 1)), base, y, games, "holdout_games")
    assert bad["verdict"] == "baseline_better"
    empty = evaluate(base[:0], base[:0], y[:0], games[:0], "holdout_season")
    assert empty["available"] is False and empty["reason"] and empty["model"] is None


def test_reliability_bins_cover_every_probability():
    P = np.array([[0.05, 0.95], [0.5, 0.5], [1.0, 0.0]])
    r = reliability(P, np.array([1, 0, 0]))
    assert sum(b["n"] for b in r["bins"]) == P.size and len(r["bins"]) == 10
    assert r["bins"][9]["n"] == 2 and r["bins"][9]["obs_rate"] == 1.0   # p == 1.0 lands in the top bin


# ---- tendencies -----------------------------------------------------------------------
def test_stand_in_rows_have_intervals_sample_sizes_and_are_ranked_by_bound(synth_run):
    """The old stand-in is no longer exported, but the tendency validation still scores against it."""
    from tendency.tendencies import team_reports
    for rep in team_reports(synth_run["poss"][synth_run["poss"]["season"] == 2091], {}).values():
        assert rep["method"]["final"] is False and 0 < len(rep["tendencies"]) <= 10
        scores = [t["rank_score"] for t in rep["tendencies"]]
        assert scores == sorted(scores, reverse=True)
        for t in rep["tendencies"]:
            lo, hi = t["interval"]
            assert lo <= t["deviation"] <= hi and t["n"] >= 25 and 0 <= t["k"] <= t["n"]
            assert t["rank_score"] == pytest.approx(lo if t["deviation"] >= 0 else -hi, abs=1e-4)
            assert t["excludes_zero"] == (lo > 0 or hi < 0)
            assert abs(t["shrunk_rate"] - t["baseline_rate"]) <= abs(t["raw_rate"] - t["baseline_rate"]) + 1e-4   # shrinks toward baseline


def test_planted_quirk_is_recovered_and_shrunk(synth_run):
    """The simulator makes team ALD take more corner threes after timeouts. The report should find it."""
    from tendency.tendencies import team_reports
    rep = team_reports(synth_run["poss"], {}, top_k=None)[9900001]      # both seasons, every candidate row
    hit = [t for t in rep["tendencies"] if t["situation"]["id"] == "after_timeout" and t["outcome"]["class"] == "corner3"]
    assert hit and hit[0]["excludes_zero"] and hit[0]["deviation"] > 0
    assert hit[0]["deviation"] < hit[0]["raw_rate"] - hit[0]["baseline_rate"]


# ---- contract -------------------------------------------------------------------------
def test_schemas_use_only_keywords_the_app_validator_supports():
    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                in_props = path and path[-1] in ("properties", "$defs")
                if not in_props:
                    assert k in contract.SUPPORTED_KEYWORDS, f"unsupported keyword {k!r} at {'/'.join(path)}"
                walk(v, path + [k])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, path + [str(i)])
    for name in contract.SCHEMAS + contract.LOG_SCHEMAS:
        walk(contract.load_schema(name), [name])


def test_export_validates_and_is_labeled_synthetic(synth_run):
    out = synth_run["out"]
    counts = validate_export_dir(out)
    assert counts["manifest"] == 1 and counts["team_report"] == 8 and counts["replay"] == 6
    man = json.loads((out / "manifest.json").read_text())
    assert man["data_kind"] == "synthetic" and "SYNTHETIC" in man["notice"] and man["field_names_verified"] is False
    for f in list((out / "replays").glob("*.json")) + list((out / "team_reports").glob("*.json")) + [out / "scorecard.json", out / "teams.json", out / "games.json"]:
        assert json.loads(f.read_text())["data_kind"] == "synthetic"
    assert all("Sample" in t["name"] for t in json.loads((out / "teams.json").read_text())["teams"])


def test_replays_are_held_out_games_with_normalised_probabilities(synth_run):
    out = synth_run["out"]
    games = json.loads((out / "games.json").read_text())["games"]
    assert {g["split"] for g in games} == {"holdout_games", "holdout_season"}
    rep = json.loads((out / "replays" / f"{games[0]['game_id']}.json").read_text())
    n_pred = 0
    for p in rep["possessions"]:
        assert (p["pred"] is not None) == p["eligible"] and (p["ineligible_reason"] is None) == p["eligible"]
        if p["pred"]:
            n_pred += 1
            for head in ("finisher", "ending_type", "shot_family"):
                for who in ("model", "baseline"):
                    assert abs(sum(p["pred"][head][who]) - 1) < 2e-3
            assert len(p["off_lineup"]) == 5
    assert n_pred > 150


def test_scorecard_has_both_splits_per_head_with_sample_sizes(synth_run):
    sc = json.loads((synth_run["out"] / "scorecard.json").read_text())
    assert [h["head"] for h in sc["heads"]][:3] == ["finisher", "ending_type", "shot_family"]      # tests/test_labels.py covers the two added in step 2.2
    for h in sc["heads"]:
        assert [s["split"] for s in h["splits"]] == ["holdout_games", "holdout_season"]
        decision = h["head"] in ("timeout", "foul_sub", "closing_five")      # step 2.4: rows are stoppages, fouls or players, and there are fewer
        for s in h["splits"]:
            assert s["available"] and s["n_possessions"] > (100 if decision else 1000) and s["n_games"] >= (5 if decision else 20)
            assert s["verdict"] in ("model_better", "baseline_better", "inconclusive")
            assert sum(b["n"] for b in s["reliability"]["model"]["bins"]) == sum(b["n"] for b in s["reliability"]["baseline"]["bins"])


def test_contract_rejects_broken_documents(synth_run):
    out = synth_run["out"]
    man = json.loads((out / "manifest.json").read_text())
    for mutate in (lambda m: m.pop("data_kind"), lambda m: m.update(data_kind="probably real"),
                   lambda m: m.update(contract_version="2.0"), lambda m: m.update(extra_field=1)):
        bad = copy.deepcopy(man)
        mutate(bad)
        with pytest.raises(ContractError):
            validate(bad, "manifest")
    rep_f = next((out / "replays").glob("*.json"))
    rep = json.loads(rep_f.read_text())
    i = next(i for i, p in enumerate(rep["possessions"]) if p["pred"])
    rep["possessions"][i]["pred"]["ending_type"]["model"] = [0.5, 0.7]      # wrong length, and no bare extra numbers
    with pytest.raises(ContractError):
        validate(rep, "replay")


def test_mixing_synthetic_and_real_raw_files_is_refused(synth_run, tmp_path):
    from tendency.pipeline import PipelineError, build_possessions
    src = synth_run["root"] / "raw"
    for kind in ("pbp", "box"):
        (tmp_path / "raw" / kind).mkdir(parents=True)
    for i, gid in enumerate(synth_run["ids"][:4]):
        for kind in ("pbp", "box"):
            doc = json.loads((src / kind / f"{gid}.json").read_text())
            if i == 0:
                doc.pop("_synthetic")                      # pretend one game is real
            (tmp_path / "raw" / kind / f"{gid}.json").write_text(json.dumps(doc))
    with pytest.raises(PipelineError, match="mixes synthetic and real"):
        build_possessions(tmp_path, log=lambda *_: None)
