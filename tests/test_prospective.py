"""Prospective log: freeze, log before tip-off, score after. Runs on the synthetic games; no network."""
from __future__ import annotations

import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from tendency import prospective as pro
from tendency.__main__ import main
from tendency.contract import ContractError, validate, validate_export_dir
from tendency.features import PREGAME_CAND, PREGAME_POSS, build_features
from tendency.prospective import ProspectiveError

quiet = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def world(synth_run, tmp_path_factory):
    """A private copy of the synthetic raw games, a model frozen before the last three game dates, and those dates."""
    root = tmp_path_factory.mktemp("prospective")
    shutil.copytree(synth_run["root"] / "raw", root / "raw")
    poss, _ = pro.load_history(root, quiet)
    dates = sorted(poss["game_date"].unique())
    d1, d2, d3 = dates[-3:]
    cut = dates[-4]
    manifest = pro.train(root, cut, log=quiet)
    return {"root": root, "poss": poss, "cut": cut, "d1": d1, "d2": d2, "d3": d3, "manifest": manifest}


def test_game_date_is_the_us_eastern_date_of_tip_off():
    tips = pd.Series(pd.to_datetime(["2026-01-11 02:41:29", "2026-01-10 17:05:00", "2026-07-04 03:59:00"]))
    assert list(pro.game_dates(tips)) == ["2026-01-10", "2026-01-10", "2026-07-03"]      # 02:41 UTC is still the night before
    assert pro.day_start("2026-01-10") == pd.Timestamp("2026-01-10 11:00")              # 06:00 Eastern, in UTC


def test_features_for_selected_games_equal_the_full_build(small_poss):
    last = list(small_poss.drop_duplicates("game_id").sort_values("tipoff")["game_id"].tail(3))
    full, part = build_features(small_poss), build_features(small_poss, only_games=last)
    want = full.poss[full.poss["game_id"].isin(last)].reset_index(drop=True)
    assert len(part.poss) == len(want) and list(part.poss["poss_idx"]) == list(want["poss_idx"])
    np.testing.assert_allclose(part.poss[PREGAME_POSS].values, want[PREGAME_POSS].values, rtol=0, atol=1e-12)
    wc = full.cand[full.cand["game_id"].astype(object).isin(last)].reset_index(drop=True)
    np.testing.assert_allclose(part.cand[PREGAME_CAND].values, wc[PREGAME_CAND].values, rtol=0, atol=1e-6)


def test_predict_refuses_played_games_unless_replay_and_never_rewrites(world, tmp_path):
    root, d1 = world["root"], world["d1"]
    with pytest.raises(ProspectiveError, match="already on disk"):
        pro.predict(root, d1, tmp_path, log=quiet)
    r1 = pro.predict(root, d1, tmp_path, replay=True, log=quiet)
    files = sorted((tmp_path / d1).glob("*.json"))
    assert r1["written"] == len(files) > 0 and (tmp_path / d1 / pro.SUMS_FILE).exists()
    before = {f.name: f.read_text() for f in files}
    r2 = pro.predict(root, d1, tmp_path, replay=True, log=quiet)
    assert r2["written"] == 0 and len(r2["skipped"]) == len(files) and before == {f.name: f.read_text() for f in files}

    doc = json.loads(files[0].read_text())
    validate(doc, "prediction_log")
    assert doc["mode"] == "replay" and doc["data_kind"] == "synthetic" and doc["game_date"] == d1
    assert doc["model"]["sha256"] == world["manifest"]["sha256"] and doc["model"]["trained_through"] == world["cut"] < d1
    assert doc["inputs"]["history_through_utc"][:10] <= d1 and "zero for the opening tip" in doc["inputs"]["ingame_assumption"]
    for t in doc["teams"]:
        assert len(t["projected_five"]) == 5 and [s["id"] for s in t["situations"]] == [s["id"] for s in pro.SITUATIONS]
        for probs in [t["baseline"]] + [s["model"] for s in t["situations"]]:
            for head in ("finisher", "ending_type", "shot_family"):
                assert abs(sum(probs[head]) - 1) < 2e-3
    bad = copy.deepcopy(doc)
    bad["teams"][0]["projected_five"].pop()
    with pytest.raises(ContractError):
        validate(bad, "prediction_log")


def test_score_accumulates_per_game_sums_and_is_idempotent(world, tmp_path):
    root = world["root"]
    for d in (world["d1"], world["d2"]):
        pro.predict(root, d, tmp_path, replay=True, log=quiet)
    s1 = pro.score(root, world["d1"], tmp_path, log=quiet)
    again = pro.score(root, world["d1"], tmp_path, log=quiet)
    assert again["total_games"] == s1["total_games"] == len(s1["scored"])           # re-scoring replaces, never double counts
    s2 = pro.score(root, world["d2"], tmp_path, log=quiet)
    assert s2["total_games"] == len(s1["scored"]) + len(s2["scored"]) and not s2["missing"] and not s2["inputs_drift"]
    card = json.loads((tmp_path / pro.SCORECARD_FILE).read_text())
    for r in card["games"]:
        assert r["mode"] == "replay" and r["inputs_max_abs_diff"] < 1e-3 and r["projected_starters_right"] <= 10
        assert set(r["heads"]) == {"finisher", "ending_type", "shot_family"}
        assert "off_lineup" not in json.dumps(r) and all(len(h["bins_model"]) == 10 for h in r["heads"].values())    # sums only, no possessions
    block = pro.scorecard_block(tmp_path, "synthetic")
    assert block["available"] and block["mode"] == "replay" and block["n_dates"] == 2 and block["n_games"] == s2["total_games"]
    assert [h["head"] for h in block["heads"]] == ["finisher", "ending_type", "shot_family"]
    for h in block["heads"]:
        r = h["result"]
        assert r["split"] == "prospective" and r["available"] and r["n_games"] == block["n_games"]
        assert r["model"]["interval"][0] <= r["model"]["log_loss"] <= r["model"]["interval"][1]
    assert block["models"] == [{"id": world["manifest"]["model_id"], "trained_through": world["cut"], "sha256": world["manifest"]["sha256"]}]
    with pytest.raises(ProspectiveError, match="holds synthetic results"):
        pro.scorecard_block(tmp_path, "real")


def test_scoring_uses_a_day_start_snapshot(world, tmp_path):
    """The baseline recomputed at scoring time equals the one logged before any game of that date was known."""
    root, d = world["root"], world["d3"]
    assert world["poss"].loc[world["poss"]["game_date"] == d, "game_id"].nunique() > 1      # several games on the date
    pro.predict(root, d, tmp_path, replay=True, log=quiet)
    res = pro.score(root, d, tmp_path, log=quiet)
    card = json.loads((tmp_path / pro.SCORECARD_FILE).read_text())
    assert max(r["inputs_max_abs_diff"] for r in card["games"]) < 1e-4 and not res["inputs_drift"]


def test_score_refuses_a_model_newer_than_the_log(world, tmp_path):
    root, d1 = world["root"], world["d1"]
    pro.predict(root, d1, tmp_path, replay=True, log=quiet)
    newer = pro.train(root, world["d2"], log=quiet)                                    # trained through a date after the logged games
    with pytest.raises(ProspectiveError, match="not before the game date"):
        pro.score(root, d1, tmp_path, model_path=pro.model_dir(root, world["d2"]), log=quiet)
    assert not (tmp_path / pro.SCORECARD_FILE).exists()
    assert newer["sha256"] != world["manifest"]["sha256"]

    # same id, retrained after the log was written: the hash no longer matches what the log names
    doc_f = sorted((tmp_path / d1).glob("*.json"))[0]
    fake = copy.deepcopy(world["manifest"])
    fake.update(sha256="0" * 64, created_at_utc="2999-01-01T00:00:00Z")
    with pytest.raises(ProspectiveError, match="after this log was written"):
        pro.check_model_matches_log(fake, json.loads(doc_f.read_text()))
    with pytest.raises(ProspectiveError, match="already holds a frozen model"):
        pro.train(root, world["cut"], log=quiet)
    assert main(["score", "--root", str(root), "--log-dir", str(tmp_path), "--date", d1, "--model", str(pro.model_dir(root, world["d2"]))]) == 2


def test_tampered_log_or_model_is_refused(world, tmp_path):
    root, d1 = world["root"], world["d1"]
    pro.predict(root, d1, tmp_path, replay=True, log=quiet)
    f = sorted((tmp_path / d1).glob("*.json"))[0]
    doc = json.loads(f.read_text())
    doc["teams"][0]["baseline"]["ending_type"] = [0.9, 0.05, 0.05]
    f.write_text(json.dumps(doc, indent=1))
    with pytest.raises(ProspectiveError, match="changed after it was logged"):
        pro.score(root, d1, tmp_path, log=quiet)
    broken = tmp_path / "model"
    shutil.copytree(pro.model_dir(root, world["cut"]), broken)
    with open(broken / "finisher.txt", "a") as fh:
        fh.write("\n# edited\n")
    with pytest.raises(ProspectiveError, match="has been altered"):
        pro.load_model(broken)


def test_live_log_from_a_schedule_file_then_nothing_to_score_yet(world, tmp_path):
    root = world["root"]
    last = world["poss"].drop_duplicates("game_id").sort_values("tipoff").iloc[-1]
    future = (pd.Timestamp(world["d3"]) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    with pytest.raises(ProspectiveError, match="no schedule"):
        pro.predict(root, future, tmp_path, log=quiet)
    (root / "schedule").mkdir(exist_ok=True)
    (root / "schedule" / f"{future}.json").write_text(json.dumps({"games": [
        {"game_id": "0099199999", "home_team_id": int(last["home_team_id"]), "away_team_id": int(last["away_team_id"]),
         "tipoff_utc": f"{future}T23:30:00Z"}]}))
    try:
        res = pro.predict(root, future, tmp_path, log=quiet)
        doc = json.loads((tmp_path / future / "0099199999.json").read_text())
        assert res["written"] == 1 and doc["mode"] == "live" and doc["scheduled_tipoff_utc"] == f"{future}T23:30:00Z"
        assert doc["model"]["trained_through"] < future
        with pytest.raises(ProspectiveError, match="are on disk as final games"):
            pro.score(root, future, tmp_path, log=quiet)
        (root / "schedule" / f"{future}.json").write_text("{not json")
        with pytest.raises(ProspectiveError, match="not a schedule file"):
            pro.predict(root, future, tmp_path / "other", log=quiet)
    finally:
        shutil.rmtree(root / "schedule")


def test_late_live_logs_are_counted_but_left_out(tmp_path):
    sums = {"n": 10, "ll_model": 7.0, "ll_base": 7.5, "top1_model": 6, "top1_base": 5,
            "bins_model": [[3, 0.9, 1.0]] * 10, "bins_base": [[3, 0.9, 1.0]] * 10}
    def rec(gid, d, before):                                                          # noqa: E306
        return {"game_id": gid, "game_date": d, "mode": "live", "model_id": "through-x", "model_sha256": "a" * 64, "model_trained_through": "2026-11-01",
                "logged_before_tip": before, "heads": {h: sums for h in ("finisher", "ending_type", "shot_family")}}
    card = {"version": 1, "data_kind": "real", "games": [rec("1", "2026-11-17", True), rec("2", "2026-11-17", False), rec("3", "2026-11-18", True),
                                                         {**rec("4", "2026-03-01", True), "mode": "replay"}]}
    (tmp_path / pro.SCORECARD_FILE).write_text(json.dumps(card))
    b = pro.scorecard_block(tmp_path, "real")
    assert (b["available"], b["mode"], b["n_games"], b["n_late"], b["n_dates"]) == (True, "live", 2, 1, 2)     # the replayed game is not mixed in
    validate({"data_kind": "real", "generated_at": "x", "interval_level": 0.95, "bootstrap": "x", "prospective": b,
              "heads": [{"head": h, "splits": [{**b["heads"][0]["result"], "split": s} for s in ("holdout_games", "holdout_season")]}
                        for h in ("finisher", "ending_type", "shot_family")]}, "scorecard")
    card["games"] = [rec("2", "2026-11-17", False)]
    (tmp_path / pro.SCORECARD_FILE).write_text(json.dumps(card))
    b = pro.scorecard_block(tmp_path, "real")
    assert not b["available"] and b["n_late"] == 1 and "after tip-off" in b["reason"]


def test_export_carries_the_block_and_an_export_without_it_still_validates(synth_run, world, tmp_path):
    sc = json.loads((synth_run["out"] / "scorecard.json").read_text())
    man = json.loads((synth_run["out"] / "manifest.json").read_text())
    assert man["contract_version"] == "1.1"
    assert sc["prospective"]["available"] is False and sc["prospective"]["heads"] == [] and sc["prospective"]["reason"]   # empty is valid
    old = copy.deepcopy(sc)
    del old["prospective"]                                                              # what a contract 1.0 export looks like
    validate(old, "scorecard")
    validate({**man, "contract_version": "1.0"}, "manifest")

    from tendency.pipeline import run
    logs, out = tmp_path / "logs", tmp_path / "export"
    pro.predict(world["root"], world["d1"], logs, replay=True, log=quiet)
    pro.score(world["root"], world["d1"], logs, log=quiet)
    run(world["root"], out, max_replays=2, log=quiet, predictions=logs)
    block = json.loads((out / "scorecard.json").read_text())["prospective"]
    assert block["available"] and block["mode"] == "replay" and block["n_dates"] == 1
    assert validate_export_dir(out)["scorecard"] == 1


def test_cli_dates_and_errors(world, tmp_path, capsys):
    root = str(world["root"])
    assert main(["predict", "--root", root, "--log-dir", str(tmp_path), "--date", "17/11/2026"]) == 2
    assert "is not a date" in capsys.readouterr().err
    assert main(["predict", "--root", root, "--log-dir", str(tmp_path), "--date", world["d1"], "--replay"]) == 0
    assert main(["score", "--root", root, "--log-dir", str(tmp_path), "--date", world["d1"]]) == 0
    assert "games scored" in capsys.readouterr().out
    assert main(["score", "--root", root, "--log-dir", str(tmp_path), "--date", world["d2"]]) == 2      # nothing logged for that date
    assert pro.date_arg(None, -1) < pro.date_arg(None, 0)
