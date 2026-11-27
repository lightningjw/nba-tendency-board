"""Step 2.3: the experimental transformer. Leakage first, then the rule that decides whether it replaces the trees.

The network is trained and checked in its own process (PyTorch and LightGBM cannot share one on
macOS), so the tests that need PyTorch drive it through `transformer.py` and are skipped without it.
The decision logic needs no PyTorch and always runs.
"""
from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tendency import features as ft
from tendency import transformer as tr
from tendency.contract import validate, validate_export_dir
from tendency.features import build_features
from tendency.models import ALL_HEADS, run_models

needs_torch = pytest.mark.skipif(not tr.torch_available(), reason='PyTorch is not installed (pip install -e ".[torch]")')
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def fitted(synth_run):
    """Features and tree models on the first 90 synthetic games: enough for both held-out splits to exist."""
    poss = synth_run["poss"]
    games = poss.drop_duplicates("game_id").sort_values("tipoff")["game_id"]
    keep = set(games.head(60)) | set(games.tail(30))                    # 60 from the first season, 30 from the second
    feat = build_features(poss[poss["game_id"].isin(keep)].reset_index(drop=True))
    return feat, run_models(feat, log=lambda *_: None)


# ---- what each position may see --------------------------------------------------------------------------
def test_the_trunk_only_gets_what_is_known_when_the_possession_starts():
    assert not set(tr.START_COLS) & set(tr.SNAP_COLS)
    mid = {"after_timeout", "def_in_penalty", "def_team_fouls_period", "off_team_fouls_period", "off_timeouts_left", "def_timeouts_left"}
    assert mid <= set(tr.SNAP_COLS) and not mid & set(tr.START_COLS)
    assert not [c for c in tr.START_COLS if c.startswith(("lineup_", "start_sty_"))]        # the five on the floor are read late too
    labels = {"action_style", "tempo", "ending_type", "shot_family", "finisher_id", "secs_to_first_action", "points"}
    assert not labels & set(tr.START_COLS + tr.SNAP_COLS + tr.CAND_COLS)
    # the transformer is given what the trees are given, no more: every column is one of their registered features
    assert set(tr.START_COLS + tr.SNAP_COLS + tr.CAND_COLS) <= set(ft.KIND)
    assert set(ft.FEATURES["tempo"]) <= set(tr.START_COLS)


def test_game_tensors_keep_every_row_once_and_in_order(fitted):
    feat, run = fitted
    gd = tr.build_game_data(feat, run)
    assert int(gd.mask.sum()) == len(feat.poss) and gd.heads == ALL_HEADS
    assert gd.xs.shape[:2] == gd.mask.shape and gd.xc.shape[2] == 5 and gd.xs.shape[2] == len(tr.START_COLS)
    y = run.probs["ending_type"]["y"]
    assert np.array_equal(gd.y["ending_type"][gd.mask], y)              # flattening by the mask gives Features.poss order back
    for h in ALL_HEADS:
        assert ((gd.y[h][gd.mask] >= 0) == run.probs[h]["has"]).all() and (gd.y[h][~gd.mask] == -1).all()
        assert np.allclose(np.exp(gd.logb[h][gd.mask]).sum(axis=1), 1, atol=1e-3)
    seen = set(feat.cand["player_id"].values[np.repeat(np.isin(run.part, ["train", "valid"]), 5)])
    assert gd.pid.max() == len(seen)                                    # players first seen in a held-out game share index 0


def test_the_network_module_never_loads_lightgbm():
    """It runs beside nothing else for a reason: two OpenMP runtimes in one process abort it on macOS."""
    tree = ast.parse((REPO / "tendency" / "transformer_net.py").read_text())
    top = {n.names[0].name.split(".")[0] for n in tree.body if isinstance(n, ast.Import)} | \
          {(n.module or "").split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert top <= {"__future__", "json", "platform", "sys", "time", "dataclasses", "numpy"}
    out = subprocess.run([sys.executable, "-c", "import sys, tendency.transformer_net; print('lightgbm' in sys.modules, 'torch' in sys.modules)"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.stdout.split() == ["False", "False"], out.stderr           # and it does not even load PyTorch until it has to


@needs_torch
def test_perturbing_later_tokens_does_not_move_earlier_predictions(fitted):
    """The main risk in a sequence model. Possession k's own outcome, and everything after it, must not reach the prediction for k."""
    feat, run = fitted
    res = tr.leak_check(tr.build_game_data(feat, run))
    clean, broken = res["clean"], res["broken_on_purpose"]
    assert clean["n_games"] >= 4
    assert clean["labels_from_k"] == 0.0                                # labels of possessions >= k: predictions for <= k hold
    assert clean["context_after_k"] == 0.0                              # features of possessions > k: predictions for <= k hold
    assert clean["late_context_at_k_tempo"] == 0.0                      # what is read at k's deciding event: tempo for k holds
    # the zeros mean something only if the same perturbations DO move what they are allowed to move
    assert clean["teeth_earlier"] > 0.01 and clean["teeth_late"] > 0.01
    # and the check catches a model that feeds possession k its own outcome
    assert broken["labels_from_k"] > 0.01 and broken["late_context_at_k_tempo"] > 0.01


@needs_torch
def test_an_untrained_network_is_exactly_the_baseline(fitted):
    feat, run = fitted
    assert tr.leak_check(tr.build_game_data(feat, run))["clean"]["untrained_minus_baseline"] < 1e-5


# ---- the rule: clear of zero on both held-out splits, or it stays experimental ----------------------------
def _attach(fitted, probs):
    feat, run = fitted
    return tr.attach(run, feat, probs, {"chosen": "test", "selection": "-", "hardware": "-", "seconds": 0.0, "n_train_games": 0,
                                        "n_parameters": 0, "configs": []})


def test_a_model_that_only_ties_the_trees_stays_a_challenger(fitted):
    feat, run = fitted
    out = _attach(fitted, {h: run.probs[h]["model"] for h in ALL_HEADS})
    for h in out.scorecard_heads:
        assert h["model_id"] == "lightgbm" and [c["model_id"] for c in h["challengers"]] == ["transformer"]
        for s in h["challengers"][0]["splits"]:
            assert s["available"] and s["vs_primary"]["log_loss"] == 0 and s["verdict"] == "level"
    assert out.models[1]["experimental"] is True and "Does not replace" in out.models[1]["note"]
    assert out.preds.equals(run.preds)                                  # the replays still show the trees


def test_a_model_clear_of_zero_on_both_splits_replaces_the_trees_for_that_head_only(fitted):
    feat, run = fitted
    probs = {h: run.probs[h]["model"] for h in ALL_HEADS}
    pr = run.probs["ending_type"]
    sharp = pr["model"] * 0.6 + np.eye(3)[pr["y"]] * 0.4                # cheats toward the truth: unmistakably better
    probs["ending_type"] = sharp
    worse = run.probs["tempo"]["baseline"] * 0.5 + 1 / 6                # flatter than the baseline: worse
    probs["tempo"] = worse / worse.sum(axis=1, keepdims=True)
    out = _attach(fitted, probs)
    by = {h["head"]: h for h in out.scorecard_heads}
    won = by["ending_type"]
    assert won["model_id"] == "transformer" and won["challengers"][0]["model_id"] == "lightgbm"
    assert all(s["verdict"] == "behind" for s in won["challengers"][0]["splits"])       # the trees, measured against the new primary
    assert won["splits"][0]["model"]["log_loss"] < run.scorecard_heads[1]["splits"][0]["model"]["log_loss"]
    assert by["tempo"]["model_id"] == "lightgbm" and all(s["verdict"] == "behind" for s in by["tempo"]["challengers"][0]["splits"])
    assert by["finisher"]["model_id"] == "lightgbm"
    assert out.models[1]["experimental"] is False and "ending_type" in out.models[1]["note"]
    held = np.isin(run.part, ["holdout_games", "holdout_season"])
    assert out.preds["ending_type_model"].iloc[0] == np.round(sharp[held][0], 4).tolist()       # replays follow the primary
    assert out.preds["tempo_model"].equals(run.preds["tempo_model"])


def test_a_win_must_hold_for_a_second_training_seed(fitted):
    """The interval is about which games were held out, not about the training run. One lucky seed does not replace the trees."""
    feat, run = fitted
    probs = {h: run.probs[h]["model"] for h in ALL_HEADS}
    pr = run.probs["ending_type"]
    sharp = dict(probs, ending_type=pr["model"] * 0.6 + np.eye(3)[pr["y"]] * 0.4)
    facts = {"chosen": "test", "selection": "-", "hardware": "-", "seconds": 0.0, "n_train_games": 0, "n_parameters": 0, "configs": []}
    lucky = {x["head"]: x for x in tr.attach(run, feat, sharp, facts, second=probs).scorecard_heads}["ending_type"]
    assert lucky["model_id"] == "lightgbm"                                # ahead once, level the second time: stays a challenger
    assert [s["verdict"] for s in lucky["challengers"][0]["splits"]] == ["ahead", "ahead"]
    assert [s["verdict"] for s in lucky["challengers"][0]["second_seed"]] == ["level", "level"]
    twice = tr.attach(run, feat, sharp, facts, second=sharp)
    assert {x["head"]: x for x in twice.scorecard_heads}["ending_type"]["model_id"] == "transformer"
    assert "two training seeds" in twice.models[1]["note"]


def test_ahead_on_one_split_only_is_not_enough(fitted):
    feat, run = fitted
    probs = {h: run.probs[h]["model"].copy() for h in ALL_HEADS}
    pr = run.probs["ending_type"]
    first = run.part == "holdout_games"
    probs["ending_type"][first] = pr["model"][first] * 0.6 + np.eye(3)[pr["y"][first]] * 0.4
    h = {x["head"]: x for x in _attach(fitted, probs).scorecard_heads}["ending_type"]
    assert [s["verdict"] for s in h["challengers"][0]["splits"]] == ["ahead", "level"] and h["model_id"] == "lightgbm"


# ---- contract 1.4 --------------------------------------------------------------------------------------------
def test_scorecard_with_and_without_a_second_model_is_valid(synth_run):
    out = synth_run["out"]
    sc = json.loads((out / "scorecard.json").read_text())
    man = json.loads((out / "manifest.json").read_text())
    assert man["contract_version"] == "1.5" and "transformer predictor" not in man["not_built"]
    if tr.torch_available():
        assert [m["model_id"] for m in sc["models"]] == ["lightgbm", "transformer"]
        fit = sc["models"][1]["fit"]
        assert len(fit["configs"]) == 3 and sum(c["chosen"] for c in fit["configs"]) == 1 and fit["seconds"] > 0 and "PyTorch" in fit["hardware"]
        assert len(fit["seeds"]) == 2 and fit["seeds"][0] != fit["seeds"][1]
        assert [c["history"] for c in fit["configs"]] == [True, True, False]
        for h in sc["heads"][5:]:                                       # the step 2.4 decision heads are trees only: no sequence to read
            assert h["model_id"] == "lightgbm" and "challengers" not in h
        for h in sc["heads"][:5]:
            assert h["model_id"] in ("lightgbm", "transformer") and len(h["challengers"]) == 1
            if h["model_id"] == "lightgbm":
                assert [s["split"] for s in h["challengers"][0]["second_seed"]] == ["holdout_games", "holdout_season"]
            for s in h["challengers"][0]["splits"]:
                lo, hi = s["vs_primary"]["interval"]
                assert s["available"] and lo <= s["vs_primary"]["log_loss"] <= hi
                assert s["verdict"] == ("ahead" if hi < 0 else "behind" if lo > 0 else "level")
    plain = copy.deepcopy(sc)                                           # what an export made without PyTorch looks like
    plain.pop("models", None)
    for h in plain["heads"]:
        h.pop("model_id", None)
        h.pop("challengers", None)
    validate(plain, "scorecard")
    assert validate_export_dir(out)["scorecard"] == 1
