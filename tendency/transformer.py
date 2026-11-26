"""A small causal transformer over a game's possessions, one head per outcome. EXPERIMENTAL unless it wins.

The question (brief section 5, model 3): does reading the game as a sequence beat boosted trees that
only see running counts? Same features, same baselines, same splits and the same `evaluate()` as
`models.py`, so the three-way table (baseline, LightGBM, transformer) compares like with like.

What sits where, and why that order is the whole leakage argument:

* Position k of a game's sequence stands for possession k. Its input is

    start-known context of k                  (clock, score, how it started, team baselines, in-game counts)
  + everything about possession k - 1         (its outcome, its lineup, the context read at its deciding event)
  + a position embedding

  The outcome of possession k is never in the input at position k or earlier, and attention is causal
  (position k sees positions <= k). So nothing from possession k onward reaches the prediction for k.
* The tempo head reads the trunk's output directly. It asks WHEN the first action comes, so it may
  not see what is read at k's deciding event (DECISIONS.md 122), and the trunk does not contain it.
* The other heads predict WHAT happens next, from the moment before it, exactly like the trees: they
  also get k's context at the deciding event and the five on the floor, fused in after the trunk.
* Every head adds its output to the log of the head's baseline and starts at zero, so an untrained
  model IS the baseline and training can only learn corrections to it. The trees start the same way.

tests/test_transformer.py perturbs later tokens and checks earlier predictions do not move.

Optional: needs PyTorch (`pip install -e ".[torch]"`). Without it the pipeline runs as before and the
scorecard simply has one model. The network itself lives in `transformer_net.py` and is trained in a
separate process (PyTorch and LightGBM cannot share one on macOS); this module only prepares the
arrays, starts it, and scores what comes back.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as ft
from .features import Features
from .models import ALL_HEADS, MODEL_ID as TREES_ID, MODEL_LABEL as TREES_LABEL, MULTICLASS, ModelRun, evaluate
from .transformer_net import GameData        # numpy only at import time; PyTorch is imported inside its functions

MODEL_ID = "transformer"
MODEL_LABEL = "Small transformer over the game's earlier possessions"
SPLITS = ("holdout_games", "holdout_season")
NET_MODULE = "tendency.transformer_net"
REPO = Path(__file__).resolve().parent.parent

# What a position may see of its OWN possession (see the module docstring).
START_COLS = (ft.START_KNOWN + ft.PREGAME_TEAM + [f"base_sty_{c}" for c in ft.ACTION_STYLES] + ft.PREGAME_TEMPO + ft.INGAME_POSS_ALL)
SNAP_COLS = ([c for c in ft.CONTEXT if c not in ft.START_KNOWN] + [c for c in ft.PREGAME_POSS if c.startswith("lineup_")]
             + [f"lineup_sty_{c}" for c in ft.ACTION_STYLES] + [f"start_sty_{c}" for c in ft.ACTION_STYLES])
CAND_COLS = ft.PREGAME_CAND + ft.INGAME_CAND
N_CLASSES = {"finisher": 5, **{h: len(MULTICLASS[h][0]) for h in MULTICLASS}}


def torch_available() -> bool:
    """Is PyTorch installed? Asked without importing it: importing it here, beside LightGBM, is the thing to avoid."""
    return importlib.util.find_spec("torch") is not None


# ---- tensors: one padded sequence per game -----------------------------------------------------
def _standardise(x: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    mu, sd = x[fit_rows].mean(axis=0), x[fit_rows].std(axis=0)
    return ((x - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)


def build_game_data(feat: Features, run: ModelRun) -> GameData:
    """Turn the feature tables into per-game sequences. Standardisation and the player vocabulary use training games only."""
    poss, cand = feat.poss, feat.cand
    n = len(poss)
    part = run.part
    fit = np.isin(part, ["train", "valid"])
    g_codes, _ = pd.factorize(poss["game_id"], sort=False)             # rows are already grouped by game, in order
    pos = poss.groupby(g_codes, sort=False).cumcount().values
    G, L = int(g_codes.max()) + 1, int(pos.max()) + 1

    def pad(a: np.ndarray, fill=0.0) -> np.ndarray:
        out = np.full((G, L) + a.shape[1:], fill, dtype=a.dtype)
        out[g_codes, pos] = a
        return out

    xs = _standardise(poss[START_COLS].values.astype(np.float64), fit)
    xn = _standardise(poss[SNAP_COLS].values.astype(np.float64), fit)
    xc = _standardise(cand[CAND_COLS].values.astype(np.float64), np.repeat(fit, 5)).reshape(n, 5, len(CAND_COLS))
    players = cand["player_id"].values.reshape(n, 5)
    vocab = {p: i + 1 for i, p in enumerate(np.unique(players[fit]))}
    pid = np.vectorize(lambda p: vocab.get(p, 0), otypes=[np.int64])(players)

    y, logb = {}, {}
    for head in ALL_HEADS:
        pr = run.probs[head]
        y[head] = pad(np.where(pr["has"], pr["y"], -1).astype(np.int64), -1)
        logb[head] = pad(np.log(np.clip(pr["baseline"], 1e-6, 1.0)).astype(np.float32))
    mask = pad(np.ones(n, bool), False)
    game_part = np.empty(G, dtype=object)
    game_part[g_codes] = part
    return GameData(pad(xs), pad(xn), pad(xc), pad(pid), y, logb, mask, game_part, list(ALL_HEADS), dict(N_CLASSES))


# ---- scoring beside the trees ----------------------------------------------------------------------
def _challenger_split(P: np.ndarray, pr: dict, m: np.ndarray, gids: np.ndarray, split: str, primary: np.ndarray) -> dict:
    """One model on one split: against the baseline, and against the head's primary model (paired, same possessions)."""
    vs_base = evaluate(P[m], pr["baseline"][m], pr["y"][m], gids[m], split)
    if not vs_base["available"]:
        return {"split": split, "available": False, "reason": vs_base["reason"], "n_possessions": 0, "n_games": 0, "model": None,
                "vs_baseline": None, "vs_primary": None, "verdict": None, "top1": None, "ece": None}
    vs_prim = evaluate(P[m], primary[m], pr["y"][m], gids[m], split)["delta"]
    lo, hi = vs_prim["interval"]
    return {"split": split, "available": True, "reason": None, "n_possessions": vs_base["n_possessions"], "n_games": vs_base["n_games"],
            "model": vs_base["model"], "vs_baseline": vs_base["delta"], "vs_primary": vs_prim,
            "verdict": "ahead" if hi < 0 else "behind" if lo > 0 else "level",
            "top1": vs_base["top1"]["model"], "ece": vs_base["reliability"]["model"]["ece"]}


def attach(run: ModelRun, feat: Features, probs: dict[str, np.ndarray], facts: dict) -> ModelRun:
    """Score the transformer on the trees' rows and write the three-way comparison into the run.

    The rule (brief section 5): the transformer replaces the trees for a head only if the interval of
    its paired difference against them is clear of zero on BOTH held-out splits. Where it does, it
    becomes that head's primary model (scorecard and replays) and the trees become the challenger.
    Otherwise it is listed as an experimental challenger and nothing else changes.
    """
    poss = feat.poss
    gids, part = poss["game_id"].values, run.part
    held = np.isin(part, SPLITS)
    preds = run.preds.copy()
    heads = []
    replaced = []
    for entry in run.scorecard_heads:
        head = entry["head"]
        pr = run.probs[head]
        masks = {s: pr["has"] & (part == s) for s in SPLITS}
        mine = [_challenger_split(probs[head], pr, masks[s], gids, s, pr["model"]) for s in SPLITS]
        wins = all(s["available"] and s["verdict"] == "ahead" for s in mine)
        if wins:
            replaced.append(head)
            theirs = [_challenger_split(pr["model"], pr, masks[s], gids, s, probs[head]) for s in SPLITS]
            splits = [evaluate(probs[head][masks[s]], pr["baseline"][masks[s]], pr["y"][masks[s]], gids[masks[s]], s) for s in SPLITS]
            preds[f"{head}_model"] = [np.round(v, 4).tolist() for v in probs[head][held]]
            heads.append({"head": head, "model_id": MODEL_ID, "splits": splits,
                          "challengers": [{"model_id": TREES_ID, "replaces_primary": False, "splits": theirs}]})
        else:
            heads.append({**entry, "model_id": TREES_ID, "challengers": [{"model_id": MODEL_ID, "replaces_primary": False, "splits": mine}]})
    models = [{"model_id": TREES_ID, "label": TREES_LABEL, "experimental": False, "note": "One model per head, starting from the baseline's odds.", "fit": None},
              {"model_id": MODEL_ID, "label": MODEL_LABEL, "experimental": not replaced,
               "note": ("Primary for: " + ", ".join(replaced) + ". " if replaced else "Does not replace the trees for any head. ")
                       + "It replaces them for a head only when its paired difference is clear of zero on both held-out splits.",
               "fit": facts}]
    return ModelRun(preds, heads, run.splits, run.rounds, part=run.part, probs=run.probs, models=models)


class TransformerError(RuntimeError):
    pass


def _net(args: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, "-m", NET_MODULE, *args], cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise TransformerError(f"the transformer process failed ({proc.returncode}): {proc.stderr.strip()[-600:]}")
    return proc


def leak_check(gd: GameData) -> dict:
    """Run the network's leakage self-check (see `transformer_net.leak_check`) in its own process."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "games.npz"
        gd.save(src)
        return json.loads(_net(["check", str(src)]).stdout.strip().splitlines()[-1])


def run_transformer(feat: Features, run: ModelRun, log=print, max_epochs: int | None = None) -> ModelRun:
    """Train the three configurations in a separate process, keep the one with the lowest validation loss, attach its scores to `run`."""
    gd = build_game_data(feat, run)
    with tempfile.TemporaryDirectory() as tmp:
        src, out, facts_f = Path(tmp) / "games.npz", Path(tmp) / "probs.npz", Path(tmp) / "facts.json"
        gd.save(src)
        proc = _net(["fit", str(src), str(out), str(facts_f)] + (["--max-epochs", str(max_epochs)] if max_epochs else []))
        for line in proc.stdout.strip().splitlines():
            log(line)
        z = np.load(out)
        probs = {h: z[h] for h in ALL_HEADS}
        facts = json.loads(facts_f.read_text())
    log(f"transformer: kept '{facts['chosen']}' of {len(facts['configs'])} configurations; {facts['seconds']:.0f} s in all on {facts['hardware']}")
    res = attach(run, feat, probs, facts)
    for h in res.scorecard_heads:
        ch = h["challengers"][0]
        for s in ch["splits"]:
            if s["available"]:
                log(f"  {h['head']:13s} {s['split']:15s} {ch['model_id']} vs {h['model_id']}: {s['vs_primary']['log_loss']:+.5f} "
                    f"({s['vs_primary']['interval'][0]:+.5f} to {s['vs_primary']['interval'][1]:+.5f}) {s['verdict']}")
    return res
