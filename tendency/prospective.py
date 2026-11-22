"""The prospective prediction log: freeze a model, log before tip-off, score after the game.

    python -m tendency train   --through 2026-11-16     # freeze models on games up to that date
    python -m tendency predict --date 2026-11-17        # before tip-off: one file per game
    python -m tendency score   --date 2026-11-17        # after the games are final

What is claimed, and what is not (DECISIONS.md, step 1.4): the model and everything it was allowed
to know were fixed before the game. The file written by `predict` names the frozen model by SHA-256
and holds what can honestly be said before tip-off: each team's baseline shares and the model's
probabilities for four fixed reference situations with the projected starting five. Per-possession
predictions need the score, the fouls and the five on the floor, so they are produced after the game
from that same frozen model. The proof of "before" is the commit timestamp of the file, not the
timestamp written inside it.

Dates are US Eastern calendar dates of tip-off. Every game on a date is given pregame features from
games before that date only (a day-start snapshot), which is what could be known when the log was
written.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import cdn_fields as F
from .contract import validate
from .features import FEATURES, INGAME_CAND, INGAME_POSS, build_features
from .models import HEADS, evaluate_sums, fit_frozen, game_sums, predict_heads
from .possessions import ENDING_CLASSES, SHOT_FAMILIES

LOG_VERSION = 1
ET = ZoneInfo("America/New_York")
SUMS_FILE = "SHA256SUMS"
SCORECARD_FILE = "scorecard_prospective.json"
INGAME_ASSUMPTION = ("In-game counts cannot be known before tip-off. They are zero for the opening tip (exact). For the other "
                     "situations they are set to their pregame expectation: the league's average possessions per team by that point "
                     "of a game, split by the team's baseline shares, with the projected five assumed on the floor throughout.")
EMPTY_REASON = "No predictions have been logged before tip-off yet"

# Four fixed situations. Context values are what the features need; nothing here is a play call.
SITUATIONS = [
    {"id": "opening_tip", "label": "First possession of the game", "period": 1, "start_clock_s": 720.0, "margin": 0,
     "start_type": "period_start", "after_timeout": False, "def_in_penalty": False, "def_fouls": 0, "off_fouls": 0, "off_to": 7, "def_to": 7},
    {"id": "q2_half_court", "label": "Q2, 6:00 left, tied, after an opponent score", "period": 2, "start_clock_s": 360.0, "margin": 0,
     "start_type": "made_basket", "after_timeout": False, "def_in_penalty": False, "def_fouls": 2, "off_fouls": 2, "off_to": 6, "def_to": 6},
    {"id": "late_after_timeout", "label": "Q4, 2:00 left, down 2, after a timeout", "period": 4, "start_clock_s": 120.0, "margin": -2,
     "start_type": "made_basket", "after_timeout": True, "def_in_penalty": False, "def_fouls": 3, "off_fouls": 3, "off_to": 2, "def_to": 3},
    {"id": "q4_penalty", "label": "Q4, 4:00 left, up 4, defense in the penalty, off a defensive rebound", "period": 4, "start_clock_s": 240.0,
     "margin": 4, "start_type": "def_rebound", "after_timeout": False, "def_in_penalty": True, "def_fouls": 5, "off_fouls": 3, "off_to": 4, "def_to": 4},
]


class ProspectiveError(RuntimeError):
    pass


# ---- small helpers ------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today_et() -> date:
    return datetime.now(ET).date()


def game_dates(tipoff: pd.Series) -> pd.Series:
    """Naive-UTC tip times -> US Eastern calendar date strings (YYYY-MM-DD)."""
    return tipoff.dt.tz_localize("UTC").dt.tz_convert(ET).dt.strftime("%Y-%m-%d")


def day_start(d: str) -> pd.Timestamp:
    """06:00 US Eastern on that date, as naive UTC: after every game of the night before, before every game of the day."""
    return pd.Timestamp(f"{d} 06:00", tz=ET).tz_convert("UTC").tz_localize(None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text)
    os.replace(tmp, path)


def _round(v, nd: int = 4) -> list[float]:
    return [round(float(x), nd) for x in v]


# ---- history ------------------------------------------------------------------------------
def load_history(root: Path, log=print) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Possession table and per-game info for everything on disk, rebuilt when raw files are newer."""
    from .pipeline import build_possessions
    proc = root / "processed"
    pq, gj, pbp_dir = proc / "possessions.parquet", proc / "games.json", root / "raw" / "pbp"
    box_dir = root / "raw" / "box"
    stale = not (pq.exists() and gj.exists())
    if not stale:
        built = min(pq.stat().st_mtime, gj.stat().st_mtime)
        newest = max((e.stat().st_mtime for d in (pbp_dir, box_dir) if d.exists() for e in os.scandir(d)), default=0.0)
        stale = newest > built
    if stale:
        log("history: raw files are newer than the possession table; rebuilding it")
        build_possessions(root, log)
    poss = pd.read_parquet(pq)
    games = {g["game_id"]: g for g in json.loads(gj.read_text())}
    poss["game_date"] = game_dates(poss["tipoff"])
    return poss, games


def _data_kind(games: dict[str, dict]) -> str:
    kinds = {bool(g["synthetic"]) for g in games.values()}
    if len(kinds) != 1:
        raise ProspectiveError("history mixes synthetic and real games")
    return "synthetic" if kinds.pop() else "real"


# ---- frozen model -------------------------------------------------------------------------
def model_dir(root: Path, through: str) -> Path:
    return root / "models" / f"through-{through}"


def train(root: Path, through: str, force: bool = False, log=print) -> dict:
    """Freeze the three heads on every game with an Eastern date on or before `through`."""
    poss, games = load_history(root, log)
    hist = poss[poss["game_date"] <= through]
    if hist.empty:
        raise ProspectiveError(f"no games on or before {through} under {root}")
    out = model_dir(root, through)
    if (out / "manifest.json").exists() and not force:
        raise ProspectiveError(f"{out} already holds a frozen model. Retraining would change its hash and orphan every log that names it. "
                               "Pick a later --through, or pass --force if no log uses it.")
    feat = build_features(hist.drop(columns=["game_date"]))
    models = fit_frozen(feat, log)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for head, (bst, rounds) in models.items():
        f = out / f"{head}.txt"
        bst.save_model(str(f), num_iteration=rounds)
        files[head] = {"file": f.name, "sha256": _sha256(f), "rounds": int(rounds), "features": FEATURES[head]}
    manifest = {
        "model_id": out.name, "trained_through": through, "created_at_utc": _now(), "data_kind": _data_kind(games),
        "n_games": int(hist["game_id"].nunique()), "n_possessions": int(len(feat.poss)),
        "last_tipoff_utc": pd.Timestamp(hist["tipoff"].max()).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lightgbm": lgb.__version__, "files": files,
        "sha256": hashlib.sha256("\n".join(f"{h}:{files[h]['sha256']}" for h in sorted(files)).encode()).hexdigest(),
    }
    _write_atomic(out / "manifest.json", json.dumps(manifest, indent=1))
    log(f"train: froze {out.name}: {manifest['n_games']:,} games, {manifest['n_possessions']:,} possessions, sha256 {manifest['sha256'][:12]}")
    return manifest


def load_model(path: Path) -> tuple[dict, dict]:
    """(manifest, {head: (booster, rounds)}). Refuses files that no longer match the manifest's hashes."""
    mf = path / "manifest.json"
    if not mf.exists():
        raise ProspectiveError(f"no frozen model at {path}; run `python -m tendency train --through <date>` first")
    manifest = json.loads(mf.read_text())
    models = {}
    for head in HEADS:
        f = path / manifest["files"][head]["file"]
        if not f.exists() or _sha256(f) != manifest["files"][head]["sha256"]:
            raise ProspectiveError(f"{f} is missing or does not match the hash in its manifest; the frozen model has been altered")
        models[head] = (lgb.Booster(model_file=str(f)), int(manifest["files"][head]["rounds"]))
    return manifest, models


def latest_model_before(root: Path, d: str) -> Path:
    """The frozen model with the latest training cut-off strictly before date `d`."""
    best = None
    for mf in sorted((root / "models").glob("through-*/manifest.json")):
        through = json.loads(mf.read_text())["trained_through"]
        if through < d and (best is None or through > best[0]):
            best = (through, mf.parent)
    if best is None:
        raise ProspectiveError(f"no frozen model trained through a date before {d} under {root / 'models'}; "
                               f"run `python -m tendency train --through <a date before {d}>`")
    return best[1]


# ---- predict ------------------------------------------------------------------------------
def _schedule(root: Path, d: str, poss: pd.DataFrame, games: dict[str, dict], replay: bool) -> list[dict]:
    """The games to log for date `d`: from a schedule file, or (replay only) from the games on disk."""
    on_disk = sorted(poss.loc[poss["game_date"] == d, "game_id"].unique())
    f = root / "schedule" / f"{d}.json"
    if f.exists():
        try:
            sched = json.loads(f.read_text())["games"]
            rows = [{"game_id": str(g["game_id"]), "home_team_id": int(g["home_team_id"]), "away_team_id": int(g["away_team_id"]),
                     "tipoff_utc": g.get("tipoff_utc") or None} for g in sched]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise ProspectiveError(f"{f} is not a schedule file ({type(e).__name__}: {e}); expected "
                                   '{"games": [{"game_id", "home_team_id", "away_team_id", "tipoff_utc"}]}') from e
        played = [g["game_id"] for g in rows if g["game_id"] in games]
        if played and not replay:
            raise ProspectiveError(f"{len(played)} of the games scheduled for {d} are already on disk ({', '.join(played[:3])}...). "
                                   "A log written now would not be a pregame log. Pass --replay to simulate one.")
        return rows
    if not replay:
        if on_disk:
            raise ProspectiveError(f"{len(on_disk)} games dated {d} are already on disk, so this would not be a pregame log. "
                                   "Pass --replay to simulate one from data on disk.")
        raise ProspectiveError(f"no schedule for {d}: expected {f}. `scripts/fetch_nba_api.py` writes it on your own machine; "
                               "or pass --replay to simulate from games already on disk.")
    if not on_disk:
        raise ProspectiveError(f"--replay: no games dated {d} on disk under {root}")
    return [{"game_id": g, "home_team_id": int(games[g]["home_team_id"]), "away_team_id": int(games[g]["away_team_id"]), "tipoff_utc": None}
            for g in on_disk]


def _last_starters(hist_games: list[dict]) -> dict[int, dict]:
    """team id -> {five, names, tricode, game_id} from each team's most recent game (list is sorted by tip time)."""
    out = {}
    for g in hist_games:
        for tid, t in g["teams"].items():
            five = sorted(p["player_id"] for p in t["players"] if p["starter"])
            out[int(tid)] = {"five": five, "names": {p["player_id"]: p["name"] for p in t["players"]}, "tricode": t["tricode"],
                             "game_id": g["game_id"]}
    return out


def predict(root: Path, d: str, log_dir: Path, replay: bool = False, force: bool = False, model_path: Path | None = None, log=print) -> dict:
    """Write one pregame file per game scheduled on Eastern date `d`. Returns counts."""
    poss, games = load_history(root, log)
    hist = poss[poss["game_date"] < d]
    if hist.empty:
        raise ProspectiveError(f"no games before {d} under {root}; nothing to base a pregame snapshot on")
    kind = _data_kind(games)
    manifest, models = load_model(model_path or latest_model_before(root, d))
    if manifest["trained_through"] >= d:
        raise ProspectiveError(f"model {manifest['model_id']} was trained through {manifest['trained_through']}, which is not before {d}")
    if manifest["data_kind"] != kind:
        raise ProspectiveError(f"model {manifest['model_id']} was trained on {manifest['data_kind']} data; this history is {kind}")
    sched = _schedule(root, d, poss, games, replay)
    hist_ids = set(hist["game_id"].unique())
    starters = _last_starters(sorted((g for gid, g in games.items() if gid in hist_ids), key=lambda g: g["tipoff_utc"]))
    exp_per_game = float(hist[hist["model_eligible"]].groupby(["game_id", "off_team_id"]).size().mean())
    tip = day_start(d)

    out = {"written": 0, "skipped": [], "date": d, "model_id": manifest["model_id"]}
    stubs, plan = [], []
    for g in sched:
        f = log_dir / d / f"{g['game_id']}.json"
        if f.exists() and not force:
            out["skipped"].append({"game_id": g["game_id"], "reason": "already logged; a pregame log is not rewritten"})
            continue
        sides = [(g["home_team_id"], g["away_team_id"], True), (g["away_team_id"], g["home_team_id"], False)]
        missing = [t for t, _, _ in sides if len(starters.get(t, {}).get("five", [])) != 5]
        if missing:
            out["skipped"].append({"game_id": g["game_id"], "reason": f"no earlier game with five known starters for team {missing[0]}"})
            continue
        idx = 0
        for off, deff, is_home in sides:
            for s in SITUATIONS:
                stubs.append({
                    "game_id": g["game_id"], "season": F.season_from_game_id(g["game_id"]), "tipoff": tip, "poss_idx": idx,
                    "period": s["period"], "start_clock_s": s["start_clock_s"], "off_team_id": off, "def_team_id": deff,
                    "off_is_home": is_home, "margin_start": s["margin"], "off_lineup": starters[off]["five"],
                    "off_team_fouls_period": s["off_fouls"], "def_team_fouls_period": s["def_fouls"], "def_in_penalty": s["def_in_penalty"],
                    "off_timeouts_left": s["off_to"], "def_timeouts_left": s["def_to"], "start_type": s["start_type"],
                    "after_timeout": s["after_timeout"], "is_heave": False, "ending_type": "reference", "finisher_id": pd.NA,
                    "shot_family": None, "lineup_ok": True, "model_eligible": True})
                idx += 1
        plan.append(g)
    if plan:
        stub_df = pd.DataFrame(stubs)
        cols = list(stub_df.columns)
        both = pd.concat([hist[cols], stub_df], ignore_index=True)
        both["finisher_id"] = both["finisher_id"].astype("Int64")
        feat = build_features(both, only_games=[g["game_id"] for g in plan])
        # in-game counts: zero at the opening tip, pregame expectation elsewhere (see INGAME_ASSUMPTION)
        sit = [SITUATIONS[i % len(SITUATIONS)] for i in feat.poss["poss_idx"].values]
        elapsed = np.array([((s["period"] - 1) * 720.0 + (720.0 - s["start_clock_s"])) / 2880.0 for s in sit])
        e_poss = exp_per_game * elapsed
        p, c = feat.poss, feat.cand
        p["ig_team_poss"] = e_poss
        for cls in ENDING_CLASSES:
            p[f"ig_end_{cls}"] = e_poss * p[f"base_end_{cls}"].values
        for fam in SHOT_FAMILIES:
            p[f"ig_fam_{fam}"] = e_poss * p["base_end_shot"].values * p[f"base_fam_{fam}"].values
        e5 = np.repeat(e_poss, 5)
        c["ig_p_on"] = e5.astype(np.float32)
        c["ig_p_fin"] = (e5 * c["p_base"].values).astype(np.float32)
        c["ig_p_last"] = np.where(e5 > 0, c["p_base"].values, 0.0).astype(np.float32)
        assert set(INGAME_POSS) <= set(p.columns) and set(INGAME_CAND) <= set(c.columns)
        pred = predict_heads(models, feat)
        row_of = {(gid, int(i)): k for k, (gid, i) in enumerate(zip(p["game_id"].values, p["poss_idx"].values))}
        history_through = pd.Timestamp(hist["tipoff"].max()).strftime("%Y-%m-%dT%H:%M:%SZ")
        for g in plan:
            teams = []
            for ti, (off, _, _) in enumerate([(g["home_team_id"], 0, 0), (g["away_team_id"], 0, 0)]):
                st = starters[off]
                first = row_of[(g["game_id"], ti * len(SITUATIONS))]
                teams.append({
                    "team_id": int(off), "tricode": st["tricode"], "projected_from_game": st["game_id"],
                    "projected_five": [{"player_id": int(x), "name": st["names"].get(x, str(x))} for x in st["five"]],
                    "baseline": {"finisher": _round(pred["finisher"]["baseline"][first]), "ending_type": _round(pred["ending_type"]["baseline"][first]),
                                 "shot_family": _round(pred["shot_family"]["baseline"][first])},
                    "situations": [{"id": s["id"], "label": s["label"],
                                    "model": {h: _round(pred[h]["model"][row_of[(g["game_id"], ti * len(SITUATIONS) + si)]]) for h in HEADS}}
                                   for si, s in enumerate(SITUATIONS)]})
            doc = {
                "log_version": LOG_VERSION, "data_kind": kind, "mode": "replay" if replay else "live",
                "game_id": g["game_id"], "game_date": d, "season": F.season_from_game_id(g["game_id"]), "created_at_utc": _now(),
                "scheduled_tipoff_utc": g["tipoff_utc"], "home_team_id": int(g["home_team_id"]), "away_team_id": int(g["away_team_id"]),
                "model": {"id": manifest["model_id"], "trained_through": manifest["trained_through"], "sha256": manifest["sha256"],
                          "files": {h: manifest["files"][h]["sha256"] for h in HEADS}},
                "inputs": {"snapshot": f"games with a US Eastern date before {d}", "history_through_utc": history_through,
                           "n_history_games": int(len(hist_ids)), "ingame_assumption": INGAME_ASSUMPTION},
                "classes": {"ending_type": ENDING_CLASSES, "shot_family": SHOT_FAMILIES, "finisher": "the projected five, in the order listed"},
                "teams": teams,
            }
            validate(doc, "prediction_log", f"prediction log {g['game_id']}")
            _write_atomic(log_dir / d / f"{g['game_id']}.json", json.dumps(doc, indent=1, ensure_ascii=False))
            out["written"] += 1
    day = log_dir / d
    if day.exists():
        sums = "".join(f"{_sha256(f)}  {f.name}\n" for f in sorted(day.glob("*.json")))
        _write_atomic(day / SUMS_FILE, sums)
    log(f"predict: {d}: wrote {out['written']} pregame files with {manifest['model_id']} ({'replay' if replay else 'live'}), "
        f"skipped {len(out['skipped'])}" + "".join(f"\n  - {s['game_id']}: {s['reason']}" for s in out["skipped"][:5]))
    return out


# ---- score --------------------------------------------------------------------------------
def _read_logs(day: Path) -> list[dict]:
    files = sorted(day.glob("*.json")) if day.exists() else []
    if not files:
        raise ProspectiveError(f"no pregame files under {day}; `python -m tendency predict --date {day.name}` writes them before tip-off")
    sums_f = day / SUMS_FILE
    if sums_f.exists():
        want = dict(reversed(line.split("  ", 1)) for line in sums_f.read_text().splitlines() if line.strip())
        for f in files:
            if want.get(f.name) != _sha256(f):
                raise ProspectiveError(f"{f} does not match {sums_f.name}: the pregame file was changed after it was logged")
    logs = []
    for f in files:
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            raise ProspectiveError(f"{f} is not valid JSON ({e})") from e
        validate(doc, "prediction_log", str(f))
        logs.append(doc)
    return logs


def check_model_matches_log(manifest: dict, doc: dict) -> None:
    """A log can only be scored by the model that was frozen before it. Raises ProspectiveError otherwise."""
    m = doc["model"]
    if manifest["trained_through"] >= doc["game_date"]:
        raise ProspectiveError(f"refusing to score {doc['game_id']}: model {manifest['model_id']} was trained through "
                               f"{manifest['trained_through']}, which is not before the game date {doc['game_date']}")
    if manifest["sha256"] != m["sha256"]:
        newer = manifest["created_at_utc"] > doc["created_at_utc"] or manifest["trained_through"] > m["trained_through"]
        why = (f"it was created at {manifest['created_at_utc']} (trained through {manifest['trained_through']}), after this log was written at "
               f"{doc['created_at_utc']} naming a model trained through {m['trained_through']}" if newer else "its hash differs")
        raise ProspectiveError(f"refusing to score {doc['game_id']}: the model on disk ({manifest['model_id']}) is not the one the pregame "
                               f"log names; {why}. A log can only be scored by the model that was frozen before it.")


def score(root: Path, d: str, log_dir: Path, model_path: Path | None = None, log=print) -> dict:
    """Score the frozen model on the logged games of Eastern date `d` and add them to the prospective scorecard."""
    logs = _read_logs(log_dir / d)
    poss, games = load_history(root, log)
    kind = _data_kind(games)
    by_model: dict[str, list[dict]] = {}
    for doc in logs:
        if doc["data_kind"] != kind:
            raise ProspectiveError(f"log {doc['game_id']} is {doc['data_kind']} data; the history under {root} is {kind}")
        by_model.setdefault(doc["model"]["id"], []).append(doc)
    loaded = {}
    for mid, docs in by_model.items():
        manifest, models = load_model(model_path or (root / "models" / mid))
        for doc in docs:
            check_model_matches_log(manifest, doc)
        loaded[mid] = (manifest, models)

    dates = poss.drop_duplicates("game_id").set_index("game_id")["game_date"]
    present = [doc for doc in logs if dates.get(doc["game_id"]) == d]
    missing = [doc["game_id"] for doc in logs if dates.get(doc["game_id"]) != d]
    if not present:
        raise ProspectiveError(f"none of the {len(logs)} games logged for {d} are on disk as final games dated {d} yet. "
                               "Load them first (`python -m tendency bulk ...` or the fetch script), then score.")
    ids = [doc["game_id"] for doc in present]
    hist = poss[poss["game_date"] < d]
    todays = poss[poss["game_id"].isin(ids)].copy()
    actual_tip = todays.drop_duplicates("game_id").set_index("game_id")["tipoff"]
    todays["tipoff"] = day_start(d)                          # day-start snapshot: no game sees another from the same date
    both = pd.concat([hist, todays], ignore_index=True).drop(columns=["game_date"])
    feat = build_features(both, only_games=ids)
    gid = feat.poss["game_id"].values

    records = {}
    for mid, (manifest, models) in loaded.items():
        mine = {doc["game_id"]: doc for doc in present if doc["model"]["id"] == mid}
        pred = predict_heads(models, feat)
        per_head = {}
        for h in HEADS:
            m = pred[h]["has"] & np.isin(gid, list(mine))
            per_head[h] = {r["game_id"]: r for r in game_sums(pred[h]["model"][m], pred[h]["baseline"][m], pred[h]["y"][m], gid[m])}
        for g, doc in mine.items():
            rows = feat.poss[gid == g]
            diffs, right = [0.0], 0
            for t in doc["teams"]:
                tr = rows[rows["off_team_id"] == t["team_id"]]
                if len(tr):
                    base = tr.iloc[0][[f"base_end_{c}" for c in ENDING_CLASSES]].values.astype(float)
                    diffs.append(float(np.abs(base / base.sum() - np.array(t["baseline"]["ending_type"])).max()))
                actual = {p["player_id"] for p in games[g]["teams"][str(t["team_id"])]["players"] if p["starter"]} \
                    if str(t["team_id"]) in games[g]["teams"] else set()
                right += len(actual & {p["player_id"] for p in t["projected_five"]})
            tip = pd.Timestamp(actual_tip[g])
            records[g] = {
                "game_id": g, "game_date": d, "mode": doc["mode"], "model_id": mid, "model_sha256": manifest["sha256"],
                "model_trained_through": manifest["trained_through"], "logged_at_utc": doc["created_at_utc"],
                "tipoff_utc": tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "logged_before_tip": bool(pd.Timestamp(doc["created_at_utc"]).tz_localize(None) < tip),
                "scored_at_utc": _now(), "n_possessions": int(len(rows)), "projected_starters_right": int(right),
                "inputs_max_abs_diff": round(max(diffs), 6),
                "heads": {h: {k: v for k, v in per_head[h][g].items() if k != "game_id"} for h in HEADS if g in per_head[h]},
            }
    sc_f = log_dir / SCORECARD_FILE
    card = json.loads(sc_f.read_text()) if sc_f.exists() else {"version": LOG_VERSION, "data_kind": kind, "games": []}
    if card.get("data_kind") != kind:
        raise ProspectiveError(f"{sc_f} holds {card.get('data_kind')} results; refusing to add {kind} games to it")
    card["games"] = sorted([r for r in card["games"] if r["game_id"] not in records] + list(records.values()),
                           key=lambda r: (r["game_date"], r["game_id"]))
    _write_atomic(sc_f, json.dumps(card, separators=(",", ":")))
    drift = [g for g, r in records.items() if r["inputs_max_abs_diff"] > 1e-3]
    summary = {h: evaluate_sums([r["heads"][h] for r in records.values() if h in r["heads"]], "prospective") for h in HEADS}
    log(f"score: {d}: {len(records)} games scored" + (f", {len(missing)} logged but not on disk for that date" if missing else "")
        + (f"; WARNING: pregame baselines differ from the log for {len(drift)} games (history changed since the log was written)" if drift else ""))
    for h in HEADS:
        s = summary[h]
        if s["available"]:
            log(f"  {h:12s} model {s['model']['log_loss']:.4f}  baseline {s['baseline']['log_loss']:.4f}  "
                f"difference {s['delta']['log_loss']:+.4f}  n {s['n_possessions']:,}")
    return {"date": d, "scored": sorted(records), "missing": missing, "inputs_drift": drift, "summary": summary,
            "total_games": len(card["games"])}


# ---- the block the app's scorecard shows ----------------------------------------------------
def scorecard_block(log_dir: Path | None, kind: str) -> dict:
    """The `prospective` part of scorecard.json. Empty is a valid state."""
    empty = {"available": False, "mode": None, "reason": EMPTY_REASON, "n_dates": 0, "n_games": 0, "n_late": 0,
             "first_date": None, "last_date": None, "models": [], "heads": []}
    f = log_dir / SCORECARD_FILE if log_dir else None
    if f is None or not f.exists():
        return empty
    card = json.loads(f.read_text())
    if card.get("data_kind") != kind:
        raise ProspectiveError(f"{f} holds {card.get('data_kind')} results but this export is {kind}; point --predictions somewhere else")
    live = [r for r in card["games"] if r["mode"] == "live"]
    mode = "live" if live else "replay"
    pool = live if live else [r for r in card["games"] if r["mode"] == "replay"]
    late = [r for r in pool if mode == "live" and not r["logged_before_tip"]]
    used = [r for r in pool if not (mode == "live" and not r["logged_before_tip"])]
    if not used:
        return {**empty, "n_late": len(late), "reason": EMPTY_REASON if not late else
                f"{len(late)} games were logged, all after tip-off, so none count"}
    dates = sorted({r["game_date"] for r in used})
    models = {}
    for r in used:
        models[r["model_id"]] = {"id": r["model_id"], "trained_through": r["model_trained_through"], "sha256": r["model_sha256"]}
    return {"available": True, "mode": mode, "reason": None, "n_dates": len(dates), "n_games": len(used), "n_late": len(late),
            "first_date": dates[0], "last_date": dates[-1], "models": sorted(models.values(), key=lambda m: m["trained_through"]),
            "heads": [{"head": h, "result": evaluate_sums([r["heads"][h] for r in used if h in r["heads"]], "prospective",
                                                          "no scored possessions for this prediction yet")} for h in HEADS]}


def date_arg(value: str | None, default_offset_days: int) -> str:
    """CLI helper: an ISO date, or today (US Eastern) plus an offset when omitted."""
    if value:
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as e:
            raise ProspectiveError(f"{value!r} is not a date; use YYYY-MM-DD") from e
    return (today_et() + timedelta(days=default_offset_days)).isoformat()
