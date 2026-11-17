"""Write the contract JSON files the app reads, then validate them against the schemas."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import CONTRACT_VERSION
from .contract import SCHEMA_DIR, validate, validate_export_dir
from .models import INTERVAL_LEVEL, ModelRun
from .possessions import ENDING_CLASSES, SHOT_FAMILIES
from .tendencies import INTERVAL_LEVEL as TENDENCY_LEVEL
from .tendencies import METHOD, team_reports

NOT_BUILT = ["transformer predictor", "coach-decision heads", "live mode", "ask box", "LLM-written summaries",
             "defense proxies", "partial-pooling tendency layer"]
SYNTHETIC_NOTICE = ("SYNTHETIC SAMPLE DATA. Teams, players, games and every number shown are invented by a simulator "
                    "to exercise the pipeline and the app. Nothing here is an NBA result.")
REAL_NOTICE = "Built from public NBA play-by-play. Predictions are shown only for games the models never trained on."


def _dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA else v


def export_all(out: Path, poss: pd.DataFrame, games: list[dict], run: ModelRun, synthetic: bool,
               max_replays: int = 30, fields_verified: bool = False, log=print) -> dict:
    """Write manifest, teams, games, scorecard, team reports and replays into `out`."""
    kind = "synthetic" if synthetic else "real"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if out.exists():
        for sub in ("team_reports", "replays", "schemas"):
            shutil.rmtree(out / sub, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SCHEMA_DIR, out / "schemas")

    # ---- teams -----------------------------------------------------------------------
    teams: dict[int, dict] = {}
    for g in sorted(games, key=lambda x: x["tipoff_utc"]):
        for tid, t in g["teams"].items():
            rec = teams.setdefault(int(tid), {"team_id": int(tid), "tricode": t["tricode"], "name": t["name"], "n_games": 0, "players": {}})
            rec["tricode"], rec["name"] = t["tricode"], t["name"]
            rec["n_games"] += 1
            for p in t["players"]:
                rec["players"][p["player_id"]] = p["name"]
    teams_doc = {"data_kind": kind, "teams": [
        {**{k: v for k, v in t.items() if k != "players"},
         "players": [{"player_id": pid, "name": nm} for pid, nm in sorted(t["players"].items())]}
        for t in sorted(teams.values(), key=lambda x: x["tricode"])]}
    validate(teams_doc, "teams")
    _dump(out / "teams.json", teams_doc)
    names = {pid: nm for t in teams.values() for pid, nm in t["players"].items()}

    # ---- team reports ----------------------------------------------------------------
    as_of = _iso(poss["tipoff"].max())
    latest = int(poss["season"].max())                       # a scouting report describes the current season's team
    reports = team_reports(poss[poss["season"] == latest], names)
    for t in teams_doc["teams"]:
        rep = reports.get(t["team_id"]) or {"team_id": t["team_id"], "n_games": 0, "n_possessions": 0, "n_candidates": 0, "method": METHOD,
                                            "interval_level": TENDENCY_LEVEL, "sort": "rank_score_desc", "tendencies": []}
        rep = {"team_id": rep["team_id"], "data_kind": kind, "as_of_utc": as_of, **{k: v for k, v in rep.items() if k != "team_id"}}
        validate(rep, "team_report", f"team report {t['team_id']}")
        _dump(out / "team_reports" / f"{t['team_id']}.json", rep)

    # ---- scorecard -------------------------------------------------------------------
    scorecard = {"data_kind": kind, "generated_at": now, "interval_level": INTERVAL_LEVEL,
                 "bootstrap": "1,000 resamples of whole games (possessions within a game are not independent)",
                 "heads": run.scorecard_heads}
    validate(scorecard, "scorecard")
    _dump(out / "scorecard.json", scorecard)

    # ---- replays (held-out games only) -------------------------------------------------
    info = {g["game_id"]: g for g in games}
    held = run.splits[run.splits.isin(["holdout_games", "holdout_season"])]
    order = poss.drop_duplicates("game_id").set_index("game_id")["tipoff"]
    chosen: list[str] = []
    for s in ("holdout_games", "holdout_season"):          # most recent games of each held-out split
        ids = sorted(held[held == s].index, key=lambda g: order[g], reverse=True)
        chosen += ids[: max(1, max_replays // 2)]
    pred = run.preds.set_index(["game_id", "poss_idx"])
    games_doc = {"data_kind": kind, "games": []}
    for gid in sorted(chosen, key=lambda g: order[g], reverse=True):
        gp = poss[poss["game_id"] == gid].sort_values("poss_idx")
        rows = []
        for r in gp.itertuples(index=False):
            key = (gid, r.poss_idx)
            has_pred = bool(r.model_eligible) and key in pred.index
            reason = None
            if not has_pred:
                reason = ("end-of-period heave" if r.is_heave else "period ended without a shot, foul or turnover" if r.ending_type == "period_end"
                          else "possession change the segmenter could not explain" if r.ending_type == "unknown"
                          else "lineup could not be reconstructed" if not r.lineup_ok else "no prediction")
            p = pred.loc[key] if has_pred else None
            rows.append({
                "idx": int(r.poss_idx), "period": int(r.period), "clock_s": float(r.start_clock_s),
                "off_team_id": int(r.off_team_id), "def_team_id": int(r.def_team_id),
                "home_score": int(r.home_score_start), "away_score": int(r.away_score_start),
                "after_timeout": bool(r.after_timeout), "start_type": r.start_type, "def_in_penalty": bool(r.def_in_penalty),
                "off_lineup": [int(x) for x in r.off_lineup],
                "actual": {"ending_type": r.ending_type, "finisher_id": None if _none(r.finisher_id) is None else int(r.finisher_id),
                           "shot_family": _none(r.shot_family), "shot_made": None if _none(r.shot_made) is None else bool(r.shot_made),
                           "points": int(r.points), "and_one": bool(r.and_one), "n_off_reb": int(r.n_off_reb)},
                "eligible": has_pred, "ineligible_reason": reason,
                "pred": None if p is None else {h: {"model": list(p[f"{h}_model"]), "baseline": list(p[f"{h}_baseline"])}
                                                for h in ("finisher", "ending_type", "shot_family")},
            })
        g = info[gid]
        doc = {"game_id": gid, "data_kind": kind, "season": int(g["season"]), "tipoff_utc": _iso(order[gid]), "split": run.splits[gid],
               "home_team_id": int(g["home_team_id"]), "away_team_id": int(g["away_team_id"]), "possessions": rows}
        validate(doc, "replay", f"replay {gid}")
        _dump(out / "replays" / f"{gid}.json", doc)
        games_doc["games"].append({"game_id": gid, "season": int(g["season"]), "tipoff_utc": doc["tipoff_utc"], "split": doc["split"],
                                   "home": {"team_id": int(g["home_team_id"]), "score": int(g["final_score"]["home"])},
                                   "away": {"team_id": int(g["away_team_id"]), "score": int(g["final_score"]["away"])},
                                   "n_possessions": len(rows), "n_eligible": sum(x["eligible"] for x in rows)})
    validate(games_doc, "games")
    _dump(out / "games.json", games_doc)

    # ---- manifest --------------------------------------------------------------------
    gsplit = poss["game_id"].map(run.splits).replace({"valid": "train"})

    def split_info(s: str) -> dict:
        m = gsplit == s
        return {"n_games": int(poss.loc[m, "game_id"].nunique()), "n_possessions": int(m.sum()),
                "seasons": sorted(int(x) for x in poss.loc[m, "season"].unique())}

    manifest = {
        "contract_version": CONTRACT_VERSION, "generated_at": now, "data_kind": kind,
        "notice": SYNTHETIC_NOTICE if synthetic else REAL_NOTICE,
        "coverage": {"seasons": sorted(int(s) for s in poss["season"].unique()), "n_games": int(poss["game_id"].nunique()),
                     "n_possessions": int(len(poss)), "n_eligible": int(poss["model_eligible"].sum()),
                     "first_tipoff_utc": _iso(poss["tipoff"].min()), "last_tipoff_utc": as_of},
        "heads": [
            {"id": "finisher", "label": "Who finishes the possession", "classes": None, "class_labels": None, "conditional_on": None,
             "baseline": "Usage shares among the five on the floor, from games before tip-off, shrunk toward one in five"},
            {"id": "ending_type", "label": "How the possession ends", "classes": ENDING_CLASSES,
             "class_labels": ["Shot", "Foul drawn", "Turnover"], "conditional_on": None,
             "baseline": "The offense's ending shares from games before tip-off, shrunk toward the league's"},
            {"id": "shot_family", "label": "Where the shot comes from", "classes": SHOT_FAMILIES,
             "class_labels": ["Rim", "Paint", "Midrange", "Corner 3", "Above-break 3"], "conditional_on": "the possession ends in a shot",
             "baseline": "The offense's shot mix from games before tip-off, shrunk toward the league's"},
        ],
        "splits": {"rule": "By game and by season, never by possession. Last season held out whole; latest 20% of earlier games held out; "
                           "models train on the rest (latest 15% of it only picks the tree count).",
                   "train": split_info("train"), "holdout_games": split_info("holdout_games"), "holdout_season": split_info("holdout_season")},
        "tendency_method": {**METHOD, "interval_level": TENDENCY_LEVEL,
                            "note": "Stand-in until the partial-pooling tendency layer is fitted. Intervals treat the baseline as fixed and are not adjusted for picking the top rows."},
        "quality": {"n_possessions": int(len(poss)), "lineup_ok_rate": round(float(poss["lineup_ok"].mean()), 4),
                    "n_anomalous_possessions": int((poss["ending_type"] == "unknown").sum()), "n_heaves": int(poss["is_heave"].sum()),
                    "n_period_end": int((poss["ending_type"] == "period_end").sum())},
        "field_names_verified": bool(fields_verified), "not_built": NOT_BUILT,
        "files": {"teams": "teams.json", "games": "games.json", "scorecard": "scorecard.json",
                  "team_reports_dir": "team_reports", "replays_dir": "replays", "schemas_dir": "schemas"},
    }
    validate(manifest, "manifest")
    _dump(out / "manifest.json", manifest)
    counts = validate_export_dir(out)
    log(f"export: wrote and validated {sum(counts.values())} contract files in {out} ({kind})")
    return counts
