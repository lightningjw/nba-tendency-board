"""Write the contract JSON files the app reads, then validate them against the schemas."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import CONTRACT_VERSION
from . import cdn_fields as F
from .contract import SCHEMA_DIR, validate, validate_export_dir
from .models import ALL_HEADS, HEADS, INTERVAL_LEVEL, ModelRun
from .possessions import ACTION_STYLES, ENDING_CLASSES, SHOT_FAMILIES, TEMPO_CLASSES
from . import tendency_validation
from .hot_hand import hot_hand_report
from .prospective import scorecard_block
from .tendency_layer import INTERVAL_LEVEL as TENDENCY_LEVEL
from .tendency_layer import METHOD, THRESHOLD, estimate_drift, team_reports
from . import coach as coach_mod

NOT_BUILT = ["live mode", "ask box", "LLM-written summaries",
             "defense proxies"]
TENDENCY_NOTE = ("Deviations are estimated on the logit scale with partial pooling: a team's rate against the league, its rate in a situation "
                 "against its other possessions, a player's share against what his usage and the four teammates on the floor predict. "
                 "Intervals include the baseline's own uncertainty and how much tendencies moved between the halves of an earlier season. "
                 "Rows are ranked by the probability that the deviation is larger than one percentage point. "
                 "Checked by fitting on the first half of a season and predicting the second; the table is on the scorecard.")
SYNTHETIC_NOTICE = ("SYNTHETIC SAMPLE DATA. Teams, players, games and every number shown are invented by a simulator "
                    "to exercise the pipeline and the app. Nothing here is an NBA result.")
REAL_NOTICE = "Built from public NBA play-by-play. Predictions are shown only for games the models never trained on."


def label_coverage(poss: pd.DataFrame, games: list[dict]) -> list[dict]:
    """The step 2.2 labels: where each comes from, when it is defined, and on what share of possessions."""
    n = int(len(poss))
    checks = sum(int(g.get("tag_checks", 0)) for g in games)
    tags = None
    if checks:
        sc, ft = sum(g["second_chance_agree"] for g in games) / checks, sum(g["from_turnover_agree"] for g in games) / checks
        tags = (f"Two tags from the same scorer can be checked against this pipeline: second-chance agrees with its offensive-rebound "
                f"count on {sc:.1%} of {checks:,} tagged possessions, and from-turnover with how the possession started on {ft:.1%}.")
    untimed = sum(bool(g.get("untimed_rebounds")) for g in games)
    lag = (f"In {untimed:,} of {len(games):,} games the feed stamps a defensive rebound with the missed shot's clock; possessions that start "
           f"off one are taken to begin {F.UNTIMED_REBOUND_LAG_S:.0f} seconds later, the usual gap where the feed does time it.") if untimed else None
    rows = [("action_style", "What kind of shot ended it", "Shot descriptor and subType, mapped to ten styles", "action_style",
             "the possession ends in a shot that is not an end-of-period heave", None),
            ("creation", "Assisted or not, and by whom", "The assist credited on the basket", "assisted",
             "the possession ends in a made basket; the feed credits no assist on a miss", None),
            ("secs_to_first_action", "Seconds to the first shot, turnover or trip to the line", "Period clock at the start and at that first action",
             "secs_to_first_action", "the possession has one", lag),
            ("tempo", f"Early (within {F.TEMPO_EARLY_MAX_S:.0f} s), middle or late ({F.TEMPO_LATE_MIN_S:.0f} s or more)", "The same clocks; the feed has no shot clock",
             "tempo", f"the period had at least {F.SHOT_CLOCK_S:.0f} seconds left when the possession started", None),
            ("fast_break", "Fast break", "The scorer's tag on the shot or its free throws, added after the fact", "fast_break",
             "the possession ends in a shot or at the line; turnovers carry no tag", tags),
            ("two_for_one", "Two-for-one taken or passed up", "Period clock", "two_for_one_taken",
             f"the possession starts with {F.TWO_FOR_ONE_MIN_START_S:.0f} to {F.TWO_FOR_ONE_MAX_START_S:.0f} seconds left in the period; taken means "
             f"the first action came with {F.TWO_FOR_ONE_SHOOT_BY_S:.0f} or more left", None)]
    out = []
    for lid, label, source, col, when, note in rows:
        if col not in poss.columns:
            continue
        d = int(poss[col].notna().sum())
        out.append({"id": lid, "label": label, "source": source, "defined_when": when, "n_defined": d, "n_possessions": n,
                    "coverage": round(d / n, 4) if n else 0.0, "note": note})
    return out


def _dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _none(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA else v


def export_all(out: Path, poss: pd.DataFrame, games: list[dict], run: ModelRun, synthetic: bool,
               max_replays: int = 30, fields_verified: bool = False, log=print, predictions: Path | None = None,
               coach: "coach_mod.CoachRun | None" = None, stops: pd.DataFrame | None = None) -> dict:
    """Write manifest, teams, games, scorecard, team reports and replays into `out`.

    `coach` and `stops` are the step 2.4 decision heads and the dead-ball table they were read from.
    Without them the export has the five possession heads only, as contract 1.4 did.

    `predictions` is the prospective log folder (`predict` / `score`). Without it, or when it holds
    nothing scored yet, the scorecard carries an empty prospective block.
    """
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
    earlier = [int(s) for s in poss["season"].unique() if s < latest]
    drift = estimate_drift(poss[poss["season"] == max(earlier)]) if earlier else None     # never measured on the season being reported
    reports = team_reports(poss[poss["season"] == latest], names, drift=drift)
    decisions = coach_mod.team_sections(coach.tables, latest, names) if coach is not None else {}
    for t in teams_doc["teams"]:
        rep = reports.get(t["team_id"]) or {"team_id": t["team_id"], "n_games": 0, "n_possessions": 0, "n_candidates": 0, "method": METHOD,
                                            "interval_level": TENDENCY_LEVEL, "threshold": THRESHOLD, "drift_included": False,
                                            "sort": "rank_score_desc", "tendencies": []}
        rep = {"team_id": rep["team_id"], "data_kind": kind, "as_of_utc": as_of, **{k: v for k, v in rep.items() if k != "team_id"}}
        if t["team_id"] in decisions:
            rep["coach_decisions"] = decisions[t["team_id"]]
        validate(rep, "team_report", f"team report {t['team_id']}")
        _dump(out / "team_reports" / f"{t['team_id']}.json", rep)

    # ---- tendency validation and the hot-hand report -------------------------------------
    val = {"data_kind": kind, **tendency_validation.report(poss)}
    validate(val, "tendency_validation")
    _dump(out / "tendency_validation.json", val)
    hot = {"data_kind": kind, **hot_hand_report(poss)}
    validate(hot, "hot_hand")
    _dump(out / "hot_hand.json", hot)

    # ---- scorecard -------------------------------------------------------------------
    scorecard = {"data_kind": kind, "generated_at": now, "interval_level": INTERVAL_LEVEL,
                 "bootstrap": "1,000 resamples of whole games (possessions within a game are not independent)",
                 "heads": run.scorecard_heads + (coach.heads if coach is not None else []), "prospective": scorecard_block(predictions, kind)}
    if run.models:                                           # more than one model was scored (transformer.py)
        scorecard["models"] = run.models
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
                           "points": int(r.points), "and_one": bool(r.and_one), "n_off_reb": int(r.n_off_reb),
                           "action_style": _none(r.action_style), "assisted": None if _none(r.assisted) is None else bool(r.assisted),
                           "assister_id": None if _none(r.assister_id) is None else int(r.assister_id),
                           "secs_to_first_action": None if _none(r.secs_to_first_action) is None else round(float(r.secs_to_first_action), 1),
                           "tempo": _none(r.tempo), "fast_break": None if _none(r.fast_break) is None else bool(r.fast_break)},
                "eligible": has_pred, "ineligible_reason": reason,
                "pred": None if p is None else {h: {"model": list(p[f"{h}_model"]), "baseline": list(p[f"{h}_baseline"])} for h in ALL_HEADS},
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
            {"id": "action_style", "label": "What kind of shot it is", "classes": ACTION_STYLES, "class_labels": F.ACTION_STYLE_LABELS,
             "conditional_on": "the possession ends in a shot",
             "baseline": "The style mix of the five on the floor (each player's own from games before tip-off, weighted by usage), shifted by how the league's mix changes with the way the possession started"},
            {"id": "tempo", "label": "How early the first action comes", "classes": TEMPO_CLASSES,
             "class_labels": [f"Early, within {F.TEMPO_EARLY_MAX_S:.0f} s", "Middle", f"Late, {F.TEMPO_LATE_MIN_S:.0f} s or more"],
             "conditional_on": f"the period had at least {F.SHOT_CLOCK_S:.0f} seconds left when the possession started",
             "baseline": "The offense's early, middle and late shares on possessions that started the same way, from games before tip-off, shrunk toward the league's"},
        ],
        "splits": {"rule": "By game and by season, never by possession. Last season held out whole; latest 20% of earlier games held out; "
                           "models train on the rest (latest 15% of it only picks the tree count).",
                   "train": split_info("train"), "holdout_games": split_info("holdout_games"), "holdout_season": split_info("holdout_season")},
        "tendency_method": {**METHOD, "interval_level": TENDENCY_LEVEL, "note": TENDENCY_NOTE},
        "quality": {"n_possessions": int(len(poss)), "lineup_ok_rate": round(float(poss["lineup_ok"].mean()), 4),
                    "n_anomalous_possessions": int((poss["ending_type"] == "unknown").sum()), "n_heaves": int(poss["is_heave"].sum()),
                    "n_period_end": int((poss["ending_type"] == "period_end").sum())},
        "labels": label_coverage(poss, games),
        "field_names_verified": bool(fields_verified), "not_built": NOT_BUILT,
        "files": {"teams": "teams.json", "games": "games.json", "scorecard": "scorecard.json",
                  "tendency_validation": "tendency_validation.json", "hot_hand": "hot_hand.json",
                  "team_reports_dir": "team_reports", "replays_dir": "replays", "schemas_dir": "schemas"},
    }
    for h in manifest["heads"]:
        h["logged_before_tipoff"] = h["id"] in HEADS          # the step 2.2 heads are scored looking back only
    if coach is not None:
        manifest["heads"] += coach_mod.manifest_heads()
        checks = {k: sum(int(g.get(k, 0)) for g in games) for k in ("pf_checks", "pf_agree")}
        manifest["stoppages"] = coach_mod.stoppage_profile(stops, checks, coach.tables)
    validate(manifest, "manifest")
    _dump(out / "manifest.json", manifest)
    counts = validate_export_dir(out)
    pro = scorecard["prospective"]
    log(f"export: wrote and validated {sum(counts.values())} contract files in {out} ({kind}); prospective log: "
        + (f"{pro['n_games']} games on {pro['n_dates']} dates ({pro['mode']})" if pro["available"] else "empty"))
    for s in val["slices"]:
        h = next(m for m in s["methods"] if m["method"] == "hierarchical")
        d = h["with_drift"] or h
        log(f"tendency validation, season {val['season']}, {s['scope']} cells ({s['n_cells']:,}): hierarchical {h['verdict']} vs the stand-in "
            f"(log-loss {d['vs_stand_in']['diff']:+.6f}, {d['vs_stand_in']['interval'][0]:+.6f} to {d['vs_stand_in']['interval'][1]:+.6f}); "
            f"90% coverage {h['coverage']:.3f} as is" + (f", {d['coverage']:.3f} with drift" if h["with_drift"] else ""))
    return counts
