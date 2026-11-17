"""Hand-check tool: print each possession of one game next to the raw actions that produced it.

    python -m tendency inspect <gameId> [--root data/real] [--only odd]

The segmenter's rows carry the first and last `actionNumber` they consumed, so every raw action can
be laid under the possession it fell in. Actions that sit between one possession's last event and
the next one's first (timeouts, substitutions, between-period events) are printed under the
possession they precede and marked with `~`; they are context, not part of the possession.

The summary at the end is the part to read first: possessions and points per team, how each
possession ended, the counts of every special case, and whether possession points plus technical
free throws add up to the final score.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import cdn_fields as F
from .possessions import FeedShapeError, segment_game

ODD_OFF_REB = 2            # this many offensive rebounds in one possession is worth a look


class InspectError(RuntimeError):
    pass


def load_game(game_id: str, root: Path) -> tuple[dict, dict]:
    pbp_f, box_f = root / "raw" / "pbp" / f"{game_id}.json", root / "raw" / "box" / f"{game_id}.json"
    for f in (pbp_f, box_f):
        if not f.exists():
            raise InspectError(f"{f} not found. Load the game first: `python -m tendency bulk --seasons <year> --download-only` "
                               f"(or `ingest`), or pass --root.")
    try:
        return json.loads(pbp_f.read_text()), json.loads(box_f.read_text())
    except json.JSONDecodeError as e:
        raise InspectError(f"game {game_id}: raw file is not valid JSON ({e})") from e


def odd_flags(r: dict) -> list[str]:
    """Flags that make a possession worth reading by hand."""
    flags = []
    if r["and_one"]:
        flags.append("and-one")
    if r["is_heave"]:
        flags.append("heave")
    if r["retained_ft"]:
        flags.append("retained-FT")
    if r["tech_ft_points_off"] or r["tech_ft_points_def"]:
        flags.append(f"tech+{r['tech_ft_points_off']}/{r['tech_ft_points_def']}")
    if not r["lineup_ok"]:
        flags.append("LINEUP-NOT-OK")
    if r["ending_type"] == "unknown":
        flags.append("UNEXPLAINED")
    if r["ending_type"] == "period_end":
        flags.append("horn")
    if r["n_off_reb"] >= ODD_OFF_REB:
        flags.append(f"{r['n_off_reb']}-oreb")
    return flags


def _clock(s: float) -> str:
    return f"{int(s // 60):2d}:{s % 60:04.1f}"


def _period(p: int) -> str:
    return f"Q{p}" if p <= 4 else f"OT{p - 4}"


def summarize(rows: list[dict], info: dict) -> dict:
    """Totals for one segmented game. `reconciles` is the check that matters."""
    home, away = info["home_team_id"], info["away_team_id"]
    per = {t: {"possessions": 0, "points": 0, "tech_points": 0} for t in (home, away)}
    for r in rows:
        off, d = r["off_team_id"], r["def_team_id"]
        per[off]["possessions"] += 1
        per[off]["points"] += r["points"]
        per[off]["tech_points"] += r["tech_ft_points_off"]
        per[d]["tech_points"] += r["tech_ft_points_def"]
    final = {home: info["final_score"]["home"], away: info["final_score"]["away"]}
    for t in per:
        per[t]["total"] = per[t]["points"] + per[t]["tech_points"]
        per[t]["final"] = final[t]
    return {
        "game_id": info["game_id"], "periods": max(r["period"] for r in rows), "n_possessions": len(rows),
        "per_team": per,
        "endings": dict(Counter(r["ending_type"] for r in rows)),
        "and_ones": sum(r["and_one"] for r in rows),
        "technical_points": sum(r["tech_ft_points_off"] + r["tech_ft_points_def"] for r in rows),
        "heaves": sum(r["is_heave"] for r in rows),
        "retained_ft": sum(r["retained_ft"] for r in rows),
        "unexplained": sum(r["ending_type"] == "unknown" for r in rows),
        "lineup_not_ok": sum(not r["lineup_ok"] for r in rows),
        "lineup_repairs": info["lineup_repairs"],
        "after_timeout": sum(r["after_timeout"] for r in rows),
        "model_eligible": sum(r["model_eligible"] for r in rows),
        "reconciles": all(per[t]["total"] == per[t]["final"] for t in per),
        "same_team_consecutive": sum(1 for a, b in zip(rows, rows[1:]) if a["period"] == b["period"] and a["off_team_id"] == b["off_team_id"]),
    }


def inspect_game(game_id: str, root: Path, only_odd: bool = False) -> tuple[str, dict]:
    """Return (printable report, summary dict) for one game on disk."""
    pbp, box = load_game(game_id, root)
    try:
        rows, info = segment_game(pbp, box)
    except FeedShapeError as e:
        raise InspectError(f"game {game_id} could not be segmented: {e}") from e
    actions = sorted(pbp[F.GAME][F.ACTIONS], key=lambda a: (a.get(F.ORDER_NUMBER, a.get(F.ACTION_NUMBER, 0)), a.get(F.ACTION_NUMBER, 0)))
    pos = {a.get(F.ACTION_NUMBER): i for i, a in enumerate(actions)}
    tri = {tid: t["tricode"] for tid, t in info["teams"].items()}
    names = {p["player_id"]: p["name"] for t in info["teams"].values() for p in t["players"]}

    out: list[str] = []
    home, away = info["home_team_id"], info["away_team_id"]
    out.append(f"Game {game_id}  {tri[away]} {info['final_score']['away']} at {tri[home]} {info['final_score']['home']}"
               f"  tip {info['tipoff_utc']}" + ("  (boxscore derived from play-by-play)" if info["derived_box"] else ""))
    out.append("idx  per clock   off  start        -> ending      finisher                 family        pts  flags")
    prev_end, period, shown = -1, 0, 0
    for r in rows:
        end_i = pos.get(r["end_action_number"], prev_end)
        start_i = pos.get(r["start_action_number"], prev_end + 1)
        flags = odd_flags(r)
        if r["after_timeout"]:
            flags.append("after-TO")
        if r["def_in_penalty"]:
            flags.append("penalty")
        if 0 < r["n_off_reb"] < ODD_OFF_REB:
            flags.append(f"{r['n_off_reb']}-oreb")
        if not only_odd or odd_flags(r):
            if r["period"] != period:
                period = r["period"]
                out.append(f"--- {_period(period)} ---")
            fin = names.get(r["finisher_id"], "-" if r["finisher_id"] is None else str(r["finisher_id"]))
            out.append(f"{r['poss_idx']:3d}  {_period(r['period']):3s} {_clock(r['start_clock_s'])}  {tri[r['off_team_id']]:3s}  {r['start_type']:12s} -> "
                       f"{r['ending_type']:10s}  {fin:24.24s} {str(r['shot_family'] or '-'):13s} {r['points']:2d}   {' '.join(flags)}")
            for i in range(prev_end + 1, end_i + 1):
                a = actions[i]
                mark = "~" if i < start_i else " "
                desc = str(a.get(F.DESCRIPTION) or "")
                out.append(f"      {mark} {str(a.get(F.CLOCK) or ''):12s} {str(a.get('teamTricode') or tri.get(a.get(F.TEAM_ID), '')):3s} "
                           f"{str(a.get(F.ACTION_TYPE) or ''):12s} {str(a.get(F.SUB_TYPE) or ''):16.16s} {str(a.get(F.DESCRIPTOR) or ''):16.16s} {desc}")
            shown += 1
        prev_end = max(prev_end, end_i)
    tail = actions[prev_end + 1:]
    if tail and not only_odd:
        out.append(f"--- {len(tail)} actions after the last possession ---")
        for a in tail:
            out.append(f"      ~ {str(a.get(F.CLOCK) or ''):12s} {str(a.get('teamTricode') or ''):3s} {str(a.get(F.ACTION_TYPE) or ''):12s} "
                       f"{str(a.get(F.SUB_TYPE) or ''):16.16s} {str(a.get(F.DESCRIPTOR) or ''):16.16s} {a.get(F.DESCRIPTION) or ''}")

    s = summarize(rows, info)
    out.append("")
    out.append(f"SUMMARY  {game_id}  {s['n_possessions']} possessions over {s['periods']} periods" + (f"  ({shown} shown)" if only_odd else ""))
    for t in (away, home):
        p = s["per_team"][t]
        ok = "ok" if p["total"] == p["final"] else "MISMATCH"
        out.append(f"  {tri[t]:3s}  possessions {p['possessions']:3d}   points {p['points']:3d} + technical {p['tech_points']} = {p['total']:3d}   final {p['final']:3d}  {ok}")
    out.append("  endings: " + ", ".join(f"{k} {v}" for k, v in sorted(s["endings"].items())))
    out.append(f"  and-ones {s['and_ones']}, technical FT points {s['technical_points']}, heaves {s['heaves']}, "
               f"retained-possession FT trips {s['retained_ft']}, unexplained {s['unexplained']}")
    out.append(f"  lineup not ok {s['lineup_not_ok']} (period-start repairs {s['lineup_repairs']}), after timeout {s['after_timeout']}, "
               f"model-eligible {s['model_eligible']}, same team on consecutive possessions {s['same_team_consecutive']}")
    out.append(f"  points reconcile with the final score: {'YES' if s['reconciles'] else 'NO'}")
    return "\n".join(out), s
