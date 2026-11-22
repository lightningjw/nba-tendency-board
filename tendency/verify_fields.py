"""Check the assumed feed names in cdn_fields.py against one real game.

    python -m tendency verify-fields data/real/raw/pbp/<id>.json data/real/raw/box/<id>.json

Prints one line per assumption: OK, MISSING or CHECK, with a sample of what the file
actually contains so a wrong name can be fixed in cdn_fields.py in one edit.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from . import cdn_fields as F
from .possessions import FeedShapeError, free_throw_kinds, is_technical, parse_clock, segment_game


def verify(pbp_path: Path, box_path: Path | None = None) -> tuple[list[tuple[str, str, str]], bool]:
    """Return ([(status, item, detail)], all_ok)."""
    out: list[tuple[str, str, str]] = []

    def add(ok: bool | None, item: str, detail: str = "") -> None:
        out.append(("OK" if ok else "CHECK" if ok is None else "MISSING", item, detail))

    pbp = json.loads(pbp_path.read_text())
    actions = (pbp.get(F.GAME) or {}).get(F.ACTIONS)
    add(isinstance(actions, list) and len(actions) > 0, f"pbp.{F.GAME}.{F.ACTIONS} is a non-empty list",
        f"top-level keys: {list(pbp)[:6]}")
    if not isinstance(actions, list) or not actions:
        return out, False
    keys = Counter(k for a in actions for k in a)
    for name in ("ACTION_NUMBER", "ORDER_NUMBER", "CLOCK", "PERIOD", "ACTION_TYPE", "SUB_TYPE", "DESCRIPTOR", "DESCRIPTION",
                 "TEAM_ID", "PERSON_ID", "SHOT_RESULT", "SCORE_HOME", "SCORE_AWAY", "AREA", "SHOT_DISTANCE", "X_LEGACY",
                 "Y_LEGACY", "FOUL_DRAWN_PERSON_ID", "ASSIST_PERSON_ID", "POSSESSION"):
        k = getattr(F, name)
        optional = name == "AREA"                         # absent from the live CDN feed; coordinates stand in (checked below)
        add(True if k in keys else (None if optional else False), f"action key {k!r}", f"present on {keys.get(k, 0)}/{len(actions)} actions")
    try:
        parse_clock(actions[0].get(F.CLOCK))
        add(True, "clock parses as an ISO-8601 duration", str(actions[0].get(F.CLOCK)))
    except FeedShapeError as e:
        add(False, "clock parses as an ISO-8601 duration", str(e))

    low = lambda a, k: str(a.get(k) or "").strip().lower()  # noqa: E731
    types = Counter(low(a, F.ACTION_TYPE) for a in actions)
    for t in (F.T_PERIOD, F.T_2PT, F.T_3PT, F.T_FREETHROW, F.T_REBOUND, F.T_TURNOVER, F.T_FOUL, F.T_SUBSTITUTION, F.T_TIMEOUT):
        add(types.get(t, 0) > 0, f"actionType value {t!r}", f"{types.get(t, 0)} actions")
    extra = sorted(set(types) - F.KNOWN_ACTION_TYPES)
    add(None if extra else True, "no unknown actionType values", f"unknown: {extra}" if extra else "")

    def subs(t: str) -> Counter:
        return Counter(low(a, F.SUB_TYPE) for a in actions if low(a, F.ACTION_TYPE) == t)

    add({F.ST_PERIOD_START, F.ST_PERIOD_END} <= set(subs(F.T_PERIOD)), "period subType has 'start' and 'end'", str(dict(subs(F.T_PERIOD))))
    add({F.ST_REBOUND_OFFENSIVE, F.ST_REBOUND_DEFENSIVE} <= set(subs(F.T_REBOUND)), "rebound subType has 'offensive' and 'defensive'", str(dict(subs(F.T_REBOUND))))
    add({F.ST_SUB_IN, F.ST_SUB_OUT} <= set(subs(F.T_SUBSTITUTION)), "substitution subType has 'in' and 'out'", str(dict(subs(F.T_SUBSTITUTION))))
    ft = subs(F.T_FREETHROW)
    add(bool(ft) and all(re.search(F.FT_TRIP_PATTERN, s) for s in ft), "freethrow subType looks like 'N of M'", str(dict(ft)))
    fouls = subs(F.T_FOUL)
    add(None, "foul subType values (need 'offensive'; 'technical' marks technicals)", str(dict(fouls)))
    ordered = sorted(actions, key=lambda a: (a.get(F.ORDER_NUMBER, a.get(F.ACTION_NUMBER, 0)), a.get(F.ACTION_NUMBER, 0)))
    kinds = free_throw_kinds(ordered)
    marked = sum(1 for a in ordered if low(a, F.ACTION_TYPE) == F.T_FREETHROW and is_technical(a))
    found = sum(1 for k in kinds if k == F.FT_TECHNICAL)
    tech_fouls = sum(1 for a in ordered if low(a, F.ACTION_TYPE) == F.T_FOUL and is_technical(a) and F.FOUL_DOUBLE not in low(a, F.DESCRIPTOR))
    add(None, "technical free throws are identified (by their own marker, or from the technical foul)",
        f"{tech_fouls} technical fouls (not double); {found} technical FTs identified, {marked} marked by the feed itself, "
        f"{found - marked} read from the foul")
    other = Counter(k for k in kinds if k and k != F.FT_TECHNICAL)
    add(None, "keep-the-ball free throws (flagrant, away-from-play, clear-path, transition take)", str(dict(other)) if other else "none in this game")
    results = Counter(low(a, F.SHOT_RESULT) for a in actions if low(a, F.ACTION_TYPE) in (F.T_2PT, F.T_3PT))
    add(set(results) <= {"made", "missed"} and bool(results), "shotResult is 'Made' or 'Missed'", str(dict(results)))
    shots = [a for a in actions if low(a, F.ACTION_TYPE) in (F.T_2PT, F.T_3PT)]
    areas = Counter(low(a, F.AREA) for a in shots)
    if set(areas) <= {""}:
        # The live CDN feed has no `area` (seen 2026-11-21); season archives do. Coordinates then decide.
        located = sum(isinstance(a.get(F.X_LEGACY), (int, float)) and isinstance(a.get(F.Y_LEGACY), (int, float)) for a in shots)
        add(bool(shots) and located == len(shots), "no shot `area` in this file: every shot has legacy coordinates instead",
            f"{located}/{len(shots)} shots have {F.X_LEGACY} and {F.Y_LEGACY}; shot family comes from coordinates")
    else:
        unknown_areas = sorted(set(areas) - set(F.AREA_TO_FAMILY))
        add(not unknown_areas, "every shot `area` value maps to a shot family", f"unmapped: {unknown_areas}; seen: {dict(areas)}")
    per_period_start_subs = sum(1 for a in actions if low(a, F.ACTION_TYPE) == F.T_SUBSTITUTION and int(a.get(F.PERIOD) or 0) > 1
                                and abs(parse_clock(a.get(F.CLOCK)) - F.REGULATION_PERIOD_SECONDS) < 0.01)
    add(None, "between-period substitutions are logged at the period start", f"{per_period_start_subs} such events (0 means the lineup repair heuristic is doing the work)")

    if box_path is None:
        add(None, "boxscore not provided", "pass it as the second argument to check starters and the full segmentation")
        return out, all(s != "MISSING" for s, _, _ in out)
    box = json.loads(box_path.read_text())
    g = box.get(F.GAME) or {}
    add(bool(g.get(F.BOX_GAME_TIME_UTC)), f"box.{F.GAME}.{F.BOX_GAME_TIME_UTC}", str(g.get(F.BOX_GAME_TIME_UTC)))
    for side in (F.BOX_HOME, F.BOX_AWAY):
        t = g.get(side) or {}
        starters = [p for p in t.get(F.BOX_PLAYERS, []) if str(p.get(F.BOX_PLAYER_STARTER)) == "1"]
        add(len(starters) == 5, f"box {side}: exactly five players with {F.BOX_PLAYER_STARTER} == '1'", f"found {len(starters)}; team keys: {list(t)[:8]}")
    try:
        rows, info = segment_game(pbp, box)
        n = len(rows)
        ok_rate = sum(r["lineup_ok"] for r in rows) / n
        pts = sum(r["points"] + r["tech_ft_points_off"] + r["tech_ft_points_def"] for r in rows)
        final = info["final_score"]["home"] + info["final_score"]["away"]
        add(150 <= n <= 280, "possession count is plausible (150-280 per game)", f"{n} possessions")
        add(pts == final, "possession points add up to the final score", f"{pts} from possessions vs {final} final")
        add(ok_rate >= 0.97, "lineups reconstruct to five-on-five on >= 97% of possessions", f"{ok_rate:.1%}; repairs: {info['lineup_repairs']}")
        add(info["anomalies"] <= 3, "at most 3 unexplained changes of possession", f"{info['anomalies']}")
        mism = _possession_field_agreement(actions, rows)
        add(None, f"agreement with the feed's own {F.POSSESSION!r} field", mism)
    except FeedShapeError as e:
        add(False, "full segmentation runs", str(e))
    return out, all(s != "MISSING" for s, _, _ in out)


def _possession_field_agreement(actions: list[dict], rows: list[dict]) -> str:
    by_num = {a.get(F.ACTION_NUMBER): a for a in actions}
    seen = agree = 0
    for r in rows:
        a = by_num.get(r["end_action_number"])
        ended_here = r["ending_type"] == "turnover" or (r["ending_type"] == "shot" and r["shot_made"] and not r["and_one"])
        if ended_here and a is not None and a.get(F.POSSESSION):
            seen += 1
            agree += int(a.get(F.POSSESSION) == r["off_team_id"])
    return f"{agree}/{seen} made-shot and turnover endings agree" if seen else "field absent or empty; nothing to compare"


def main(pbp: str, box: str | None) -> int:
    results, ok = verify(Path(pbp), Path(box) if box else None)
    width = max(len(i) for _, i, _ in results)
    for status, item, detail in results:
        print(f"{status:8} {item:{width}}  {detail}")
    print("\nAll hard checks passed." if ok else "\nSome assumed names are MISSING. Fix them in tendency/cdn_fields.py and re-run.")
    return 0 if ok else 1
