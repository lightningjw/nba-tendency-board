"""Possession segmentation: raw play-by-play + boxscore -> one row per possession.

Rules (each has a test in tests/test_possessions.py):

* A possession ends on a made field goal, a defensive rebound, a turnover, the
  last free throw of a trip (made, or missed and rebounded by the defense), or
  the end of the period.
* Offensive rebound: the same possession continues. Labels come from the event
  that finally ends it; `n_off_reb` counts the extra chances.
* And-one: a made field goal followed by a "1 of 1" free throw is ONE possession
  ending in a shot (finisher = shooter, `and_one` = True). The free throw's point
  is added to the possession.
* Technical free throws never start, end or relabel a possession. Their points
  are kept apart in `tech_ft_points_off` / `tech_ft_points_def`.
* Free throws where the shooting team keeps the ball afterwards (flagrant, clear-path, transition
  take, and a lone away-from-play free throw) add their points to the possession but do not end or
  relabel it. A basket made on the same play ends its own possession first; the free throws open the
  next one for the same team. An away-from-play free throw after a made basket, or a two-shot
  away-from-play trip, ends the possession like any other trip.
* A technical free throw (and its dead-ball rebound) between a made basket and its and-one free
  throw does not split the and-one.
* End-of-period heave: a `heave` action (2025-26 onward), or a shot from >= 40 ft (or area "Backcourt") with <= 3 s on
  the period clock is flagged `is_heave`, given shot_family "heave", and excluded
  from every prediction head (`model_eligible` = False).

Context columns (lineups, team fouls, timeouts) are snapshotted just BEFORE the
event that decides the label, so they only hold information available at that time.
"""
from __future__ import annotations

import re
from typing import Any

from . import cdn_fields as F

ENDING_CLASSES = ["shot", "foul_drawn", "turnover"]
SHOT_FAMILIES = ["rim", "paint", "midrange", "corner3", "above_break3"]
START_TYPES = ["period_start", "made_basket", "def_rebound", "turnover", "free_throw", "other"]

_CLOCK_RE = re.compile(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")
_FT_RE = re.compile(F.FT_TRIP_PATTERN)


class FeedShapeError(ValueError):
    """The raw JSON does not look like the feed this code was written against."""


def parse_clock(value: Any) -> float:
    """'PT11M42.50S' -> 702.5 seconds left in the period."""
    m = _CLOCK_RE.match(str(value or "").strip())
    if not m or (m.group(1) is None and m.group(2) is None):
        raise FeedShapeError(f"unparseable clock value: {value!r}")
    return int(m.group(1) or 0) * 60 + float(m.group(2) or 0.0)


def period_length(period: int) -> float:
    return F.REGULATION_PERIOD_SECONDS if period <= 4 else F.OVERTIME_PERIOD_SECONDS


def _low(a: dict, key: str) -> str:
    return str(a.get(key) or "").strip().lower()


def _int_or_none(v: Any) -> int | None:
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if i > 0 else None


def is_technical(a: dict) -> bool:
    m = F.TECHNICAL_MARKER
    return m in _low(a, F.SUB_TYPE) or m in _low(a, F.DESCRIPTOR) or m in _low(a, F.DESCRIPTION)


def shot_family(a: dict) -> str | None:
    """Shot family from the `area` field, falling back to legacy coordinates, then distance."""
    fam = F.AREA_TO_FAMILY.get(_low(a, F.AREA))
    if fam:
        return fam
    is3 = _low(a, F.ACTION_TYPE) == F.T_3PT
    x, y = a.get(F.X_LEGACY), a.get(F.Y_LEGACY)
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        dist = (x * x + y * y) ** 0.5 / 10.0
        if is3:
            return "corner3" if y <= 92.5 else "above_break3"
        if dist <= 4.0:
            return "rim"
        return "paint" if abs(x) <= 80 and y <= 137.5 else "midrange"
    d = a.get(F.SHOT_DISTANCE)
    if isinstance(d, (int, float)):
        if is3:
            return "above_break3"      # cannot separate corners without coordinates
        return "rim" if d <= 4 else ("paint" if d <= 8 else "midrange")
    return None


def is_heave(a: dict, clock_s: float) -> bool:
    if clock_s > F.HEAVE_MAX_CLOCK_S:
        return False
    d = a.get(F.SHOT_DISTANCE)
    far = isinstance(d, (int, float)) and d >= F.HEAVE_MIN_DISTANCE_FT
    return far or _low(a, F.AREA) == "backcourt"


def game_meta(box: dict) -> dict:
    """Teams, rosters, starters and tip time from the boxscore."""
    try:
        g = box[F.GAME]
        home, away = g[F.BOX_HOME], g[F.BOX_AWAY]
    except (KeyError, TypeError) as e:
        raise FeedShapeError(f"boxscore missing {e}") from e
    teams = {}
    for side in (home, away):
        tid = int(side[F.BOX_TEAM_ID])
        players = [
            {"player_id": int(p[F.BOX_PLAYER_ID]), "name": str(p.get(F.BOX_PLAYER_NAME) or p[F.BOX_PLAYER_ID]),
             "starter": str(p.get(F.BOX_PLAYER_STARTER, "0")) == "1"}
            for p in side.get(F.BOX_PLAYERS, [])
        ]
        name = " ".join(x for x in (side.get(F.BOX_TEAM_CITY), side.get(F.BOX_TEAM_NAME)) if x) or str(tid)
        teams[tid] = {"team_id": tid, "tricode": str(side.get(F.BOX_TEAM_TRICODE) or tid), "name": name, "players": players}
    return {
        "game_id": str(g[F.GAME_ID]),
        "tipoff_utc": str(g.get(F.BOX_GAME_TIME_UTC) or ""),
        "home_team_id": int(home[F.BOX_TEAM_ID]),
        "away_team_id": int(away[F.BOX_TEAM_ID]),
        "teams": teams,
    }


def _repair_period_lineup(period_actions: list[dict], carried: set[int], team: int, player_team: dict[int, int]) -> tuple[set[int], bool]:
    """Infer who started a period when between-period substitutions are missing.

    A player seen acting (or being subbed out) before any sub-in this period must
    have started it. Returns (lineup, was_repaired).
    """
    entered: set[int] = set()
    must_start: list[int] = []
    started = False
    for a in period_actions:
        t, st = _low(a, F.ACTION_TYPE), _low(a, F.SUB_TYPE)
        if t == F.T_PERIOD and st == F.ST_PERIOD_START:
            started = True
            continue
        if not started:
            continue                                        # subs logged before the period-start event are already applied
        pid = _int_or_none(a.get(F.PERSON_ID))
        if t == F.T_SUBSTITUTION:
            if pid and player_team.get(pid) == team:
                if st == F.ST_SUB_IN:
                    entered.add(pid)
                elif st == F.ST_SUB_OUT and pid not in entered and pid not in must_start:
                    must_start.append(pid)
            continue
        if t not in F.ON_FLOOR_ACTION_TYPES or is_technical(a):
            continue                                        # bench technicals, ejections and timeouts prove nothing
        for key in (F.PERSON_ID, F.ASSIST_PERSON_ID, F.FOUL_DRAWN_PERSON_ID, F.STEAL_PERSON_ID, F.BLOCK_PERSON_ID):
            p = _int_or_none(a.get(key))
            if p and player_team.get(p) == team and p not in entered and p not in must_start:
                must_start.append(p)
    missing = [p for p in must_start if p not in carried]
    if not missing and len(carried) == 5:
        return set(carried), False
    lineup = list(dict.fromkeys(must_start))[:5]
    for p in sorted(carried):
        if len(lineup) >= 5:
            break
        if p not in lineup:
            lineup.append(p)
    return set(lineup), True


def segment_game(pbp: dict, box: dict) -> tuple[list[dict], dict]:
    """Return (possession rows, game info). Raises FeedShapeError on an unusable file."""
    meta = game_meta(box)
    try:
        raw_actions = pbp[F.GAME][F.ACTIONS]
    except (KeyError, TypeError) as e:
        raise FeedShapeError(f"play-by-play missing {e}") from e
    if not raw_actions:
        raise FeedShapeError(f"game {meta['game_id']}: play-by-play has no actions")
    actions = sorted(raw_actions, key=lambda a: (a.get(F.ORDER_NUMBER, a.get(F.ACTION_NUMBER, 0)), a.get(F.ACTION_NUMBER, 0)))

    home, away = meta["home_team_id"], meta["away_team_id"]
    other = {home: away, away: home}
    player_team = {p["player_id"]: tid for tid, t in meta["teams"].items() for p in t["players"]}
    lineups = {tid: {p["player_id"] for p in t["players"] if p["starter"]} for tid, t in meta["teams"].items()}
    fouls = {home: 0, away: 0}
    fouls_l2m = {home: 0, away: 0}                            # team fouls in the last two minutes of the period
    timeouts_used = {home: 0, away: 0}
    score = {home: 0, away: 0}
    season = F.season_from_game_id(meta["game_id"])

    by_period: dict[int, list[dict]] = {}
    for a in actions:
        by_period.setdefault(int(a.get(F.PERIOD) or 0), []).append(a)

    pending_tech = {home: 0, away: 0}                        # technical FT points made between periods
    rows: list[dict] = []
    stats = {"anomalies": 0, "lineup_repairs": 0, "unknown_action_types": set(), "penalty_checks": 0, "penalty_agree": 0}
    cur: dict | None = None
    period = 0

    def in_penalty(team: int | None, clock_s: float) -> bool:
        if team is None:
            return False
        limit = F.PENALTY_TEAM_FOULS if period <= 4 else F.PENALTY_TEAM_FOULS_OT
        return fouls[team] >= limit or (clock_s <= 120.0 and fouls_l2m[team] >= 1)

    def snapshot(off: int | None, clock_s: float = 720.0) -> dict:
        d = other.get(off) if off else None
        return {
            "def_in_penalty": in_penalty(d, clock_s),
            "after_timeout": bool(cur and cur["after_timeout"]),   # a timeout called after the deciding event must not count
            "off_lineup": sorted(lineups[off]) if off else [],
            "def_lineup": sorted(lineups[d]) if d else [],
            "off_team_fouls_period": fouls[off] if off else 0,
            "def_team_fouls_period": fouls[d] if d else 0,
            "off_timeouts_used": timeouts_used[off] if off else 0,
            "def_timeouts_used": timeouts_used[d] if d else 0,
        }

    def new_poss(off: int | None, clock_s: float, start_type: str) -> dict:
        carry = dict(pending_tech)
        pending_tech[home] = pending_tech[away] = 0
        return {
            "off": off, "period": period, "start_clock_s": clock_s, "start_type": start_type,
            "score_home_start": score[home], "score_away_start": score[away],
            "after_timeout": False, "n_off_reb": 0, "points": 0, "and_one": False,
            "tech": carry, "rebounded_label": None, "label": None, "awaiting_rebound": False,
            "keep_shot_label": False, "pending_foul": None, "live": False, "first_action": None, "retained_ft": False,
        }

    def emit(p: dict, label: dict | None, end_clock_s: float, end_action: int | None) -> None:
        off = p["off"]
        if label is None:
            label = {"ending": "period_end", "finisher": None, "family": None, "made": None, "heave": False, "ctx": snapshot(off, end_clock_s)}
        ctx = label["ctx"]
        finisher = label["finisher"]
        lineup_ok = len(ctx["off_lineup"]) == 5 and len(ctx["def_lineup"]) == 5 and (finisher is None or finisher in ctx["off_lineup"])
        heave = bool(label["heave"])
        family = "heave" if heave else label["family"]
        off_home = off == home
        s_off = p["score_home_start"] if off_home else p["score_away_start"]
        s_def = p["score_away_start"] if off_home else p["score_home_start"]
        used_off, used_def = ctx["off_timeouts_used"], ctx["def_timeouts_used"]
        rows.append({
            "game_id": meta["game_id"], "season": season, "tipoff_utc": meta["tipoff_utc"],
            "poss_idx": len(rows), "period": p["period"],
            "start_clock_s": round(p["start_clock_s"], 2), "end_clock_s": round(end_clock_s, 2),
            "off_team_id": off, "def_team_id": other[off], "home_team_id": home, "away_team_id": away,
            "off_is_home": off_home,
            "home_score_start": p["score_home_start"], "away_score_start": p["score_away_start"],
            "margin_start": s_off - s_def,
            "off_lineup": ctx["off_lineup"], "def_lineup": ctx["def_lineup"],
            "off_team_fouls_period": ctx["off_team_fouls_period"], "def_team_fouls_period": ctx["def_team_fouls_period"],
            "def_in_penalty": ctx["def_in_penalty"],
            "off_timeouts_used": used_off, "def_timeouts_used": used_def,
            "off_timeouts_left": max(0, F.TIMEOUTS_PER_GAME - used_off), "def_timeouts_left": max(0, F.TIMEOUTS_PER_GAME - used_def),
            "start_type": p["start_type"], "after_timeout": ctx["after_timeout"],
            "n_off_reb": p["n_off_reb"], "and_one": p["and_one"],
            "tech_ft_points_off": p["tech"][off], "tech_ft_points_def": p["tech"][other[off]], "retained_ft": p["retained_ft"],
            "is_heave": heave, "ending_type": label["ending"], "finisher_id": finisher,
            "shot_family": family, "shot_made": label["made"], "points": p["points"],
            "lineup_ok": lineup_ok,
            "model_eligible": lineup_ok and not heave and label["ending"] in ENDING_CLASSES,
            "start_action_number": p["first_action"], "end_action_number": end_action,
        })

    def finalize(a: dict, clock_s: float, next_start: str) -> None:
        nonlocal cur
        off = cur["off"]
        emit(cur, cur["label"], clock_s, a.get(F.ACTION_NUMBER))
        cur = new_poss(other[off], clock_s, next_start)

    def claim(team: int, a: dict, clock_s: float) -> None:
        """Make `team` the offense of the current possession, closing a stale one if needed."""
        nonlocal cur
        if cur["off"] is None or cur["off"] == team:
            cur["off"] = team
            return
        if cur["label"] is not None and cur["awaiting_rebound"]:
            finalize(a, clock_s, "def_rebound")           # miss whose defensive rebound was not logged
        elif not cur["live"]:
            cur["off"] = team                              # e.g. team keeps the ball after flagrant FTs
        else:
            stats["anomalies"] += 1                         # change of possession we cannot explain
            cur["label"] = {"ending": "unknown", "finisher": None, "family": None, "made": None, "heave": False, "ctx": snapshot(cur["off"], clock_s)}
            finalize(a, clock_s, "other")
        cur["off"] = team

    def and_one_follows(i: int, team: int) -> bool:
        """Is the made basket at index i followed by a single free throw that belongs to the same possession?

        A rebound cannot end the scan: no shot has been missed yet, so any rebound seen here is the
        dead-ball one logged after a missed technical free throw. A flagrant, clear-path or take-foul
        free throw is not an and-one: the team keeps the ball, so the basket ends its possession and
        the free throws open the next one. An away-from-play "1 of 1" after a made basket IS one: the
        ball goes to the other team afterwards (33 of 34 such trips in 2023-26; the other ended a period).
        """
        for b in actions[i + 1:]:
            t = _low(b, F.ACTION_TYPE)
            if t in (F.T_2PT, F.T_3PT, F.T_TURNOVER, F.T_JUMPBALL, F.T_PERIOD, F.T_HEAVE):
                return False
            if t == F.T_FREETHROW and not is_technical(b):
                if _low(b, F.DESCRIPTOR) in F.FT_KEEP_BALL_DESCRIPTORS:
                    return False
                m = _FT_RE.search(_low(b, F.SUB_TYPE))
                return bool(m) and m.group(2) == "1" and b.get(F.TEAM_ID) == team
        return False

    for i, a in enumerate(actions):
        t, st = _low(a, F.ACTION_TYPE), _low(a, F.SUB_TYPE)
        team = _int_or_none(a.get(F.TEAM_ID))
        if team is not None and team not in other:
            team = None
        pid = _int_or_none(a.get(F.PERSON_ID))
        if t not in F.KNOWN_ACTION_TYPES:
            stats["unknown_action_types"].add(t)

        if t == F.T_PERIOD:
            if st == F.ST_PERIOD_START:
                period = int(a.get(F.PERIOD) or period + 1)
                fouls[home] = fouls[away] = 0
                fouls_l2m[home] = fouls_l2m[away] = 0
                if period > 1:
                    for tid in (home, away):
                        lineups[tid], repaired = _repair_period_lineup(by_period.get(period, []), lineups[tid], tid, player_team)
                        stats["lineup_repairs"] += int(repaired)
                cur = new_poss(None, period_length(period), "period_start")
            elif st == F.ST_PERIOD_END and cur is not None:
                if cur["off"] is not None:
                    if cur["label"] is not None and cur["awaiting_rebound"]:
                        emit(cur, cur["label"], 0.0, a.get(F.ACTION_NUMBER))
                    elif cur["label"] is None and cur["rebounded_label"] is not None:
                        emit(cur, cur["rebounded_label"], 0.0, a.get(F.ACTION_NUMBER))   # buzzer shot, then a rebound as time expired
                    elif cur["start_clock_s"] > 0.0:
                        emit(cur, None, 0.0, a.get(F.ACTION_NUMBER))
                cur = None
            continue
        if t == F.T_SUBSTITUTION:
            if pid and team:                                # between-period subs are logged before the period-start event
                (lineups[team].add if st == F.ST_SUB_IN else lineups[team].discard)(pid)
            continue
        if cur is None:
            if t == F.T_FREETHROW and team and _low(a, F.SHOT_RESULT) == F.SHOT_MADE:
                pending_tech[team] += 1                     # e.g. a technical shot between periods
            for tid, key in ((home, F.SCORE_HOME), (away, F.SCORE_AWAY)):
                try:
                    score[tid] = int(a[key])
                except (KeyError, TypeError, ValueError):
                    pass
            continue                                        # other events outside any period

        clock_s = parse_clock(a.get(F.CLOCK))
        for tid, key in ((home, F.SCORE_HOME), (away, F.SCORE_AWAY)):
            try:
                score[tid] = int(a[key])
            except (KeyError, TypeError, ValueError):
                pass
        if cur["first_action"] is None and t != F.T_TIMEOUT:
            cur["first_action"] = a.get(F.ACTION_NUMBER)

        if t == F.T_TIMEOUT:
            if team:
                timeouts_used[team] += 1
            cur["after_timeout"] = True
        elif t in (F.T_2PT, F.T_3PT) and team:
            claim(team, a, clock_s)
            cur["live"] = True
            made = _low(a, F.SHOT_RESULT) == F.SHOT_MADE
            cur["label"] = {"ending": "shot", "finisher": pid, "family": shot_family(a), "made": made,
                            "heave": is_heave(a, clock_s), "ctx": snapshot(team, clock_s)}
            cur["awaiting_rebound"] = False
            if made:
                cur["points"] += 3 if t == F.T_3PT else 2
                if and_one_follows(i, team):
                    cur["and_one"] = cur["keep_shot_label"] = True
                else:
                    finalize(a, clock_s, "made_basket")
            else:
                cur["awaiting_rebound"] = True
        elif t == F.T_HEAVE and team:
            claim(team, a, clock_s)                         # 2025-26 onward: team-charged missed heave, no shooter
            cur["live"] = True
            cur["label"] = {"ending": "shot", "finisher": None, "family": None, "made": False, "heave": True, "ctx": snapshot(team, clock_s)}
            cur["awaiting_rebound"] = True
        elif t == F.T_REBOUND:
            if not cur["awaiting_rebound"]:
                continue                                    # dead-ball rebound between free throws
            cur["awaiting_rebound"] = False
            if st == F.ST_REBOUND_OFFENSIVE:
                cur["n_off_reb"] += 1
                cur["rebounded_label"] = cur["label"]       # kept in case the horn sounds before anything else happens
                cur["label"] = None                         # the possession is still alive
                cur["keep_shot_label"] = False
                cur["pending_foul"] = None
            else:
                finalize(a, clock_s, "def_rebound")
        elif t == F.T_TURNOVER and team:
            claim(team, a, clock_s)
            cur["live"] = True
            cur["label"] = {"ending": "turnover", "finisher": pid, "family": None, "made": None, "heave": False, "ctx": snapshot(team, clock_s)}
            finalize(a, clock_s, "turnover")
        elif t == F.T_FOUL and team:
            if is_technical(a):
                continue
            offensive = st == F.ST_FOUL_OFFENSIVE
            if cur["off"] is None and not offensive:
                cur["off"] = other[team]
            if cur["off"] is not None and team == other[cur["off"]]:
                cur["live"] = True
                cur["pending_foul"] = {"finisher": _int_or_none(a.get(F.FOUL_DRAWN_PERSON_ID)), "ctx": snapshot(cur["off"], clock_s),
                                       "penalty_before": in_penalty(team, clock_s)}
            if not offensive and F.FOUL_DOUBLE not in _low(a, F.DESCRIPTOR):
                quals = a.get(F.QUALIFIERS)
                if isinstance(quals, list):                 # cross-check our penalty rule against the feed's own tag
                    stats["penalty_checks"] += 1
                    stats["penalty_agree"] += int(("inpenalty" in quals) == in_penalty(team, clock_s))
                fouls[team] += 1                            # offensive and double fouls do not count toward the penalty
                if clock_s <= 120.0:
                    fouls_l2m[team] += 1
        elif t == F.T_FREETHROW and team:
            made = _low(a, F.SHOT_RESULT) == F.SHOT_MADE
            if is_technical(a):
                cur["tech"][team] += int(made)
                continue
            claim(team, a, clock_s)
            cur["live"] = True
            m = _FT_RE.search(st)
            k, n = (int(m.group(1)), int(m.group(2))) if m else (1, 1)
            desc = _low(a, F.DESCRIPTOR)
            # An away-from-play trip keeps the ball only as a lone free throw with no basket before it.
            # After a made basket it is an and-one shot by a teammate, and a two-shot trip is an
            # ordinary penalty trip; both end the possession the usual way.
            ordinary_away = desc == F.FT_AWAY_FROM_PLAY and (cur["keep_shot_label"] or n >= 2)
            if desc in F.FT_RETAIN_DESCRIPTORS and not ordinary_away:
                cur["points"] += int(made)                  # shooting team keeps the ball: no label, no ending
                cur["retained_ft"] = True
                continue
            if made:
                cur["points"] += 1
            if not cur["keep_shot_label"] and (cur["label"] is None or cur["label"]["ending"] != "foul_drawn"):
                pf = cur["pending_foul"] or {}
                cur["label"] = {"ending": "foul_drawn", "finisher": pf.get("finisher") or pid, "family": None,
                                "made": None, "heave": False, "ctx": pf.get("ctx") or snapshot(team, clock_s)}
            cur["awaiting_rebound"] = False
            if k >= n:
                if made:
                    finalize(a, clock_s, "free_throw")
                else:
                    cur["awaiting_rebound"] = True

    if not rows:
        raise FeedShapeError(f"game {meta['game_id']}: no possessions found; check actionType values in cdn_fields.py")
    final = {"home": score[home], "away": score[away]}
    info = {**meta, "season": season, "final_score": final,
            "anomalies": stats["anomalies"], "lineup_repairs": stats["lineup_repairs"],
            "penalty_checks": stats["penalty_checks"], "penalty_agree": stats["penalty_agree"],
            "derived_box": bool(box.get("_derived")),
            "unknown_action_types": sorted(stats["unknown_action_types"]),
            "synthetic": bool(pbp.get(F.SYNTHETIC_MARKER) or box.get(F.SYNTHETIC_MARKER))}
    return rows, info
