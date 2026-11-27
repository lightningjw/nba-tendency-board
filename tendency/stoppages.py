"""The dead-ball table: one row per stoppage, beside the possession table (step 2.4).

Coaches decide at stoppages, so this table holds what was known when play stopped and what each bench
did before it resumed. It describes what coaches DID. Nothing here says what they should have done.

A stoppage is one whistle (or one made basket) and everything logged before play resumes. A live action
(a shot, a rebound a player pulls down, a steal, a block, a turnover with a steal) ends it, and so does a
new whistle after the clock has run. Its `kind` is what stopped play:

* `foul`, `turnover` (one with no steal: out of bounds, travel, offensive foul...), `violation`,
  `jump_ball`, `team_rebound` (a missed shot nobody secured: out of bounds).
* `made_basket`, `made_free_throw` (the last free throw of a trip, made): not a dead ball for
  substitutions, but a timeout can be called before the inbound, so each is a row with the new score.
  `subs_allowed` is False.
* `timeout_live`: a timeout called with the ball in play (more than 8 seconds of game clock after the
  last whistle or basket, or after a live action). The row exists only because the timeout was called,
  so it is never a decision row: there is no matching row for the moments nobody called one.
* `period_break`: between periods. Its substitutions are the difference between the five who ended
  one period and the five who started the next (the feed often leaves those substitutions out).
* `unlogged`: substitutions with nothing before them that stops play. The feed missed the whistle.

Each row carries: clock and score when play stopped; for both fives, each player's id, seconds in his
current stint, seconds played and personal fouls; full timeouts used; the run in progress (points each
team scored in the last `RUN_POSSESSIONS` possessions, and points scored since the other team last
did); and what happened next (timeouts and challenges by team, players taken out and sent in, a take
foul). Personal fouls are counted here and checked against the "(3 PF)" the feed prints.

`mandatory_due` marks a stoppage where the rules stop play anyway: no timeout yet this quarter with
6:59 or less left, or fewer than two with 2:59 or less left. The feed does not say whether a timeout
taken then was asked for or charged by the scorer, so those rows are not coach decisions.

`build_stoppages` also returns one row per player and game (seconds by period, fouls through three
quarters, seconds in the last five minutes of regulation) for the closing-five decision.
"""
from __future__ import annotations

import re
from bisect import bisect_right

from . import cdn_fields as F
from .possessions import (FeedShapeError, _int_or_none, _low, _repair_period_lineup, free_throw_kinds, game_meta, is_technical,
                          parse_clock, period_length)

RUN_POSSESSIONS = 10                 # the run in progress looks back this many possessions (both teams together)
CLOSING_WINDOW_S = 300.0             # the closing five: the last five minutes of the fourth quarter
TIMEOUT_ATTACH_S = 8.0               # a timeout this long after the whistle belongs to it; later, it was called in live play
STOPPAGE_KINDS = ["foul", "turnover", "violation", "jump_ball", "team_rebound", "made_basket", "made_free_throw", "timeout_live",
                  "period_break", "unlogged"]
SOFT_KINDS = ("made_basket", "made_free_throw")      # the ball is inbounded without a whistle: a timeout is possible, a substitution is not
_PF_RE = re.compile(r"\((\d+) PF\)")
_FT_RE = re.compile(F.FT_TRIP_PATTERN)
_LIVE = {F.T_2PT, F.T_3PT, F.T_HEAVE, "steal", "block"}


def mandatory_due(period: int, clock_s: float, timeouts_this_period: int) -> bool:
    """Is a mandatory timeout due at the next dead ball? Regulation only: under 7:00 with none taken, under 3:00 with fewer than two."""
    if period > 4:
        return False
    return (clock_s <= F.MANDATORY_FIRST_S and timeouts_this_period < 1) or (clock_s <= F.MANDATORY_SECOND_S and timeouts_this_period < 2)


def build_stoppages(pbp: dict, box: dict, poss_rows: list[dict] | None = None) -> tuple[list[dict], list[dict], dict]:
    """Return (stoppage rows, player-game rows, checks). `poss_rows` are this game's possession rows (for the run)."""
    meta = game_meta(box)
    try:
        raw = pbp[F.GAME][F.ACTIONS]
    except (KeyError, TypeError) as e:
        raise FeedShapeError(f"play-by-play missing {e}") from e
    actions = sorted(raw, key=lambda a: (a.get(F.ORDER_NUMBER, a.get(F.ACTION_NUMBER, 0)), a.get(F.ACTION_NUMBER, 0)))
    gid, home, away = meta["game_id"], meta["home_team_id"], meta["away_team_id"]
    sides = (("home", home), ("away", away))
    other = {home: away, away: home}
    season = F.season_from_game_id(gid)
    player_team = {p["player_id"]: tid for tid, t in meta["teams"].items() for p in t["players"]}
    starters = {p["player_id"] for t in meta["teams"].values() for p in t["players"] if p["starter"]}
    lineups = {tid: {p["player_id"] for p in t["players"] if p["starter"]} for tid, t in meta["teams"].items()}

    by_period: dict[int, list[dict]] = {}
    for a in actions:
        by_period.setdefault(int(a.get(F.PERIOD) or 0), []).append(a)
    ft_kind = free_throw_kinds(actions)
    pos_of = {a.get(F.ACTION_NUMBER): i for i, a in enumerate(actions)}
    ends = sorted((pos_of.get(r["end_action_number"], -1), r["off_team_id"], r["points"]) for r in (poss_rows or []))
    end_pos = [e[0] for e in ends]

    pf: dict[int, int] = {}                                  # personal fouls per player
    secs: dict[int, float] = {}                              # seconds played
    secs_q: dict[int, list[float]] = {}                      # ... by regulation quarter
    secs_close: dict[int, float] = {}                        # ... in the last five minutes of the fourth
    stint_start: dict[int, float] = {p: 0.0 for tid in lineups for p in lineups[tid]}
    q3: dict[int, dict] = {}                                 # per player, as the third quarter ended
    score = {home: 0, away: 0}
    unanswered = {home: 0, away: 0}                          # points scored since the other team last scored
    full_used = {home: 0, away: 0}
    period_tos = 0                                           # timeouts of any kind so far this period, both teams
    last_timeout_t: float | None = None
    period, clock_s, now, in_period = 0, 720.0, 0.0, False
    period_base = 0.0                                        # game seconds elapsed before this period
    margin_q3: int | None = None
    margin_close: int | None = None
    checks = {"pf_checks": 0, "pf_agree": 0}
    rows: list[dict] = []
    cur: dict | None = None

    def advance(to_clock: float) -> None:
        nonlocal clock_s, now, margin_close
        if to_clock >= clock_s:
            return                                           # clocks that run backwards are feed noise
        dt = clock_s - to_clock
        close_dt = max(0.0, min(clock_s, CLOSING_WINDOW_S) - to_clock) if period == 4 else 0.0
        if period == 4 and margin_close is None and to_clock <= CLOSING_WINDOW_S:
            margin_close = score[home] - score[away]         # the score when the last five minutes began
        for tid in (home, away):
            for p in lineups[tid]:
                secs[p] = secs.get(p, 0.0) + dt
                if period <= 4:
                    secs_q.setdefault(p, [0.0] * 4)[period - 1] += dt
                if close_dt:
                    secs_close[p] = secs_close.get(p, 0.0) + close_dt
        clock_s, now = to_clock, now + dt

    def run_points(i: int) -> dict[int, int]:
        """Points by each team in the last RUN_POSSESSIONS possessions that ended at or before action index i."""
        hi = bisect_right(end_pos, i)
        pts = {home: 0, away: 0}
        for _, off, p in ends[max(0, hi - RUN_POSSESSIONS):hi]:
            if off in pts:
                pts[off] += int(p)
        return pts

    def open_stop(kind: str, i: int, a: dict, team: int | None) -> dict:
        run = run_points(i)
        row = {"game_id": gid, "season": season, "tipoff_utc": meta["tipoff_utc"], "stop_idx": len(rows), "period": period,
               "clock_s": round(clock_s, 2), "game_secs": round(now, 2), "kind": kind, "subs_allowed": kind not in SOFT_KINDS,
               "trigger_team_id": team, "trigger_action_number": a.get(F.ACTION_NUMBER),
               "home_team_id": home, "away_team_id": away, "home_score": score[home], "away_score": score[away],
               "timeouts_this_period": period_tos, "mandatory_due": mandatory_due(period, clock_s, period_tos),
               "secs_since_timeout": None if last_timeout_t is None else round(now - last_timeout_t, 2),
               "n_free_throws": 0, "take_foul_team_id": None, "timeout_gap_s": None,
               "foul_player_ids": [], "foul_team_ids": [], "foul_counts": [], "foul_technical": []}
        for side, tid in sides:
            five = sorted(lineups[tid])
            row.update({f"{side}_lineup": five, f"{side}_stint_s": [round(now - stint_start.get(p, now), 1) for p in five],
                        f"{side}_secs_played": [round(secs.get(p, 0.0), 1) for p in five], f"{side}_fouls": [pf.get(p, 0) for p in five],
                        f"{side}_timeouts_used": full_used[tid], f"{side}_run_points": run[tid], f"{side}_unanswered": unanswered[tid],
                        f"{side}_timeout": False, f"{side}_challenge": False, f"{side}_subs_out": [], f"{side}_subs_in": []})
        return row

    def close_stop() -> None:
        nonlocal cur
        if cur is not None:
            rows.append(cur)
            cur = None

    def side_of(tid: int) -> str:
        return "home" if tid == home else "away"

    for i, a in enumerate(actions):
        t, st = _low(a, F.ACTION_TYPE), _low(a, F.SUB_TYPE)
        team = _int_or_none(a.get(F.TEAM_ID))
        if team is not None and team not in other:
            team = None
        pid = _int_or_none(a.get(F.PERSON_ID))

        if t == F.T_PERIOD:
            if st == F.ST_PERIOD_START:
                new_period = int(a.get(F.PERIOD) or period + 1)
                before = {tid: set(lineups[tid]) for tid in (home, away)}
                if new_period > 1:
                    for tid in (home, away):
                        lineups[tid], _ = _repair_period_lineup(by_period.get(new_period, []), lineups[tid], tid, player_team)
                if cur is not None and cur["kind"] == "period_break":
                    for side, tid in sides:                  # who actually changed, whether or not the feed logged it
                        cur[f"{side}_subs_out"] = sorted(set(cur[f"{side}_lineup"]) - lineups[tid])
                        cur[f"{side}_subs_in"] = sorted(lineups[tid] - set(cur[f"{side}_lineup"]))
                close_stop()
                period_base += period_length(period) if period else 0.0
                period, clock_s, now, in_period, period_tos = new_period, period_length(new_period), period_base, True, 0
                for tid in (home, away):
                    for p in lineups[tid] - before[tid]:
                        stint_start[p] = now
            elif st == F.ST_PERIOD_END and in_period:
                advance(0.0)
                close_stop()
                if period == 3:
                    margin_q3 = score[home] - score[away]
                    for p, tid in player_team.items():
                        q3[p] = {"fouls_q3": pf.get(p, 0), "on_floor_q3": p in lineups[tid]}
                cur = open_stop("period_break", i, a, None)
                in_period = False
            continue

        if t == F.T_SUBSTITUTION:
            if pid and team:
                if in_period:
                    advance(parse_clock(a.get(F.CLOCK)))
                    if cur is None:
                        cur = open_stop("unlogged", i, a, None)
                if st == F.ST_SUB_IN:
                    lineups[team].add(pid)
                    stint_start[pid] = now
                else:
                    lineups[team].discard(pid)
                if cur is not None and cur["kind"] != "period_break":
                    cur[f"{side_of(team)}_subs_in" if st == F.ST_SUB_IN else f"{side_of(team)}_subs_out"].append(pid)
            continue
        if not in_period:
            continue

        advance(parse_clock(a.get(F.CLOCK)))
        before_score = dict(score)

        def take_score() -> None:
            for tid, key in ((home, F.SCORE_HOME), (away, F.SCORE_AWAY)):
                try:
                    score[tid] = int(a[key])
                except (KeyError, TypeError, ValueError):
                    pass
            for tid in (home, away):
                gain = score[tid] - before_score[tid]
                if gain > 0:
                    unanswered[tid] += gain
                    unanswered[other[tid]] = 0

        made_fg = t in (F.T_2PT, F.T_3PT) and _low(a, F.SHOT_RESULT) == F.SHOT_MADE
        steal_next = i + 1 < len(actions) and _low(actions[i + 1], F.ACTION_TYPE) == "steal"
        live = (t in _LIVE or (t == F.T_REBOUND and pid is not None)
                or (t == F.T_TURNOVER and (_int_or_none(a.get(F.STEAL_PERSON_ID)) is not None or steal_next)))
        if live:
            close_stop()
            take_score()
            if made_fg:
                cur = open_stop("made_basket", i, a, team)   # the score in the row includes the basket
            continue

        take_score()
        kind = {F.T_FOUL: "foul", F.T_TURNOVER: "turnover", "violation": "violation", F.T_JUMPBALL: "jump_ball",
                F.T_REBOUND: "team_rebound"}.get(t)
        if t == F.T_JUMPBALL and _low(a, F.DESCRIPTOR) == F.JUMPBALL_START:
            continue                                         # the opening tip is not a stoppage
        if t == F.T_TIMEOUT:
            if cur is None or cur["clock_s"] - clock_s > TIMEOUT_ATTACH_S:
                close_stop()                                 # called with the ball in play, not at the whistle before it
                cur = open_stop("timeout_live", i, a, team)
        elif kind is not None:
            if cur is None:
                cur = open_stop(kind, i, a, team)
            elif cur["clock_s"] != round(clock_s, 2):        # the clock ran since the last whistle: this is a new one
                close_stop()
                cur = open_stop(kind, i, a, team)
            elif cur["kind"] in SOFT_KINDS and kind in ("foul", "violation", "jump_ball"):
                cur["kind"], cur["subs_allowed"] = kind, True    # an and-one, or a whistle before the inbound: a real dead ball
                cur["trigger_team_id"] = team
        elif cur is None:
            continue                                         # free throws, ejections and replays never open a stoppage

        if t == F.T_FOUL and pid and team:
            tech = is_technical(a)
            if not tech:
                pf[pid] = pf.get(pid, 0) + 1
                m = _PF_RE.search(str(a.get(F.DESCRIPTION) or ""))
                if m:
                    checks["pf_checks"] += 1
                    checks["pf_agree"] += int(int(m.group(1)) == pf[pid])
                if _low(a, F.DESCRIPTOR) == F.FOUL_TAKE:
                    cur["take_foul_team_id"] = team
                five = cur[f"{side_of(team)}_lineup"]
                if pid in five:                              # the foul that stopped play is known when play stops
                    cur[f"{side_of(team)}_fouls"][five.index(pid)] = pf[pid]
            cur["foul_player_ids"].append(pid)
            cur["foul_team_ids"].append(team)
            cur["foul_counts"].append(pf.get(pid, 0))
            cur["foul_technical"].append(tech)
        elif t == F.T_FREETHROW:
            cur["n_free_throws"] += 1
            m = _FT_RE.search(st)
            last = (not m) or int(m.group(1)) >= int(m.group(2))
            if last and ft_kind[i] != F.FT_TECHNICAL and _low(a, F.SHOT_RESULT) == F.SHOT_MADE:
                close_stop()                                 # like a made basket: the ball is inbounded next, with the new score
                cur = open_stop("made_free_throw", i, a, team)
        elif t == F.T_TIMEOUT:
            period_tos += 1
            last_timeout_t = now
            cur["timeout_gap_s"] = round(cur["clock_s"] - clock_s, 2)
            if team:
                if st == F.ST_TIMEOUT_CHALLENGE:
                    cur[f"{side_of(team)}_challenge"] = True
                else:
                    cur[f"{side_of(team)}_timeout"] = True
                    full_used[team] += 1

    if cur is not None and cur["kind"] != "period_break":
        rows.append(cur)                                     # a period break with no period after it is the end of the game

    final = {home: score[home], away: score[away]}
    players = []
    for tid, t in meta["teams"].items():
        for p in t["players"]:
            p_id = p["player_id"]
            q = secs_q.get(p_id, [0.0] * 4)
            players.append({"game_id": gid, "season": season, "tipoff_utc": meta["tipoff_utc"], "team_id": tid, "opp_team_id": other[tid],
                            "player_id": p_id, "starter": p_id in starters, "is_home": tid == home,
                            "secs_q1": round(q[0], 1), "secs_q2": round(q[1], 1), "secs_q3": round(q[2], 1), "secs_q4": round(q[3], 1),
                            "secs_total": round(secs.get(p_id, 0.0), 1), "secs_close": round(secs_close.get(p_id, 0.0), 1),
                            "fouls_q3": q3.get(p_id, {}).get("fouls_q3"), "on_floor_q3": q3.get(p_id, {}).get("on_floor_q3"),
                            "fouls_final": pf.get(p_id, 0),
                            "margin_q3": None if margin_q3 is None else (margin_q3 if tid == home else -margin_q3),
                            "margin_close": None if margin_close is None else (margin_close if tid == home else -margin_close),
                            "team_won": final[tid] > final[other[tid]]})
    return rows, players, checks
