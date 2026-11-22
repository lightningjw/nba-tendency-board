#!/usr/bin/env python3
"""Fetch current-season games and today's schedule with nba_api. RUN THIS ON YOUR OWN MACHINE ONLY.

    pip install nba_api                      # not a project dependency; see DECISIONS.md 71
    python scripts/fetch_nba_api.py --season 2026-27
    python scripts/fetch_nba_api.py --season 2026-27 --through 2026-11-20 --max-games 40

What it does:
  1. Reads the season schedule (stats.nba.com, `ScheduleLeagueV2`).
  2. For every regular-season game that is final, dated on or before --through (US Eastern, default
     yesterday) and not yet on disk: saves the play-by-play and boxscore JSON exactly as received to
     <root>/raw/pbp/<gameId>.json and <root>/raw/box/<gameId>.json (`nba_api.live`, the CDN feed).
  3. Writes <root>/schedule/<date>.json for today and tomorrow, which `python -m tendency predict` reads.

It was written against nba_api 1.12 and tested only with a fake client: it has NEVER been run against
nba.com. stats.nba.com and cdn.nba.com refuse cloud hosts, and nba_api reaches the CDN by sending
browser-like headers. Whether you are comfortable with that is the terms decision in roadmap step
1.0a; read that before the first run. One request at a time, a pause between requests, no retries.

After the first run, check the real boxscore field names before trusting anything built on it:
    python -m tendency verify-fields data/real/raw/pbp/<gameId>.json data/real/raw/box/<gameId>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
REPO = Path(__file__).resolve().parents[1]
REGULAR_SEASON_PREFIX = "002"
STATUS_FINAL = 3


class FetchError(RuntimeError):
    pass


class NbaApiClient:
    """The three calls this script makes. nba_api is imported here so tests and the package never need it."""

    def __init__(self, timeout: int = 30):
        try:
            from nba_api.live.nba.endpoints import boxscore, playbyplay
            from nba_api.stats.endpoints import scheduleleaguev2
        except ImportError as e:
            raise FetchError("nba_api is not installed. On your own machine: pip install nba_api") from e
        self._box, self._pbp, self._sched, self.timeout = boxscore, playbyplay, scheduleleaguev2, timeout

    def schedule(self, season: str) -> dict:
        return self._sched.ScheduleLeagueV2(season=season, timeout=self.timeout).get_dict()

    def pbp(self, game_id: str) -> dict:
        return self._pbp.PlayByPlay(game_id, timeout=self.timeout).get_dict()

    def box(self, game_id: str) -> dict:
        return self._box.BoxScore(game_id, timeout=self.timeout).get_dict()


def _write_atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    os.replace(tmp, path)


def parse_schedule(doc: dict) -> list[dict]:
    """Regular-season games as {game_id, date (US Eastern), tipoff_utc, status, home_team_id, away_team_id}."""
    try:
        dates = doc["leagueSchedule"]["gameDates"]
    except (KeyError, TypeError) as e:
        raise FetchError(f"the schedule response has no leagueSchedule.gameDates ({e!r}); nba_api or the endpoint changed shape") from e
    out = []
    for d in dates:
        for g in d.get("games", []):
            try:
                gid = str(g["gameId"])
                if not gid.startswith(REGULAR_SEASON_PREFIX):
                    continue
                tip = datetime.fromisoformat(str(g["gameDateTimeUTC"]).replace("Z", "+00:00"))
                out.append({"game_id": gid, "date": tip.astimezone(ET).date().isoformat(), "tipoff_utc": tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "status": int(g["gameStatus"]), "home_team_id": int(g["homeTeam"]["teamId"]), "away_team_id": int(g["awayTeam"]["teamId"])})
            except (KeyError, TypeError, ValueError) as e:
                raise FetchError(f"schedule game {g.get('gameId')!r} is missing a field this script needs ({e!r})") from e
    if not out:
        raise FetchError("the schedule holds no regular-season games; check --season (format 2026-27)")
    return sorted(out, key=lambda g: (g["tipoff_utc"], g["game_id"]))


def fetch(api, root: Path, season: str, through: str, today: str, max_games: int = 60, pause: float = 0.8,
          log=print, sleep=time.sleep) -> dict:
    """Download finished games up to `through`, then write schedule files for `today` and the day after."""
    games = parse_schedule(api.schedule(season))
    want = [g for g in games if g["status"] == STATUS_FINAL and g["date"] <= through
            and not ((root / "raw" / "pbp" / f"{g['game_id']}.json").exists() and (root / "raw" / "box" / f"{g['game_id']}.json").exists())]
    out = {"downloaded": [], "failed": [], "left_for_next_run": max(0, len(want) - max_games), "schedules": []}
    for g in want[:max_games]:
        gid = g["game_id"]
        try:
            pbp = api.pbp(gid)
            sleep(pause)
            box = api.box(gid)
            sleep(pause)
            if not pbp.get("game", {}).get("actions"):
                raise FetchError("play-by-play has no game.actions")
            if "homeTeam" not in box.get("game", {}):
                raise FetchError("boxscore has no game.homeTeam")
        except Exception as e:                               # one bad game must not stop the rest; nothing partial is written
            out["failed"].append({"game_id": gid, "error": f"{type(e).__name__}: {e}"})
            log(f"fetch: {gid} failed: {type(e).__name__}: {e}")
            if len(out["failed"]) >= 3 and not out["downloaded"]:
                raise FetchError("the first three games all failed; stopping. The host may be refusing this machine.") from e
            continue
        _write_atomic(root / "raw" / "box" / f"{gid}.json", box)   # boxscore first: a game counts as present only with both files
        _write_atomic(root / "raw" / "pbp" / f"{gid}.json", pbp)
        out["downloaded"].append(gid)
    for d in (today, (date.fromisoformat(today) + timedelta(days=1)).isoformat()):
        day = [{k: g[k] for k in ("game_id", "home_team_id", "away_team_id", "tipoff_utc")} for g in games if g["date"] == d]
        if day:
            _write_atomic(root / "schedule" / f"{d}.json", {"date": d, "source": "nba_api ScheduleLeagueV2", "games": day})
            out["schedules"].append({"date": d, "n_games": len(day)})
    log(f"fetch: downloaded {len(out['downloaded'])} games, failed {len(out['failed'])}, {out['left_for_next_run']} left for the next run; "
        "schedules written: " + (", ".join("%s (%d)" % (s["date"], s["n_games"]) for s in out["schedules"]) or "none"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--season", required=True, help="season in nba.com form, e.g. 2026-27")
    ap.add_argument("--root", type=Path, default=REPO / "data" / "real")
    ap.add_argument("--through", help="last US Eastern game date to download, YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--max-games", type=int, default=60, help="stop after this many downloads; run again for the rest")
    ap.add_argument("--pause", type=float, default=0.8, help="seconds between requests")
    a = ap.parse_args(argv)
    today = datetime.now(ET).date()
    try:
        through = date.fromisoformat(a.through).isoformat() if a.through else (today - timedelta(days=1)).isoformat()
        res = fetch(NbaApiClient(), a.root, a.season, through, today.isoformat(), a.max_games, a.pause)
    except (FetchError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
