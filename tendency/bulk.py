"""Bulk history from shufinskiy/nba_data (Apache-2.0), a GitHub mirror of the cdn.nba.com play-by-play.

Each `cdnnba_<season>.csv` holds one row per action with the feed's own key names as columns, so a
row converts straight back into a feed-shaped action. This module downloads a season archive,
rebuilds per-game play-by-play JSON in the feed's shape, and derives the three things the pipeline
normally takes from the boxscore, which the archive does not include:

* home/away: whichever of scoreHome/scoreAway moves when a team scores
* tip time:  the first action's timeActual
* starters:  players who act (or are subbed out) in period 1 before any sub-in

The derived boxscore carries `_derived: true`. Team names are tricodes; player names are the
feed's "J. Tatum" form. Raw files land in the same layout as `ingest`, so everything downstream is
unchanged.
"""
from __future__ import annotations

import json
import math
import tarfile
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from . import cdn_fields as F
from .ingest import HostBlockedError, IngestError, _write_atomic

LIST_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"
RAW_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets/{name}.tar.xz"
INT_COLS = {"actionNumber", "orderNumber", "period", "personId", "teamId", "possession", "isFieldGoal", "assistPersonId",
            "foulDrawnPersonId", "stealPersonId", "blockPersonId", "shotActionNumber", "jumpBallWonPersonId",
            "jumpBallLostPersonId", "jumpBallRecoverdPersonId", "officialId", "pointsTotal", "reboundTotal",
            "reboundDefensiveTotal", "reboundOffensiveTotal", "turnoverTotal", "assistTotal", "foulPersonalTotal", "foulTechnicalTotal"}
LIVE = {"2pt", "3pt", "freethrow", "rebound", "turnover", "foul", "steal", "block", "jumpball", "violation"}


def download_season(season: int, root: Path, session: requests.Session | None = None, log: Callable[[str], None] = print) -> Path:
    """Fetch and unpack cdnnba_<season>.csv under root/bulk/. Skips work already done."""
    name = f"cdnnba_{season}"
    bulk = root / "bulk"
    csv = bulk / f"{name}.csv"
    if csv.exists() and csv.stat().st_size > 0:
        return csv
    s = session or requests.Session()
    archive = bulk / f"{name}.tar.xz"
    if not archive.exists():
        try:
            listing = s.get(LIST_URL, timeout=30)
        except requests.exceptions.ProxyError as e:
            raise HostBlockedError(f"A proxy refused the connection to {LIST_URL}.") from e
        except requests.exceptions.RequestException as e:
            raise IngestError(f"could not read the nba_data file list: {e}") from e
        if listing.status_code != 200 or f"{name}=" not in listing.text:
            raise IngestError(f"nba_data has no {name} archive (HTTP {listing.status_code}). Seasons start at 2020.")
        url = RAW_URL.format(name=name)
        log(f"bulk: downloading {name} (about 19 MB)")
        try:
            r = s.get(url, timeout=300)
        except requests.exceptions.RequestException as e:
            raise IngestError(f"download of {url} failed: {e}") from e
        if r.status_code != 200:
            raise IngestError(f"{url} answered HTTP {r.status_code}")
        _write_atomic(archive, r.content)
    with tarfile.open(archive) as tar:
        member = next((m for m in tar.getmembers() if m.name == f"{name}.csv" and m.isfile()), None)
        if member is None:
            raise IngestError(f"{archive} does not contain {name}.csv")
        tar.extract(member, bulk, filter="data")
    return csv


def _clean(v, col: str):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if col in INT_COLS:
        return int(v)
    if col in ("scoreHome", "scoreAway"):
        return str(int(v))                               # the feed sends running scores as strings
    if col == "qualifiers":
        return [q.strip() for q in str(v).split(",") if q.strip()]
    if hasattr(v, "item"):
        v = v.item()
    return v


def game_to_raw(game_id: str, g: pd.DataFrame) -> tuple[dict, dict]:
    """One game's rows -> (feed-shaped play-by-play, derived boxscore)."""
    g = g.sort_values(["orderNumber", "actionNumber"])
    cols = [c for c in g.columns if c != "gameId"]
    actions = []
    for row in g[cols].itertuples(index=False, name=None):
        a = {c: _clean(v, c) for c, v in zip(cols, row)}
        actions.append({k: v for k, v in a.items() if v is not None})

    # home/away from which running score a team's points move
    home = away = None
    prev_h = prev_a = 0
    for a in actions:
        h, w, t = int(a.get("scoreHome", prev_h)), int(a.get("scoreAway", prev_a)), a.get("teamId")
        if t and a.get("actionType") in ("2pt", "3pt", "freethrow"):
            if h > prev_h and home is None:
                home = t
            if w > prev_a and away is None:
                away = t
        prev_h, prev_a = h, w
        if home and away:
            break
    teams = [t for t in dict.fromkeys(a.get("teamId") for a in actions) if t]
    if home is None or away is None or home == away or len(teams) != 2:
        raise IngestError(f"game {game_id}: could not tell home from away")

    names: dict[int, dict[int, str]] = {home: {}, away: {}}
    tricode = {}
    for a in actions:
        t, p = a.get("teamId"), a.get("personId")
        if t in names:
            tricode.setdefault(t, a.get("teamTricode", str(t)))
            if p and a.get("actionType") != "ejection":
                names[t].setdefault(p, a.get("playerNameI") or a.get("playerName") or str(p))
    player_team = {p: t for t, ps in names.items() for p in ps}
    for a in actions:                                        # players who only appear as assister / fouled / stealer / blocker
        for key, nm in (("assistPersonId", "assistPlayerNameInitial"), ("foulDrawnPersonId", "foulDrawnPlayerName"),
                        ("stealPersonId", "stealPlayerName"), ("blockPersonId", "blockPlayerName")):
            p = a.get(key)
            if p and p not in player_team and a.get("teamId") in names:
                same_side = key == "assistPersonId"
                t = a["teamId"] if same_side else (away if a["teamId"] == home else home)
                names[t][p] = a.get(nm) or str(p)
                player_team[p] = t

    starters: dict[int, list[int]] = {home: [], away: []}
    entered: set[int] = set()
    for a in actions:
        if a.get("period") != 1:
            break
        typ, sub, p = a.get("actionType"), a.get("subType"), a.get("personId")
        if typ == "substitution":
            if sub == "in":
                entered.add(p)
            elif p and p not in entered and p in player_team and p not in starters[player_team[p]]:
                starters[player_team[p]].append(p)
            continue
        if typ not in LIVE:
            continue
        for key in ("personId", "assistPersonId", "foulDrawnPersonId", "stealPersonId", "blockPersonId"):
            q = a.get(key)
            if q and q not in entered and q in player_team and q not in starters[player_team[q]]:
                starters[player_team[q]].append(q)

    def side(t: int) -> dict:
        five = set(starters[t][:5])
        return {F.BOX_TEAM_ID: t, F.BOX_TEAM_TRICODE: tricode.get(t, str(t)), F.BOX_TEAM_NAME: tricode.get(t, str(t)), F.BOX_TEAM_CITY: "",
                F.BOX_PLAYERS: [{F.BOX_PLAYER_ID: p, F.BOX_PLAYER_NAME: n, F.BOX_PLAYER_STARTER: "1" if p in five else "0"}
                                for p, n in sorted(names[t].items())]}

    tip = next((a["timeActual"] for a in actions if a.get("timeActual")), "")
    pbp = {"meta": {"source": "shufinskiy/nba_data cdnnba archive"}, F.GAME: {F.GAME_ID: game_id, F.ACTIONS: actions}}
    box = {"_derived": True, "meta": {"source": "derived from play-by-play; no boxscore in the bulk archive"},
           F.GAME: {F.GAME_ID: game_id, F.BOX_GAME_TIME_UTC: tip, F.BOX_HOME: side(home), F.BOX_AWAY: side(away)}}
    return pbp, box


def load_seasons(seasons: list[int], root: Path, session: requests.Session | None = None, log: Callable[[str], None] = print) -> dict:
    """Download each season and write raw/pbp and raw/box files. Idempotent per game."""
    out = {"written": 0, "skipped": 0, "failed": []}
    for season in seasons:
        csv = download_season(season, root, session, log)
        df = pd.read_csv(csv, low_memory=False)
        log(f"bulk: {season}: {df['gameId'].nunique():,} games, {len(df):,} actions")
        for gid_int, g in df.groupby("gameId", sort=True):
            gid = f"{int(gid_int):010d}"
            p_pbp, p_box = root / "raw" / "pbp" / f"{gid}.json", root / "raw" / "box" / f"{gid}.json"
            if p_pbp.exists() and p_box.exists():
                out["skipped"] += 1
                continue
            try:
                pbp, box = game_to_raw(gid, g)
            except (IngestError, KeyError, ValueError) as e:
                out["failed"].append({"game_id": gid, "error": str(e)})
                continue
            _write_atomic(p_pbp, json.dumps(pbp, separators=(",", ":")).encode())
            _write_atomic(p_box, json.dumps(box, separators=(",", ":")).encode())
            out["written"] += 1
        del df
    return out
