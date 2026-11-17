"""Fetch raw game JSON from cdn.nba.com and store it exactly as landed.

* Idempotent: a game whose files already exist and parse as JSON is skipped.
* Retries with exponential backoff on timeouts, connection errors, 429 and 5xx.
* A 403 (or a proxy refusing the tunnel) raises HostBlockedError with a plain
  explanation. There is deliberately no fallback host and no header spoofing.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import requests

from . import cdn_fields as F

USER_AGENT = "tendency-board/0.1 (personal research project)"
RETRY_STATUS = {429, 500, 502, 503, 504}


class HostBlockedError(RuntimeError):
    pass


class IngestError(RuntimeError):
    pass


def fetch_json(url: str, session: requests.Session | None = None, retries: int = 4, backoff: float = 1.5,
               timeout: float = 20.0, sleep: Callable[[float], None] = time.sleep) -> tuple[dict, bytes]:
    """GET a URL and return (parsed JSON, raw bytes). Raw bytes are what gets stored."""
    s = session or requests.Session()
    last = ""
    for attempt in range(retries + 1):
        try:
            r = s.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        except requests.exceptions.ProxyError as e:
            raise HostBlockedError(
                f"A proxy refused the connection to {url}. This network does not allow the host; "
                "run ingest from a machine that can reach cdn.nba.com. Not retrying, not working around it."
            ) from e
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if r.status_code == 403:
                raise HostBlockedError(
                    f"{url} answered 403 Forbidden. The host is blocking this network (common for cloud IPs). "
                    "Run ingest from a home connection. Not retrying, not working around it."
                )
            if r.status_code == 404:
                raise IngestError(f"{url} answered 404: no such game, or the feed has not been published yet.")
            if r.status_code in RETRY_STATUS:
                last = f"HTTP {r.status_code}"
            elif r.status_code != 200:
                raise IngestError(f"{url} answered HTTP {r.status_code}")
            else:
                try:
                    return json.loads(r.content), r.content
                except json.JSONDecodeError as e:
                    last = f"invalid JSON ({e})"
        if attempt < retries:
            sleep(backoff * (2 ** attempt))
    raise IngestError(f"gave up on {url} after {retries + 1} attempts; last error: {last}")


def _valid_json_file(path: Path) -> bool:
    try:
        json.loads(path.read_bytes())
        return True
    except (OSError, json.JSONDecodeError):
        return False


def _write_atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(raw)
    tmp.replace(path)


def game_ids_for_range(schedule: dict, start: date, end: date) -> list[str]:
    """Final games whose UTC tip date falls in [start, end]."""
    try:
        dates = schedule[F.SCHED_ROOT][F.SCHED_GAME_DATES]
    except (KeyError, TypeError) as e:
        raise IngestError(f"schedule JSON missing {e}; check SCHED_* names in cdn_fields.py") from e
    ids = []
    for d in dates:
        for g in d.get(F.SCHED_GAMES, []):
            try:
                tip = datetime.fromisoformat(str(g[F.SCHED_GAME_TIME_UTC]).replace("Z", "+00:00")).date()
            except (KeyError, ValueError) as e:
                raise IngestError(f"schedule game without a parseable {F.SCHED_GAME_TIME_UTC}: {e}") from e
            if start <= tip <= end and g.get(F.SCHED_GAME_STATUS) == F.SCHED_STATUS_FINAL:
                ids.append(str(g[F.SCHED_GAME_ID]))
    return sorted(set(ids))


def ingest_games(game_ids: list[str], root: Path, session: requests.Session | None = None, pause: float = 0.6,
                 sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print) -> dict:
    """Download play-by-play and boxscore for each game into root/raw/{pbp,box}/<id>.json."""
    s = session or requests.Session()
    out = {"fetched": 0, "skipped": 0, "failed": []}
    for gid in game_ids:
        targets = [(root / "raw" / "pbp" / f"{gid}.json", F.PBP_URL), (root / "raw" / "box" / f"{gid}.json", F.BOX_URL)]
        if all(_valid_json_file(p) for p, _ in targets):
            out["skipped"] += 1
            continue
        try:
            for path, url in targets:
                if not _valid_json_file(path):
                    _, raw = fetch_json(url.format(game_id=gid), s, sleep=sleep)
                    _write_atomic(path, raw)
                    sleep(pause)
            out["fetched"] += 1
        except HostBlockedError:
            raise                                            # every later game would fail the same way
        except IngestError as e:
            out["failed"].append({"game_id": gid, "error": str(e)})
            log(f"  ! {gid}: {e}")
    return out


def ingest_range(start: date, end: date, root: Path, session: requests.Session | None = None,
                 log: Callable[[str], None] = print) -> dict:
    s = session or requests.Session()
    schedule, raw = fetch_json(F.SCHEDULE_URL, s)
    _write_atomic(root / "raw" / "schedule" / f"schedule_{date.today().isoformat()}.json", raw)
    ids = game_ids_for_range(schedule, start, end)
    if not ids:
        raise IngestError(f"no final games between {start} and {end} in the schedule (it covers the current season only)")
    log(f"ingest: {len(ids)} final games between {start} and {end}")
    return {**ingest_games(ids, root, s, log=log), "game_ids": ids}
