"""Ingest: idempotent, retries with backoff, and a clear stop when the host blocks us. No network used."""
from __future__ import annotations

import json
from datetime import date

import pytest
import requests

from tendency import cdn_fields as F
from tendency.ingest import HostBlockedError, IngestError, fetch_json, game_ids_for_range, ingest_games


class Resp:
    def __init__(self, status, body=b"{}"):
        self.status_code, self.content = status, body


class FakeSession:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def get(self, url, **kw):
        self.calls.append(url)
        item = self.script.pop(0) if self.script else Resp(200, b'{"ok": true}')
        if isinstance(item, Exception):
            raise item
        return item


def test_retries_with_exponential_backoff_then_succeeds():
    s = FakeSession([Resp(503), requests.exceptions.ConnectTimeout("slow"), Resp(429), Resp(200, b'{"game": 1}')])
    sleeps = []
    obj, raw = fetch_json("https://example.test/x.json", s, backoff=1.0, sleep=sleeps.append)
    assert obj == {"game": 1} and raw == b'{"game": 1}' and len(s.calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_gives_up_after_the_retry_budget():
    s = FakeSession([Resp(500)] * 10)
    with pytest.raises(IngestError, match="gave up .* after 3 attempts"):
        fetch_json("https://example.test/x.json", s, retries=2, sleep=lambda _: None)
    assert len(s.calls) == 3


def test_blocked_host_stops_immediately_with_a_plain_message():
    s = FakeSession([Resp(403)])
    with pytest.raises(HostBlockedError, match="403 Forbidden.*home connection"):
        fetch_json("https://cdn.nba.com/x.json", s, sleep=lambda _: None)
    assert len(s.calls) == 1                           # no retry, no alternative host
    s = FakeSession([requests.exceptions.ProxyError("tunnel 403")])
    with pytest.raises(HostBlockedError, match="proxy refused"):
        fetch_json("https://cdn.nba.com/x.json", s, sleep=lambda _: None)


def test_ingest_is_idempotent_and_stores_bytes_as_landed(tmp_path):
    body = b'{"game": {"gameId": "0022500001"},   "spacing": "kept"}'
    s = FakeSession([Resp(200, body), Resp(200, body)])
    r1 = ingest_games(["0022500001"], tmp_path, s, sleep=lambda _: None)
    assert r1 == {"fetched": 1, "skipped": 0, "failed": []} and len(s.calls) == 2
    assert (tmp_path / "raw" / "pbp" / "0022500001.json").read_bytes() == body
    r2 = ingest_games(["0022500001"], tmp_path, s, sleep=lambda _: None)
    assert r2["skipped"] == 1 and len(s.calls) == 2    # nothing fetched again


def test_a_corrupt_file_is_refetched_and_one_bad_game_does_not_stop_the_rest(tmp_path):
    (tmp_path / "raw" / "pbp").mkdir(parents=True)
    (tmp_path / "raw" / "pbp" / "A.json").write_text("{truncated")
    s = FakeSession([Resp(200), Resp(200), Resp(404), Resp(200), Resp(200)])
    r = ingest_games(["A", "B", "C"], tmp_path, s, sleep=lambda _: None, log=lambda _: None)
    assert r["fetched"] == 2 and [f["game_id"] for f in r["failed"]] == ["B"]
    assert json.loads((tmp_path / "raw" / "pbp" / "A.json").read_text()) == {}


def test_blocked_host_aborts_the_whole_ingest(tmp_path):
    with pytest.raises(HostBlockedError):
        ingest_games(["A", "B"], tmp_path, FakeSession([Resp(403)]), sleep=lambda _: None)


def test_schedule_filter_keeps_final_games_in_range():
    def g(gid, when, status):
        return {F.SCHED_GAME_ID: gid, F.SCHED_GAME_TIME_UTC: when, F.SCHED_GAME_STATUS: status}
    sched = {F.SCHED_ROOT: {F.SCHED_GAME_DATES: [{F.SCHED_GAMES: [
        g("0022500001", "2025-10-21T23:30:00Z", 3), g("0022500002", "2025-10-22T02:00:00Z", 3),
        g("0022500003", "2025-10-25T00:00:00Z", 3), g("0022500004", "2025-10-22T00:00:00Z", 1)]}]}}
    assert game_ids_for_range(sched, date(2025, 10, 21), date(2025, 10, 22)) == ["0022500001", "0022500002"]
    with pytest.raises(IngestError, match="schedule JSON missing"):
        game_ids_for_range({"nope": 1}, date(2025, 10, 21), date(2025, 10, 22))


def test_cli_reports_a_block_with_its_own_exit_code(monkeypatch, tmp_path, capsys):
    from tendency import __main__ as cli
    from tendency import ingest

    def blocked(*a, **k):
        raise HostBlockedError("cdn.nba.com answered 403 Forbidden")
    monkeypatch.setattr(ingest, "ingest_range", blocked)
    code = cli.main(["run", "--start", "2025-10-21", "--end", "2025-10-22", "--root", str(tmp_path), "--out", str(tmp_path / "o")])
    assert code == 3 and "BLOCKED" in capsys.readouterr().err
