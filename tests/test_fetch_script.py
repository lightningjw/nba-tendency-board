"""scripts/fetch_nba_api.py against a fake client. The real one is never called from tests or from the cloud."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("fetch_nba_api", Path(__file__).resolve().parents[1] / "scripts" / "fetch_nba_api.py")
fx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fx)


def _game(gid, utc, status, home=1610612741, away=1610612752):
    return {"gameId": gid, "gameDateTimeUTC": utc, "gameStatus": status, "homeTeam": {"teamId": home}, "awayTeam": {"teamId": away}}


SCHEDULE = {"leagueSchedule": {"gameDates": [
    {"games": [_game("0012600001", "2026-10-05T23:00:00Z", 3)]},                      # preseason: ignored
    {"games": [_game("0022600201", "2026-11-20T00:30:00Z", 3), _game("0022600202", "2026-11-20T03:00:00Z", 3)]},   # Nov 19 Eastern
    {"games": [_game("0022600210", "2026-11-21T00:00:00Z", 3)]},                      # Nov 20 Eastern
    {"games": [_game("0022600220", "2026-11-22T01:00:00Z", 1), _game("0022600221", "2026-11-22T03:30:00Z", 1)]},   # Nov 21 Eastern, not played
]}}


class Fake:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def schedule(self, season):
        self.calls.append(("schedule", season))
        return SCHEDULE

    def pbp(self, gid):
        self.calls.append(("pbp", gid))
        if gid in self.fail:
            raise ConnectionError("refused")
        return {"game": {"gameId": gid, "actions": [{"actionNumber": 1}]}}

    def box(self, gid):
        self.calls.append(("box", gid))
        return {"game": {"gameId": gid, "homeTeam": {}, "awayTeam": {}}}


def run(api, root, **kw):
    return fx.fetch(api, root, "2026-27", kw.pop("through", "2026-11-20"), kw.pop("today", "2026-11-21"), log=lambda _: None, sleep=lambda _: None, **kw)


def test_schedule_dates_are_us_eastern_and_preseason_is_dropped():
    games = fx.parse_schedule(SCHEDULE)
    assert [g["game_id"] for g in games] == ["0022600201", "0022600202", "0022600210", "0022600220", "0022600221"]
    assert games[1]["date"] == "2026-11-19" and games[1]["tipoff_utc"] == "2026-11-20T03:00:00Z"      # 03:00 UTC is still the night before
    with pytest.raises(fx.FetchError, match="leagueSchedule.gameDates"):
        fx.parse_schedule({"unexpected": 1})
    with pytest.raises(fx.FetchError, match="missing a field"):
        fx.parse_schedule({"leagueSchedule": {"gameDates": [{"games": [{"gameId": "0022600001"}]}]}})


def test_fetch_saves_final_games_skips_what_is_on_disk_and_writes_the_schedule(tmp_path):
    api = Fake()
    r1 = run(api, tmp_path)
    assert r1["downloaded"] == ["0022600201", "0022600202", "0022600210"] and not r1["failed"]
    assert json.loads((tmp_path / "raw" / "pbp" / "0022600201.json").read_text())["game"]["actions"]
    sched = json.loads((tmp_path / "schedule" / "2026-11-21.json").read_text())
    assert [g["game_id"] for g in sched["games"]] == ["0022600220", "0022600221"] and sched["games"][0]["home_team_id"] == 1610612741
    assert not (tmp_path / "raw" / "pbp" / "0022600220.json").exists()                 # not final: never downloaded
    api2 = Fake()
    r2 = run(api2, tmp_path)
    assert r2["downloaded"] == [] and [c[0] for c in api2.calls] == ["schedule"]       # idempotent: one request, the schedule


def test_fetch_respects_through_and_max_games_and_isolates_failures(tmp_path):
    r = run(Fake(), tmp_path, through="2026-11-19", max_games=1)
    assert r["downloaded"] == ["0022600201"] and r["left_for_next_run"] == 1
    r = run(Fake(fail={"0022600202"}), tmp_path)
    assert r["downloaded"] == ["0022600210"] and [f["game_id"] for f in r["failed"]] == ["0022600202"]
    assert not (tmp_path / "raw" / "box" / "0022600202.json").exists()                 # nothing partial on disk
    with pytest.raises(fx.FetchError, match="first three games all failed"):
        run(Fake(fail={"0022600201", "0022600202", "0022600210"}), tmp_path / "fresh")


def test_schedule_file_is_what_predict_reads(tmp_path):
    from tendency.prospective import ProspectiveError, _schedule
    import pandas as pd
    run(Fake(), tmp_path)
    poss = pd.DataFrame({"game_id": ["0022600201"], "game_date": ["2026-11-19"]})
    rows = _schedule(tmp_path, "2026-11-21", poss, {"0022600201": {}}, replay=False)
    assert [(g["game_id"], g["tipoff_utc"]) for g in rows] == [("0022600220", "2026-11-22T01:00:00Z"), ("0022600221", "2026-11-22T03:30:00Z")]
    with pytest.raises(ProspectiveError, match="no schedule"):
        _schedule(tmp_path, "2026-11-25", poss, {}, replay=False)
