"""Bulk archive path: CSV rows -> feed-shaped game -> same possessions as the original files. No network."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tendency import cdn_fields as F
from tendency.bulk import download_season, game_to_raw, load_seasons
from tendency.ingest import IngestError
from tendency.possessions import segment_game


def _as_archive_rows(pbp: dict) -> pd.DataFrame:
    """Flatten a game the way the nba_data CSV does: one row per action, ints for scores, joined qualifiers."""
    rows = []
    for a in pbp[F.GAME][F.ACTIONS]:
        r = dict(a)
        r["scoreHome"], r["scoreAway"] = int(r["scoreHome"]), int(r["scoreAway"])
        r["qualifiers"] = ", ".join(r.get("qualifiers") or []) or None
        r["playerNameI"] = r.get("playerNameI") or r.get("playerName")
        r["gameId"] = int(pbp[F.GAME][F.GAME_ID])
        rows.append(r)
    return pd.DataFrame(rows)


def test_archive_rows_rebuild_the_same_possessions(synth_run):
    root = synth_run["root"]
    for gid in synth_run["ids"][:6]:
        pbp = json.loads((root / "raw" / "pbp" / f"{gid}.json").read_text())
        box = json.loads((root / "raw" / "box" / f"{gid}.json").read_text())
        want, want_info = segment_game(pbp, box)
        pbp2, box2 = game_to_raw(gid, _as_archive_rows(pbp))
        got, got_info = segment_game(pbp2, box2)
        assert box2["_derived"] and got_info["derived_box"]
        assert (got_info["home_team_id"], got_info["away_team_id"]) == (want_info["home_team_id"], want_info["away_team_id"])
        assert got_info["final_score"] == want_info["final_score"]
        key = lambda rows: [(r["period"], r["off_team_id"], r["ending_type"], r["finisher_id"], r["points"], tuple(r["off_lineup"])) for r in rows]  # noqa: E731
        assert key(got) == key(want)                       # includes starters inferred from play-by-play alone
        assert isinstance(pbp2[F.GAME][F.ACTIONS][0][F.SCORE_HOME], str)


def test_load_seasons_is_idempotent_and_never_touches_the_network_twice(synth_run, tmp_path):
    root = synth_run["root"]
    frames = [_as_archive_rows(json.loads((root / "raw" / "pbp" / f"{g}.json").read_text())) for g in synth_run["ids"][:3]]
    (tmp_path / "bulk").mkdir()
    pd.concat(frames).to_csv(tmp_path / "bulk" / "cdnnba_2090.csv", index=False)

    class NoNetwork:
        def get(self, *a, **k):
            raise AssertionError("network used although the season CSV is already on disk")
    r1 = load_seasons([2090], tmp_path, NoNetwork(), log=lambda _: None, full_names=False)
    r2 = load_seasons([2090], tmp_path, NoNetwork(), log=lambda _: None, full_names=False)
    assert (r1["written"], r1["skipped"], r1["failed"]) == (3, 0, []) and (r2["written"], r2["skipped"]) == (0, 3)
    assert len(list((tmp_path / "raw" / "box").glob("*.json"))) == 3


def test_missing_season_is_a_clear_error(tmp_path):
    class Listing:
        status_code, text = 200, "cdnnba_2024=https://example.test/x.tar.xz\n"

    class S:
        def get(self, *a, **k):
            return Listing()
    with pytest.raises(IngestError, match="no cdnnba_1999 archive"):
        download_season(1999, tmp_path, S(), log=lambda _: None)


def test_bench_technical_does_not_make_a_coach_a_starter():
    """Game 0022500704: a team technical at 12:00 carries a personId with no player name. It is not a player."""
    from conftest import AWAY, HOME, Game
    g = Game().start()
    g.ev("foul", 720, "technical", HOME, 2794, description="TEAM foul technical")
    for i, pid in enumerate([101, 102, 103, 104, 105]):
        g.shot(700 - 20 * i, HOME, pid, made=True, playerName=f"P{pid}")
        g.shot(690 - 20 * i, AWAY, 201 + i, made=True, playerName=f"P{201 + i}")
    g.end()
    _, box = game_to_raw("0022500001", _as_archive_rows(g.pbp()))
    home = box[F.GAME][F.BOX_HOME]
    assert sorted(p[F.BOX_PLAYER_ID] for p in home[F.BOX_PLAYERS] if p[F.BOX_PLAYER_STARTER] == "1") == [101, 102, 103, 104, 105]
    assert 2794 not in [p[F.BOX_PLAYER_ID] for p in home[F.BOX_PLAYERS]]


# ---- step 1.3: real team names and full player names ----
REAL_HOME, REAL_AWAY = 1610612741, 1610612752            # ids as they appear in the feed


def _real_game(gid="0022500001", home_tri="CHI", away_tri="NYK", traded=False):
    """A tiny game between two real team ids. Player 9001 plays for the away side when `traded`."""
    from conftest import clock
    acts, score = [], {REAL_HOME: 0, REAL_AWAY: 0}

    def ev(t, clk, sub, team=None, pid=None, name=None, pts=0, **kw):
        if pts:
            score[team] += pts
        a = {"actionNumber": len(acts) + 1, "orderNumber": (len(acts) + 1) * 10000, "period": 1, "clock": clock(clk), "actionType": t,
             "subType": sub, "scoreHome": str(score[REAL_HOME]), "scoreAway": str(score[REAL_AWAY]), **kw}
        if team:
            a.update(teamId=team, teamTricode=home_tri if team == REAL_HOME else away_tri)
        if pid:
            a.update(personId=pid, playerName=name.split(". ")[1], playerNameI=name)
        acts.append(a)
    ev("period", 720, "start")
    home_ids = [9002, 9003, 9004, 9005, 9006] if traded else [9001, 9002, 9003, 9004, 9005]
    away_ids = [9001, 9012, 9013, 9014, 9015] if traded else [9011, 9012, 9013, 9014, 9015]
    for i, (h, w) in enumerate(zip(home_ids, away_ids)):
        ev("2pt", 700 - 30 * i, "Layup", REAL_HOME, h, f"A. Home{h}", pts=2, shotResult="Made")
        ev("2pt", 690 - 30 * i, "Layup", REAL_AWAY, w, f"B. Away{w}", pts=2, shotResult="Made")
    ev("period", 0, "end")
    return {F.GAME: {F.GAME_ID: gid, F.ACTIONS: acts}}


def test_team_table_matches_the_shape_the_feed_uses():
    from tendency.teams import TEAMS, city_and_nickname
    assert len(TEAMS) == 30 and len({t[0] for t in TEAMS.values()}) == 30
    assert all(1610612737 <= k <= 1610612766 for k in TEAMS)
    assert city_and_nickname(1610612741, "CHI") == ("Chicago", "Bulls")
    assert city_and_nickname(42, "XYZ") == ("", "XYZ")              # unknown id: fall back to the tricode, never guess


def test_derived_boxscore_has_city_nickname_and_full_names():
    full = {9001: "Alexander Homeowner", 9011: "Bartholomew Awayday"}
    _, box = game_to_raw("0022500001", _as_archive_rows(_real_game()), full)
    home, away = box[F.GAME][F.BOX_HOME], box[F.GAME][F.BOX_AWAY]
    assert (home[F.BOX_TEAM_CITY], home[F.BOX_TEAM_NAME], home[F.BOX_TEAM_TRICODE]) == ("Chicago", "Bulls", "CHI")
    assert (away[F.BOX_TEAM_CITY], away[F.BOX_TEAM_NAME]) == ("New York", "Knicks")
    names = {p[F.BOX_PLAYER_ID]: p[F.BOX_PLAYER_NAME] for side in (home, away) for p in side[F.BOX_PLAYERS]}
    assert names[9001] == "Alexander Homeowner" and names[9011] == "Bartholomew Awayday"
    assert names[9002] == "A. Home9002"                              # not in the map: the feed's form stays
    _, info = segment_game(_real_game(), box)
    assert info["teams"][REAL_HOME]["name"] == "Chicago Bulls"       # what the app's team report header shows


def test_tricode_comes_from_the_game_and_names_follow_the_player_id():
    """A rebrand shows the tricode of the day; a traded player keeps one name on either team."""
    full = {9001: "Alexander Homeowner"}
    _, box1 = game_to_raw("0022500001", _as_archive_rows(_real_game()), full)
    _, box2 = game_to_raw("0022500002", _as_archive_rows(_real_game("0022500002", home_tri="CHC", traded=True)), full)
    assert box2[F.GAME][F.BOX_HOME][F.BOX_TEAM_TRICODE] == "CHC" and box2[F.GAME][F.BOX_HOME][F.BOX_TEAM_NAME] == "Bulls"
    on = lambda box, side: {p[F.BOX_PLAYER_ID]: p[F.BOX_PLAYER_NAME] for p in box[F.GAME][side][F.BOX_PLAYERS]}  # noqa: E731
    assert on(box1, F.BOX_HOME)[9001] == on(box2, F.BOX_AWAY)[9001] == "Alexander Homeowner"


def _matchups_csv(path, rows):
    pd.DataFrame(rows, columns=["person_id", "first_name", "family_name", "matchups_person_id", "matchups_first_name",
                                "matchups_family_name"]).to_csv(path, index=False)


def test_full_names_are_optional_cached_and_refresh_old_boxscores(tmp_path):
    import requests
    from tendency.bulk import BOX_VERSION, player_names
    (tmp_path / "bulk").mkdir()
    _as_archive_rows(_real_game()).to_csv(tmp_path / "bulk" / "cdnnba_2025.csv", index=False)

    class Down:
        def get(self, *a, **k):
            raise requests.exceptions.ConnectionError("offline")
    logs = []
    r1 = load_seasons([2025], tmp_path, Down(), log=logs.append)              # matchups archive unreachable
    assert (r1["written"], r1["named_players"]) == (1, 0) and any("no full player names for 2025" in m for m in logs)
    box = json.loads((tmp_path / "raw" / "box" / "0022500001.json").read_text())
    assert box["_derived_version"] == BOX_VERSION and box[F.GAME][F.BOX_HOME][F.BOX_PLAYERS][0][F.BOX_PLAYER_NAME] == "A. Home9001"
    pbp_mtime = (tmp_path / "raw" / "pbp" / "0022500001.json").stat().st_mtime_ns

    _matchups_csv(tmp_path / "bulk" / "matchups_2025.csv", [[9001, "Alexander", "Homeowner", 9011, "Bartholomew", "Awayday"]])
    r2 = load_seasons([2025], tmp_path, Down(), log=logs.append)              # now on disk: boxscore rewritten, pbp untouched
    assert (r2["written"], r2["refreshed"], r2["skipped"], r2["named_players"]) == (0, 1, 0, 2)
    box = json.loads((tmp_path / "raw" / "box" / "0022500001.json").read_text())
    assert box[F.GAME][F.BOX_HOME][F.BOX_PLAYERS][0][F.BOX_PLAYER_NAME] == "Alexander Homeowner"
    assert (tmp_path / "raw" / "pbp" / "0022500001.json").stat().st_mtime_ns == pbp_mtime
    r3 = load_seasons([2025], tmp_path, Down(), log=logs.append)
    assert (r3["refreshed"], r3["skipped"]) == (0, 1)
    assert player_names([2025], tmp_path, Down(), log=logs.append) == {9001: "Alexander Homeowner", 9011: "Bartholomew Awayday"}   # from the cache


def test_a_truncated_archive_is_removed_with_a_clear_error(tmp_path):
    from tendency.bulk import download_archive
    (tmp_path / "bulk").mkdir()
    (tmp_path / "bulk" / "matchups_2025.tar.xz").write_bytes(b"not an archive")
    with pytest.raises(IngestError, match="not a readable archive"):
        download_archive("matchups_2025", tmp_path, object(), log=lambda _: None)
    assert not (tmp_path / "bulk" / "matchups_2025.tar.xz").exists()
