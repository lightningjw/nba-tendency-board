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
        r["playerNameI"] = r.get("playerName")
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
    r1 = load_seasons([2090], tmp_path, NoNetwork(), log=lambda _: None)
    r2 = load_seasons([2090], tmp_path, NoNetwork(), log=lambda _: None)
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
