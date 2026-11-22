"""The hand-check tool: possessions beside their raw actions, plus a summary that must reconcile."""
from __future__ import annotations

import json

import pytest

from tendency.__main__ import main
from tendency.inspect import InspectError, inspect_game, odd_flags, summarize
from tendency.possessions import segment_game

from conftest import AWAY, HOME, Game, make_box

GID = "0022500001"


@pytest.fixture
def root(tmp_path):
    g = Game().start()
    g.shot(700, HOME, 101, made=True).foul(700, AWAY, 201, drawn=101, descriptor="shooting").ft(700, HOME, 101, 1, 1, made=True)   # and-one
    g.shot(680, AWAY, 201, made=False).reb(678, AWAY, 202, offensive=True).shot(676, AWAY, 202, made=True)
    g.timeout(676, HOME).tov(660, HOME, 102)
    g.foul(650, HOME, 103, sub="technical").ft(650, AWAY, 203, 1, 1, made=True, technical=True).shot(640, AWAY, 203, made=True, three=True)
    g.shot(620, HOME, 104, made=False).reb(618, AWAY, 204).end()
    for sub, doc in (("pbp", g.pbp(GID)), ("box", make_box(GID))):
        (tmp_path / "raw" / sub).mkdir(parents=True)
        (tmp_path / "raw" / sub / f"{GID}.json").write_text(json.dumps(doc))
    return tmp_path


def test_report_lists_every_possession_with_its_actions_and_reconciles(root):
    text, s = inspect_game(GID, root)
    assert s["reconciles"] and s["n_possessions"] == 6 and s["and_ones"] == 1 and s["technical_points"] == 1
    assert s["per_team"][HOME]["total"] == 3 and s["per_team"][AWAY]["total"] == 6
    assert s["endings"] == {"shot": 4, "turnover": 1, "period_end": 1}     # the horn cuts off the last possession
    lines = text.splitlines()
    assert sum(1 for ln in lines if ln[:3].strip().isdigit()) == 6              # one header line per possession
    assert "and-one" in text and "after-TO" in text and "points reconcile with the final score: YES" in text
    n_actions = len(json.loads((root / "raw" / "pbp" / f"{GID}.json").read_text())["game"]["actions"])
    shown = sum(1 for ln in lines if ln.startswith("        PT") or ln.startswith("      ~ PT"))
    assert shown == n_actions                                                     # every raw action appears exactly once


def test_only_odd_shows_fewer_possessions_but_the_same_summary(root):
    full, s1 = inspect_game(GID, root)
    odd, s2 = inspect_game(GID, root, only_odd=True)
    assert s1 == s2 and len(odd) < len(full)
    heads = [ln for ln in odd.splitlines() if ln[:3].strip().isdigit()]
    assert len(heads) == 3 and "and-one" in heads[0] and "tech+" in heads[1] and "horn" in heads[2]


def test_odd_flags_and_summary_catch_a_mismatch(root):
    pbp = json.loads((root / "raw" / "pbp" / f"{GID}.json").read_text())
    rows, info = segment_game(pbp, make_box(GID))
    assert odd_flags(rows[0]) == ["and-one"] and odd_flags(rows[1]) == []
    info["final_score"]["home"] += 1
    assert not summarize(rows, info)["reconciles"]


def test_missing_game_and_cli_exit_codes(root, capsys):
    with pytest.raises(InspectError, match="not found"):
        inspect_game("0022599999", root)
    assert main(["inspect", GID, "--root", str(root), "--only", "odd"]) == 0
    assert "SUMMARY" in capsys.readouterr().out
    assert main(["inspect", "0022599999", "--root", str(root)]) == 2
