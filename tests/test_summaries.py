"""Step 3.2: the written summary above a team's tendency table.

The rule under test: the writer sees one team's table and nothing else, and a summary is shown only
if every sentence is backed by the rows it cites. None of these tests touches the network. The model
is a fake that returns what the test tells it to; one test drives the real SDK against a local transport.
"""
from __future__ import annotations

import copy
import json
import shutil
from types import SimpleNamespace

import pytest

from tendency import summaries as S
from tendency.contract import ContractError, validate, validate_export_dir

STAR = "Nickeil Alexander-Walker"


def row(i, scope="team", situation="After a timeout", label="Rim share of shots", head="shot_family", raw=0.3412, base=0.2890,
        dev=0.0410, lo=0.0120, hi=0.0730, n=140, kind="team_elsewhere_plus_league_shift", pid=None):
    return {"id": f"r{i}", "scope": scope, "player_id": pid, "situation": {"id": f"s{i}", "label": situation},
            "outcome": {"head": head, "class": None, "label": label}, "baseline_kind": kind, "baseline_rate": base, "raw_rate": raw,
            "shrunk_rate": base + dev, "deviation": dev, "interval": [lo, hi], "n": n, "k": round(raw * n), "prior_strength": 50.0,
            "rank_score": 0.9, "prob_beyond": 0.9, "excludes_zero": lo > 0 or hi < 0}


def report():
    """Four rows: a clear rise, a clear fall, a player row whose interval includes zero, and a row with numbers in its labels."""
    return {"team_id": 7, "interval_level": 0.9, "tendencies": [
        row(1),
        row(2, situation="Off a defensive rebound", label="Late share of possessions (first action after 16 s or more)", head="tempo",
            raw=0.0177, base=0.0483, dev=-0.0284, lo=-0.0344, hi=-0.0199, n=1073),
        row(3, scope="player", situation="Last 5:00 of Q4/OT, within 5", label=f"{STAR}: share of finishes while on the floor", head="finisher",
            raw=0.3286, base=0.2399, dev=0.0400, lo=-0.0178, hi=0.1062, n=70, kind="player_lineup_adjusted", pid=1629638),
        row(4, situation="Two-for-one window (30 to 45 s left in the period)", label="Above-the-break-three share of shots", head="shot_family",
            raw=0.2000, base=0.3100, dev=-0.0550, lo=-0.0990, hi=-0.0120, n=55)]}


GOOD = [
    {"rows": [1], "text": "After a timeout, the team's rim share of shots is 4.1 points above baseline (1.2 to 7.3 above) on 140 shots."},
    {"rows": [2], "text": "Off a defensive rebound, the team's late share of possessions is 2.8 points below baseline (2.0 to 3.4 below) on 1,073 possessions."},
    {"rows": [3], "text": f"In the last 5:00 of Q4/OT within 5, {STAR}'s share of finishes is 4.0 points above baseline on 70 possessions on the floor, "
                          "but the interval runs from 1.8 points below to 10.6 above and includes zero."},
]


def table():
    return S.summary_table(report())


def changed(i, old, new):
    s = copy.deepcopy(GOOD)
    assert old in s[i]["text"]
    s[i]["text"] = s[i]["text"].replace(old, new)
    return s


def why(sentences):
    return " | ".join(S.check(sentences, table()))


# ---------- the table -----------------------------------------------------------------------------

def test_the_table_is_the_reports_rows_as_the_app_prints_them():
    t = table()
    assert t["interval_level"] == "90%" and [r["row"] for r in t["rows"]] == [1, 2, 3, 4]
    a, b, c, _ = t["rows"]
    assert (a["about"], a["observed"], a["baseline"], a["sample"]) == ("the team", "34.1%", "28.9%", "140 shots")
    assert a["difference"] == "4.1 points above baseline" and a["interval"] == "1.2 to 7.3 points above baseline" and not a["interval_includes_zero"]
    assert b["difference"] == "2.8 points below baseline" and b["interval"] == "2.0 to 3.4 points below baseline" and b["sample"] == "1,073 possessions"
    assert (c["about"], c["outcome"], c["sample"]) == (STAR, "share of finishes while on the floor", "70 possessions on the floor")
    assert c["interval"] == "1.8 points below to 10.6 points above baseline" and c["interval_includes_zero"]
    assert set(json.dumps(t)) & set("0123456789") and "team_id" not in json.dumps(t) and "prior_strength" not in json.dumps(t)


def test_a_tie_rounds_the_way_the_apps_tofixed_does():
    assert S._fixed1(2.25) == "2.3" and S._fixed1(0.0225 * 100) == "2.3" and S._fixed1(-0.34) == "0.3"     # format(2.25, ".1f") says 2.2


def test_the_hash_moves_with_any_cell_and_with_nothing_else():
    r = report()
    h = S.table_hash(S.summary_table(r))
    assert h.startswith("sha256:") and h == S.table_hash(S.summary_table(copy.deepcopy(r)))
    r2 = copy.deepcopy(r)
    r2["tendencies"][1]["n"] += 1
    r3 = copy.deepcopy(r)
    r3["tendencies"][0]["prior_strength"] = 1.0                 # not printed, so not part of what the summary was written from
    assert S.table_hash(S.summary_table(r2)) != h and S.table_hash(S.summary_table(r3)) == h


def test_the_model_is_sent_no_name():
    masked, mapping = S.mask(table())
    body = json.dumps(S.request(masked))
    assert mapping == {"PLAYER_A": STAR} and masked["rows"][2]["about"] == "PLAYER_A"
    for word in ("Nickeil", "Alexander", "Walker", "1629638", "team_id"):
        assert word not in body
    assert S.unmask("PLAYER_A's share. PLAYER_A again, not PLAYER_AB.", mapping) == f"{STAR}'s share. {STAR} again, not PLAYER_AB."
    assert [S._placeholder(i) for i in (0, 25, 26)] == ["PLAYER_A", "PLAYER_Z", "PLAYER_AA"]


# ---------- the check -----------------------------------------------------------------------------

def test_a_summary_written_from_its_rows_passes():
    assert S.check(GOOD, table()) == []
    both = [{"rows": [1, 2], "text": "After a timeout the team's rim share of shots is 4.1 points above baseline on 140 shots, and off a defensive rebound "
                                     "its late share of possessions is 2.8 points below baseline on 1,073 possessions."}]
    assert S.check(both, table()) == []


@pytest.mark.parametrize("old,new,says", [
    ("4.1 points", "4 points", "4 is not a number in row 1"),                       # rounded
    ("4.1 points", "5.2 points", "5.2 is not a number"),                            # observed minus baseline: arithmetic the table does not do
    ("on 140 shots", "on 1,073 shots", "1073 is not a number in row 1"),            # a real number, from another row
    ("on 140 shots", "on 140 shots in 28 games", "28 is not a number"),
    ("4.1 points above", "about four points above", "'four' is a number written as a word"),
    ("is 4.1 points", "is roughly double the league's and 4.1 points", "'double'"),
    ("After a timeout,", "After a timeout in the second half,", "'second'"),
])
def test_a_number_that_is_not_in_the_row_is_caught(old, new, says):
    assert says in why(changed(0, old, new))


@pytest.mark.parametrize("i,old,new,says", [
    (0, "the team's rim share", "the Celtics' rim share", "'Celtics' is capitalised and is not a name or label in row 1"),
    (0, "After a timeout, the team's", "Under Mazzulla, after a timeout the team's", "'Mazzulla'"),
    (0, "After a timeout, the team's", "Tatum drives it: after a timeout, the team's", "'Tatum' is capitalised"),          # even first in the sentence
    (1, "the team's late share", f"{STAR}'s late share", "'Nickeil' is capitalised and is not a name or label in row 2"),   # a real name, wrong row
    (2, f"{STAR}'s", "PLAYER_B's", "PLAYER_B"),                                                                             # a placeholder left behind
])
def test_a_name_that_is_not_in_the_row_is_caught(i, old, new, says):
    assert says in why(changed(i, old, new))


@pytest.mark.parametrize("i,old,new,says", [
    (0, "above baseline (1.2 to 7.3 above)", "below baseline (1.2 to 7.3 below)", "say row 1's direction with the word 'above'"),
    (0, "is 4.1 points above", "is lower than usual, 4.1 points above", "'lower' points the other way"),
    (1, "late share of possessions is 2.8", "late share of possessions (first action after 16 s or more) comes more often, 2.8", "'more' points the other way"),
    (0, " (1.2 to 7.3 above) on 140 shots", "", "row 1's difference needs its interval"),
    (0, "is 4.1 points above baseline (1.2 to 7.3 above) on 140 shots", "is well above baseline", "give row 1's difference"),
    (2, " and includes zero", "", "row 3's interval includes zero"),
    (0, "the team's rim share", "the team's **rim** share", "plain prose only"),
    (0, "on 140 shots.", "on 140 shots", "ends with a full stop"),
])
def test_a_sentence_that_misstates_its_row_is_caught(i, old, new, says):
    assert says in why(changed(i, old, new))


def test_each_rows_direction_has_to_follow_its_own_figure():
    """Two rows that point opposite ways in one sentence: both words are there either way, so the order is what is checked."""
    pace = "After a timeout the team's rim share of shots is 4.1 points {} baseline on 140 shots, and off a defensive rebound its late share of possessions is 2.8 points {} baseline on 1,073 possessions."
    assert S.check([{"rows": [1, 2], "text": pace.format("above", "below")}], table()) == []
    assert "after row 1's difference, 4.1, the next direction word has to be 'above'" in why([{"rows": [1, 2], "text": pace.format("below", "above")}])
    wide = changed(2, "is 4.0 points above baseline", "is 4.0 points below baseline")        # an interval that includes zero still has a side
    assert "after row 3's difference, 4.0, the next direction word has to be 'above'" in why(wide)


def test_a_label_may_carry_a_word_that_would_otherwise_point_the_wrong_way():
    late = [{"rows": [2], "text": "Off a defensive rebound, late possessions (first action after 16 s or more) are 2.8 points below baseline on 1,073 possessions."}]
    arc = [{"rows": [4], "text": "In the two-for-one window (30 to 45 s left in the period), the team's above-the-break-three share of shots is "
                                 "5.5 points below baseline (1.2 to 9.9 below) on 55 shots."}]
    assert S.check(late, table()) == [] and S.check(arc, table()) == []
    arc[0]["text"] = arc[0]["text"].replace("is 5.5", "comes more often, 5.5")           # row 4's labels have no "more"; nothing excuses it there
    assert "'more' points the other way" in why(arc)


def test_the_shape_of_a_summary_is_checked():
    assert S.check([], table()) and S.check(GOOD * 2, table())                                # none, and six
    assert "cites one or two different rows" in why([{**GOOD[0], "rows": [9]}])
    assert "cites one or two different rows" in why([{**GOOD[0], "rows": [1, 2, 3]}])
    assert "already used" in why([GOOD[0], GOOD[0]])
    assert "needs a text" in why([{"rows": [1]}])


# ---------- the template --------------------------------------------------------------------------

def test_the_template_fills_in_the_first_three_rows_and_passes_its_own_check():
    s = S.template_summary(table())
    assert s["generated_by"] == {"kind": "template", "model": None, "prompt_sha": None} and (s["attempts"], s["rejected"]) == (0, 0)
    assert [x["rows"] for x in s["sentences"]] == [[1], [2], [3]] and s["table_hash"] == S.table_hash(table())
    assert s["sentences"][0]["text"] == "After a timeout: rim share of shots, 4.1 points above baseline (90% interval 1.2 to 7.3 points above) on 140 shots."
    assert f"{STAR}'s share of finishes" in s["sentences"][2]["text"] and "which includes zero" in s["sentences"][2]["text"]
    assert S.template_summary({"interval_level": "90%", "rows": []}) is None


def test_every_team_in_the_sample_gets_a_summary_that_matches_its_table(synth_run):
    man = json.loads((synth_run["out"] / "manifest.json").read_text())
    assert man["contract_version"] == "1.7" and "LLM-written summaries" not in man["not_built"]
    reports = [json.loads(p.read_text()) for p in sorted((synth_run["out"] / "team_reports").glob("*.json"))]
    assert all(r["summary"]["generated_by"]["kind"] == "template" and S.verify(r) == [] for r in reports)
    b = man["summaries"]
    assert (b["n_reports"], b["n_llm"], b["n_template"], b["n_asked"], b["models"], b["prompt_sha"]) == (len(reports), 0, len(reports), 0, [], None)


# ---------- the model, faked ------------------------------------------------------------------------

def reply(sentences, stop="end_turn", model="claude-opus-5-5"):
    text = sentences if isinstance(sentences, str) else json.dumps({"sentences": sentences})
    return SimpleNamespace(stop_reason=stop, model=model, content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)])


class Fake:
    """Stands in for anthropic.Anthropic(): records each request and returns the next scripted reply."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r


def masked(sentences):
    return [{**s, "text": s["text"].replace(STAR, "PLAYER_A")} for s in sentences]


BAD = masked(changed(0, "4.1 points", "about 4 points"))


def test_a_draft_that_passes_is_kept_and_the_request_was_the_table_and_nothing_else():
    pytest.importorskip("anthropic")
    fake = Fake(reply(masked(GOOD)))
    s = S.llm_summary(table(), fake)
    assert s["generated_by"] == {"kind": "llm", "model": "claude-opus-5-5", "prompt_sha": S.prompt_sha()} and (s["attempts"], s["rejected"]) == (1, 0)
    assert s["sentences"] == GOOD and s["table_hash"] == S.table_hash(table())               # the name is back in the text that is shown
    (call,) = fake.calls
    assert set(call) == {"model", "max_tokens", "system", "messages", "output_config", "betas", "fallbacks"}        # no tools, nothing to look anything up with
    assert call["system"] == S.PROMPT_PATH.read_text() and call["model"] == S.MODEL
    assert call["messages"] == [{"role": "user", "content": json.dumps(S.mask(table())[0], indent=1, ensure_ascii=False)}]
    assert call["output_config"]["format"]["schema"] == S.OUT_SCHEMA and "Nickeil" not in json.dumps(call)


def test_a_discarded_draft_gets_one_more_try_that_is_told_why_and_still_no_name():
    pytest.importorskip("anthropic")
    logged = []
    fake = Fake(reply(BAD), reply(masked(GOOD)))
    s = S.llm_summary(table(), fake, log=logged.append)
    assert s["generated_by"]["kind"] == "llm" and (s["attempts"], s["rejected"]) == (2, 1) and s["sentences"] == GOOD
    second = fake.calls[1]["messages"]
    assert len(second) == 1 and second[0]["content"].startswith(fake.calls[0]["messages"][0]["content"])
    assert "4 is not a number in row 1" in second[0]["content"] and "about 4 points" in second[0]["content"]
    assert "Nickeil" not in json.dumps(fake.calls) and "4 is not a number in row 1" in logged[0]


@pytest.mark.parametrize("first", [reply(BAD), reply([], stop="refusal"), reply("{", stop="max_tokens"), reply("not json at all")])
def test_two_discarded_drafts_fall_back_to_the_template(first):
    pytest.importorskip("anthropic")
    fake = Fake(first)
    s = S.llm_summary(table(), fake)
    assert len(fake.calls) == S.MAX_ATTEMPTS == 2
    assert s["generated_by"]["kind"] == "template" and (s["attempts"], s["rejected"]) == (2, 2)
    assert s["sentences"] == S.template_summary(table())["sentences"] and S.check(s["sentences"], table()) == []


def test_the_model_that_answered_is_the_one_recorded():
    pytest.importorskip("anthropic")
    s = S.llm_summary(table(), Fake(reply(masked(GOOD), model="claude-opus-5")))             # a server-side fallback answered
    assert s["generated_by"]["model"] == "claude-opus-5"


def test_an_api_error_is_not_a_rejection_and_does_not_sink_an_export():
    anthropic = pytest.importorskip("anthropic")
    import httpx2
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    with pytest.raises(S.SummaryError, match="could not reach the API"):
        S.llm_summary(table(), Fake(err))
    fake, logged = Fake(err), []
    w = S.Writer("llm", cl=fake, log=logged.append)
    a, b = w.write(report()), w.write({**report(), "team_id": 8})
    assert len(fake.calls) == 1 and w.stopped and "summarize" in logged[0]                    # it stops asking after the first failure
    assert a["generated_by"]["kind"] == b["generated_by"]["kind"] == "template" and (a["attempts"], a["rejected"]) == (0, 0)


def test_no_key_is_an_error_before_anything_is_built(monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(S.SummaryError, match="ANTHROPIC_API_KEY"):
        S.client()
    from tendency.__main__ import main
    assert main(["build", "--summaries", "llm", "--root", "/nonexistent"]) == 2


def test_the_real_sdk_sends_the_same_request_and_reads_the_reply():
    """The SDK itself, pointed at a local transport: what leaves the process, header and body."""
    anthropic = pytest.importorskip("anthropic")
    import httpx2
    seen = []

    def handler(req):
        seen.append(req)
        return httpx2.Response(200, json={"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "stop_reason": "end_turn",
                                          "stop_sequence": None, "usage": {"input_tokens": 900, "output_tokens": 200},
                                          "content": [{"type": "thinking", "thinking": "", "signature": "x"},
                                                      {"type": "text", "text": json.dumps({"sentences": masked(GOOD)})}]})

    cl = anthropic.Anthropic(api_key="test-key", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    s = S.llm_summary(table(), cl)
    assert s["generated_by"]["kind"] == "llm" and s["sentences"] == GOOD
    (req,) = seen
    body = json.loads(req.content)
    assert req.url.path == "/v1/messages" and S.FALLBACK_BETA in req.headers["anthropic-beta"]
    assert set(body) == {"model", "max_tokens", "system", "messages", "output_config", "fallbacks"} and body["fallbacks"] == "default"
    assert body["system"] == S.PROMPT_PATH.read_text() and len(body["messages"]) == 1 and json.loads(body["messages"][0]["content"]) == S.mask(table())[0]
    assert "Nickeil" not in req.content.decode()


# ---------- the export, the validator, the command --------------------------------------------------

def test_a_summary_already_paid_for_is_kept_only_while_its_table_stands():
    r = report()
    old = {**S.template_summary(table()), "generated_by": {"kind": "llm", "model": "claude-opus-5-5", "prompt_sha": "abc"}, "sentences": GOOD, "attempts": 1}
    assert S.Writer("keep", {7: old}).write(r) is old                                         # no client, no network
    moved = copy.deepcopy(r)
    moved["tendencies"][0]["n"] = 141
    assert S.Writer("keep", {7: old}).write(moved)["generated_by"]["kind"] == "template"      # the table changed under it
    stale = {**old, "sentences": changed(0, "4.1 points", "4.2 points")}
    assert S.Writer("keep", {7: stale}).write(r)["generated_by"]["kind"] == "template"        # right hash, wrong text
    assert S.Writer("none", {7: old}).write(r) is None
    assert S.Writer("keep").write({**r, "tendencies": []}) is None
    with pytest.raises(S.SummaryError):
        S.Writer("always")


def test_the_validator_refuses_a_summary_that_no_longer_matches_its_table(synth_run, tmp_path):
    out = tmp_path / "export"
    shutil.copytree(synth_run["out"], out)
    validate_export_dir(out)
    path = sorted((out / "team_reports").glob("*.json"))[0]
    rep = json.loads(path.read_text())
    edited = copy.deepcopy(rep)
    edited["tendencies"][0]["n"] += 1                                                     # the table moved; the summary did not
    path.write_text(json.dumps(edited))
    with pytest.raises(ContractError, match="written from a different table"):
        validate_export_dir(out)
    edited = copy.deepcopy(rep)
    t = edited["summary"]["sentences"][0]["text"]
    edited["summary"]["sentences"][0]["text"] = t.replace(" on ", " on about 1,000,000 of ", 1)   # the summary moved; the table did not
    path.write_text(json.dumps(edited))
    with pytest.raises(ContractError, match="does not match its table"):
        validate_export_dir(out)
    validate({k: v for k, v in rep.items() if k != "summary"}, "team_report")              # a report without one is still a report


def test_summarize_fills_an_export_on_disk_and_does_not_ask_twice(synth_run, tmp_path):
    pytest.importorskip("anthropic")
    out = tmp_path / "export"
    shutil.copytree(synth_run["out"], out)

    class Echo(Fake):
        """Answers each table with a passing summary of its first three rows, written the way the prompt asks."""
        def create(self, **kw):
            self.calls.append(kw)
            t = json.loads(kw["messages"][0]["content"].split("\n\nA first draft")[0])
            if len(self.calls) == 2:                                                      # one team's first draft is thrown away
                return reply([{"rows": [1], "text": "The team is simply better than 29 other teams."}])
            return reply([{"rows": [r["row"]], "text": f"In this situation ({r['situation']}), {r['about']} has {r['outcome']} {r['difference']} "
                                                       f"({r['interval']}{', which includes zero' if r['interval_includes_zero'] else ''}) on {r['sample']}."}
                          for r in t["rows"][:3]])

    fake, lines = Echo(), []
    block = S.summarize_dir(out, cl=fake, log=lines.append, n_show=2)
    n = block["n_reports"]
    assert (block["n_llm"], block["n_template"], block["n_asked"], block["n_first_draft_rejected"], block["n_fell_back"]) == (n, 0, n, 1, 0)
    assert (block["n_drafts"], block["n_drafts_rejected"], block["models"]) == (n + 1, 1, ["claude-opus-5-5"]) and len(fake.calls) == n + 1
    man = json.loads((out / "manifest.json").read_text())
    assert man["summaries"] == block and validate_export_dir(out)["team_report"] == n
    text = "\n".join(lines)
    assert text.count("   table:") == 2 and "discarded the first draft for 1 of" in text and "1 draft discarded" in text
    names = {p["name"] for t in json.loads((out / "teams.json").read_text())["teams"] for p in t["players"]}
    assert not any(nm in json.dumps(fake.calls) for nm in names)                           # the whole export's roster, and none of it left the process
    again = Echo()
    S.summarize_dir(out, cl=again, log=lambda *_: None)
    assert again.calls == []                                                              # every table already has its summary
    S.summarize_dir(out, cl=again, log=lambda *_: None, force=True, limit=2)
    assert len(again.calls) >= 2


def test_dry_run_prints_the_request_and_needs_no_key(synth_run, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    lines = []
    assert S.summarize_dir(synth_run["out"], dry_run=True, log=lines.append) == {}
    body = json.loads(lines[0])
    assert body["system"] == S.PROMPT_PATH.read_text() and body["messages"][0]["role"] == "user" and "Nothing was sent" in lines[1]


def test_an_export_without_summaries_is_still_valid(synth_run):
    man = json.loads((synth_run["out"] / "manifest.json").read_text())
    validate({k: v for k, v in man.items() if k != "summaries"}, "manifest")
    validate({**{k: v for k, v in man.items() if k != "summaries"}, "contract_version": "1.6", "not_built": man["not_built"] + ["LLM-written summaries"]}, "manifest")


def test_the_prompt_is_a_file_in_the_repo_and_states_the_rule():
    p = S.prompt_text()
    assert S.PROMPT_PATH.name == "team_summary.txt" and len(S.prompt_sha()) == 16
    for must in ("it is everything you know", "PLAYER_A", "Never do arithmetic", "zero", "above", "below"):
        assert must in p
    assert "{" not in p                                                                   # sent as it is; nothing is filled in


def test_the_pregame_path_never_reaches_the_summaries():
    from pathlib import Path
    src = (Path(S.__file__).parent / "prospective.py").read_text()
    assert "summaries" not in src and "anthropic" not in src
