"""Step 3.2: a short written summary above each team report's tendency table.

The rule from the brief is absolute: the writer sees one team's tendency table and writes only
from it. Three things here hold it to that.

1. `summary_table` turns a team report's rows into the table as the app prints it (the same
   strings, to the same decimal). That table, as JSON, is the whole user message. The model is
   not told which team it is, and a player reaches it only as a placeholder (`mask`), so there is
   nothing for it to remember.
2. `check` reads the result one sentence at a time against the rows that sentence cites. Any
   number that is not in those rows, any capitalised word that is not a name or label in them,
   a direction the row does not have, or a figure given without its interval or sample size,
   and the whole summary is thrown away.
3. `template_summary` is what is shown instead: three sentences filled in from the first three
   rows by a fixed template. It passes the same check.

Nothing in this module needs the network or the SDK until `client()` is called.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "team_summary.txt"
MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 8000                     # thinking counts toward it; the summary itself is about 150 tokens
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_ATTEMPTS = 2                      # one draft, and one more that is shown why the first was discarded
MIN_ROWS_FOR_LLM = 3                  # a table shorter than this is its own summary
TEMPLATE_ROWS = 3
MAX_SENTENCES = 5
MAX_WORDS = 150
MODES = ("keep", "llm", "none")

OUT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["sentences"], "properties": {
    "sentences": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["rows", "text"], "properties": {
        "rows": {"type": "array", "items": {"type": "integer"}}, "text": {"type": "string"}}}}}}

# The same words the team page prints beside each row (app/app.js, renderTeam).
UNIT = {"shot_family": "shots", "action_style": "shots", "creation": "made baskets", "fast_break": "possessions ending in a shot or at the line"}
VERSUS = {"league_same_situation": "the league rate",
          "team_overall_plus_league_shift": "expected from the team's overall rate and the league-wide shift in this situation",
          "player_overall": "his share across all situations",
          "team_elsewhere_plus_league_shift": "expected from the team's rate in its other possessions and the league-wide shift in this situation",
          "player_lineup_adjusted": "what his usage and the four teammates on the floor predict"}
TEAM = "the team"

RULE = ("Each summary is written from that team's tendency table and nothing else. The writer is not told which team it is and sees players "
        "as placeholders. A program then checks every sentence against the rows it cites: each number must appear in those rows as printed, "
        "each name must be in them, the direction must match, and a difference must come with its interval or its sample size. "
        "A summary that fails is discarded and three sentences filled in from the first three rows by a fixed template are shown instead.")


class SummaryError(RuntimeError):
    """The summaries could not be written at all: no SDK, no credentials, or the API turned the request down."""


# ---------- the table ---------------------------------------------------------------------------

def _fixed1(x: float) -> str:
    """abs(x) to one decimal, rounding a tie up: what JavaScript's toFixed(1) prints in the app."""
    return str(Decimal(abs(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _points(x: float) -> str:
    return _fixed1(x * 100)


def _interval(lo: float, hi: float) -> str:
    a, b = _points(lo), _points(hi)
    if lo >= 0:
        return f"{a} to {b} points above baseline"
    if hi <= 0:
        return f"{b} to {a} points below baseline"
    return f"{a} points below to {b} points above baseline"


def summary_table(report: dict) -> dict:
    """The tendency table as the team page prints it. Built from the report's rows and nothing else."""
    rows = []
    for i, x in enumerate(report["tendencies"], 1):
        label = x["outcome"]["label"]
        who, outcome = TEAM, label[:1].lower() + label[1:]
        if x["scope"] == "player":
            who, _, outcome = label.partition(": ")       # tendency_layer writes "<name>: share of finishes while on the floor"
        unit = "possessions on the floor" if x["scope"] == "player" else UNIT.get(x["outcome"]["head"], "possessions")
        rows.append({
            "row": i, "about": who, "situation": x["situation"]["label"], "outcome": outcome,
            "observed": f"{_fixed1(x['raw_rate'] * 100)}%", "sample": f"{x['n']:,} {unit}",
            "baseline": f"{_fixed1(x['baseline_rate'] * 100)}%", "baseline_is": VERSUS.get(x["baseline_kind"], x["baseline_kind"]),
            "difference": f"{_points(x['deviation'])} points {'above' if x['deviation'] >= 0 else 'below'} baseline",
            "interval": _interval(*x["interval"]), "interval_includes_zero": not x["excludes_zero"]})
    return {"interval_level": f"{round(report['interval_level'] * 100)}%", "rows": rows}


def table_hash(table: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def prompt_text() -> str:
    return PROMPT_PATH.read_text()


def prompt_sha() -> str:
    return hashlib.sha256(prompt_text().encode()).hexdigest()[:16]


def _placeholder(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return f"PLAYER_{s}"


def mask(table: dict) -> tuple[dict, dict[str, str]]:
    """The table the model is sent: every player's name replaced by a placeholder. Returns it with placeholder -> name."""
    names: dict[str, str] = {}
    rows = []
    for r in table["rows"]:
        if r["about"] != TEAM:
            names.setdefault(r["about"], _placeholder(len(names)))
        rows.append({**r, "about": names.get(r["about"], TEAM)})
    return {**table, "rows": rows}, {ph: name for name, ph in names.items()}


def unmask(text: str, mapping: dict[str, str]) -> str:
    for ph in sorted(mapping, key=len, reverse=True):
        text = re.sub(rf"\b{ph}\b", lambda _m, n=mapping[ph]: n, text)
    return text


# ---------- the check ---------------------------------------------------------------------------

_NUM = re.compile(r"[A-Za-z#]*\d(?:[\d,.:]*\d)?[A-Za-z]*")
_WORD = re.compile(r"[^\W\d_][\w'’\-]*")
_BARE = re.compile(r"\d+(?:\.\d+)?")
CARDINALS = set("two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
                "thirty forty fifty sixty seventy eighty ninety hundred hundreds thousand thousands million dozen dozens couple".split())
ORDINALS = set("first second third fourth fifth sixth seventh eighth ninth tenth".split())
MULTIPLES = set("half halves twice double doubles doubled triple tripled thirds".split())
COUNT_NOUNS = set("point points shot shots possession possessions game games time times percent basket baskets foul fouls turnover turnovers".split())
STARTERS = set("the this that these those its their it they there a an in on at with within across after off over when while during late early "
               "for from by against among both each every all most coming out of and but also still only inside near around".split())
OPPOSITE = {"above": {"below", "less", "fewer", "lower", "under", "rarer"}, "below": {"above", "more", "higher", "greater", "over", "extra"}}
FORBIDDEN = set("<>{}[]*_`|\\\n\r\t")
ALIASES = {"q4": {"fourth"}, "q1": {"first"}, "q2": {"second"}, "q3": {"third"}}       # a row labelled Q4 may be called the fourth quarter


def _strip(tok: str) -> str:
    return re.sub(r"['’]s$", "", tok).strip(".-'’")


def _numbers(text: str) -> set:
    out = set()
    for tok in _NUM.findall(text):
        t = tok.replace(",", "")
        out.add(Decimal(t) if _BARE.fullmatch(t) else t.lower())
    return out


def _vocab(row: dict, level: str) -> dict:
    strings = [str(v) for k, v in row.items() if k not in ("row", "interval_includes_zero")]
    text = " ".join(strings)
    words = {_strip(w).lower() for w in _WORD.findall(text)}
    words |= {p for w in list(words) for p in w.split("-")}
    for tok in _NUM.findall(text):
        words |= ALIASES.get(tok.lower(), set())
    caps = {_strip(w) for w in _WORD.findall(text) if w[0].isupper()}
    if row["about"] != TEAM:
        caps |= {_strip(p) for p in row["about"].split()}
    return {"numbers": _numbers(text) | _numbers(level), "words": words, "caps": caps}


def _show(n) -> str:
    return str(n)


def check(sentences: list, table: dict) -> list[str]:
    """Why this summary cannot be shown above this table. An empty list means it can."""
    rows = {r["row"]: r for r in table["rows"]}
    if not isinstance(sentences, list) or not 1 <= len(sentences) <= MAX_SENTENCES:
        return [f"a summary has 1 to {MAX_SENTENCES} sentences"]
    why, used, n_words = [], set(), 0
    for i, s in enumerate(sentences, 1):
        at = f"sentence {i}"
        if not isinstance(s, dict) or not isinstance(s.get("text"), str) or not isinstance(s.get("rows"), list):
            why.append(f"{at}: needs a text and a list of rows")
            continue
        text, cited = s["text"].strip(), s["rows"]
        if not 1 <= len(cited) <= 2 or len(set(cited)) != len(cited) or any(type(r) is not int or r not in rows for r in cited):
            why.append(f"{at}: cites one or two different rows of the table by number; got {cited}")
            continue
        if used & set(cited):
            why.append(f"{at}: row {sorted(used & set(cited))[0]} is already used by an earlier sentence")
        used |= set(cited)
        probe = text
        for r in cited:                                       # a placeholder such as PLAYER_A is the row's own name until it is swapped back
            probe = probe.replace(rows[r]["about"], "")
        bad = sorted(FORBIDDEN & set(probe))
        if bad:
            why.append(f"{at}: plain prose only; remove {' '.join(repr(c) for c in bad)}")
        if not 20 <= len(text) <= 420 or not text.endswith("."):
            why.append(f"{at}: one sentence of 20 to 420 characters that ends with a full stop")
        n_words += len(text.split())
        voc = [_vocab(rows[r], table["interval_level"]) for r in cited]
        numbers = set().union(*(v["numbers"] for v in voc))
        words = set().union(*(v["words"] for v in voc))
        caps = set().union(*(v["caps"] for v in voc))
        said = _numbers(text)
        for n in sorted(said - numbers, key=str):
            why.append(f"{at}: {_show(n)} is not a number in row {' or '.join(map(str, cited))}")
        toks = [(m.start(), m.group()) for m in _WORD.finditer(text)]
        lower = [_strip(t).lower() for _, t in toks]
        for j, w in enumerate(lower):
            for part in w.split("-"):
                counted = (part in CARDINALS or part in ORDINALS or part in MULTIPLES
                           or (part == "one" and w == "one" and j + 1 < len(lower) and lower[j + 1] in COUNT_NOUNS))
                if counted and part not in words:
                    why.append(f"{at}: '{part}' is a number written as a word and is not in the row; copy the row's digits")
        for j, (pos, tok) in enumerate(toks):
            if not tok[0].isupper():
                continue
            t = _strip(tok)
            first = j == 0 or re.search(r"[.!?]\s+$", text[:pos]) is not None
            known = t in caps or (first and (t.lower() in words or t.lower() in STARTERS))
            if not known:
                why.append(f"{at}: '{t}' is capitalised and is not a name or label in row {' or '.join(map(str, cited))}"
                           + ("; start with the row's situation or with 'The team'" if first else ""))
        def bag(t: str) -> list[str]:                         # "above-the-break" is a place on the floor, not a direction
            return [_strip(w) for w in _WORD.findall(re.sub(r"above[- ]the[- ]break", " ", t.lower()))]
        plain = bag(text)
        plain_words = set(plain)
        for r in cited:
            row = rows[r]
            direction = "above" if "above baseline" in row["difference"] else "below"
            diff = Decimal(row["difference"].split()[0])
            ends = _numbers(row["interval"])
            size = Decimal(row["sample"].split()[0].replace(",", ""))
            if diff not in said:
                why.append(f"{at}: give row {r}'s difference, {row['difference']}")
            elif size not in said and not ends <= said:
                why.append(f"{at}: row {r}'s difference needs its interval ({row['interval']}) or its sample ({row['sample']}) beside it")
            if row["interval_includes_zero"]:
                if "zero" not in plain_words:
                    why.append(f"{at}: row {r}'s interval includes zero and the sentence has to say so with the word 'zero'")
            else:
                if direction not in plain_words:
                    why.append(f"{at}: say row {r}'s direction with the word '{direction}'")
                if len(cited) == 1:                           # a row's own label may hold such a word ("16 s or more"); the sentence gets no more of them than that
                    label = bag(f"{row['situation']} {row['outcome']}")
                    for w in sorted(OPPOSITE[direction] & plain_words):
                        if plain.count(w) > label.count(w):
                            why.append(f"{at}: row {r} is {direction} baseline, and '{w}' points the other way")
    if n_words > MAX_WORDS:
        why.append(f"the summary is {n_words} words; keep it under {MAX_WORDS}")
    return why


# ---------- the two writers ------------------------------------------------------------------------

def template_sentences(table: dict) -> list[dict]:
    out = []
    for r in table["rows"][:TEMPLATE_ROWS]:
        who = "" if r["about"] == TEAM else f"{r['about']}'s "
        zero = ", which includes zero" if r["interval_includes_zero"] else ""
        out.append({"rows": [r["row"]], "text": f"{r['situation']}: {who}{r['outcome']}, {r['difference']} "
                                                f"({table['interval_level']} interval {r['interval'].removesuffix(' baseline')}{zero}) on {r['sample']}."})
    return out


def _block(kind: str, model: str | None, table: dict, sentences: list[dict], attempts: int, rejected: int) -> dict:
    return {"generated_by": {"kind": kind, "model": model, "prompt_sha": prompt_sha() if kind == "llm" else None},
            "table_hash": table_hash(table), "sentences": [{"rows": list(s["rows"]), "text": s["text"].strip()} for s in sentences],
            "attempts": attempts, "rejected": rejected}


def template_summary(table: dict, attempts: int = 0, rejected: int = 0) -> dict | None:
    """Three sentences from the first three rows. None when the table has no rows: there is nothing to say."""
    if not table["rows"]:
        return None
    sentences = template_sentences(table)
    why = check(sentences, table)
    if why:                                                   # a label the template cannot carry; say nothing rather than something unchecked
        raise SummaryError("the templated summary failed its own check: " + "; ".join(why))
    return _block("template", None, table, sentences, attempts, rejected)


def client():
    """An Anthropic client with credentials read from the environment. Raises SummaryError when there are none."""
    try:
        import anthropic
    except ImportError as e:
        raise SummaryError('the Anthropic SDK is not installed; pip install -e ".[llm]"') from e
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SummaryError("no API key in the environment; set ANTHROPIC_API_KEY (it is read from the environment and never written anywhere)")
    return anthropic.Anthropic()


def request(table_for_model: dict, model: str = MODEL, feedback: tuple[list, list[str]] | None = None) -> dict:
    """The exact request body: the prompt file as the system prompt, and the table as the user message.

    On a second attempt the user message also carries the discarded draft and what the check said about it.
    Both are in placeholder form, so the request still names no team and no player.
    """
    content = json.dumps(table_for_model, indent=1, ensure_ascii=False)
    if feedback is not None:
        draft, why = feedback
        content += ("\n\nA first draft was discarded by the checker. The draft:\n" + json.dumps({"sentences": draft}, indent=1, ensure_ascii=False)
                    + "\n\nWhat the checker reported:\n" + "\n".join(f"- {w}" for w in why[:12])
                    + "\n\nWrite the summary again from the table so that every sentence passes.")
    return {"model": model, "max_tokens": MAX_TOKENS, "system": prompt_text(), "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": OUT_SCHEMA}},
            # If a safety classifier declines the request, the API reruns it on the model Anthropic recommends
            # instead. Whichever model answered is the one recorded in `generated_by`.
            "betas": [FALLBACK_BETA], "fallbacks": "default"}


def _draft(resp) -> tuple[list | None, str | None]:
    """(sentences, None), or (None, why there are none)."""
    if resp.stop_reason == "refusal":
        return None, "the model declined to write a summary"
    if resp.stop_reason == "max_tokens":
        return None, "the reply was cut off before it finished"
    text = next((b.text for b in resp.content if b.type == "text"), None)
    try:
        return json.loads(text)["sentences"], None
    except (TypeError, ValueError, KeyError):
        return None, "the reply was not the JSON that was asked for"


def llm_summary(table: dict, cl, model: str = MODEL, log=None) -> dict:
    """Ask for a summary of this table, check it, ask once more if it fails, and fall back to the template.

    Raises SummaryError when the API cannot be used at all (credentials, a rejected request, no connection).
    A summary that fails the check is never an error: the template is returned, marked with how many drafts were discarded.
    """
    import anthropic
    masked, mapping = mask(table)
    feedback, rejected = None, 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = cl.beta.messages.create(**request(masked, model, feedback))
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise SummaryError(f"the API key was not accepted: {e}") from e
        except anthropic.APIStatusError as e:                 # 400, 404, and 429 or 5xx once the SDK's own retries are spent
            raise SummaryError(f"the API turned the request down ({e.status_code}): {e}") from e
        except anthropic.APIConnectionError as e:
            raise SummaryError(f"could not reach the API: {e}") from e
        draft, why = _draft(resp)
        if draft is not None:
            why = check(draft, masked)
            if not why:
                sentences = [{"rows": s["rows"], "text": unmask(s["text"], mapping)} for s in draft]
                if not check(sentences, table):              # the same check again on the text as it will be shown
                    return _block("llm", resp.model, table, sentences, attempt, rejected)
                # Said in placeholder terms, so that no name goes back to the model with the reasons.
                why = ["the text no longer passed once the players' names were put back; write each placeholder exactly as given, on its own"]
        else:
            why = [why]
        rejected += 1
        if log:
            log(f"    draft {attempt} discarded: " + "; ".join(why[:6]) + (f"; and {len(why) - 6} more" if len(why) > 6 else ""))
        feedback = (draft, why) if draft is not None else None
    return template_summary(table, attempts=MAX_ATTEMPTS, rejected=rejected)


# ---------- wiring: the export, the validator, the command -------------------------------------------

def verify(report: dict) -> list[str]:
    """Is the summary stored in this report still about the table beside it? Reasons it is not; empty when it is."""
    s = report.get("summary")
    if s is None:
        return []
    table = summary_table(report)
    if s["table_hash"] != table_hash(table):
        return ["the summary was written from a different table (table_hash does not match the rows in this report)"]
    return check(s["sentences"], table)


def existing(out: Path) -> dict[int, dict]:
    """Summaries already in an export folder, by team. Read before the export clears it, so a paid-for summary can be kept."""
    found = {}
    for f in (out / "team_reports").glob("*.json") if (out / "team_reports").is_dir() else []:
        try:
            rep = json.loads(f.read_text())
            if "summary" in rep:
                found[int(rep["team_id"])] = rep["summary"]
        except (ValueError, KeyError, TypeError):
            continue
    return found


class Writer:
    """Chooses a summary for each report as the export writes it.

    keep: a machine-written summary already in the output folder is kept when it was written from exactly this
          table and still passes the check; every other report gets the template. No network.
    llm:  the same, except that a report with nothing to keep is sent to the model.
    none: no summaries.
    """

    def __init__(self, mode: str = "keep", previous: dict[int, dict] | None = None, cl=None, model: str = MODEL, log=print, force: bool = False):
        if mode not in MODES:
            raise SummaryError(f"--summaries is one of {', '.join(MODES)}")
        self.mode, self.previous, self.cl, self.model, self.log, self.force = mode, previous or {}, cl, model, log, force
        self.stopped: str | None = None
        self.done: list[dict] = []
        if mode == "llm" and cl is None:
            self.cl = client()

    def write(self, report: dict) -> dict | None:
        if self.mode == "none":
            return None
        table = summary_table(report)
        if not table["rows"]:
            return None
        old = self.previous.get(int(report["team_id"]))
        if (old and not self.force and old.get("generated_by", {}).get("kind") == "llm" and old.get("table_hash") == table_hash(table)
                and not check(old.get("sentences"), table)):
            s = old
        elif self.mode == "llm" and not self.stopped and len(table["rows"]) >= MIN_ROWS_FOR_LLM:
            try:
                s = llm_summary(table, self.cl, self.model, self.log)
            except SummaryError as e:                       # the export is not thrown away over a network error
                self.stopped = str(e)
                self.log(f"summaries: {e}. The remaining reports get the templated summary; run `python -m tendency summarize` on the export to finish.")
                s = template_summary(table)
        else:
            s = template_summary(table)
        self.done.append(s)
        return s


def tally(summaries: list[dict]) -> dict:
    """The manifest's count of who wrote what, and how often the check threw a draft away."""
    llm = [s for s in summaries if s["generated_by"]["kind"] == "llm"]
    asked = [s for s in summaries if s["attempts"] > 0]
    models = sorted({s["generated_by"]["model"] for s in llm})
    return {"rule": RULE, "n_reports": len(summaries), "n_llm": len(llm), "n_template": len(summaries) - len(llm),
            "n_asked": len(asked), "n_first_draft_rejected": sum(1 for s in asked if s["rejected"] > 0),
            "n_fell_back": sum(1 for s in asked if s["generated_by"]["kind"] == "template"),
            "n_drafts": sum(s["attempts"] for s in asked), "n_drafts_rejected": sum(s["rejected"] for s in asked),
            "models": models, "prompt_sha": prompt_sha() if llm else None}


def show(report: dict, name: str) -> str:
    """One team's summary beside the table it was written from, as text."""
    table, s = summary_table(report), report.get("summary")
    lines = [f"== {name} =="]
    if s:
        g = s["generated_by"]
        lines.append(f"   written by: {g['model'] if g['kind'] == 'llm' else 'the fixed template'}"
                     + (f" (drafts asked for: {s['attempts']}, discarded: {s['rejected']})" if s["attempts"] else ""))
        lines += [f"   [{', '.join(map(str, x['rows']))}] {x['text']}" for x in s["sentences"]]
    else:
        lines.append("   no summary")
    lines.append("   table:")
    for r in table["rows"]:
        lines.append(f"   {r['row']:>2}. {r['situation']} | {r['about']}: {r['outcome']} | {r['observed']} on {r['sample']}; baseline {r['baseline']} | "
                     f"{r['difference']} ({r['interval']}{', includes zero' if r['interval_includes_zero'] else ''})")
    return "\n".join(lines)


def summarize_dir(out: Path, model: str = MODEL, force: bool = False, limit: int | None = None, n_show: int = 5, dry_run: bool = False,
                  cl=None, log=print) -> dict:
    """Write machine-written summaries into an export that is already on disk, without rebuilding it.

    Each report is rewritten as soon as its summary is done, so a run that stops partway can be started again
    and picks up where it left off. Returns the manifest's `summaries` block.
    """
    import shutil

    from . import CONTRACT_VERSION
    from .contract import SCHEMA_DIR, validate, validate_export_dir
    validate_export_dir(out)                                  # against this checkout's schemas, before anything is spent
    manifest = json.loads((out / "manifest.json").read_text())
    files = manifest["files"]
    teams = json.loads((out / files["teams"]).read_text())["teams"]
    paths = {t["team_id"]: out / files["team_reports_dir"] / f"{t['team_id']}.json" for t in teams}
    reports = {tid: json.loads(p.read_text()) for tid, p in paths.items()}
    if dry_run:
        tid = next((t for t, r in reports.items() if len(r["tendencies"]) >= MIN_ROWS_FOR_LLM), None)
        if tid is None:
            raise SummaryError("no report in this export has enough rows to summarise")
        masked, _ = mask(summary_table(reports[tid]))
        log(json.dumps(request(masked, model), indent=1, ensure_ascii=False))
        log(f"\nThat is the whole request for one team ({len(reports)} in this export). Nothing was sent.")
        return {}
    w = Writer("llm", {tid: r["summary"] for tid, r in reports.items() if "summary" in r}, cl, model, log, force)
    todo = [t for t in teams if reports[t["team_id"]]["tendencies"]][:limit]
    stopped = None
    for i, t in enumerate(todo, 1):
        rep = reports[t["team_id"]]
        before = rep.get("summary")
        s = w.write(rep)
        if w.stopped:
            stopped = f"{w.stopped}. {i - 1} of {len(todo)} reports were done and are saved; run the command again to continue."
            break
        rep["summary"] = s
        validate(rep, "team_report", f"team report {t['team_id']}")
        paths[t["team_id"]].write_text(json.dumps(rep, separators=(",", ":"), allow_nan=False))
        g = s["generated_by"]
        log(f"  {i:>2}/{len(todo)} {t['tricode']}: " + ("kept" if s is before else g["model"] if g["kind"] == "llm" else "TEMPLATE (every draft was discarded)")
            + (f", {s['rejected']} draft{'s' if s['rejected'] != 1 else ''} discarded" if s["rejected"] and s is not before else ""))
    block = tally([r["summary"] for r in reports.values() if "summary" in r])
    manifest["summaries"] = block
    manifest["contract_version"] = CONTRACT_VERSION           # an export from before summaries existed is brought up to this contract
    manifest["not_built"] = [x for x in manifest["not_built"] if x != "LLM-written summaries"]
    validate(manifest, "manifest")
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":"), allow_nan=False))
    shutil.copytree(SCHEMA_DIR, out / files["schemas_dir"], dirs_exist_ok=True)      # both validators read the export's own copy
    validate_export_dir(out)
    if stopped:
        raise SummaryError(stopped)
    for t in [t for t in todo if reports[t["team_id"]]["summary"]["generated_by"]["kind"] == "llm"][:n_show]:
        log("\n" + show(reports[t["team_id"]], t["name"]))
    log("\n" + describe(block))
    return block


def describe(b: dict) -> str:
    if not b or not b["n_reports"]:
        return "summaries: none"
    line = f"summaries: {b['n_llm']} machine-written" + (f" ({', '.join(b['models'])})" if b["models"] else "") + f", {b['n_template']} templated, of {b['n_reports']} reports"
    if b["n_asked"]:
        line += (f". The check discarded the first draft for {b['n_first_draft_rejected']} of {b['n_asked']} teams "
                 f"({b['n_first_draft_rejected'] / b['n_asked']:.0%}) and {b['n_drafts_rejected']} of {b['n_drafts']} drafts in all; "
                 f"{b['n_fell_back']} team{'s' if b['n_fell_back'] != 1 else ''} fell back to the template")
    return line + "."
