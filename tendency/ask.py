"""Step 3.3: the ask box. A question becomes one lookup in the tendency table; the answer is written from the rows.

The rule from the brief: the language model turns a question into a query and never answers from
its own memory. Here that is true by construction, not by instruction:

1. A translator fills a small schema (team, player, situation, outcome, season, comparison,
   compare_to) or says the question cannot be put that way and picks one of six reasons. The model
   does this by calling one of two tools. It can return nothing but values from fixed lists and a
   player's name.
2. `resolve` checks the query against the table's own lists and matches the name to a player.
3. `run` executes one parameterised SQL statement in DuckDB against the Parquet table.
4. `compose` writes one sentence from the rows with a fixed template. Every rate in it carries its
   interval and its sample size. A refusal is one of six fixed sentences.

No text written by a model reaches the reader. The worst a wrong translation can do is look up the
wrong row, and the response always shows the query that was run.

`RulesTranslator` is the baseline: keyword matching, no model, no network. `LLMTranslator` has to
beat it on the thirty questions in ask_questions.json (`python -m tendency ask-eval`).

This module needs DuckDB and the standard library. The SDK is imported only when a model is asked.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from statistics import NormalDist

from .summaries import _fixed1, _interval, _points

HERE = Path(__file__).resolve().parent
PROMPT_PATH = HERE / "prompts" / "ask_translate.txt"
QUESTIONS_PATH = HERE / "ask_questions.json"
MODEL = "claude-opus-5-5"
EFFORT = "low"                        # the job is choosing from short lists
MAX_TOKENS = 4000                     # thinking counts toward it; a tool call is about 100 tokens
MAX_ATTEMPTS = 2                      # one more try when the model answers in words and calls no tool
MAX_QUESTION = 300                    # characters
MAX_ROWS = 30                         # rows returned to the reader; a ranking of players is cut to its top ten
TOP_PLAYERS = 10
INTERVAL_LEVEL = 0.90                 # the tendency layer's level; the table's intervals are at this level
Z = NormalDist().inv_cdf((1 + INTERVAL_LEVEL) / 2)
PLAYER_OUTCOME = "finisher"
FIELDS = ("team", "player", "situation", "outcome", "season", "comparison", "compare_to")
COMPARISONS = {"baseline": "against the row's own baseline", "league": "ranked across the league",
               "team": "against another team", "season": "against another season"}
OPENING = "This data cannot answer that, because "
REASONS = {
    "play_call": "play-by-play records how a possession ended and who was on the floor, not the play that was called.",
    "coverage": "play-by-play does not record where defenders stood or who guarded whom, so it cannot identify a coverage, a scheme or a matchup.",
    "injury": "it holds nothing about injuries, health or who is available.",
    "forecast_or_opinion": "it holds measured rates from games already played, and the question asks for a forecast, a reason or a judgment.",
    "not_in_table": ("the ask box looks up one row at a time: a team's rate for one outcome in one situation and season, "
                     "or a player's share of finishes in one situation. This question needs something else."),
    "outside_data": "it is outside what the table holds, which is offensive tendencies from play-by-play for the seasons loaded.",
}
PHRASE = {"all": "across all possessions", "after_timeout": "after a timeout", "clutch": "in the last 5:00 of Q4/OT within 5",
          "start_def_rebound": "off a defensive rebound", "start_turnover": "off a turnover", "start_made_basket": "after an opponent score",
          "def_in_penalty": "with the defense in the penalty", "q4": "in the fourth quarter",
          "two_for_one": "in the two-for-one window (30 to 45 s left in the period)"}
RULE = ("A question is translated into one lookup in the tendency table (team, player, situation, outcome, season, comparison). "
        "The translator returns only values from fixed lists and a player's name. A program runs the lookup and writes the answer "
        "from the rows with a fixed template, so no text written by a language model is shown. A question the table cannot answer "
        "gets a fixed sentence saying why.")


class AskError(RuntimeError):
    """The question could not be turned into a query at all: no SDK, no key, the API turned it down, or no tool was called."""


class Invalid(ValueError):
    """The translated query does not fit the table. The message is plain enough to show."""


# ---------- the lists a question is translated against ------------------------------------------

def season_label(y: int) -> str:
    return f"{int(y)}-{(int(y) + 1) % 100:02d}"


def norm(s: str) -> str:
    """Lower case, no accents, no punctuation: how names are compared."""
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9#]+", " ", s.lower()).split())


class Directory:
    """Teams, seasons, situations, outcomes and players. Everything a translator may name comes from here."""

    def __init__(self, teams, seasons, situations, outcomes, players, data_kind: str = "synthetic", as_of: str | None = None):
        self.teams = sorted((str(t), str(n)) for t, n in teams)                     # (tricode, name)
        self.seasons = sorted(int(s) for s in seasons)
        self.situations = [(str(i), str(lab)) for i, lab in situations]             # (id, label)
        self.outcomes = [(str(i), str(lab)) for i, lab in outcomes]                 # team outcomes; "finisher" is added for players
        self.players = sorted({(int(i), str(n)) for i, n in players}, key=lambda x: (x[1], x[0]))
        self.data_kind, self.as_of = data_kind, as_of

    @property
    def latest(self) -> int:
        return self.seasons[-1]

    @property
    def season_labels(self) -> list[str]:
        return [season_label(s) for s in self.seasons]

    def team_name(self, tricode: str) -> str:
        return dict(self.teams).get(tricode, tricode)

    def match_players(self, name: str) -> list[tuple[int, str]]:
        """Players the name could mean: the full name, else every player whose name holds all of its words."""
        want = norm(name)
        if not want:
            return []
        exact = [p for p in self.players if norm(p[1]) == want]
        if exact:
            return exact
        words = set(want.split())
        return [p for p in self.players if words <= set(norm(p[1]).split())]


class Table(Directory):
    """The ask table: a DuckDB view over the Parquet file `ask_table.write` made, and the lists read from it."""

    def __init__(self, path: Path | str):
        import duckdb
        self.path = Path(path)
        if not self.path.is_file():
            raise AskError(f"no ask table at {self.path}; `python -m tendency run` (or `sample`) writes it")
        self.con = duckdb.connect()
        self.con.execute("CREATE VIEW cells AS SELECT * FROM read_parquet('" + str(self.path).replace("'", "''") + "')")
        q = lambda s: self.con.execute(s).fetchall()                                # noqa: E731
        kind = q("SELECT any_value(data_kind), max(as_of_utc) FROM cells")[0]
        super().__init__(
            teams=q("SELECT tricode, arg_max(team_name, season) FROM cells GROUP BY tricode"),
            seasons=[r[0] for r in q("SELECT DISTINCT season FROM cells")],
            situations=q("SELECT situation, any_value(situation_label) FROM cells GROUP BY situation ORDER BY situation"),
            outcomes=q("SELECT outcome, any_value(outcome_label) FROM cells WHERE scope = 'team' GROUP BY outcome ORDER BY outcome"),
            players=q("SELECT DISTINCT player_id, player_name FROM cells WHERE scope = 'player'"),
            data_kind=kind[0] or "synthetic", as_of=kind[1])
        if not self.seasons:
            raise AskError(f"the ask table at {self.path} is empty")

    def query(self, sql: str, params: list) -> list[dict]:
        cur = self.con.cursor()                                                     # one cursor per call: the service answers from a thread pool
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def eval_sets() -> tuple[Directory, dict[str, list[dict]]]:
    """The fixed lists, and both sets of questions: `written` (the thirty) and `reworded` (the same thirty, put another way)."""
    d, written = eval_directory()
    doc = json.loads(QUESTIONS_PATH.read_text())
    return d, {"written": written, "reworded": doc.get("reworded", [])}


def eval_directory() -> tuple[Directory, list[dict]]:
    """The fixed lists the thirty questions are translated against (30 franchises, four seasons), and the questions."""
    from . import cdn_fields as F
    from .possessions import ENDING_CLASSES, SHOT_FAMILIES
    from .teams import TEAMS
    from .tendencies import END_LABEL, FAM_LABEL, SITUATIONS, STYLE_LABEL, TEMPO_LABEL
    doc = json.loads(QUESTIONS_PATH.read_text())
    outcomes = ([(f"ending_type:{c}", END_LABEL[c]) for c in ENDING_CLASSES] + [(f"shot_family:{c}", FAM_LABEL[c]) for c in SHOT_FAMILIES]
                + [(f"action_style:{c}", STYLE_LABEL[c]) for c in F.ACTION_STYLES] + [("creation:assisted", "Assisted share of made baskets")]
                + [(f"tempo:{c}", TEMPO_LABEL[c]) for c in ("early", "late")]
                + [("fast_break:fast_break", "Fast-break share of possessions ending in a shot or at the line")])
    d = Directory(teams=[(t, f"{city} {nick}") for t, city, nick in TEAMS.values()], seasons=doc["context"]["seasons"], situations=SITUATIONS,
                  outcomes=sorted(outcomes), players=list(enumerate(doc["context"]["players"], 1)), data_kind="real")
    return d, doc["questions"]


# ---------- the query: check it, run it ----------------------------------------------------------

def _clean_name(s) -> str | None:
    """A player's name from the translator, cut down to what a name can hold. Nothing else it writes is kept."""
    if s is None:
        return None
    s = " ".join(re.sub(r"[^\w .'’#-]", " ", str(s)).split())[:40].strip()
    return s or None


def resolve(q: dict, d: Directory) -> tuple[dict, list[int]]:
    """The query with every field checked against the table's lists, and the ids of the player it names.

    Raises Invalid with a sentence that can be shown as it is.
    """
    if not isinstance(q, dict) or set(q) - set(FIELDS):
        raise Invalid("the translation was not a query the ask box knows how to run.")
    q = {f: q.get(f) for f in FIELDS}
    tricodes, labels = {t for t, _ in d.teams}, d.season_labels
    situations, outcomes = {i for i, _ in d.situations}, {i for i, _ in d.outcomes} | {PLAYER_OUTCOME}
    if q["season"] is None:
        q["season"] = season_label(d.latest)
    if q["season"] not in labels:
        raise Invalid(f"the table holds {_listing(labels)} and the question asks about another season.")
    if q["situation"] not in situations or q["outcome"] not in outcomes or q["comparison"] not in COMPARISONS:
        raise Invalid("the translation named a situation, an outcome or a comparison that is not in the table.")
    if q["team"] is not None and q["team"] not in tricodes:
        raise Invalid("the translation named a team that is not in the table.")
    q["player"] = _clean_name(q["player"])
    ids: list[int] = []
    if q["player"] is not None:
        if q["outcome"] != PLAYER_OUTCOME:
            raise Invalid("a player's rows hold one thing only: the share of the team's possessions the player finishes while on the floor, in one situation.")
        found = d.match_players(q["player"])
        if not found:
            raise Invalid("no player by that name is in the table.")
        if len({n for _, n in found}) > 1:
            names = sorted({n for _, n in found})
            raise Invalid(f"more than one player fits that name ({_listing(names[:5])}{' and others' if len(names) > 5 else ''}); ask again with the full name.")
        q["player"], ids = found[0][1], [i for i, _ in found]
        if q["comparison"] == "team":
            raise Invalid("a player's row can be set against its baseline, the league or another season, not against a team.")
    elif q["outcome"] == PLAYER_OUTCOME and q["comparison"] != "league":
        raise Invalid("the share of finishes belongs to a player, and the question names none.")
    if q["team"] is None and q["player"] is None and q["comparison"] != "league":
        raise Invalid("the question names no team and no player, and only a ranking across the league can do without one.")
    if q["comparison"] == "team":
        if q["compare_to"] not in tricodes or q["compare_to"] == q["team"]:
            raise Invalid("comparing two teams needs a second team that is in the table.")
    elif q["comparison"] == "season":
        if q["compare_to"] not in labels or q["compare_to"] == q["season"]:
            raise Invalid(f"comparing two seasons needs a second season, and the table holds {_listing(labels)}.")
    else:
        q["compare_to"] = None
    return q, ids


COLS = ("season, scope, tricode, team_name, player_id, player_name, situation, situation_label, outcome, outcome_label, unit, k, n, estimated, "
        "baseline_is, baseline_rate, raw_rate, deviation, dev_lo, dev_hi, excludes_zero, rate, rate_lo, rate_hi")


def build_sql(q: dict, ids: list[int], d: Directory) -> tuple[str, list]:
    """One SELECT over `cells` for a resolved query, with its parameters. Values never enter the SQL text."""
    year = {lab: s for lab, s in zip(d.season_labels, d.seasons)}
    scope = "player" if q["outcome"] == PLAYER_OUTCOME else "team"
    where, params = ["scope = ?", "situation = ?", "outcome = ?"], [scope, q["situation"], q["outcome"]]
    marks = lambda xs: ", ".join("?" for _ in xs)                                    # noqa: E731
    if q["comparison"] == "league":
        where += ["season = ?", "estimated"]
        params += [year[q["season"]]]
        return (f"SELECT {COLS}, rank() OVER (ORDER BY rate DESC) AS rank, count(*) OVER () AS ranked FROM cells "
                f"WHERE {' AND '.join(where)} ORDER BY rate DESC, tricode, player_name"), params
    seasons = [q["season"]] + ([q["compare_to"]] if q["comparison"] == "season" else [])
    where.append(f"season IN ({marks(seasons)})")
    params += [year[s] for s in seasons]
    teams = [t for t in ([q["team"]] + ([q["compare_to"]] if q["comparison"] == "team" else [])) if t]
    if teams:
        where.append(f"tricode IN ({marks(teams)})")
        params += teams
    if ids:
        where.append(f"player_id IN ({marks(ids)})")
        params += ids
    return f"SELECT {COLS} FROM cells WHERE {' AND '.join(where)} ORDER BY n DESC, season DESC, tricode", params


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _row(r: dict) -> dict:
    """A table row as the response carries it."""
    est = bool(r["estimated"])
    out = {"season": season_label(r["season"]), "scope": r["scope"], "team": r["tricode"], "team_name": r["team_name"],
           "player": r["player_name"] if r["scope"] == "player" else None, "situation": r["situation"], "situation_label": r["situation_label"],
           "outcome": r["outcome"], "outcome_label": r["outcome_label"], "unit": r["unit"], "k": int(r["k"]), "n": int(r["n"]),
           "estimated": est, "raw_rate": _num(r["raw_rate"])}
    if est:
        out.update({"baseline_rate": r["baseline_rate"], "baseline_is": r["baseline_is"], "deviation": r["deviation"],
                    "interval": [r["dev_lo"], r["dev_hi"]], "excludes_zero": bool(r["excludes_zero"]), "rate": r["rate"],
                    "rate_interval": [r["rate_lo"], r["rate_hi"]]})
    if r.get("rank") is not None:
        out.update({"rank": int(r["rank"]), "ranked": int(r["ranked"])})
    return out


# ---------- the sentence -------------------------------------------------------------------------

def _listing(xs) -> str:
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _own(s: str) -> str:
    return s + ("'" if s.endswith("s") else "'s")


def _lower(label: str) -> str:
    return label[:1].lower() + label[1:]


def _pc(x: float) -> str:
    return f"{_fixed1(x * 100)}%"


def _signed(x: float) -> str:
    t = _points(x)
    return ("-" if x < 0 and float(t) != 0 else "") + t


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _phrase(r: dict) -> str:
    return PHRASE.get(r["situation"], _lower(r["situation_label"]))


def _who(r: dict) -> str:
    return r["player"] if r["scope"] == "player" else f"the {r['team_name']}"


LEVEL = f"{round(INTERVAL_LEVEL * 100)}%"


def _rate(r: dict) -> str:
    return f"{_pc(r['rate'])} ({LEVEL} interval {_pc(r['rate_interval'][0])} to {_pc(r['rate_interval'][1])}, {r['n']:,} {r['unit']})"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _thin(r: dict) -> str:
    return (f"{_cap(_own(_who(r)))} {_lower(r['outcome_label'])}, {_phrase(r)} in {r['season']}, "
            f"rests on {r['k']:,} of {r['n']:,} {r['unit']} ({_pc(r['raw_rate'])}), too few for the table to estimate a tendency.")


def _overlaps(mine: dict, rows: list[dict]) -> int:
    lo, hi = mine["rate_interval"]
    return sum(1 for r in rows if r is not mine and r["rate_interval"][0] <= hi and r["rate_interval"][1] >= lo)


def difference(a: dict, b: dict) -> dict:
    """a's rate minus b's with an interval. The two rows are treated as independent."""
    sd = [(r["rate_interval"][1] - r["rate_interval"][0]) / (2 * Z) for r in (a, b)]
    d, s = a["rate"] - b["rate"], math.hypot(*sd)
    lo, hi = d - Z * s, d + Z * s
    return {"value": round(d, 4), "interval": [round(lo, 4), round(hi, 4)], "excludes_zero": bool(lo > 0 or hi < 0)}


def _no_row(q: dict, d: Directory) -> str:
    who = q["player"] or (f"the {d.team_name(q['team'])}" if q["team"] else "any team")
    out = _lower(dict(d.outcomes).get(q["outcome"], "share of finishes while on the floor"))
    hint = ""
    if q["outcome"] == PLAYER_OUTCOME and q["situation"] == "all":
        hint = " A player's row exists only inside a situation, such as after a timeout or in the fourth quarter."
    elif q["player"] and q["team"]:
        hint = " The player may have no possessions with that team in that season."
    return (OPENING + f"the table has no row for {who}, {out}, {PHRASE.get(q['situation'], q['situation'])} in "
            f"{_listing([q['season']] + ([q['compare_to']] if q['comparison'] == 'season' else []))}.{hint}")


def compose(q: dict, rows: list[dict], d: Directory) -> tuple[str, str, dict | None]:
    """(status, the one sentence, the difference block when two rows are compared). Written from `rows` and nothing else."""
    if not rows:
        return "no_row", _no_row(q, d), None
    cmp_ = q["comparison"]
    if cmp_ == "baseline":
        r = rows[0]
        if not r["estimated"]:
            return "no_estimate", _thin(r), None
        zero = "" if r["excludes_zero"] else ", so zero is not ruled out"
        more = f" This is the larger of {len(rows)} rows for the player." if len(rows) > 1 else ""
        return "answered", (f"In {r['season']}, {_phrase(r)}, {_own(_who(r))} {_lower(r['outcome_label'])} is {_pc(r['raw_rate'])} on {r['n']:,} {r['unit']} "
                            f"against a baseline of {_pc(r['baseline_rate'])} ({r['baseline_is']}); after pooling that is "
                            f"{_points(r['deviation'])} points {'above' if r['deviation'] >= 0 else 'below'} baseline "
                            f"({LEVEL} interval {_interval(*r['interval'])}{zero}).{more}"), None
    if cmp_ == "league":
        kind = "players with a row" if q["outcome"] == PLAYER_OUTCOME else "teams"
        asked = q["player"] or q["team"]
        if asked:
            mine = next((r for r in rows if (r["player"] == q["player"] if q["player"] else r["team"] == q["team"])), None)
            if mine is None:
                return "no_row", _no_row(q, d), None
            verb = "ranks" if q["player"] else "rank"
            return "answered", (f"In {mine['season']}, {_phrase(mine)}, {_who(mine)} {verb} {_ordinal(mine['rank'])} of {mine['ranked']} {kind} in "
                                f"{_lower(mine['outcome_label'])} at {_rate(mine)}; the intervals of {_overlaps(mine, rows)} others overlap it."), None
        top, low = rows[0], rows[-1]
        if q["outcome"] == PLAYER_OUTCOME:
            return "answered", (f"In {top['season']}, {_phrase(top)}, of {top['ranked']} {kind} {top['player']} ({top['team']}) has the highest "
                                f"{_lower(top['outcome_label'])} at {_rate(top)}; the intervals of {_overlaps(top, rows)} others overlap it."), None
        return "answered", (f"In {top['season']}, {_phrase(top)}, of {top['ranked']} teams the {top['team_name']} have the highest "
                            f"{_lower(top['outcome_label'])} at {_rate(top)} and the {low['team_name']} the lowest at {_rate(low)}; "
                            f"the intervals of {_overlaps(top, rows)} others overlap the highest."), None
    # two rows side by side
    if cmp_ == "team":
        pick = lambda key: next((r for r in rows if r["team"] == key), None)        # noqa: E731
        a, b, names = pick(q["team"]), pick(q["compare_to"]), ("team", "team_name")
    else:
        pick = lambda key: next((r for r in rows if r["season"] == key), None)      # noqa: E731
        a, b, names = pick(q["season"]), pick(q["compare_to"]), ("season", "season")
    if a is None or b is None:
        return "no_row", _no_row(q, d), None
    thin = next((r for r in (a, b) if not r["estimated"]), None)
    if thin is not None:
        return "no_estimate", _thin(thin), None
    diff = difference(a, b)
    tail = (f"the difference, {a[names[0]]} minus {b[names[0]]}, is {_signed(diff['value'])} points ({LEVEL} interval {_signed(diff['interval'][0])} to "
            f"{_signed(diff['interval'][1])}), which {'rules out' if diff['excludes_zero'] else 'does not rule out'} zero.")
    if cmp_ == "team":
        return "answered", (f"In {a['season']}, {_phrase(a)}, {_lower(a['outcome_label'])} is {_rate(a)} for the {a['team_name']} and {_rate(b)} "
                            f"for the {b['team_name']}; {tail}"), diff
    return "answered", (f"{_cap(_own(_who(a)))} {_lower(a['outcome_label'])}, {_phrase(a)}, is {_rate(a)} in "
                        f"{a['season']} and {_rate(b)} in {b['season']}; {tail}"), diff


# ---------- translators --------------------------------------------------------------------------

def prompt_text() -> str:
    return PROMPT_PATH.read_text()


def prompt_sha() -> str:
    return hashlib.sha256(prompt_text().encode()).hexdigest()[:16]


def lists_text(d: Directory) -> str:
    """The table's own lists, appended to the prompt. The model is given these and the question, nothing else."""
    cur = season_label(d.latest)
    lines = ["The lists", "", "Seasons: " + ", ".join(d.season_labels) + f". The current season is {cur}.", "", "Teams (tricode: name):"]
    lines += [f"- {t}: {n}" for t, n in d.teams]
    lines += ["", "Situations (id: what it means):"] + [f"- {i}: {lab}" for i, lab in d.situations]
    lines += ["", "Outcomes (id: what it means):"] + [f"- {i}: {lab}" for i, lab in d.outcomes]
    lines += [f"- {PLAYER_OUTCOME}: A player's share of the team's possessions finished while on the floor (player rows only)"]
    return "\n".join(lines)


def tools(d: Directory) -> list[dict]:
    """The two tools. `strict` makes the API hold the arguments to these schemas, so an enum value cannot be invented."""
    opt = lambda enum: {"anyOf": [{"type": "string", "enum": list(enum)}, {"type": "null"}]}     # noqa: E731
    tricodes = [t for t, _ in d.teams]
    return [
        {"name": "run_query", "strict": True,
         "description": "Look up one row of the tendency table, or one row set against the league, another team or another season.",
         "input_schema": {"type": "object", "additionalProperties": False, "required": list(FIELDS), "properties": {
             "team": opt(tricodes),
             "player": {"anyOf": [{"type": "string"}, {"type": "null"}]},
             "situation": {"type": "string", "enum": [i for i, _ in d.situations]},
             "outcome": {"type": "string", "enum": [i for i, _ in d.outcomes] + [PLAYER_OUTCOME]},
             "season": {"type": "string", "enum": d.season_labels},
             "comparison": {"type": "string", "enum": list(COMPARISONS)},
             "compare_to": opt(tricodes + d.season_labels)}}},
        {"name": "cannot_answer", "strict": True,
         "description": "Say that no single lookup in the table answers the question, and why.",
         "input_schema": {"type": "object", "additionalProperties": False, "required": ["reason"], "properties": {
             "reason": {"type": "string", "enum": list(REASONS)}}}},
    ]


def request(question: str, d: Directory, model: str = MODEL, effort: str = EFFORT, nudge: bool = False) -> dict:
    """Everything sent to the API for one question: the prompt, the table's lists, the two tools and the question."""
    content = question if not nudge else question + "\n\n(Call run_query or cannot_answer. Do not reply in words.)"
    return {"model": model, "max_tokens": MAX_TOKENS, "system": prompt_text() + "\n" + lists_text(d), "tools": tools(d),
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},     # this model family turns down a forced tool call
            "output_config": {"effort": effort}, "messages": [{"role": "user", "content": content}]}


def client():
    from . import summaries
    try:
        return summaries.client()
    except summaries.SummaryError as e:
        raise AskError(str(e)) from e


def _translation(name: str, args) -> dict | None:
    if name == "cannot_answer" and isinstance(args, dict) and args.get("reason") in REASONS:
        return {"kind": "refuse", "reason": args["reason"]}
    if name == "run_query" and isinstance(args, dict):
        return {"kind": "query", "query": {f: args.get(f) for f in FIELDS}}
    return None


class LLMTranslator:
    kind = "llm"

    def __init__(self, d: Directory, cl=None, model: str | None = None, effort: str | None = None):
        self.d, self.cl, self.model, self.effort = d, cl or client(), model or MODEL, effort or EFFORT
        self.calls = 0

    def translate(self, question: str) -> dict:
        import anthropic
        for attempt in range(MAX_ATTEMPTS):
            self.calls += 1
            try:
                resp = self.cl.messages.create(**request(question, self.d, self.model, self.effort, nudge=attempt > 0))
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                raise AskError(f"the API key was not accepted: {e}") from e
            except anthropic.APIStatusError as e:
                raise AskError(f"the API turned the request down ({e.status_code}): {e}") from e
            except anthropic.APIConnectionError as e:
                raise AskError(f"could not reach the API: {e}") from e
            if resp.stop_reason == "refusal":
                raise AskError("the model declined to translate the question")
            call = next((b for b in resp.content if b.type == "tool_use"), None)
            got = _translation(call.name, call.input) if call is not None else None
            if got is not None:
                return {**got, "by": {"kind": "llm", "model": resp.model, "prompt_sha": prompt_sha()}}
        raise AskError("the model did not call a tool, so there is no query to run")       # whatever it wrote instead is dropped


# Keyword lists for the baseline. Checked in this order; the first list with a hit decides.
_REFUSE = [
    ("play_call", r"play call|what play|which play|plays? (do|does) .* (run|call)|call(s|ed)? a (set|play)|set play|playbook|horns|formation"),
    ("coverage", r"coverage|\bdrop\b|\bswitch|\bzone\b|blitz|\bhedge|\btrap|who guards|guarding|matchup|put on the"),
    ("injury", r"injur|\bhurt\b|healthy|health\b|sidelined|availab|out for the"),
    ("forecast_or_opinion", r"\bwill\b|going to|tonight|tomorrow|\bshould\b|^why\b|\bwhy do|\bbest\b|predict|\bodds\b|\bbet\b"),
    ("outside_data", r"payroll|salary|contract|\btrade|\bdraft|college|wnba|nfl|\bmvp\b|weather|forget the|ignore (the|your|all)"),
    ("not_in_table", r"points per game|per game|averag|percentage|defens(e|ive) (rating|ranking)|\ballow|rebounds per|\bthrees\b(?<!corner threes)"),
]
_SITUATION = [
    ("two_for_one", r"two[- ]for[- ]one|2[- ]for[- ]1|30 to 45 s"),
    ("after_timeout", r"after (a )?time ?outs?|out of (a )?time ?outs?|\bato\b"),
    ("clutch", r"clutch|last (five|5) minutes|crunch time|late in close games|close games late"),
    ("start_def_rebound", r"defensive rebounds?|off (a )?rebounds?|after a stop"),
    ("start_turnover", r"off (a )?turnovers?|after (a )?(steal|turnover)s?|off (a )?steals?"),
    ("start_made_basket", r"after (an |the )?(opponent|other team) scores?|after (a )?made baskets?|after being scored on|after makes"),
    ("def_in_penalty", r"penalty|\bbonus\b"),
    ("q4", r"fourth quarter|4th quarter|\bq4\b|in the fourth"),
]
_OUTCOME = [
    ("shot_family:corner3", r"corner[- ](three|3)s?"),
    ("shot_family:above_break3", r"above[- ]the[- ]break"),
    ("shot_family:rim", r"\brim\b|at the basket"),
    ("shot_family:paint", r"\bpaint\b"),
    ("shot_family:midrange", r"mid-?range"),
    ("action_style:step_back", r"step-? ?backs?"),
    ("action_style:pullup", r"pull-? ?ups?"),
    ("action_style:floater", r"floaters?"),
    ("action_style:putback", r"put-? ?backs?|tip-? ?ins?"),
    ("action_style:cut_lob", r"\bcuts?\b|\blobs?\b|alley"),
    ("action_style:turnaround_hook", r"hooks?\b|turnaround"),
    ("action_style:running", r"running (shot|jumper)"),
    ("action_style:drive", r"\bdriv(e|es|ing)\b"),
    ("fast_break:fast_break", r"fast[- ]breaks?|transition"),
    ("creation:assisted", r"assist"),
    ("ending_type:turnover", r"turn(s|ed)? (it|the ball) over|turnovers?"),
    ("ending_type:foul_drawn", r"fouls?\b|free throws?|to the line"),
    ("tempo:early", r"\bearly\b|quick(ly)?\b|push the pace"),
    ("tempo:late", r"\blate\b|walk it up|slow(ly)?\b"),
    ("ending_type:shot", r"end(s|ing)? in a shot|get a shot"),
]
_LEAGUE = r"\brank|\bmost\b|\bleast\b|\bfewest\b|highest|lowest|which team|^who\b|\bleague\b"


class RulesTranslator:
    """The baseline: keyword matching against the table's own lists. No model, no network, and no judgment."""
    kind = "rules"

    def __init__(self, d: Directory):
        self.d = d
        alias: dict[str, set[str]] = {}
        for tri, name in d.teams:
            words = name.split()
            for a in {name, words[-1], " ".join(words[:-1]), " ".join(words[1:])}:
                if norm(a):
                    alias.setdefault(norm(a), set()).add(tri)
        self.alias = {a: next(iter(t)) for a, t in alias.items() if len(t) == 1}      # a word two teams share names neither
        self.tricodes = {t for t, _ in d.teams}
        self.calls = 0

    def _teams(self, text: str, raw: str) -> list[str]:
        hits = [(m.start(), self.alias[a]) for a in self.alias for m in re.finditer(rf"\b{re.escape(a)}\b", text)]
        hits += [(m.start(), m.group()) for m in re.finditer(r"\b[A-Z]{3}\b", raw) if m.group() in self.tricodes]
        out = []
        for _, t in sorted(hits):
            if t not in out:
                out.append(t)
        return out

    def _player(self, text: str) -> str | None:
        full = [n for _, n in self.d.players if re.search(rf"\b{re.escape(norm(n))}\b", text)]
        if full:
            return max(full, key=len)
        last = {}
        for _, n in self.d.players:
            last.setdefault(norm(n).split()[-1], set()).add(n)
        for w, names in last.items():
            if len(names) == 1 and re.search(rf"\b{re.escape(w)}\b", text):
                return next(iter(names))
        return None

    def translate(self, question: str) -> dict:
        by = {"kind": "rules", "model": None, "prompt_sha": None}
        text = norm(question)
        low = question.lower()
        for reason, pat in _REFUSE:
            if re.search(pat, low):
                return {"kind": "refuse", "reason": reason, "by": by}
        situation = "all"
        for sid, pat in _SITUATION:
            m = re.search(pat, low)
            if m:
                situation, low = sid, low[:m.start()] + " " + low[m.end():]            # its words are spent: "off turnovers" is not also an outcome
                break
        player = self._player(text)
        outcome = PLAYER_OUTCOME if player else next((oid for oid, pat in _OUTCOME if re.search(pat, low)), None)
        if outcome is None and re.search(r"\bfinish", low):
            outcome = PLAYER_OUTCOME
        if outcome is None:
            return {"kind": "refuse", "reason": "not_in_table", "by": by}
        teams = self._teams(text, question)
        labels, latest = self.d.season_labels, season_label(self.d.latest)
        named = [season_label(int(y)) for y in re.findall(r"\b(20\d\d)[-–/]\d\d\b", question)]
        if re.search(r"last (season|year)|previous season", low) and len(self.d.seasons) > 1:
            named.append(labels[-2])
        if re.search(r"this (season|year)", low) and named:
            named.append(latest)
        named = [s for i, s in enumerate(named) if s not in named[:i]]
        season = named[0] if named else latest
        comparison, compare_to = "baseline", None
        if len(named) > 1:
            comparison, compare_to = "season", named[1]
        elif len(teams) > 1 and not player:
            comparison, compare_to = "team", teams[1]
        elif re.search(_LEAGUE, low) or not (teams or player):
            comparison = "league"
        team = teams[0] if teams and not player else None
        return {"kind": "query", "by": by, "query": {"team": team, "player": player, "situation": situation, "outcome": outcome,
                                                      "season": season, "comparison": comparison, "compare_to": compare_to}}


# ---------- one question, start to finish --------------------------------------------------------

def ask(question: str, table: Table, translator) -> dict:
    """The whole response for one question: who translated it, the query, the SQL, the rows and the sentence."""
    from . import CONTRACT_VERSION
    out = {"contract_version": CONTRACT_VERSION, "data_kind": table.data_kind, "as_of_utc": table.as_of, "question": question,
           "status": "failed", "translated_by": {"kind": translator.kind, "model": getattr(translator, "model", None), "prompt_sha": None},
           "query": None, "refusal": None, "sql": None, "interval_level": INTERVAL_LEVEL, "n_rows": 0, "rows": [], "difference": None, "answer": ""}
    try:
        got = translator.translate(question)
    except AskError as e:
        out["answer"] = f"The question could not be turned into a query: {e}."
        return out
    out["translated_by"] = got["by"]
    if got["kind"] == "refuse":
        out.update({"status": "refused", "refusal": {"reason": got["reason"]}, "answer": OPENING + REASONS[got["reason"]]})
        return out
    try:
        q, ids = resolve(got["query"], table)
    except Invalid as e:
        out["answer"] = OPENING + str(e)
        return out
    sql, params = build_sql(q, ids, table)
    rows = [_row(r) for r in table.query(sql, params)]
    status, sentence, diff = compose(q, rows, table)
    shown = rows
    if q["comparison"] == "league" and q["outcome"] == PLAYER_OUTCOME:                 # hundreds of players: the top ten and the one asked about
        shown = [r for r in rows if r["rank"] <= TOP_PLAYERS or r["player"] == q["player"]]
    out.update({"status": status, "query": q, "sql": {"text": sql, "params": params}, "n_rows": len(rows), "rows": shown[:MAX_ROWS],
                "difference": diff, "answer": sentence})
    return out


# ---------- accuracy on the thirty questions -----------------------------------------------------

def _same(got, want) -> bool:
    ok = want if isinstance(want, list) else [want]
    return any((norm(got) == norm(w)) if isinstance(got, str) and isinstance(w, str) else got == w for w in ok)


def score(got: dict, expect: dict) -> tuple[bool, str]:
    """(right, what was wrong). A query is right when every field matches; a refusal when it refuses for an accepted reason."""
    if expect["kind"] == "refuse":
        if got["kind"] != "refuse":
            return False, "ran a query for a question that had to be refused"
        return (True, "") if got["reason"] in expect["reasons"] else (False, f"refused, but as {got['reason']}")
    if got["kind"] != "query":
        return False, f"refused ({got['reason']}) a question the table can answer"
    q = dict(got["query"])
    for a, b in (("team", "compare_to"), ("season", "compare_to")):                    # a pair being compared may come in either order
        if q.get("comparison") == a and expect.get("comparison") == a and _same(q.get(b), expect[a]) and _same(q.get(a), expect[b]):
            q[a], q[b] = q[b], q[a]
    wrong = [f for f in FIELDS if not _same(q.get(f), expect.get(f))]
    return (not wrong), ("" if not wrong else "wrong " + ", ".join(f"{f} ({q.get(f)!r})" for f in wrong))


def evaluate(translator, questions: list[dict], log=None) -> dict:
    """Translate every question and score it. Nothing is looked up: this measures translation only."""
    items = []
    for x in questions:
        try:
            got = translator.translate(x["question"])
            right, why = score(got, x["expect"])
        except AskError as e:
            got, right, why = {"kind": "error"}, False, f"no translation: {e}"
        items.append({"id": x["id"], "question": x["question"], "expect": x["expect"]["kind"], "right": right, "why": why,
                      "got": {k: v for k, v in got.items() if k != "by"}})
        if log:
            log(f"  {'ok ' if right else 'NO '} {x['id']:>2}  {x['question']}" + (f"\n         {why}" if why else ""))
    ask_ = [i for i in items if i["expect"] == "query"]
    ref = [i for i in items if i["expect"] == "refuse"]
    return {"translator": translator.kind, "model": getattr(translator, "model", None), "n": len(items), "right": sum(i["right"] for i in items),
            "queries": {"n": len(ask_), "right": sum(i["right"] for i in ask_),
                        "refused_wrongly": sum(i["got"].get("kind") == "refuse" for i in ask_)},
            "refusals": {"n": len(ref), "right": sum(i["right"] for i in ref), "refused": sum(i["got"].get("kind") == "refuse" for i in ref),
                         "answered_wrongly": sum(i["got"].get("kind") == "query" for i in ref)},
            "calls": getattr(translator, "calls", None), "items": items}


def describe(rep: dict) -> str:
    who = "keyword rules (no model)" if rep["translator"] == "rules" else f"{rep['model']}"
    q, r = rep["queries"], rep["refusals"]
    return (f"{who}: {rep['right']} of {rep['n']} right. Questions the table can answer: {q['right']} of {q['n']} became the expected query "
            f"({q['refused_wrongly']} refused by mistake). Questions that must be refused: {r['refused']} of {r['n']} refused, "
            f"{r['right']} for an accepted reason; {r['answered_wrongly']} got a query and an answer they should not have.")
