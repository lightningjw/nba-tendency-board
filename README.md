# Tendency Board

[![CI](https://github.com/lightningjw/nba-tendency-board/actions/workflows/ci.yml/badge.svg)](https://github.com/lightningjw/nba-tendency-board/actions/workflows/ci.yml)

Next-possession predictions and scouting tendencies from public NBA play-by-play, with a static web app on top.
Two halves meet at one data contract (`contract/schemas/`): the Python pipeline writes JSON that validates against
it, and the app validates every file against the same schemas before rendering.

**State of this build:** runs end to end on synthetic games (bundled in `app/data`) and on three real regular
seasons loaded from a public GitHub mirror of the feed (3,690 games, 738,771 possessions). On real held-out
games and a held-out season, all three heads beat their baselines by small, clearly non-zero margins; see
"Real results" below. The direct cdn.nba.com path (`run`, `ingest`) is still untested. Details in
`ASSUMPTIONS.md`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                  # about 20 s
python -m tendency sample               # synthetic games -> app/data (no network)
python -m http.server 8000 -d app       # open http://localhost:8000
```

## Real data, one command

```bash
python -m tendency bulk --seasons 2023 2024 2025
```

That downloads whole seasons from `shufinskiy/nba_data` (about 19 MB each, plus 4 MB for full player names), rebuilds per-game files, builds
possessions, features and models, and writes validated contract JSON to `app/data`. Three seasons take about
five minutes and 4 GB of RAM. Reload the app; no front-end change is needed. Add `--out data/real_export` to
keep the bundled synthetic sample in place. Archives exist from 2020 and are refreshed by their maintainer, not
nightly.

For a date range straight from the CDN (untested, and the CDN may refuse the request):

```bash
python -m tendency run --start 2026-10-20 --end 2026-11-15
```

Raw files land in `data/real/raw/` and are never re-downloaded. `data/` is gitignored: do not publish a real
export until you have read NBA.com's terms.

Before trusting it, check the assumed feed names against one game (about a minute):

```bash
python -m tendency ingest --start 2026-11-14 --end 2026-11-14
python -m tendency verify-fields data/real/raw/pbp/<gameId>.json data/real/raw/box/<gameId>.json
```

Fix anything it flags in `tendency/cdn_fields.py` (the only file that names feed fields), then pass
`--fields-verified` to `run` or `build` so the app drops its "not checked" notice.

To read one game by hand, possession by possession beside the raw actions:

```bash
python -m tendency inspect 0022500001              # every possession
python -m tendency inspect 0022500001 --only odd   # and-ones, heaves, technicals, anything unexplained
```

## Prospective log

The scorecard's retrospective splits were computed after the games were played. The prospective log is the
stricter test: freeze a model, write a file for each game before tip-off that names the model by SHA-256, commit
it, and score that same model once the game is final. The commit time is the proof.

What a pregame file can honestly hold is limited: the score, the fouls and the five on the floor do not exist
yet. So `predictions/YYYY-MM-DD/<gameId>.json` holds each team's projected five, its baseline shares, and the
model's probabilities for four fixed reference situations. Possession-by-possession predictions are made after
the game by the frozen model and scored against the baseline. Design and limits: `DECISIONS.md` 57-71.

Once, and again whenever you want a fresher model:

```bash
python -m tendency train --through 2026-11-20      # freezes data/real/models/through-2026-11-20 (not in git; keep it)
```

Every day, before the first tip (the two commands):

```bash
python -m tendency score                           # yesterday: score the frozen model on the games logged for it
python -m tendency predict                         # today: one pregame file per scheduled game
```

Dates are US Eastern and default to yesterday and today; pass `--date YYYY-MM-DD` to override. `predict` needs
the day's schedule in `data/real/schedule/<date>.json` and `score` needs the finished games on disk. Until the
mirror carries the current season, both come from `scripts/fetch_nba_api.py`, which must run on your own machine
(nba.com refuses cloud hosts) and has never been run against nba.com: read its header first.
`scripts/nightly.sh` chains fetch, score and predict. A crontab line for a Mac in Chicago, 9:30 am, which is
after the last West Coast game is final and before the earliest noon-Eastern tip:

```
30 9 * * * cd $HOME/code/nba-tendency-board && PUBLISH=none ./scripts/nightly.sh >> data/nightly.log 2>&1
```

`PUBLISH=none` commits nothing; `sums` commits only each date's `SHA256SUMS` (timestamps the log without
publishing names or numbers); `full` commits the per-game files. Which one is a terms decision, not a technical
one. cron does not run while the laptop sleeps: a log written after tip-off is kept, counted as late, and left
out of the prospective numbers.

`score` refuses any model but the one the log names (by hash), refuses a model trained through a date on or
after the game, and refuses a pregame file that no longer matches its checksum. To rehearse the whole flow on
games already on disk, add `--replay` to `predict` and point `--log-dir` somewhere outside `predictions/`; a
replay is marked as such in every file and the app labels it "not prospective evidence".

```bash
python -m tendency build --root data/real --out data/real_export --predictions predictions   # export with the log
```

Other commands: `python -m tendency build` (everything after ingest, from files on disk),
`python -m tendency validate app/data`, `node app/tools/check-contract.mjs` (same check with the app's validator).

Exit codes: 0 ok, 2 pipeline or data error, 3 the host blocked the request (run from a home connection; there is
no fallback by design).

Notes on ranges: the schedule file covers the current season only. One season gives a held-out-games split and
an explicit "not available" for the held-out season. You need at least 8 training games; a few hundred is where
the scorecard starts to mean something.

## Layout

```
contract/schemas/     the data contract (six JSON Schemas)
tendency/             pipeline package
  cdn_fields.py       every assumed feed name, URL and rule of thumb
  ingest.py           fetch raw JSON as landed; idempotent; backoff; clear stop on a blocked host
  possessions.py      one row per possession; edge-case rules in the module docstring
  inspect.py          one game's possessions beside the raw actions, with a reconciliation summary
  prospective.py      freeze a model, log before tip-off, score after; the scorecard's prospective block
  features.py         context / pregame / in-game features and the baselines (DuckDB)
  models.py           splits, LightGBM per head, log-loss with game-bootstrap intervals, reliability
  tendencies.py       simple shrinkage estimate for the team report (labeled as such)
  export.py           writes and validates the contract JSON
  bulk.py             whole seasons from the GitHub mirror; derives the boxscore the archive lacks
  teams.py            the 30 franchises by team id (city, nickname)
  synth.py            synthetic raw games in the assumed feed shape
  verify_fields.py    checks assumptions against a real game
  stubs/              named placeholders for everything outside the MVP
tests/                pytest: segmentation edge cases, leakage, ingest, models, contract
app/                  static front end (no build step); app/data is the bundled sample export
```

## The app

Four views: Method (what it does and its limits), Team report, Replay with "guess before the model", Scorecard.
It is a plain static site: host the `app/` folder anywhere (GitHub Pages works). Served over HTTPS it installs
to a phone home screen and works offline after the first load.

Sample data is synthetic and says so on every page. The banner is driven by `data_kind` in the contract, and the
pipeline sets that from a marker inside the raw files, so synthetic output cannot be exported as real.

## Synthetic data, and what it does not show

`python -m tendency sample` simulates 224 games between eight placeholder teams over two fake seasons (2090 and
2091). A few situational quirks are planted (`QUIRKS` in `tendency/synth.py`) so the team report has ground truth
to recover; a test checks that it does. Scorecard numbers on this data show that the evaluation code runs. They
say nothing about whether the models beat their baselines on NBA games.

## Real results (2026-11-16 run)

Trained on 1,968 games from 2023-24 and 2024-25. Log-loss, lower is better; intervals are 95% from a bootstrap
over whole games.

| Head | Split | Possessions | Model | Baseline | Difference (95% interval) |
|---|---|---|---|---|---|
| Finisher | late 2024-25, 492 games | 96,272 | 1.5675 | 1.5715 | -0.0040 (-0.0045 to -0.0035) |
| Finisher | 2025-26, 1,230 games | 242,591 | 1.5668 | 1.5711 | -0.0043 (-0.0046 to -0.0040) |
| Ending type | late 2024-25 | 97,073 | 0.6859 | 0.6970 | -0.0112 (-0.0122 to -0.0101) |
| Ending type | 2025-26 | 244,348 | 0.7077 | 0.7196 | -0.0118 (-0.0125 to -0.0111) |
| Shot family | late 2024-25 | 74,547 | 1.4900 | 1.5119 | -0.0219 (-0.0237 to -0.0201) |
| Shot family | 2025-26 | 184,754 | 1.4933 | 1.5122 | -0.0189 (-0.0200 to -0.0179) |

Read these as small. The finisher model picks the right player 29.0% of the time against the baseline's 28.7%.
The lift is real and consistent across both splits, and it is modest, which is what the brief predicted. The
held-out season is retrospective, not prospective: the scorecard becomes prospective only once nightly
predictions are logged before games are played.
