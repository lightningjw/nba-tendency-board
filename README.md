# Tendency Board

[![CI](https://github.com/lightningjw/nba-tendency-board/actions/workflows/ci.yml/badge.svg)](https://github.com/lightningjw/nba-tendency-board/actions/workflows/ci.yml)

Tendency Board predicts five things about the next NBA possession from public play-by-play: who finishes it,
how it ends, where the shot comes from, what kind of shot it is, and how early in the possession it comes. Each
prediction is scored against a simple baseline on games the model never saw, and the result is shown with its uncertainty whether it flatters the model or not. A tendency
layer turns the same possessions into scouting-style deviations ("more corner threes after timeouts than this
team's other possessions predict"), each with an interval and a sample size. A static web app sits on top: a
scorecard, team tendency reports, a hot-hand report, and a possession-by-possession replay.

**Live app:** https://nba-tendency-board-app.vercel.app/ (simulated games with placeholder teams; real results
are in the table below, and [docs/terms-check.md](docs/terms-check.md) explains why real play-by-play is not
published).

![Team report on the bundled synthetic sample: each tendency is a deviation from baseline with an interval and a sample size](docs/screenshot-team-report.png)

## Results on real games

Three regular seasons, 2023-24 to 2025-26: 3,690 games, 738,812 possessions. Models trained on 1,968 games from
the first two seasons. Log-loss, lower is better; intervals are 95% from a bootstrap over whole games.

| Prediction | Held out | Possessions | Model | Baseline | Difference (95% interval) | Top pick right, model vs baseline |
|---|---|---|---|---|---|---|
| Who finishes | late 2024-25, 492 games | 96,302 | 1.5675 | 1.5716 | -0.0041 (-0.0046 to -0.0035) | 28.9% vs 28.7% |
| Who finishes | 2025-26, 1,230 games | 242,648 | 1.5668 | 1.5711 | -0.0043 (-0.0047 to -0.0040) | 29.0% vs 28.7% |
| How it ends | late 2024-25, 492 games | 97,103 | 0.6860 | 0.6970 | -0.0111 (-0.0121 to -0.0099) | 77.1% vs 76.8% |
| How it ends | 2025-26, 1,230 games | 244,406 | 0.7078 | 0.7194 | -0.0117 (-0.0124 to -0.0110) | 75.9% vs 75.6% |
| Where the shot comes from | late 2024-25, 492 games | 74,571 | 1.4782 | 1.5010 | -0.0228 (-0.0246 to -0.0211) | 34.6% vs 31.6% |
| Where the shot comes from | 2025-26, 1,230 games | 184,811 | 1.4801 | 1.4999 | -0.0197 (-0.0208 to -0.0187) | 34.9% vs 32.4% |
| What kind of shot it is | late 2024-25, 492 games | 74,571 | 1.9838 | 1.9977 | -0.0139 (-0.0151 to -0.0128) | 29.6% vs 29.2% |
| What kind of shot it is | 2025-26, 1,230 games | 184,811 | 1.9744 | 1.9878 | -0.0133 (-0.0141 to -0.0126) | 29.9% vs 29.7% |
| How early the first action comes | late 2024-25, 492 games | 94,836 | 0.9180 | 0.9439 | -0.0259 (-0.0276 to -0.0242) | 52.5% vs 50.8% |
| How early the first action comes | 2025-26, 1,230 games | 238,739 | 0.9221 | 0.9449 | -0.0228 (-0.0239 to -0.0217) | 52.2% vs 50.5% |

The baselines: usage shares among the five on the floor (finisher); the offense's season-to-date shares shrunk
toward the league's (ending type, shot location); the style mix of the five on the floor, shifted by how the
possession started (shot style, ten classes); and the offense's early / middle / late mix on possessions that
started the same way (tempo). All from games that tipped off earlier. The models are
gradient-boosted trees that start from the baseline's odds and learn corrections from clock, score, fouls,
timeouts, lineup and what has happened so far in the game.

**Read these as small.** Every interval is clear of zero and the two splits agree, so the lift is real. It is also
a fraction of a percentage point of top-pick accuracy for two of the three predictions. Next-possession outcomes
are mostly noise once you know who is on the floor, and next-shooter models are not new
([HoopsOracle](https://github.com/anovick1/HoopsOracle) does it from the same feed). What this project adds is
the checking: possession segmentation that reconciles to the final score in 3,690 of 3,690 games with 11
unexplained changes of possession in 738,812, leakage tests, baselines for every head, and intervals on every
number.

## What the feed lets us label

The last two rows of the table rest on labels added in step 2.2. Each is an outcome of the possession and is
defined on only part of them (738,812 possessions, three seasons):

| Label | Read from | Defined when | Share of possessions |
|---|---|---|---|
| Shot style (ten styles) | shot `descriptor` and `subType` | it ends in a shot that is not a heave | 75.4% |
| Assisted, and by whom | the assist on the basket | it ends in a made basket | 41.8% |
| Seconds to the first action | period clock | it has a shot, turnover or trip to the line | 99.0% |
| Tempo: early, middle, late | the same clocks | at least 24 s were left at the start | 96.4% |
| Fast break | the scorer's tag | it ends in a shot or at the line | 84.6% |
| Two-for-one taken | period clock | it starts with 30 to 45 s left in the period | 2.4% |

- **A style is the scorer's word for the shot** ("driving", "pullup", "cutting"), not the play that was called.
  32% of shots carry no word at all; a jump shot with none is 94% threes and 97% assisted when it goes in.
- **The first style baseline was too easy and was replaced.** Against the lineup's style mix alone the model
  won by 0.086. A baseline that also knows how the possession started took 85% of that away. The table shows
  the lift over the tougher one.
- **Tempo may only see what was known when the possession began.** Most context here is read just before the
  deciding event, and for tempo that leaks: a timeout in the middle of a possession says time has passed. A
  quarter of the first tempo result came from that and was given back.
- **Two scorer's tags double as a check on the pipeline**: second-chance agrees with its count of offensive
  rebounds on 99.6% of 368,467 tagged possessions, from-turnover with how the possession started on 99.99%.
- **The two new predictions are not in the pregame log.** They are scored looking back only.

## Team-report tendencies, checked

A tendency is a deviation from a baseline in a defined situation. The layer (`tendency/tendency_layer.py`) is a
hierarchical logit fitted by empirical Bayes: a team's rate against the league; its rate in a situation against
its own other possessions; a player's share of finishes against what his usage and the four teammates on the
floor predict. Small samples shrink toward the baseline by an amount estimated from how much teams and players
really differ. Rows are ranked by the probability that the deviation is larger than one percentage point.

The check that decided whether it replaced the earlier stand-in: fit on the first half of 2025-26 (615 games),
predict the second half (615 games), against raw first-half rates and the stand-in. Every method is scored on the
same cells (one team or player, one situation, one outcome, at least 25 trials in each half). The table was
recomputed after step 2.2 added shot style, creation, tempo and fast-break rows, which tripled the team cells.

| Cells | Method | Log-loss | Against the stand-in (95% interval) | 90% intervals held, as fitted | With the drift term | Interval width |
|---|---|---|---|---|---|---|
| Team, 5,815 | Raw rate | 0.39714 | +0.000328 (+0.000299 to +0.000357) | 81.7% | 90.5% | 10.2 pts |
| Team, 5,815 | Stand-in | 0.39682 | reference | 78.6% | 90.0% | 8.6 pts |
| Team, 5,815 | Hierarchical | 0.39682 | +0.000003 (-0.000003 to +0.000009) | 79.8% | 90.0% | 8.8 pts |
| Team, 5,815 | Hierarchical, PyMC | 0.39681 | -0.000005 (-0.000011 to +0.000000) | 80.1% | 90.3% | 8.7 pts |
| Player, 2,644 | Raw rate | 0.49630 | +0.000791 (+0.000714 to +0.000865) | 85.9% | 89.3% | 15.4 pts |
| Player, 2,644 | Stand-in | 0.49549 | reference | 86.0% | 90.1% | 14.3 pts |
| Player, 2,644 | Hierarchical | 0.49477 | -0.000740 (-0.000837 to -0.000650) | 85.8% | 89.8% | 13.6 pts |
| Player, 2,644 | Hierarchical, PyMC | 0.49491 | -0.000602 (-0.000687 to -0.000523) | 85.5% | 89.4% | 13.5 pts |

Log-loss, the difference and the width are with the drift term; the stand-in reference is as it was exported.
5,174,871 team trials and 737,144 player trials in the second half; the interval resamples whole games.

- **On team rows the new model does not beat the stand-in.** It is level, on 2025-26 and on 2024-25. On player
  rows it does, in both seasons, because it accounts for who else is on the floor. It replaced the stand-in on
  that, on coverage of the rows a report shows, and on fixing what the stand-in got wrong by construction.
- **90% intervals held about 80% until drift was modeled.** Splitting the same season by alternating games
  (no time between halves) gives 89.6% and 91.5%, so the shortfall is teams and players changing. How much they
  change is measured on the season before (a spread of about 0.1 on the logit scale, roughly 2 percentage points
  at a 25% rate) and added to every interval. It does nothing for log-loss.
- **The rows a report shows are the hard case**, because picking ten of a few hundred favours the lucky ones.
  For the ten each team would have been shown from the first half, 90% intervals held 61.9% under the stand-in
  and 87.0% under the new model with drift (293 rows; 87.7% on 2024-25). That is short of 90%, and it was 90.5%
  before step 2.2 doubled the rows to pick from: more candidates, more selection, and nothing corrects for it yet.
- **A full Bayesian fit was tried and not used.** The same model in PyMC (NUTS, 70 seconds on an M3 Max) is a
  hair better on team rows and worse on player rows than the one-second fit, so it stays an optional extra:
  `pip install -e ".[bayes]"`.

## Hot hand

Separate from the team reports, league-wide, 3,931 games. After a player's make he finishes his team's next
possession 22.8% of the time against 21.8% expected from his usage and the lineup; after his miss, 18.1% against
21.8%. Make minus miss: +4.7 percentage points (95% interval 4.5 to 5.0; 541,792 possessions). The possession
itself scores no differently: +0.001 points (-0.005 to +0.007). So the ball finds the player who just scored and
there is no sign that it pays, or costs. This is an association: play-by-play cannot tell a team looking for the
hot player from a player hunting his own shot.

## Limits

- **Retrospective so far.** The table was computed after the games were played. The stricter test is built
  (`train`, `predict`, `score`: a pregame file names a frozen model by hash, and that model is scored after the
  game) and was rehearsed on 14 dates of 2025-26, but nothing has been logged before a real game yet.
- **Tendencies are validated half-season to half-season, on one league.** The drift term is a half-season
  number applied to a report of any age, and "bigger than one point" is a judgment call. A report 15 games into
  a season is mostly prior.
- **The shot-location baseline is weak.** It is the team's shot mix and knows nothing about the lineup, which is
  probably why that head shows the largest lift. A lineup-weighted mix is the tougher baseline and is not built.
- **No play calls, no defensive coverages.** Play-by-play records what ended a possession, not the set that was
  called or how a pick-and-roll was guarded. Nothing here predicts or implies them. Teams with tracking and
  video have richer labels; the point is the method, not exclusive insight.
- **Lineups are inferred.** The season archives have no boxscore, so starters are the first five players to act
  and lineups follow from substitutions. Sixteen possessions (one game with a missing substitution) do not
  reconstruct to five a side and are left out.
- **Two data paths, one lightly tested.** The results above come from a public GitHub mirror of the feed.
  `scripts/fetch_nba_api.py` has fetched 2026-27 games from nba.com and one passed `verify-fields`; the live feed
  has no shot `area`, so shot location there comes from coordinates (99.6% agreement with `area` on 2023-26).
  The direct cdn.nba.com commands (`run`, `ingest`) are still untested.

More detail: `ASSUMPTIONS.md` (what was checked against real games and what was wrong), `DECISIONS.md` (every
choice made along the way), `NEXT.md`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                  # about a minute
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
contract/schemas/     the data contract (eight JSON Schemas the app reads, plus the pregame log's)
tendency/             pipeline package
  cdn_fields.py       every assumed feed name, URL and rule of thumb
  ingest.py           fetch raw JSON as landed; idempotent; backoff; clear stop on a blocked host
  possessions.py      one row per possession; edge-case rules in the module docstring
  inspect.py          one game's possessions beside the raw actions, with a reconciliation summary
  prospective.py      freeze a model, log before tip-off, score after; the scorecard's prospective block
  features.py         context / pregame / in-game features and the baselines (DuckDB)
  models.py           splits, LightGBM per head, log-loss with game-bootstrap intervals, reliability
  tendency_layer.py   the team report: partial pooling across teams, situations and players, with drift
  tendency_validation.py  fit on half a season, predict the other half, against raw rates and the stand-in
  tendencies.py       the earlier simple-shrinkage stand-in, kept as the reference the validation scores against
  tendency_pymc.py    the same model by NUTS; optional extra, used only by the validation
  hot_hand.py         after a make or a miss: who finishes the next possession, and what it scores
  export.py           writes and validates the contract JSON
  bulk.py             whole seasons from the GitHub mirror; derives the boxscore the archive lacks
  teams.py            the 30 franchises by team id (city, nickname)
  synth.py            synthetic raw games in the assumed feed shape
  verify_fields.py    checks assumptions against a real game
  stubs/              named placeholders for everything outside the MVP
scripts/              fetch_nba_api.py and nightly.sh, for your own machine only
docs/                 terms check, screenshot, release notes
tests/                pytest: segmentation edge cases, leakage, ingest, models, contract, prospective log, tendency layer
app/                  static front end (no build step); app/data is the bundled sample export
```

## The app

Five views: Method (what it does, what the feed can label, and its limits), Team report, Replay with "guess before the model", Scorecard
(which also carries the tendency layer's half-season check), and Hot hand.
It is a plain static site: host the `app/` folder anywhere (GitHub Pages works). Served over HTTPS it installs
to a phone home screen and works offline after the first load.

Sample data is synthetic and says so on every page. The banner is driven by `data_kind` in the contract, and the
pipeline sets that from a marker inside the raw files, so synthetic output cannot be exported as real.

## Synthetic data, and what it does not show

`python -m tendency sample` simulates 224 games between eight placeholder teams over two fake seasons (2090 and
2091). A few situational quirks are planted (`QUIRKS` in `tendency/synth.py`) so the team report has ground truth
to recover; a test checks that it does. Each team's habits also shift at mid-season, so the tendency layer's
drift term has something to measure. Scorecard numbers on this data show that the evaluation code runs. They
say nothing about whether the models beat their baselines on NBA games.
