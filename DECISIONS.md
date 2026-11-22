# Decisions made during the MVP build (2026-11-16)

Each was made without asking, per the build instructions. Reverse any of them freely.

## Contract
1. **Six schemas, one manifest as the entry point**: manifest, teams, team_report, games, replay, scorecard.
   Contract version 1.0 is a `const`; the app refuses anything else.
2. **Schemas are limited to a small keyword subset** so the app can validate with a ~60-line validator and no
   dependency. A test fails if a schema uses anything outside the subset.
3. **`data_kind` (`synthetic` / `real`) is required on every file.** The pipeline derives it from a marker in the
   raw files and refuses a raw folder that mixes both.
4. **Replays are exported only for held-out games**, so the app can never show an in-sample prediction.
5. **`additionalProperties: false` everywhere.** Adding a field is a deliberate contract change.

## Possessions
6. **Labels come from the event that ends the possession.** After an offensive rebound the earlier shot is
   forgotten for labeling; `n_off_reb` records the extra chances.
7. **And-one is one possession ending in a shot**, finisher = shooter, the free throw's point included.
8. **Technical free throws never start, end or relabel a possession**; their points are stored separately so
   possession points still reconcile with the final score.
9. **End-of-period heaves are excluded from every head**, not only shot family. A heave is the clock's choice,
   not a tendency. They stay in the table with `shot_family = "heave"`.
10. **Possessions cut off by the horn end as `period_end`** and are excluded from modeling.
11. **Context is snapshotted just before the deciding event** (the shot, the turnover, or the foul that sends
    someone to the line), so lineups reflect mid-possession substitutions and nothing after the event.
12. **Unexplained changes of possession are kept as `ending_type = "unknown"` and counted**, never dropped
    silently. More than 20% unsegmentable games aborts the run.
13. **`foul_drawn` finisher is the fouled player** (`foulDrawnPersonId`), falling back to the free-throw shooter.

## Heads, baselines, models
14. **Shot family is conditional on the possession ending in a shot** (five classes), not a seven-class head.
    Cleaner to explain and to calibrate.
15. **Finisher is modeled per candidate** (five rows per possession, binary LightGBM, renormalised across the
    five). Team turnovers have no finisher and are excluded from that head.
16. **Baselines are shrunk**: season-to-date counts plus half-weight earlier seasons, pulled toward the league
    (teams, 200 pseudo-possessions) or toward one in five (player usage, 100). First games get a uniform prior.
    The brief says "season usage shares"; unshrunk shares are undefined in week one.
17. **LightGBM starts from the baseline's log-odds** (`init_score`), so it learns corrections to the baseline.
18. **Splits**: last season held out whole; latest 20% of earlier games held out; latest 15% of the remainder is
    used only for early stopping. All by game. Single-season data reports the season split as unavailable.
19. **Intervals are a 1,000-draw bootstrap over whole games.** Verdict is "model ahead" or "baseline ahead" only
    when the 95% interval of the difference excludes zero; otherwise "no clear difference".
20. **No opponent or defender features.** Defense proxies are out of scope, and keeping them out avoids implying
    coverage knowledge.
21. **No hyperparameter search.** One fixed, conservative parameter set.

## Tendencies (team report)
22. **Empirical-Bayes beta-binomial shrinkage, labeled "simple shrinkage estimate"** with `final: false`.
23. **Situational rows measure the team-by-situation interaction**: observed rate versus the team's overall rate
    shifted by the league-wide situation effect (logit scale). Otherwise a three-heavy team shows up under every
    situation. "All possessions" rows compare with the league.
24. **Player rows** compare a player's share of finishes in a situation with his own overall share; fixed prior
    strength of 60. Not adjusted for teammates on the floor (stated in the app).
25. **Ranking is by the interval bound nearest zero, either direction** (`rank_score`), so strong negative
    tendencies rank alongside positive ones.
26. **90% intervals treat the baseline as fixed and are not multiplicity-adjusted.** The app says how many
    candidates the top ten came from and how many would clear zero by chance.
27. Minimum 25 possessions per row; top 10 per team.

## Replay uncertainty
28. **Each probability is shown with its held-out calibration band** ("30-40% calls hit 34%, n 8,441") instead of
    a per-prediction interval, which gradient boosting does not give honestly.
29. **Guess mode scores top-pick hits** for you, the model and the baseline on the same possessions, always with
    the count, and says a few dozen possessions cannot separate them.

## App
30. **No framework, no build step**: one HTML file, one stylesheet, two ES modules. Nothing to maintain.
31. **Visual identity: a coach's dry-erase clipboard.** White board, blue marker for the model, pencil grey for
    the baseline, green for what happened, yellow floor tape for the synthetic banner; the half-court zone map
    is the signature element. Type: Big Shoulders Display (Chicago's civic typeface) and Archivo.
32. **Routes are plain hashes** (`#method`, `#team`, `#replay`, `#scorecard`).
33. **Service worker**: shell cache-first, data network-first, so a fresh export shows on the next load.

## Dependencies beyond the listed stack
- `pandas`, `numpy`, `pyarrow`: dataframes and Parquet I/O around DuckDB and LightGBM.
- `scipy`: beta quantiles for tendency intervals (already a LightGBM dependency).
- `requests`: HTTP for ingest.
- `jsonschema`: validates exports against the contract.

## Added after the real-data run (2026-11-16, same day)
34. **History comes from `shufinskiy/nba_data`** (`python -m tendency bulk`), a source the brief already lists.
    cdn.nba.com refused both the cloud host and a plain desktop browser request; I did not spoof headers to get
    past it. The archive is the same feed, so the possession code is unchanged.
35. **The bulk path derives a stand-in boxscore** (home/away, tip time, starters) from play-by-play and marks it
    `_derived`.
36. **Retained-possession free throws** (flagrant, clear-path, transition take, away-from-play) add points to
    the possession without ending it.
37. **`heave` actions (2025-26 onward) are team misses excluded from every head**, consistent with decision 9.
38. **`after_timeout`, lineups, fouls and penalty are all read at the deciding event**, never later. This closed
    a real leak in `after_timeout`.
39. **Team reports use the latest season only.** Three seasons of one franchise are several different teams.
    Models and baselines still use every season.
40. **Candidate features use integer keys and float32** so three seasons fit in about 4 GB.
41. **Real exports are not committed.** `data/` is gitignored and the repo ships the synthetic sample, until
    NBA.com's terms have been read.

## Housekeeping
- The workspace held a few files from an earlier interrupted start (a `pipeline/tendency_board` skeleton and an
  older schema). I removed them and built on one layout.

## Phase 1 finish (2026-11-16/17 session)

### Step 1.1, CI
42. **One workflow, one job, Python 3.12 and Node 22.** Runs pytest, the synthetic pipeline into `/tmp`, and both
    contract validators on the fresh export. No matrix: one supported interpreter is enough for a solo project.
43. **CI also validates the bundled `app/data`.** A contract change that forgets to regenerate the sample the
    deployed app serves fails the build instead of failing in the browser.
44. **No real data in CI.** The real path needs a 60 MB download and 4 GB of RAM; it stays a local check.

### Step 1.2, hand-check tool and what it found
45. **`inspect` lays raw actions under possessions by the first and last `actionNumber` each row consumed.** No
    second pass over the game; what is printed is exactly what the segmenter saw. Actions between possessions
    are marked `~`. Exit code 2 if points do not reconcile, so it can be scripted.
46. **"Odd" means**: and-one, heave, retained free throw, technical points, lineup not five-a-side, unexplained
    ending, horn, or two or more offensive rebounds.
47. **A free throw with a keep-the-ball descriptor is never an and-one, except away-from-play after a made
    basket.** Flagrant, clear-path and take fouls: the basket ends its possession and the free throws open the
    next for the same team. Away-from-play after a basket: and-one by a teammate, possession ends. Two-shot
    away-from-play: ordinary trip. Rule set by counting who had the ball next in 3,690 games (ASSUMPTIONS.md).
48. **Rebounds never stop the and-one look-ahead.** Before a second shot there is nothing to rebound but a
    free throw, so a rebound seen there is the dead-ball one after a technical.
49. **A `personId` with no player name is not a player** (bulk path): not on the roster, never a starter.
50. **Debatable segmentation cases were listed, not changed** (ASSUMPTIONS.md, "Debatable"). Each moves a few
    possessions a season; none moves a scorecard number at the precision shown.

### Step 1.3, real names
51. **Team city and nickname come from a static 30-row table keyed by team id** (`tendency/teams.py`), built from
    the mirror's matchups archives and checked against the team ids in all 3,690 games on disk. Not from memory.
    The tricode still comes from each game's own feed, so a rebrand shows the tricode of the day.
52. **Full player names come from the mirror's matchups archives** (`first_name`, `family_name` per `person_id`,
    about 4 MB a season, 2017-18 onward). They cover all 802 players in the three seasons; the shot-detail archive
    would miss anyone who never took a shot. Keyed by person id alone, so a traded player has one name.
53. **Names are optional.** If the archive cannot be fetched the run carries on with "J. Tatum" and says so once.
    `--no-full-names` skips it. Cached in `data/real/bulk/player_names.json`.
54. **Derived boxscores carry a version.** `bulk` rewrites one made by an older version (or without names when
    names are now available) and leaves the play-by-play beside it alone. No need to delete `data/real/raw`.
55. **No contract change.** `teams.json` already carried `name` and player `name` as free strings.
56. **Spelling follows the source.** The matchups archive has "Nikola Jokić" where the feed's short form has
    "N. Jokic"; nine players differ only by diacritics. The full name wins.

### Step 1.4, prospective prediction log (design, written before the code)

**The claim, exactly.** "The model and everything it was allowed to know were fixed before the game." Proof is a
file committed before tip-off that names the frozen model by hash. It is not "these per-possession numbers were
published before the game": score, fouls and the five on the floor do not exist before tip-off.

57. **Source for 2026-27 games: wait for the mirror, and replay 2025-26 as if live to test the flow.** That was
    the default given for this session. The mirror has no `cdnnba_2026` archive as of 2026-11-21 (it lists `wnba_cdnnba_2026`, which `bulk` used to mistake for it). A script for
    your own machine (`scripts/fetch_nba_api.py`) is written and unit-tested against a fake client but was never
    run against nba.com, and nothing in this repo calls cdn.nba.com or stats.nba.com from the cloud.
58. **Three commands.** `train --through D` freezes models on games up to D. `predict --date D` writes one file
    per game before tip-off. `score --date D` scores the frozen model on that date's games once they are final.
59. **What a pregame file holds** (`predictions/YYYY-MM-DD/<gameId>.json`, schema
    `contract/schemas/prediction_log.schema.json`): when it was written; the frozen model's id, training cut-off
    and SHA-256; the last game the inputs include; and for each team its projected five (the starters of its
    most recent game), its baseline shares, and baseline and model probabilities for all three heads in four
    fixed reference situations (opening tip; Q2 half court; after a timeout late and close; defense in the
    penalty in Q4).
60. **Reference situations are fake possessions pushed through the real feature code.** No second implementation
    of the shrinkage formulas to drift. In-game counts are not knowable pregame: they are zero at the opening
    tip (exact) and set to their pregame expectation elsewhere (team's usual possessions by that point times its
    baseline shares; the projected five assumed on the floor). The file says so in `ingame_assumption`.
61. **A game date is the US Eastern calendar date of tip-off.** UTC would split one night across two dates.
62. **Scoring uses a day-start snapshot.** Every game on date D gets pregame features from games before D only,
    not from an earlier game the same night. That is what could be known when the log was written, and it lets
    `score` recompute each team's baseline and compare it with the logged one (`inputs_max_abs_diff`). It is
    slightly stricter than the retrospective splits, which let a late game see that night's early games.
63. **`score` only accepts the model the log names.** It recomputes the SHA-256 of the model files and refuses
    on any mismatch, refuses a model whose training cut-off is not before the game date, and says so plainly
    when the model on disk was created after the log was written. Retraining over an existing model id is
    refused by `train` unless `--force`, for the same reason.
64. **Models live under `data/<root>/models/`, not in git.** A model file is trained on NBA data; only its hash
    is in the log. Keep the files: without them a log cannot be scored. LightGBM runs deterministically here,
    so the same data and versions should rebuild the same file, but that is not guaranteed across machines.
65. **`predict` refuses a date whose games are already on disk unless `--replay`**, and a replayed log is marked
    `mode: "replay"` in every file and in the scorecard. The app labels a replay as a flow test, not as
    prospective evidence. A live log written after tip-off is kept but counted as late and left out of the
    prospective numbers.
66. **Per-game sums, not per-possession rows, are what accumulates** (`predictions/scorecard_prospective.json`):
    per head, the count, summed log-loss for model and baseline, top-pick hits, and calibration-bin sums. That
    is enough to rebuild log-loss, the game-bootstrap interval and the reliability table exactly, and it holds
    no play-by-play. Re-scoring a date replaces its games; it never double counts.
67. **Each date folder also gets `SHA256SUMS`.** If the terms check says per-game files with real names should
    not be public, committing only the checksum file still timestamps the log; the files can be shown later.
    `score` verifies the checksums when the file is present.
68. **Contract 1.1: `scorecard.json` gains an optional `prospective` block**, and the manifest accepts versions
    1.0 and 1.1. An export without the block (any 1.0 export) still validates and the app shows the same
    "nothing logged yet" state as an empty block. The retrospective `splits` are untouched.
69. **The bundled synthetic sample ships with an empty prospective block.** Nothing has been logged before a
    real game, and a simulated one on the public demo would read as more than it is. The populated view is
    covered by a test that runs train, predict and score on synthetic games, and by the 2025-26 replay.
70. **`evaluate()` was split into per-game sums plus one function that turns sums into the scorecard entry**, so
    retrospective and prospective numbers come from the same arithmetic.
71. **`nba_api` is not a project dependency.** The fetch script imports it lazily and tells you to
    `pip install nba_api` on your own machine. It sends browser-like headers to reach the CDN; whether to use it
    is the terms decision in step 1.0a.
72. **`scripts/nightly.sh` commits nothing by default** (`PUBLISH=none`). `sums` commits only the checksum files,
    `full` the per-game files. What gets published is the owner's decision after the terms check.
73. **Replay of 2025-26, as built:** model frozen through 2026-03-01 (3,366 games), then 2026-03-02 to 03-15
    predicted and scored one date at a time: 14 dates, 105 games, about 4 s per `predict` and per `score`.
    Logged and recomputed baselines agree to 5e-05 (rounding). Last game's starters were 83% of the next game's.
    The replay log lives in `data/real/predictions_replay` (gitignored), never in `predictions/`.
