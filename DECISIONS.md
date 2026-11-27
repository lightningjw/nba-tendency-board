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

### Step 1.0a, terms check
74. **Research only; nothing published changed.** Finding and recommendation are in `docs/terms-check.md`. The
    conservative reading of NBA.com Terms section 9 bars any public product that presents live or archived
    play-by-play, which covers a public real-data replay view, real replay files in the repo, and Phase 3 live
    mode. Code, the model scorecard and attributed, non-commercial tendency tables are the defensible set.

### Step 1.5, close-out
75. **README screenshot is the synthetic sample.** A real-data screenshot would publish real names and numbers,
    and what gets published is the owner's call after the terms check.
76. **Results moved to the top of the README with top-pick accuracy beside log-loss**, and the limits listed
    directly under them, including two the old README did not state: the tendency layer is a stand-in, and the
    shot-location baseline ignores the lineup.
77. **Method page: three sentences changed, no features.** "Ships only if it beats its baseline" now says the
    margins are small; a card states that results are retrospective until logged ahead; the lineup card says
    starters are inferred when there is no boxscore. With real data the footer adds "Statistics from NBA.com".
78. **No tag.** Draft notes are in `docs/release-notes-v0.1.0.md`.

### After the first real fetch (2026-11-21)
79. **A two-point shot logged exactly at the hoop is a rim shot whatever `area` says.** The feed labels 44% of
    them "Mid-Range". This changes 7,480 training labels, so shot-location numbers in the README were rebuilt.
80. **Coordinate rule retuned to the feed's own boundaries** (4.4 ft, y = 89.5), because for the live feed, which
    has no `area`, it is the only rule. Constants live in `cdn_fields.py`.
81. **`verify-fields` treats a missing `area` as a note when every shot has coordinates.**
82. **Any model frozen before this change should be retrained**: its shot-location labels differ. None had been
    used for a live log.
83. **A free throw's kind comes from its own markers, else from the foul that awarded it** (`free_throw_kinds`).
    Technical fouls owe one "1 of 1"; away-from-play, flagrant, clear-path and transition-take fouls owe a trip;
    double fouls owe nothing; a debt holds only at its own period and clock; marked free throws are paid first.
    Needed because the live CDN feed leaves technical and away-from-play free throws unmarked.
84. **The cached possession table is rebuilt when the segmentation code is newer than it**, not only when raw
    files are. Otherwise a `git pull` that changes the rules would leave `train`, `predict` and `score` on old rows.
85. **The model frozen before this fix stays frozen.** `through-2026-11-20` has pregame files naming it; it is
    not retrained. Its 2026-27 training rows had this defect (well under 0.1% of its training possessions).
    Scoring uses the corrected rows, so `score` will report a small baseline drift for logs written before the fix.

## Step 2.1, the partial-pooling tendency layer (2026-11-23)

All numbers below: 2025-26 split in half by date (615 games to fit, 615 to predict), with 2024-25 (drift from
2023-24) as the second check in brackets. Log-loss differences are against the stand-in, with a 95% interval
from 1,000 resamples of whole second-half games.

### What ships, and why
86. **The empirical-Bayes hierarchical logit ships and `method.final` is true.** The roadmap's rule is "the simplest
    method that wins out of sample". On team cells it does not win: log-loss 0.45614 against the stand-in's
    0.45613, difference +0.000008 (-0.000002 to +0.000018), level [2024-25: -0.000004, level]. On player cells it
    wins: 0.49470 against 0.49541, difference -0.000734 (-0.000830 to -0.000645) [2024-25: -0.000518
    (-0.000605 to -0.000429)]. It ships on the second clause of the rule: level or better on log-loss everywhere,
    better coverage where it counts (the rows a report shows, decision 91), and each stated weakness of the
    stand-in fixed (decisions 88 to 91).
87. **`final` is a constant in the code, decided on real seasons, not recomputed per export.** The bundled
    sample is simulated; a flag that flipped with a simulator seed would mean nothing. What is recomputed on
    every export is the validation table itself, verdicts included, and the app shows it.
88. **Teammates: player rows are measured against a lineup-adjusted expectation.** Strengths w are fitted so
    that P(i finishes) = w_i / (sum over the five on the floor), on the team's possessions outside the
    situation, then averaged over the lineups he played with inside it. This is where the out-of-sample gain
    is. A test builds a star who closes games with a weak bench: the stand-in reports +9.5 points, the layer +0.6.
89. **Baseline uncertainty: a situational row compares the situation with the team's OTHER possessions**, not
    all of them, so the two sides are independent and the baseline's variance adds to the deviation's exactly.
90. **Prior strength is estimated, never fixed.** Each group (one situation and outcome across teams; one
    situation across all players) has its own between-unit variance by marginal maximum likelihood.
91. **Selection: rows are ranked by the posterior probability that the deviation exceeds one percentage point
    in its own direction**, and the app prints, from those probabilities, how many of the ten shown are expected
    not to. Checked directly: for the top ten rows each method would have shown from the first half, 90%
    intervals held 71.2% for the stand-in and 77.9% for the layer as fitted, 90.5% for the layer with the drift
    term (285 rows) [2024-25: 71.7%, 75.7%, 87.9%; 280 rows]. One point is a judgment call, not a fitted number.

### Two bugs in the WIP, both of which changed the first result
92. **Player cells were missing because their keys were NaN.** Under pandas 3 a `None` outcome class in a string
    column becomes NaN, and a string joined to NaN is NaN. Keys are now built by one function (`cell_key`) for
    every method; `test_every_method_is_scored_on_the_same_cells_players_included` fails if a method lacks a cell.
93. **The WIP predicted a cell's rate as baseline plus shrunk deviation, which treats an uncertain baseline as
    exact.** That is the right posterior for the deviation (what the report shows) and the wrong one for the
    rate (what the validation scores). The rate now weighs the cell's own count against "baseline plus a typical
    deviation" by precision. `estimate()` returns both and its docstring says which is which.

### Coverage and the drift term
94. **Why 90% intervals covered about 80%: teams and players change, and nothing modeled it.** Evidence: split
    the same season by alternating games, so no time passes between the halves, and raw-rate intervals hold
    88.6% of team cells and 91.6% of player cells [88.0%, 90.8%]. Split by date and they hold 81.7% and 85.7%.
    The last point or two on team cells is possessions in one game not being independent.
95. **Drift is measured on an earlier season and added as variance on the logit scale.** For rates, per scope:
    sd 0.108 for team cells and 0.101 for player cells from 2024-25 (about 2 percentage points at a 25% rate);
    0.090 and 0.124 from 2023-24; 0.095 and 0.124 from 2025-26. It is the variance of the half-to-half change
    that binomial sampling does not explain, by maximum likelihood, around zero.
    Result on 2025-26: coverage 79.8% to 91.7% on team cells and 86.1% to 90.0% on player cells, log-loss
    0.45613 to 0.45614 and 0.49471 to 0.49470. It does not hurt log-loss and does not help it: it moves the
    predicted rate by the width of a rounding error and only widens the interval. [2024-25: 79.2% to 88.8%,
    86.4% to 93.0%.] The same term fixes the raw and stand-in intervals as well (91.3% and 91.3% on team
    cells); calibration on all cells is not what separates the methods.
96. **For the deviation a report row shows, drift is measured per kind of row, not per scope.** I was asked for
    per scope and measured both. A team's overall rates drift (sd 0.099 against the league) and its situational
    rates drift with them, so the situational DEVIATION barely moves (0.051 in 2024-25, zero in the other two
    seasons); the player deviation's drift is 0.000 to 0.075 depending on the season. One number per scope
    would widen 1,680 situational rows to cover for 240 main-effect rows. With per-kind drift the displayed
    intervals held 89.2% (team, all possessions), 91.0% (team, in a situation) and 90.5% (player) of
    second-half deviations [86.3%, 88.6%, 93.3%].
97. **A report's intervals include drift, so they describe the games to come, and the app says so.** Drift for
    a report comes from the season before the one reported, never from itself. With no earlier season there is
    no drift term and the row says that instead.
98. **Not done: a mean-reverting version** (shrink harder because the thing being estimated will move). It is
    the obvious way the drift could buy log-loss on team rows. I stopped at the term that was asked for.

### PyMC
99. **`pymc` is an optional extra (`pip install -e ".[bayes]"`), one line of justification: it is the only way
    to ask whether a full Bayesian fit predicts better than the plug-in one.** Nothing in the pipeline, the
    tests' required path or CI imports it; the one test that uses it is skipped without it.
100. **It was run and it does not ship.** Same model with exact binomial likelihoods for the cell and for the
     baseline's raw material, variance components pooled across the 56 situation-outcome groups, each team
     with its own scale for player deviations, lineup adjustment as an offset. NUTS, 4 chains, 1,000 tuning and
     1,000 kept draws, 9,512 parameters: 28 seconds on an Apple M3 Max (14 cores, 36 GB), 4 cores used, PyMC
     6.4.0. Largest r-hat 1.012, 2 divergences in 4,000 draws. Team cells: -0.000005 (-0.000013 to +0.000003),
     level. Player cells: -0.000597 (-0.000681 to -0.000517), ahead of the stand-in but behind the
     empirical-Bayes fit's -0.000734. Coverage with drift 91.6% and 89.4%. [2024-25: -0.000011 and -0.000472.]
     A sampler and a dependency for no gain, so the half-second fit ships. ADVI was not needed and not run.

### Hot hand
101. **Its own contract file and its own view, league-wide, all seasons on disk.** A row is a possession whose
     previous trip by the same team, in the same period, ended in a field-goal attempt by a player still on the
     floor. Expected share is the same lineup-adjusted strength the layer uses, fitted per team and season.
     Points are compared with that team's season average.
102. **The make-minus-miss difference is the reported number, not make against zero.** On 3,931 games: after his
     make he finishes the next possession 22.8% against 21.8% expected (+0.9 points, 0.8 to 1.1; 306,381
     possessions); after his miss 18.1% against 21.8% (-3.8, -3.9 to -3.6; 235,411). Difference +4.7 points
     (4.5 to 5.0). Points on that possession: +0.001 per possession (-0.005 to +0.007), no clear difference.
     A placebo I ran but did not export: the gap shrinks with distance (4.4, 2.8 and 1.6 points one, two and
     three trips later in 2025-26), which is what a short-lived effect looks like and not what a fixed
     matchup advantage would. It is still an association; the app says what it cannot separate.
103. **Nothing here implies a play call.** The report says "finishes the next possession more often" and states
     that play-by-play does not record who a play was run for.

### Contract, app, sample
104. **Contract 1.2.** Two new files, both optional and listed in `manifest.files` only when present:
     `tendency_validation.json` and `hot_hand.json`. Team reports gain optional `threshold`, `drift_included`
     and per-row `prob_beyond`, and two baseline kinds. Schemas stay inside the supported keyword subset. A 1.0
     or 1.1 export validates and renders: checked by serving the 1.1 export on `main` under the new app.
105. **The validation table goes on the scorecard; the team report links to it** and quotes the one number that
     is about the rows it shows.
106. **Which season is validated: the latest complete one** (at least 80% as many games as the fullest season
     on disk), with drift from the season before it. So a season in progress is never split in half.
107. **The simulator now shifts each team's habits at mid-season**, so the sample has drift to measure and
     exercises the "includes drift" wording. Every number in `app/data` changed; they were invented before too.
108. **`tendencies.py` stays** as the reference the validation scores against. It is no longer exported.
109. **Known soft spots, left as they are.** The player baseline's variance is his own count elsewhere; when
     lineups barely overlap that is too small (a hand-built world showed it; real seasons did not). A group whose
     estimated spread is zero gets intervals a fraction of a point wide; those rows rank last and are never
     shown. Drift is a half-season number applied to reports of any age.

## Step 2.2, Phase 2 labels from play-by-play (2026-11-23)

Head results below are from three seasons only (2023-24 to 2025-26, `data/real3`), the same footing as the README
table: trained on 1,968 games, held out late 2024-25 (492 games) and all of 2025-26 (1,230 games).

### Asked before building, as the step says
110. **The action-style mapping was shown to the owner before it was wired in, and the ten-style version was
     chosen** over a coarser seven and a finer twelve. It lives in `cdn_fields.ACTION_STYLE_RULES`: ordered rules
     on the words in `descriptor`, then the shot `subType`. All 50 (subType, descriptor) pairs seen in three
     seasons are pinned to their style by a test.
111. **The two heads were the owner's choice too: action style and tempo.** An "assisted or not" head was offered
     and declined: the feed credits an assist only on a make, so it would be conditional on an outcome.

### Labels
112. **A style is the scorer's description of the shot, not a play.** `cut_lob` means the shot was a cutting layup
     or an alley-oop. The names, the app copy and the docstrings all say so.
113. **Every attempt gets a style, including the 32% with no descriptor.** Two "plain" classes instead of an
     "unknown": a jump shot with no move noted is 94% threes and 97% of its makes are assisted, which is information.
114. **Creation is defined on made baskets only** (41.8% of possessions). `assister_id` is on the table; nothing
     reads it yet, so there are no passer-to-shooter rows in this step.
115. **Tempo is seconds from the start of the possession to its first shot, turnover or trip to the line.** An
     offensive rebound does not restart it. Early is 8 seconds or less, late is 16 or more (27% / 38% / 35%).
     It uses the period clock because the feed has no shot clock, and is defined only when at least 24 seconds
     were left at the start (96.4% of possessions).
116. **The live feed's rebound clocks are corrected for the tempo labels only** (ASSUMPTIONS.md 15). Games whose
     defensive rebounds share the missed shot's clock are detected one by one, and in those a possession that
     starts off a rebound is taken to begin 3 seconds later. `start_clock_s` is left alone: the frozen model was
     trained on it as the feed gives it.
117. **Fast break is the scorer's tag and nothing more.** Defined when the possession ends in a shot or at the
     line (84.6%). The sibling tags, second-chance and from-turnover, are not stored because the pipeline already
     derives both; they are used as a check instead and agree on 99.6% and 99.99% of 368,467 tagged possessions.
118. **Two-for-one is a situation plus a label.** The window (30 to 45 seconds left at the start) is known when
     the possession begins, so it is a context feature and a team-report situation. "Taken" (first action with
     27 or more left) is the outcome inside it. Thresholds are judgment calls, in `cdn_fields.py`.

### Heads
119. **The two new heads are not in the pregame log.** `models.HEADS` is still the three a frozen model carries;
     `EXTRA_HEADS` are fitted and scored by `run_models` only. The model frozen on 2026-11-20 keeps scoring, and a
     test pins the three logged heads' feature lists column by column. The app says "looking back only".
120. **Action style's baseline knows who is on the floor and how the possession started.** The first version was
     the lineup's usage-weighted style mix alone, and the model beat it by 0.0856. Before believing that I scored
     a baseline that also shifts the mix by the league's start-type effect: it recovered 85% of the gap. That is
     the same weakness the README already admits for shot location, so the tougher one became the baseline.
     Result: 1.9744 against 1.9878 on 2025-26, difference -0.0133 (-0.0141 to -0.0126); 1.9838 against 1.9977 on
     late 2024-25, -0.0139 (-0.0151 to -0.0128). Top pick right 29.9% against 29.7%. It beats its baseline and
     ships; the lift is small.
121. **Tempo's baseline is the team's early / middle / late mix on possessions that started the same way**,
     shrunk toward the league's by 100 pseudo-possessions. Without the start type the model would win by knowing
     the ball came off a turnover.
122. **The tempo head sees only what was known when the possession started.** Context is normally read just
     before the deciding event (decision 11), which is right for "what happens next". For "when does it happen"
     it is a leak by construction: a timeout, a foul or a substitution in the middle of a possession says time
     has passed (late share is 57% after a timeout against 34% otherwise). With that context the lift was
     -0.0295; without it, and without the lineup features, it is -0.0228 (-0.0239 to -0.0217) on 2025-26 and
     -0.0259 (-0.0276 to -0.0242) on late 2024-25. Top pick right 52.2% against 50.5%. The smaller number ships.
123. **The older heads keep their context as it was.** Predicting what happens next from the moment before it
     is what they were built and logged to do.
124. **Leakage tests cover the new features and have teeth for the new labels**: scrambling now rewrites style
     and tempo, the standard pregame and in-game checks run over every new column, and two tests feed the guard
     a feature built from the possession's own style or tempo and require it to be caught.

### Team reports
125. **Four outcome families and one situation were added**: style share of shots (ten rows), assisted share of
     made baskets, early and late share, fast-break share; and the two-for-one window. A team has 301 to 391
     candidate rows on a full season, and 215 to 285 sixteen games into 2026-27 where it had 115 to 154 before
     this step. The stand-in was given the same outcomes so the half-season validation still scores every
     method on the same cells.
126. **No tempo rows in "after a timeout" or "defense in the penalty".** Both flags are read mid-possession,
     so a late first action there is partly arithmetic.
127. **The validation was rerun on the larger cell set and the 2.1 decision holds, with one number worse.**
     2025-26: 5,815 team cells level with the stand-in, +0.000003 (-0.000003 to +0.000009); 2,644 player cells
     ahead, -0.000740 (-0.000837 to -0.000650); coverage with drift 0.900 and 0.898. But for the top ten rows a
     report would show, 90% intervals held 87.0% with drift (293 rows; 87.7% on 2024-25), down from 90.5% when
     there were about half as many candidates to pick from. The stand-in as exported: 61.9%; with drift added,
     86.2%. Picking ten from twice as many rows is harder on an interval, and nothing here corrects for it. The
     README says 87.0%.
128. **PyMC rerun on the larger set: still not worth a sampler.** 70 seconds now (17,695 parameters, r-hat 1.009,
     no divergences). 2025-26: team cells -0.000005 (-0.000011 to +0.000000), player cells -0.000602 against the
     fast fit's -0.000740. 2024-25: team cells -0.000013 (-0.000019 to -0.000008), clear of zero for the first
     time, player cells -0.000468 against -0.000515. Summed over all trials the fast fit is ahead on 2025-26 and
     the two are even on 2024-25.
129. **Rows are ordered by how many posterior standard deviations the deviation clears the threshold by**, which
     ranks exactly as the probability does but does not saturate. After a full season dozens of rows show
     "100% likely", and the old tie-break was the row id.
130. **On a full season most of a team's top ten are "all possessions" rows** (how its style or tempo mix differs
     from the league's), because those have the most data. I did not add a rule to mix in situational rows.

### Contract, app, sample
131. **Contract 1.3.** Manifest: three to five heads, `logged_before_tipoff` on each, and a `labels` list with
     each label's source, when it is defined, and its share of possessions. Scorecard: up to five heads.
     Replay: step 2.2 fields on `actual`, two more entries in `pred`. Team report: four more outcome families.
     All additions are optional; 1.0 to 1.2 exports validate and render.
132. **The replay shows the two new heads but the guess mode still asks about three.** Picking one of ten shot
     styles before every possession is a chore, not a game.
133. **The simulator writes descriptors, assists that depend on the shot, scorer tags and a pace per team**, plus
     one planted style quirk (a team that triples its pull-ups in the fourth quarter) that a test recovers. Pace
     is drawn after everything else so the eight teams are the ones the older tests know.
134. **`inspect` prints the new labels beside each possession**, so they can be read against the raw actions.

## Step 2.3, a transformer over possession tokens (2026-11-26)

Numbers are from three seasons (`data/real3`), the README's footing: 1,673 training games plus 295 for early
stopping, held out late 2024-25 (492 games) and 2025-26 (1,230 games). Differences are transformer minus trees,
paired on the same possessions, 95% interval from resampling whole games.

### The answer
135. **The transformer replaces the trees for one prediction of five: shot style.** -0.0031 (-0.0046 to -0.0017)
     on late 2024-25 and -0.0018 (-0.0026 to -0.0008) on 2025-26, with a second training seed agreeing, and with
     seeds 7, 11 and 23 agreeing in a separate check. Top pick right 30.2% against the trees' 29.9% on 2025-26.
     That is between an eighth and a quarter of what the trees already gained over the baseline. Real, and small.
136. **Everywhere else it is level or behind, and stays an experiment there.** Who finishes: level, then behind
     (+0.0008, +0.0004 to +0.0011). How it ends: behind on both (+0.0044, +0.0052). Shot location: ahead on one
     split, behind on the other. Tempo: behind on both (+0.0047, +0.0042).
137. **What the sequence is worth, measured.** The third configuration is the same network with the sequence
     switched off. Without it the network is behind the trees on every head (by 0.002 to 0.013). Adding the
     sequence buys 0.006 to 0.007 on shot style, 0.008 to 0.009 on tempo, 0.002 on finisher, 0.001 to 0.004 on
     shot location, and nothing on ending type (slightly worse). So the gain on style is the sequence, and on the other heads the
     sequence closes most of the gap to the trees without crossing it.

### Two things the rule, read literally, would have got wrong
138. **Tempo looked like a win and was an input advantage.** First run: the transformer ahead of the trees on
     tempo on both splits, and again with another seed. The reason was in the inputs. Position k receives
     possession k - 1 as a token, including its fouls, timeouts and outcome, and step 2.2 had denied the tree
     tempo head anything read at its own deciding event without giving it a substitute. Giving the trees the
     previous possession's late context and outcome (`features.LAG_TEMPO`, nine columns, registered as in-game
     and leak-tested) improved them by 0.006 on both splits, and the transformer is now behind them. The fix
     ships: the tree tempo head goes from -0.0228 to -0.0290 against its baseline on 2025-26. I changed the
     trees after looking at held-out results; the effect is seven times the width of its interval.
139. **A win must hold for a second training seed.** The interval that decides a replacement is about which
     games were held out; it says nothing about the training run. Between two runs with different inputs, shot
     style went from "ahead, level" to "ahead, ahead". So the kept configuration is trained once more with the
     next seed (about a minute), and a head changes hands only if it is ahead on both splits both times. This is
     stricter than the rule I was given and can only prevent a replacement. Shot location shows why: ahead on
     the first split with seed 7, level with seed 8.
140. **Not done: averaging seeds.** The mean of three runs was better than any one (shot style -0.0065 and
     -0.0052; shot location ahead on both splits). An ensemble is a fourth configuration; the step allows three.

### The model
141. **Three configurations and no more**: "small" (64 wide, 2 layers, 123,126 parameters), "wide" (128, 3 layers,
     523,638) and "no_history" (small with the sequence off). The one with the lowest validation loss is kept:
     small, 6.6334 against 6.6369 and 6.6551. Held-out games play no part in the choice.
142. **Hardware and time.** Apple M3 Max, CPU only, 8 threads, PyTorch 2.15.0. All three configurations and the
     second seed: 327 seconds on 1,673 training games, far inside the hour a season the step allows. The kept one
     alone: 62 seconds, 8 epochs.
143. **Same inputs as the trees, same baselines, same rows.** Every column the transformer reads is one of the
     trees' registered features (a test checks it). Each head adds its output to the log of the baseline and
     starts at zero, so an untrained network is exactly the baseline, as LightGBM's `init_score` makes the
     trees. Scoring uses `evaluate()` on the trees' own row masks.
144. **Leakage is handled by where things sit.** Position k gets the start-known context of possession k and
     everything about possession k - 1, under a causal mask. The tempo head reads the trunk alone; the other
     heads also get k's late context and its five, fused in after the trunk. The self-check perturbs labels
     from k onward, features after k, and k's own late context, and requires predictions up to k (and tempo at
     k) not to move at all; it also requires earlier changes to move later predictions, and it catches a copy
     of the model that is fed possession k's own outcome.
145. **Player identity is an embedding, learned from training games; anyone first seen later shares one.** The
     trees have no player identity, only each player's pregame numbers. I count this as the model, not as an
     extra input: it is what a network can do with an id and a tree cannot.
146. **The sequence is the game's eligible possessions.** Heaves and possessions cut off by the horn are skipped.

### Plumbing
147. **PyTorch runs in its own process.** PyTorch and LightGBM each load an OpenMP runtime, and on macOS two in
     one process abort it. The documented escape hatch (`KMP_DUPLICATE_LIB_OK`) is described by its own error
     message as unsafe, so `transformer.py` writes the arrays to a temporary file, starts
     `python -m tendency.transformer_net`, and reads probabilities back. A test checks that module never loads
     LightGBM. PyTorch is an optional extra; without it the pipeline says so and the scorecard has one model.
148. **Contract 1.4.** The scorecard gains an optional `models` list and, per head, `model_id` (its primary
     model) and `challengers` (the other model's results on the same possessions, its paired difference against
     the primary, and the second seed's verdict). Replays show each head's primary model. Exports made without
     PyTorch, and 1.0 to 1.3 exports, validate and render.
149. **The pregame log is untouched.** A frozen model is still three LightGBM heads; shot style is not logged.
150. **CI installs CPU-only PyTorch** so the leakage test runs there, from PyTorch's CPU wheel index.

## Step 2.4: coach-decision heads (2026-11-26)

Profiled on 3,931 real games (2023-24 to 2026-27, 241 of them from the live feed) before anything was designed.

### The dead-ball table
151. **A second table beside the possession table, built in the same pass over the same files**
     (`tendency/stoppages.py`, written to `processed/stoppages.parquet`). One row per stoppage: 634,384 rows,
     161 a game. A row is one whistle or one made basket and everything logged before play resumes. A live
     action (a shot, a rebound a player pulls down, a steal, a block, a turnover with a steal) ends it. A new
     whistle after the clock has run opens a new row; events at the same clock are one whistle.
152. **A made basket and a made last free throw are rows.** No substitution is allowed there, but a timeout is,
     and 23,724 of 42,178 full timeouts follow a made basket. The team scored on calls 99.5% of those (23,724
     against 124), so after a basket or a made free throw only that team gets a decision row. Splitting the row
     at the last free throw keeps the score and the run current for the timeout that follows the free throws.
153. **A timeout belongs to the stoppage before it when it comes within 8 seconds of game clock.** Of the
     timeouts that follow a whistle or a basket, 66% carry the same clock and 93% come within 2 seconds. Later
     than 8 seconds, or after a live action, the timeout was called with the ball in play: its own row
     (`timeout_live`, 4,392 of them) and never a decision row, because there is no matching row for the moments
     nobody called one.
154. **Mandatory timeouts are not decisions, and the feed's own tag does not find them.** The `mandatory`
     qualifier sits on 72% of full timeouts: it marks the ones that fill a quarter's two television slots,
     asked for or not. So the table applies the rule instead: a timeout is due at the next dead ball when none
     has been taken in the quarter with 6:59 left, or fewer than two with 2:59 left. Checked against the data:
     at the first foul, dead-ball turnover or loose ball while one is due, a timeout follows 97.5% of the time
     (5,879 cases; 99.4% at fouls), and the first of a quarter is charged to the home team 99.1% of the time.
     Stoppages where one is due are left out of the timeout head.
155. **Personal fouls are counted here and checked against the feed.** The feed prints each player's running
     count on every foul ("(3 PF)"). This table's count matches on 151,195 of 151,195.
156. **The run in progress is two numbers per team.** Points in the last 10 possessions (both teams' together,
     read from the possession table), and points scored since the other team last scored.
157. **Substitutions between periods come from the lineups, not the log.** The feed often leaves them out, so a
     `period_break` row holds the difference between the five who ended one period and the five who started
     the next. Lineups are rebuilt exactly as the possession table rebuilds them.
158. **Substitutions with no whistle before them get a row of their own** (`unlogged`, 0.7 a game). The feed
     missed what stopped play; they are in the table and in no decision.
159. **The pregame log does not build this table.** `build_possessions` takes a flag that only the full
     pipeline sets, so `train` and `predict` read and write exactly what they did before. A test checks it.

### Which three, and which were left
160. **Kept the three the roadmap suggested**, after checking each label on real games: a timeout at a
     stoppage, a player sitting after a foul, and who closes a close game. I did not stop to ask; the roadmap
     says to pick the cleanest three and say which were dropped.
161. **Late deliberate fouls: not built.** The scorer's `take` descriptor marks them (2,711 fouls; about two in
     three come in the last minute of the fourth quarter or overtime, nearly all by the trailing team). The
     label is clean. It is also one narrow situation that the score and clock nearly decide, and the step
     allows three heads.
162. **Challenges: not built.** 1,073 in 3,931 games. The feed logs a challenge as a timeout with subType
     `challenge` and has no replay events at all, so it never says which call was challenged or whether it
     worked. Without the call at stake there is nothing to condition on.
163. **The next substitution in general: not built.** Who replaces whom is a choice among every pairing on the
     roster. Foul trouble is the one slice with a yes-or-no label at a single whistle.

### The three heads
164. **Timeout.** One row per stoppage and team, in regulation, when no mandatory timeout is due and the team
     has used fewer than seven. Yes means the feed charged that team a full timeout there. Overtime is left out
     because the table does not track overtime grants. 761,850 rows on three seasons, 3.9% yes.
165. **The timeout baseline already knows the run.** It is the league's rate for the kind of stoppage (after a
     score, a foul, anything else), the part of the game (first three quarters, fourth, last three minutes) and
     the opponent's unanswered points (0 to 3, 4 to 6, 7 or more). Step 2.2 taught this: a baseline that lacks
     the obvious thing makes the model look clever for learning it. On 2025-26 the run bucket alone is worth
     0.0096 of log-loss (-0.0101 to -0.0090) over the same table without it. The trees then add 0.0288, mostly
     from clock structure the table cannot hold: seconds until a mandatory timeout would come anyway, time
     since the last one, the exact stoppage. Finer run features add only 0.0013 beyond the three buckets.
166. **A player sits after a foul.** One row per personal foul, first to fifth, by a player on the floor.
     Yes means he was substituted out before the clock ran again (free throws included). The sixth foul is a
     disqualification, not a decision. Baseline: the league's rate for that foul number in that period. The
     strongest added input is how long he has been on the floor: a player near the end of a normal stint comes
     out anyway.
167. **A foul near the end of a period is a "no" even if he sits to start the next one.** The label is what
     happened at that whistle. The clock is a feature, so the model learns it; the report section should be
     read the same way.
168. **Who closes.** A game is close when it is within five points as the last five minutes of the fourth
     quarter begin (27.8% of games). The closing five are the team's five with the most seconds in those five
     minutes, which holds up through offense-defense substitutions; they play 93.9% of the team's seconds
     there. Each player who had played before the fourth quarter gets a row, predicted as the third quarter
     ends. Baseline: how often he closed for this team in earlier close games, pulled toward the league's rate
     for starters (79%) or bench players (22%).
169. **The closing head is scored only on games that turned out close.** Nobody knows that when the third
     quarter ends, and the model is not told. It is a statement about who closes given that the game is close,
     which is the question a scout asks. Eighteen closers of about 10,900 had not played before the fourth
     quarter and are not rows.
170. **Every baseline and prior counts games that tipped off strictly earlier**, this season in full and
     earlier seasons at half weight, the same discount the possession features use. The first night of the
     first season starts from a fixed guess (5%, 20%, 50%) with 20 rows of weight.
171. **One team feature per head: how far the team has run from its baseline in earlier games.** A coach's
     habit, as a number. It is worth 0.001 to 0.002.
172. **Same splits, same `evaluate()`.** A yes-or-no head goes in as two columns. The scorecard does not print
     "top pick right" for these: with 96% of timeout rows a no, it says nothing.
173. **Trees only.** The transformer reads possessions; these rows are stoppages, fouls and players. Not tried.
174. **All three beat their baselines on both splits, so none was cut.** Numbers are in the README.

### Leakage
175. **Two guards and a registry** (`tests/test_coach.py`). Cutting a game's feed off at any point leaves every
     earlier stoppage row identical, so a row holds only what had happened. Rewriting what every bench did in a
     game and in all later games (every team calls time, every fouler sits, the bench closes) leaves that
     game's baselines and priors unchanged; rewriting earlier games moves them. A third test lists every
     feature of every head and fails if one is neither a whistle snapshot nor an earlier-games count.
176. **Lineup repair still reads ahead within a period**, as it does for possessions: a player seen acting
     before any substitution is taken to have started the period. That is data repair, not an outcome.

### The report, the contract, the app
177. **The team report gets one section, "What the bench did".** Timeout rate by opponent run, the rate of
     sitting a player at seven foul-and-period situations, and who closed. Each team rate is a beta posterior
     that borrows from the league, with the prior's strength set from how much teams differ in that cell; the
     row shows the count, the rate, a 90% interval and the league's rate with its n. Closers carry a plain
     interval on k of n. The season's games are all used: this is description, not a held-out score.
178. **Descriptive framing is in the data, not only the page.** The section's note says these are descriptions
     and not grades, and travels in the JSON. Nothing computes or shows what a coach should have done.
179. **Contract 1.5.** Manifest: three more head ids, optional `kind` and `unit` on a head, an optional
     `stoppages` block (rows by kind, the foul-count check, timeouts set aside, decisions not modelled and
     why). Scorecard: up to eight heads. Team report: optional `coach_decisions`. A 1.4 export validates and
     renders; the app shows the new sections only when they are there.
180. **No new dependency.** The beta interval uses SciPy, already required.

### The simulator
181. **The simulator now behaves at stoppages**: it counts personal fouls and prints them the way the feed does,
     sits a player whose fouls outrun the period (one team more readily), fouls players out, charges mandatory
     timeouts, calls more timeouts the longer the other team's run, and logs a few challenges and take fouls.
     These draws come from a second generator so the possession stream keeps its own.
182. **Two older tests were made less lucky, not looser.** One used a single "control" team whose
     fourth-quarter pull-up share happened to sit at zero; closing lineups really move that mix a little, so it
     now requires every other team's deviation to be under half the planted one. The other probed possession
     60 of one game, which became a heave; it now takes the first eligible possession from 60 on.
183. **A decision head needs 200 training rows with both outcomes**, or it is reported as not fitted with the
     reason. The synthetic sample has 293 closing rows.
