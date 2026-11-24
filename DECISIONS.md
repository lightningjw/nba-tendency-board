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
