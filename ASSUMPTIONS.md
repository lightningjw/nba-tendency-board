# Assumptions: what is verified, and against what

Updated 2026-11-16 after running on real data.

**How this was checked.** cdn.nba.com refuses both the cloud build host (proxy 403) and an ordinary desktop
browser request for the JSON files (the CDN's own "Access Denied"). I did not spoof past that. Instead the
play-by-play was checked against `shufinskiy/nba_data` (GitHub, Apache-2.0), which republishes the CDN feed as one
CSV row per action with the feed's own key names: three full regular seasons, 2023-24 to 2025-26, 3,690 games,
2.07M actions. Boxscore and endpoint shapes were checked against the `swar/nba_api` documentation on GitHub.

## Verified on 3,690 real games

| Item | Result |
|---|---|
| Per-action keys (`actionNumber`, `orderNumber`, `clock`, `period`, `teamId`, `personId`, `actionType`, `subType`, `descriptor`, `description`, `shotResult`, `scoreHome`, `scoreAway`, `area`, `shotDistance`, `xLegacy`, `yLegacy`, `foulDrawnPersonId`, `assistPersonId`, `possession`, `qualifiers`) | all present |
| `actionType` values | as assumed, plus **`heave`** (new in 2025-26, see below) |
| `subType` values for period, rebound, substitution, freethrow | as assumed |
| Shot `area` strings | as assumed (one null in a season) |
| Technical free throws carry `descriptor = "technical"` | yes |
| And-one = made field goal, shooting foul, `1 of 1` free throw | yes |
| Offensive fouls come with their own `turnover` action | yes (3,410 fouls, 3,431 turnovers in 2024-25) |
| Dead-ball team rebounds between free throws | present, tagged `deadball`; ignored correctly |
| Season from `gameId` characters 3-4 | yes |
| Possession points add up to the final score | 3,690 of 3,690 games |
| Lineups reconstruct to five a side | 738,796 of 738,812 possessions (the 16 are one game with a missing substitution) |
| Unexplained changes of possession | 11 of 738,812 (47 of 738,771 before the fixes of 2026-11-17, below) |
| Penalty rule vs the feed's own `inpenalty` tag | agrees on 99.99% of 88,347 fouls |

## Assumptions that were wrong, and the fix

1. **Between-period substitutions arrive before the period's `start` event** (tagged `startperiod`). The segmenter
   ignored everything outside a period, so it dropped them. Fixed: substitutions always apply.
2. **Four kinds of free throw leave the ball with the shooting team**: `flagrant`, `clear-path`,
   `transition take`, `away-from-play` (in `descriptor`). They were ending possessions. Fixed: their points join
   the possession, which carries on.
3. **`heave` is its own action type from 2025-26**: a missed end-of-period heave charged to the team, with no
   shooter and no `shotResult`. Fixed: handled as a team miss and excluded from every head. Older seasons still
   use the distance-and-clock rule.
4. **A buzzer shot followed by a team offensive rebound as time expires** was being relabeled as "period ended".
   Fixed: the shot keeps its label.
5. **Technical free throws shot between periods** were lost, leaving ten games a point short. Fixed.
6. **Penalty**: added the last-two-minutes rule and the overtime threshold; double fouls no longer count.
7. **`after_timeout` leaked the outcome**: a timeout called between a foul and its free throws flagged the
   possession that had already been decided. Fixed; it cost the ending-type model about a quarter of its lift,
   which is the evidence it was a real leak.

## Hand-check of three real games (2026-11-16, `python -m tendency inspect`)

Read: `0022500001` HOU at OKC (two overtimes, one-point game), `0022500547` CHA at UTA (55-point blowout),
`0022500017` BOS at DET (one-point game). Every possession with an unusual flag was read in all three, plus both
overtime periods of the first in full; the rest was checked by summary (points reconcile, possessions per team,
no unexplained changes). Then every unexplained change of possession and every pair of consecutive possessions
by the same team in 2025-26 was read, because those two lists are where segmentation errors show up.

### Wrong, and fixed (each has a test named for the game it came from)

8. **A technical free throw between a made basket and its and-one free throw split the and-one in two**
   (`0022500166`, `0022500201`). The dead-ball rebound logged after a missed technical stopped the look-ahead.
   The basket became one possession and the free throw a second, labeled "foul drawn". Fixed.
9. **An away-from-play free throw does not always mean the team keeps the ball.** After a made basket it is an
   and-one shot by a teammate and the other team gets the ball (33 of 34 such trips in three seasons; the other
   ended a period). A two-shot away-from-play trip is an ordinary penalty trip (3 of 4 went to the other team).
   Only the lone "1 of 1" with no basket keeps the ball (268 of 270). The old rule treated all three as
   "keeps the ball", which produced most of the unexplained changes of possession. Fixed.
10. **Basket plus a one-shot flagrant** left the possession open with a stale shot label, so a later free-throw
    trip could not relabel it. Now the basket ends its possession (as it already did for two-shot flagrants) and
    the free throw opens the next one for the same team.
11. **Bulk path: a team technical in the first minutes made a coach a starter** (`0022500704`, 27 possessions
    with six "players"). A `personId` with no player name is the bench. Fixed in `bulk.py`; the roster no longer
    lists them either.

Effect over 3,690 games: unexplained changes of possession 47 -> 11; possessions without a clean five-a-side
lineup 86 -> 16 (all 16 are one game, `0022400061`, where the feed is missing a substitution in garbage time).

### Debatable, left as they are (your call; each is a one-line change)

- **Zero-length turnover possessions.** A lane violation or offensive goaltending on an and-one free throw is
  logged as a turnover with no free throw. The basket's possession ends normally and the turnover becomes a
  second possession for the same team with no time on it. About 3 a season. Alternative: fold it into the basket.
- **Free-throw trips that end in an offensive rebound lose the "foul drawn" label.** Decision 6 labels a
  possession by the event that finally ends it, so a trip whose last miss is rebounded by the offense is
  relabeled by whatever follows. 2,375 of 88,307 trips (2.7%). Alternative: keep the first deciding event.
- **`after_timeout` is true for a timeout taken in the middle of a possession**, not only for one taken before
  the inbound. Both are "first action after a timeout", but only the second is a set play from a dead ball.
- **Start type after an and-one is `free_throw`**, not `made_basket`.
- **Basket plus flagrant is two possessions for the same team** (the basket, then the free throws and what
  follows). About 20 a season. Alternative: one long possession.
- **Late non-heave shots stay model-eligible.** A 19-footer at 0:00.3 is a shot; only 40 ft or more inside 3 s
  (or the feed's own `heave`) is excluded.
- **The 11 remaining unexplained changes** are jump balls after a challenge that hand the ball over with no
  turnover logged, and gaps in the feed (a miss, then the other team shooting, nothing between).

## First real fetch with nba_api (2026-11-21, game `0022600001`)

Run by the owner on his own machine. `verify-fields` passed everything except shot `area`:

12. **The live CDN feed has no `area` on shots.** The season archives do. `xLegacy`, `yLegacy` and `shotDistance`
    are there on every shot, so shot family comes from coordinates. The coordinate rule was retuned against
    `area` on all 657,380 shots of 2023-26 (restricted area runs to 4.4 ft, corner threes end at y = 89.5) and now
    agrees on 99.6%. `verify-fields` reports a missing `area` as a note, not a failure, when coordinates exist.
13. **Wrong in every season, and fixed: shots at the hoop labeled "Mid-Range".** Tips, dunks and layups logged at
    x = y = 0, distance 0, carry `area` "Mid-Range" 44% of the time (it follows which basket). 7,480 shots, 1.1%
    of all shots and about one "midrange" label in nine, were dunks and tip-ins. They are now rim shots whatever
    `area` says. Shot-location log-loss moved by about 0.012 for model and baseline alike; the difference between
    them, and every verdict, held.
14. **The live CDN feed does not mark technical or away-from-play free throws.** The archives put `technical`,
    `away-from-play`, `clear-path` and `transition take` on the free throw itself; the live feed marks only
    `flagrant` (seen in 240 fetched 2026-27 games). Unmarked, a technical "1 of 1" reads as an ordinary trip and
    ends or invents a possession, silently; one of them surfaced as an unexplained change, which is how it was
    found. The kind is now read from the foul that awarded the free throw when the free throw has no marker.
    Checked by stripping the markers from all 3,690 archive games: 30 of about 4,460 special free throws are
    read wrongly (0.7%), 12 games differ by 16 possessions in total, and with markers left in place nothing
    changes at all.
- Still unexplained in the fetched games, and left that way: two possessions in `0022600009` where the feed
  lists a two-shot foul but no free throws; an offensive-foul turnover charged to a team that, by every other
  event, did not have the ball (`0022600003`, `0022600114`); and a flagrant by the scoring team between a basket
  and its and-one (`0022600225`, `0022600254`).
- Also confirmed on that game: the real boxscore has five starters a side, 201 possessions, points reconcile
  (234 of 234), lineups reconstruct on 100%, no unexplained changes, and the segmenter agrees with the feed's
  own `possession` field on 102 of 102 made-shot and turnover endings.

## Phase 2 labels (step 2.2, 2026-11-23): what was profiled, and what the live feed does differently

Profiled on 657,380 field-goal attempts in the three archive seasons and 42,347 in the 241 fetched 2026-27 games.

| Item | Result |
|---|---|
| Shot `subType` values | four: Jump Shot, Layup, DUNK, Hook (plus one stray `shot` in 2026-27) |
| Shot `descriptor` values | 31, the same set in every season and in the live feed; 32% of attempts have none |
| (subType, descriptor) pairs | 50, each mapped to one of ten styles (`cdn_fields.ACTION_STYLE_RULES`; a test pins all 50) |
| `assistPersonId` | on 63.5% of made baskets, never on a miss |
| `qualifiers` on shots and free throws | `fastbreak`, `2ndchance`, `fromturnover`, `pointsinthepaint`, `defensivegoaltending`; on makes and misses alike |
| Scorer's `2ndchance` tag vs this pipeline's offensive-rebound count | agree on 99.6% of 368,467 tagged possessions |
| Scorer's `fromturnover` tag vs how the possession started | agree on 99.99% |
| Mirror vs raw JSON | the CSV mirror joins qualifiers with ", " (leading space on all but the first); stripped on load |

15. **The live CDN feed stamps a defensive rebound with the missed shot's clock.** 7,130 of 7,130 in 120 fetched
    2026-27 games; in the archives the rebound is stamped when the ball was secured, a median 3 seconds later
    (same clock on 6%). A possession that starts off a rebound therefore looks 3 seconds slower in the live feed:
    early-clock share read 19.5% against 26 to 27% in every archive season. Games are now checked one by one
    (at least 90% of their defensive rebounds sharing the shot's clock), and in those the tempo labels start the
    possession 3 seconds after the rebound's clock. After that, the same 241 early-season games read 27.3% early
    in 2026-27 against 27.8% in 2025-26. Only the tempo labels and the two-for-one window move; `start_clock_s`
    stays as the feed gave it, because the frozen model was trained on it that way.

Assumed, not checked:

- **A style is the scorer's word for the shot**, entered by a person at the table. Nobody audited it against video.
  `plain_jumper` (no descriptor) is 94% threes and 97% of its makes are assisted, so it is mostly catch-and-shoot,
  but the feed never says so.
- **Tempo uses the period clock**, since the feed has no shot clock. After a made basket it includes the inbound;
  after a non-shooting foul the shot clock may have reset and the label cannot see it. Early is 8 seconds or
  less, late is 16 or more: judgment calls that split possessions 27% / 38% / 35%.
- **Two-for-one**: 30 to 45 seconds left at the start, taken if the first action comes with 27 or more left. Also
  judgment calls. 2.4% of possessions; taken on 52%.
- **Fast break is whatever the scorer tagged.** It exists only on possessions that end in a shot or at the line;
  a fast break that ends in a turnover is not in the data.

## What the tendency layer assumes (step 2.1, 2026-11-23)

Checked where it says checked; the rest is assumed.

- **Deviations within a group are normal on the logit scale.** Checked only through coverage: displayed 90%
  intervals held 89%, 91% and 90% of second-half deviations on 2025-26 by kind of row (86%, 89%, 93% on 2024-25).
- **A count's noise is binomial.** Nearly: halves made by alternating games hold 88.6% (team) and 91.6% (player)
  against a nominal 90%. The team shortfall is possessions within a game moving together; the drift term absorbs it.
- **Change between halves has the same size from one season to the next.** Measured three times: 0.090, 0.108,
  0.095 for team rates and 0.124, 0.101, 0.124 for player rates (sd, logit scale). Close enough to borrow.
- **Half a season of drift is the right amount for a report of any age.** Not checked.
- **A player's finishing share is his strength over the sum of the five strengths on the floor**, the same in
  every situation unless the row says otherwise. Checked only through the player rows' out-of-sample gain.
- **A teammate never seen outside the situation is an average finisher.** Rare on real rosters; it decided one
  hand-built test world.
- **Hot hand: the make-minus-miss gap is an association.** A night-long matchup advantage would produce part of it.
  The gap halving by two trips later says part of it is short-lived, and nothing here separates the two.

## Stoppages and coach decisions (step 2.4, 2026-11-26): what was profiled, and what is assumed

Profiled on 3,931 games (3,690 from the season archives, 241 from the live feed).

Checked against the feed:
- **Timeouts** always name the team charged. subType is `full` (42,478), `challenge` (1,073), `short` (3) or
  `officials` (1). There are no instant-replay events, so a challenge has no subject and no result.
- **Personal fouls.** The table's running count per player equals the "(n PF)" the feed prints on all 151,195
  fouls. Technical fouls are not personal fouls and are not counted.
- **Take fouls** carry descriptor `take` (2,711); `transition take` is a different call and is not counted as one.
- **Substitutions sit at the stoppage's clock**: 96% of sub-outs come right after a timeout, foul, free throw,
  turnover, rebound or period end logged at the same clock. Those with nothing before them that stops play get
  an `unlogged` row (0.7 a game).
- **The mandatory-timeout rule** (none yet in the quarter at 6:59, fewer than two at 2:59) is from the rulebook
  as I remember it, then checked: when one is due, the first foul, dead-ball turnover or loose ball is followed
  by a timeout 97.5% of the time, and the first of a quarter goes to the home team 99.1% of the time.

Assumed, and worth knowing:
- **A stoppage ends at the next live action in the feed.** An inbound pass is not logged, so a foul, an inbound
  and a second foul two seconds later are two rows only because the clock moved.
- **"Called at that stoppage" allows 8 seconds.** A team scored on usually calls time at once or as it crosses
  half court. A timeout 9 seconds after a basket counts as called in live play. The cut is a judgment call.
- **A team rebound with no player is a dead ball** (the miss went out of bounds). So is any turnover without a
  steal. A kicked ball or a delay-of-game warning is a `violation` row, though play barely stops.
- **Timeouts left is seven minus full timeouts used.** The fourth-quarter caps and overtime grants are not
  modelled, which is why the timeout head stops at the end of regulation.
- **Sitting after a foul means substituted before the clock ran again.** A player who stays for one more
  possession and then comes out is a "no". So is one who fouls with two seconds left and sits to start the next
  period.
- **The closing five are the five with the most seconds in the last five minutes of the fourth quarter.** A
  specialist who comes in for one defensive possession is not a closer by this definition. Ties go to the player
  with more seconds in the game. Overtime is ignored.
- **"Close" is within five points as those five minutes begin.** A game that was tied at 6:00 and a nine-point
  game at 5:00 is not close; a nine-point game that tightens at 2:00 is not either.
- **Stints run through period breaks.** A player who finishes one quarter and starts the next has one stint.
- **A player's seconds are game-clock seconds between logged substitutions**, so they inherit any substitution
  the feed logged late.

## Defense proxies (step 3.1, 2026-11-27): what was checked, and what is assumed

Checked on the three complete seasons (3,690 games):
- **How a possession starts moves where its shot comes from.** Rim share of shots in 2025-26: 41% off a
  turnover, 32% off a defensive rebound, 27% after a made basket or a free throw. Turnover and foul rates barely
  move with it (13.6% to 15.2%; 9.3% to 10.4%, apart from the first possession of a period).
- **The defending five is known on all but 16 possessions.** Those count for the team's row and for no lineup.
- **Five-man lineups are thin.** In 2025-26 a team has a median of 8 lineups with 100 defensive possessions and
  3 with 200; 22% of possessions are played by a lineup with at least 100. Across the league, 51 lineups reach
  100 in the first half of the season and 25 in the second; 26 reach 100 in both.
- **The matchups archive** (not used; see DECISIONS.md 198): one row per offensive player and defender per
  game, 1,230 games a season. `person_id` is the offensive player and `team_id` is his team, checked against
  shot counts in the possession table. In the 2025-26 file `switches_on`, `help_*` and
  `matchup_potential_assists` are all zero.

Assumed, and worth knowing:
- **Expected is the opposing TEAM's offense, not the five it had on the floor.** A defense's starters mostly
  face starters. Team rows average over that; it is one reason a lineup's or a player's row is hard to read,
  and those rows are not shown.
- **Expected comes from the same season to date.** Twenty games in, an opponent's "usual" is twenty games of
  it, pooled toward the league. The interval carries that.
- **The start-type step is one league-wide contrast.** Every offense is taken to gain the same amount (on the
  logit scale) from starting off a turnover. Start types are made basket, defensive rebound, turnover, and
  everything else together (free throws, period starts).
- **Rates are combined on the logit scale at the level of a team's whole set of possessions**, not possession
  by possession: the expected rate is the mean of the per-possession expectations, and the deviation is the
  gap between its logit and the observed rate's. With effects of a few points the difference is negligible.
- **A foul drawn is a possession that ended at the line.** A non-shooting foul before the bonus does not end a
  possession and is not counted. A turnover is any turnover, forced or not.
- **Home and away, rest, and garbage time are not adjusted for.** Over a season a defense's opponents are
  about half at home.
- **Deviations are relative to the other defenses in the league** (DECISIONS.md 188).
- **Possessions in one game are treated as independent** when the interval is computed. The drift term, which
  is everything the binomial does not explain between two halves of a season, absorbs some of what that misses.

## Written summaries (step 3.2, 2026-11-27)

- **Not verified against the live API.** The request was built from the SDK's documentation and driven through
  the installed SDK (2.14) against a local transport, which checks what is sent and how a reply is read. It
  does not check that the API accepts the request. In particular the combination of a JSON output format with
  the server-side fallback option has not been tried; if the first real run is turned down for it, remove
  `betas` and `fallbacks` from `summaries.request` and call `cl.messages.create`.
- **Player names are found by their capital letters.** The check treats a capitalised word that is not in the
  cited rows as a name. A name written in lower case would not be seen.
- **A player row's label is "<name>: share of finishes while on the floor".** The table splits on the first
  colon and a space. A player name containing ": " would be split in the wrong place; none does.
- **A summary speaks of "the team".** The model is never given the team's name, so no sentence can use it.

## Verified from documentation only (nba_api docs, not a live file)

- Endpoints: `cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gameId}.json` and
  `.../boxscore/boxscore_{gameId}.json`.
- Boxscore: `game.gameTimeUTC`, `homeTeam` / `awayTeam` with `teamId`, `teamName`, `teamCity`, `teamTricode`,
  `players[]` with `personId`, `name`, `starter` (`"1"`).
- `scoreHome` / `scoreAway` are strings in the raw JSON.

## Still unverified

- **The schedule file** (`staticData/scheduleLeagueV2.json`: `leagueSchedule.gameDates[].games[]` with `gameId`,
  `gameDateTimeUTC`, `gameStatus`). Only `python -m tendency run/ingest` depends on it.
- **Whether `ingest` works from your home connection.** The CDN refused a plain browser request from your Mac.
  nba_api reaches it by sending a browser User-Agent and an nba.com Referer. `ingest` sends an honest
  User-Agent and will stop with exit code 3 if refused. Using nba_api for nightly pulls is your call; read the
  terms first.
- **Raw JSON vs the CSV mirror.** The mirror flattens `qualifiers` to a comma-joined string and scores to
  integers; `bulk.py` converts them back. A field the mirror drops would not show up here.
- **Bulk path stand-ins for the boxscore**: home/away from which score moves, tip time from the first action's
  `timeActual` (a few minutes after the scheduled tip), starters from who acts in period 1 before any sub-in.
  Team city and nickname come from a static table keyed by team id; full player names from the mirror's
  matchups archives (all 802 players in 2023-26 found), with "J. Tatum" as the fallback.
- **`off_timeouts_left`** still assumes 7 per game with no Q4 cap or overtime grants. `*_timeouts_used` is exact.
  Challenges count as a timeout used.
- **Playoffs, play-in and cup games** are untested; only regular-season archives were loaded.
- **NBA.com terms** for publishing derived data: not read. The real export is gitignored and was not pushed.
