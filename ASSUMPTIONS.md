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
