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
| Lineups reconstruct to five a side | 99.99% of 738,771 possessions |
| Unexplained changes of possession | 47 of 738,771 |
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
  Team names are tricodes and player names are "J. Tatum" style.
- **`off_timeouts_left`** still assumes 7 per game with no Q4 cap or overtime grants. `*_timeouts_used` is exact.
  Challenges count as a timeout used.
- **Playoffs, play-in and cup games** are untested; only regular-season archives were loaded.
- **NBA.com terms** for publishing derived data: not read. The real export is gitignored and was not pushed.
