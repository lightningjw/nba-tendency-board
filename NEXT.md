# Next

## Done on 2026-11-16
- Field names verified on 3,690 real games; seven wrong assumptions fixed (ASSUMPTIONS.md).
- Three real seasons run end to end; all three heads beat their baselines on held-out data (README).

## Before publishing anything real
1. ~~Read NBA.com terms~~: done, `docs/terms-check.md`. Recommendation: code and derived aggregates may be public;
   play-by-play (raw rows, real replay files, live mode) may not. The decision on what to publish is still yours.
2. ~~Hand-check one real game~~: done with `python -m tendency inspect` on three games; see ASSUMPTIONS.md.
3. Decide how current-season games arrive: the mirror is refreshed by its maintainer, the CDN refuses plain
   requests, and nba_api gets through with browser-like headers.

## Then, still MVP
- ~~Prospective log~~: built (`train`, `predict`, `score`; README "Prospective log"). Rehearsed on 2025-26 as a
  replay: 14 dates, 105 games. **Nothing has been logged before a real game yet.** That starts on your machine:
  1. `pip install nba_api`, then `python scripts/fetch_nba_api.py --season 2026-27` until it reports nothing left.
  2. `python -m tendency verify-fields` on one fetched game: the real boxscore's field names are from docs only.
  3. `python -m tendency train --through <yesterday>`, then the cron line in the README.
- 2026-27 games: still not in the mirror on 2026-11-21 (only the WNBA 2026 archives are).
- ~~Real team names and full player names~~: done (`tendency/teams.py`, matchups archives).
- ~~CI~~: done, `.github/workflows/ci.yml`.

## Stubs, and where each plugs in (`tendency/stubs/`)
| Stub | Phase | Slots in at |
|---|---|---|
| `live.serve_live` | 3 | FastAPI service that serves the replay contract for a game in progress |
| `llm.summarize_team_report`, `llm.question_to_query` | 3 | reads team report JSON only; never answers from memory |
| `defense.defense_fingerprint` | 3 | post-game proxies; no coverage labels |

## Step 2.1, done 2026-11-23: the tendency layer (DECISIONS.md 86-109)
- `tendency/tendency_layer.py` replaces the stand-in in the export. `tendencies.py` stays, only as the thing the
  validation scores against. `tendency_validation.py` and `hot_hand.py` write two new contract files (1.2).
- Left out, on purpose or for lack of evidence:
  - A mean-reverting drift term (shrink harder because teams change) was not tried. The drift term only widens
    intervals. It is the obvious next thing to test on team rows, where the model only ties the stand-in.
  - Drift is measured half-season to half-season. A report 20 games into a season borrows that number as is.
  - The player baseline's uncertainty uses his own count elsewhere. When lineups barely overlap it is too small
    (a test world showed it); on real seasons the deviation intervals held 90% and 92%.
  - The PyMC fit is a validation tool. It is not wired into the export and has no report rows of its own.
  - The hot-hand report is league-wide only: no per-team or per-player rows, no shot difficulty (not in the feed).

## Step 2.2, done 2026-11-23: Phase 2 labels (DECISIONS.md 110 onward)
- Labels on the possession table: action style, assisted and assister, seconds to the first action and tempo,
  fast break, two-for-one. Two heads on top (action style, tempo), scored on held-out games only.
- Left out:
  - The two new heads are not in the pregame log. `train`, `predict` and `score` still handle three heads, so the
    model frozen on 2026-11-20 keeps scoring. Adding them means a new frozen model and a log schema change.
  - No passer-to-shooter pair rows: `assister_id` is on the table, nothing reads it yet.
  - No player-scope rows for the new labels (a player's pull-up share after timeouts, say).
  - `after_timeout` still counts a timeout taken in the middle of a possession (ASSUMPTIONS.md, "Debatable").

## Step 2.3, done 2026-11-26: the transformer (DECISIONS.md 135 onward)
- `tendency/transformer.py` and `transformer_net.py`: a small causal transformer over each game's possessions,
  scored beside the trees on the same rows. Optional (`pip install -e ".[torch]"`); it trains in its own process.
- It replaces the trees for shot style only. Everywhere else it is level or behind and stays an experiment.
- Left out:
  - Not in the pregame log. A frozen model is still three LightGBM heads.
  - No ensemble of seeds, although averaging three runs was better than any one of them: that would have been
    a fourth configuration and the step allows three.
  - The table has no start-of-possession fouls, timeouts or lineup, only the values read at the deciding event.
    The tempo head now borrows the previous possession's; a proper start snapshot in `possessions.py` is cleaner.
  - The sequence holds eligible possessions only (heaves and possessions cut off by the horn are skipped).

## Step 2.4, done 2026-11-26: coach-decision heads (DECISIONS.md 151 onward)
- `tendency/stoppages.py`: the dead-ball table (`processed/stoppages.parquet`) and a player-game table.
  `tendency/coach.py`: three decisions on top (a timeout at a stoppage, a player sitting after a foul, who closes
  a close game), each with a baseline from earlier games, and the "What the bench did" section of a team report.
- All three beat their baselines on both held-out splits. Phase 2 of the roadmap is complete.
- Left out:
  - Late deliberate fouls, challenges, and the next substitution in general. The manifest says why and how many
    of each the data holds. Late fouls are the easy one to add: the label (`take_foul_team_id`) is in the table.
  - Overtime. The timeout head stops at the end of regulation and the closing head ignores overtime.
  - Who comes IN. The foul head says a player sat; nothing says who replaced him.
  - Per-coach rows. Sections are per team and season; a mid-season coaching change is averaged over.
  - Not in the pregame log, and not tried with the transformer.
  - The timeout head's "timeouts left" is seven minus used. The fourth-quarter caps would sharpen it late.
  - The closing head is told nothing about injuries or ejections during the game beyond minutes and fouls.

## Known weaknesses worth fixing early
- Timeouts-left and the penalty flag use simplified rules (see ASSUMPTIONS.md).
- The finisher model renormalises independent binary scores; a ranking objective may calibrate better.
- No tempo features (shot clock is not in the feed; possession length would leak the outcome).
