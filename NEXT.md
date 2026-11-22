# Next

## Done on 2026-11-16
- Field names verified on 3,690 real games; seven wrong assumptions fixed (ASSUMPTIONS.md).
- Three real seasons run end to end; all three heads beat their baselines on held-out data (README).

## Before publishing anything real
1. Read NBA.com terms. Until then the real export stays local.
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
| `tendency_layer.fit_tendency_layer` | 2 | replaces `tendencies.team_reports`; same schema, set `method.final = true` |
| `transformer.run_transformer` | 2 | beside `models.run_models`; same `ModelRun` out |
| `coach_heads.label_coach_decisions` | 2 | new labels in `possessions.py`, new head ids in the manifest schema |
| `live.serve_live` | 3 | FastAPI service that serves the replay contract for a game in progress |
| `llm.summarize_team_report`, `llm.question_to_query` | 3 | reads team report JSON only; never answers from memory |
| `defense.defense_fingerprint` | 3 | post-game proxies; no coverage labels |

## Known weaknesses worth fixing early
- Player tendency rows ignore who else is on the floor.
- Tendency intervals are not adjusted for selecting the top rows.
- Timeouts-left and the penalty flag use simplified rules (see ASSUMPTIONS.md).
- The finisher model renormalises independent binary scores; a ranking objective may calibrate better.
- No tempo features (shot clock is not in the feed; possession length would leak the outcome).
