# v0.1.0 (draft; not tagged)

Tag after the six Phase 1 pull requests are merged: `git tag -a v0.1.0 -m "MVP" && git push origin v0.1.0`.

## What is in it

- **Pipeline**: raw play-by-play to a possession table, features, three LightGBM heads (who finishes, how the
  possession ends, where the shot comes from), each with a baseline, and contract JSON for the app.
- **Real results** on three seasons (3,690 games, 738,812 possessions): all three heads beat their baselines
  on 492 held-out games and on a held-out season of 1,230 games, by small margins with 95% intervals clear of
  zero. Table in the README.
- **App**: Method, Team report, Replay with "guess before the model", Scorecard. Static, installable, offline.
- **Prospective log**: `train`, `predict`, `score`. A pregame file names a frozen model by SHA-256; the model is
  scored after the game. Contract 1.1 adds the scorecard's `prospective` block.
- **Hand-check tool**: `python -m tendency inspect <gameId>`.
- **Real team and player names** in the bulk path.
- **CI**: tests, the synthetic pipeline and both contract validators on every push.

## Honest limits

- Results are retrospective. The prospective flow was rehearsed on 14 dates of 2025-26 (105 games); nothing has
  been logged before a real game.
- The tendency layer is a labeled stand-in, not the partial-pooling model.
- The shot-location baseline ignores the lineup.
- The cdn.nba.com path and `scripts/fetch_nba_api.py` have never run against nba.com.
- The public app shows simulated games; see `docs/terms-check.md`.

## Changed since the first real run (2026-11-16)

- Four segmentation and lineup bugs fixed after reading real games: unexplained changes of possession 47 to 11,
  possessions without a clean lineup 86 to 16. Possession count 738,771 to 738,812. No scorecard number moved by
  more than 0.0003.
