"""Orchestration: raw files -> possessions.parquet -> features -> models -> contract JSON."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .export import export_all
from .features import build_features
from .models import run_models
from .possessions import FeedShapeError, segment_game
from .stoppages import build_stoppages


class PipelineError(RuntimeError):
    pass


def build_possessions(root: Path, log=print, stoppages: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Segment every raw game under root/raw into possessions and write root/processed/*.parquet.

    With `stoppages`, also write the dead-ball table and the player-game table beside it (step 2.4).
    The prospective log does not ask for them, so `train` and `predict` do exactly what they did before.
    """
    pbp_dir, box_dir = root / "raw" / "pbp", root / "raw" / "box"
    files = sorted(pbp_dir.glob("*.json")) if pbp_dir.exists() else []
    if not files:
        raise PipelineError(f"no raw play-by-play under {pbp_dir}; run `python -m tendency ingest` (or `sample`) first")
    frames, games, bad, stop_frames, player_frames, stop_bad = [], [], [], [], [], []
    for f in files:
        box_f = box_dir / f.name
        try:
            if not box_f.exists():
                raise FeedShapeError("boxscore file missing")
            pbp, box = json.loads(f.read_text()), json.loads(box_f.read_text())
            r, info = segment_game(pbp, box)
            if not info["tipoff_utc"]:
                raise FeedShapeError("boxscore has no tip time; needed to order games")
        except (FeedShapeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            bad.append((f.stem, f"{type(e).__name__}: {e}"))
            continue
        if stoppages:                                     # its own guard: a game the dead-ball table cannot read keeps its possessions
            try:
                st, pl, checks = build_stoppages(pbp, box, r)
                info.update(checks)
                stop_frames.append(pd.DataFrame(st))
                player_frames.append(pd.DataFrame(pl))
            except (FeedShapeError, KeyError, TypeError, ValueError) as e:
                stop_bad.append((f.stem, f"{type(e).__name__}: {e}"))
        frames.append(pd.DataFrame(r))                    # per game, so 700k+ row dicts never sit in memory at once
        games.append(info)
    for gid, err in bad[:10]:
        log(f"  ! skipped {gid}: {err}")
    if not games:
        raise PipelineError(f"none of {len(files)} raw games could be segmented. First error: {bad[0][1]}. "
                            "The feed's field names probably differ from tendency/cdn_fields.py; run `python -m tendency verify-fields`.")
    if len(bad) > 0.2 * len(files):
        raise PipelineError(f"{len(bad)} of {len(files)} games failed segmentation; refusing to model on a biased subset.")
    kinds = {g["synthetic"] for g in games}
    if len(kinds) > 1:
        raise PipelineError("raw directory mixes synthetic and real games; keep them in separate --root folders")
    poss = pd.concat(frames, ignore_index=True)
    del frames
    poss["tipoff"] = pd.to_datetime(poss["tipoff_utc"], utc=True, format="ISO8601").dt.tz_localize(None)
    poss["finisher_id"] = poss["finisher_id"].astype("Int64")
    poss["shot_made"] = poss["shot_made"].astype("boolean")
    for c in ("assisted", "fast_break", "two_for_one_taken"):         # true / false / not defined
        poss[c] = poss[c].astype("boolean")
    poss["assister_id"] = poss["assister_id"].astype("Int64")
    for c in ("first_action_clock_s", "secs_to_first_action"):
        poss[c] = poss[c].astype("float64")
    proc = root / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    poss.to_parquet(proc / "possessions.parquet", index=False)
    (proc / "games.json").write_text(json.dumps(games))
    if stoppages:
        if not stop_frames:
            raise PipelineError(f"the dead-ball table could not be built for any game. First error: {stop_bad[0][1]}")
        stops = pd.concat(stop_frames, ignore_index=True)
        stops["trigger_team_id"] = stops["trigger_team_id"].astype("Int64")
        stops["take_foul_team_id"] = stops["take_foul_team_id"].astype("Int64")
        for c in ("secs_since_timeout", "timeout_gap_s"):
            stops[c] = stops[c].astype("float64")
        players = pd.concat(player_frames, ignore_index=True)
        for c in ("fouls_q3", "margin_q3", "margin_close"):
            players[c] = players[c].astype("Int64")
        players["on_floor_q3"] = players["on_floor_q3"].astype("boolean")
        stops.to_parquet(proc / "stoppages.parquet", index=False)
        players.to_parquet(proc / "player_games.parquet", index=False)
        n_pf, ok_pf = sum(g.get("pf_checks", 0) for g in games), sum(g.get("pf_agree", 0) for g in games)
        for gid, err in stop_bad[:10]:
            log(f"  ! no stoppage rows for {gid}: {err}")
        log(f"stoppages: {len(stops):,} stoppages ({len(stops) / max(1, len(games) - len(stop_bad)):.0f} a game)"
            + (f"; {len(stop_bad)} games could not be read" if stop_bad else "")
            + (f"; personal-foul counts match the feed's own on {ok_pf:,} of {n_pf:,} fouls" if n_pf else ""))
    unknown = sorted({t for g in games for t in g["unknown_action_types"]})
    log(f"possessions: {len(poss):,} possessions from {len(games)} games ({len(bad)} skipped); "
        f"{int((poss['ending_type'] == 'unknown').sum())} unexplained changes of possession; "
        f"lineup ok on {poss['lineup_ok'].mean():.1%}" + (f"; unknown actionType values: {unknown}" if unknown else ""))
    return poss, games


def run(root: Path, out: Path, max_replays: int = 30, fields_verified: bool = False, log=print, predictions: Path | None = None,
        transformer: bool | None = None, transformer_epochs: int | None = None,
        summaries: str = "keep", summary_model: str | None = None, summary_client=None) -> dict:
    """Everything after ingest. `transformer`: None trains the experimental transformer when PyTorch is installed,
    True insists on it, False skips it. `transformer_epochs` caps its training (tests use it to stay quick)."""
    poss, games = build_possessions(root, log, stoppages=True)
    synthetic = bool(games[0]["synthetic"])
    feat = build_features(poss)
    feat.poss.drop(columns=["off_lineup"]).to_parquet(root / "processed" / "features_possession.parquet", index=False)
    feat.cand.to_parquet(root / "processed" / "features_candidate.parquet", index=False)
    log(f"features: {len(feat.poss):,} model-eligible possessions, {len(feat.cand):,} candidate rows")
    model_run = run_models(feat, log)
    from .transformer import run_transformer, torch_available
    if transformer and not torch_available():
        raise PipelineError('the transformer was asked for but PyTorch is not installed; pip install -e ".[torch]"')
    if transformer or (transformer is None and torch_available()):
        model_run = run_transformer(feat, model_run, log, transformer_epochs)
    else:
        log("transformer: skipped" + ("" if torch_available() else " (PyTorch is not installed)") + "; the scorecard has the tree models only")
    from .coach import run_coach
    stops = pd.read_parquet(root / "processed" / "stoppages.parquet")
    coach = run_coach(stops, pd.read_parquet(root / "processed" / "player_games.parquet"), model_run.splits, log)
    from .summaries import MODEL
    return export_all(out, poss, games, model_run, synthetic, max_replays, fields_verified, log, predictions, coach=coach, stops=stops,
                      summaries=summaries, summary_model=summary_model or MODEL, summary_client=summary_client)
