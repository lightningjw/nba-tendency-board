"""Command line.

    python -m tendency run --start 2026-10-20 --end 2026-11-15     # ingest + possessions + features + models + export
    python -m tendency bulk --seasons 2023 2024 2025               # whole seasons from the GitHub mirror, then build
    python -m tendency sample                                      # same pipeline on synthetic games (no network)
    python -m tendency ingest --start ... --end ...                # download only
    python -m tendency build                                       # everything after ingest, on what is already on disk
    python -m tendency verify-fields <pbp.json> [<box.json>]       # check assumed feed names against one real game
    python -m tendency validate app/data                           # validate an export directory against the schemas
    python -m tendency inspect <gameId> [--only odd]               # one game's possessions beside the raw actions

Prospective log (see tendency/prospective.py):
    python -m tendency train --through 2026-11-16                  # freeze models on games up to that date
    python -m tendency predict [--date YYYY-MM-DD]                 # before tip-off: one pregame file per game (default: today, US Eastern)
    python -m tendency score [--date YYYY-MM-DD]                   # after the games: score the frozen model (default: yesterday)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tendency", description="Tendency Board pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, root_default: str) -> None:
        p.add_argument("--root", type=Path, default=REPO / root_default, help="data folder (raw/ and processed/ live here)")
        p.add_argument("--out", type=Path, default=REPO / "app" / "data", help="where the app's contract JSON is written")
        p.add_argument("--max-replays", type=int, default=30, help="how many held-out games to export for the replay view")
        p.add_argument("--no-transformer", action="store_true",
                       help="skip the experimental transformer (it trains by default when PyTorch is installed)")
        p.add_argument("--predictions", type=Path, default=None,
                       help="prospective log folder to show on the scorecard (default: predictions/ for real data, none for `sample`)")

    def dates(p: argparse.ArgumentParser) -> None:
        p.add_argument("--start", type=date.fromisoformat, required=True, help="first UTC tip date, YYYY-MM-DD")
        p.add_argument("--end", type=date.fromisoformat, required=True, help="last UTC tip date, YYYY-MM-DD")

    p = sub.add_parser("run", help="ingest a date range, then build and export")
    dates(p)
    common(p, "data/real")
    p.add_argument("--fields-verified", action="store_true", help="set once verify-fields passes on a real game")
    p = sub.add_parser("ingest", help="download raw games only")
    dates(p)
    p.add_argument("--root", type=Path, default=REPO / "data" / "real")
    p = sub.add_parser("build", help="possessions, features, models and export from raw files already on disk")
    common(p, "data/real")
    p.add_argument("--fields-verified", action="store_true")
    p = sub.add_parser("sample", help="generate synthetic games and run the same pipeline on them")
    common(p, "data/synthetic")
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("bulk", help="load whole seasons from the shufinskiy/nba_data GitHub mirror, then build and export")
    p.add_argument("--seasons", type=int, nargs="+", required=True, help="season start years, e.g. 2023 2024 2025")
    common(p, "data/real")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--no-full-names", action="store_true", help="skip the matchups archives; players keep the feed's 'J. Tatum' form")
    p = sub.add_parser("verify-fields", help="check assumed feed names against one real game")
    p.add_argument("pbp")
    p.add_argument("box", nargs="?")
    p = sub.add_parser("validate", help="validate an export directory against the contract schemas")
    p.add_argument("dir", type=Path)
    for name, text in (("train", "freeze the three models on games up to a date"),
                       ("predict", "write the pregame log for a date, before tip-off"),
                       ("score", "score the frozen model on a logged date once its games are final")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--root", type=Path, default=REPO / "data" / "real")
        if name == "train":
            p.add_argument("--through", required=True, help="last US Eastern game date to train on, YYYY-MM-DD")
            p.add_argument("--force", action="store_true", help="overwrite an existing frozen model (orphans logs that name it)")
            continue
        p.add_argument("--date", help="US Eastern game date, YYYY-MM-DD (default: today for predict, yesterday for score)")
        p.add_argument("--log-dir", type=Path, default=REPO / "predictions", help="where pregame files and the prospective scorecard live")
        p.add_argument("--model", type=Path, help="frozen model folder (default: the one the log names, or the latest trained before the date)")
        if name == "predict":
            p.add_argument("--replay", action="store_true", help="simulate a pregame log for a date whose games are already on disk")
            p.add_argument("--force", action="store_true", help="rewrite pregame files that already exist")
    p = sub.add_parser("inspect", help="print one game's possessions beside the raw actions that produced them")
    p.add_argument("game_id", help="ten-digit game id, e.g. 0022500001")
    p.add_argument("--root", type=Path, default=REPO / "data" / "real")
    p.add_argument("--only", choices=["odd"], help="'odd' shows only possessions with unusual flags")
    a = ap.parse_args(argv)

    from .contract import ContractError
    from .ingest import HostBlockedError, IngestError
    from .inspect import InspectError
    from .models import NotEnoughData
    from .pipeline import PipelineError
    from .prospective import ProspectiveError
    try:
        if a.cmd == "verify-fields":
            from .verify_fields import main as vf
            return vf(a.pbp, a.box)
        if a.cmd == "validate":
            from .contract import validate_export_dir
            print(f"valid: {validate_export_dir(a.dir)}")
            return 0
        if a.cmd == "inspect":
            from .inspect import inspect_game
            text, summary = inspect_game(a.game_id, a.root, a.only == "odd")
            print(text)
            return 0 if summary["reconciles"] else 2
        if a.cmd in ("train", "predict", "score"):
            from . import prospective as pro
            if a.cmd == "train":
                pro.train(a.root, pro.date_arg(a.through, 0), a.force)
            elif a.cmd == "predict":
                res = pro.predict(a.root, pro.date_arg(a.date, 0), a.log_dir, a.replay, a.force, a.model)
                if not res["written"] and res["skipped"]:
                    print("predict: nothing new to log")
            else:
                pro.score(a.root, pro.date_arg(a.date, -1), a.log_dir, a.model)
            return 0
        if a.cmd in ("run", "ingest"):
            if a.end < a.start:
                ap.error("--end is before --start")
            from .ingest import ingest_range
            res = ingest_range(a.start, a.end, a.root)
            print(f"ingest: fetched {res['fetched']}, already had {res['skipped']}, failed {len(res['failed'])}")
            if a.cmd == "ingest":
                return 0
        if a.cmd == "bulk":
            from .bulk import load_seasons
            res = load_seasons(a.seasons, a.root, full_names=not a.no_full_names)
            print(f"bulk: wrote {res['written']} games, refreshed {res['refreshed']} boxscores, already had {res['skipped']}, "
                  f"failed {len(res['failed'])}; full names for {res['named_players']} players")
            for f in res["failed"][:5]:
                print(f"  ! {f['game_id']}: {f['error']}")
            if a.download_only:
                return 0
            a.fields_verified = True                       # the archive is the feed itself; see ASSUMPTIONS.md
        if a.cmd == "sample":
            from .synth import write_synthetic_raw
            ids = write_synthetic_raw(a.root, a.seed)
            print(f"sample: wrote {len(ids)} synthetic games to {a.root / 'raw'}")
        from .pipeline import run
        predictions = a.predictions if a.predictions is not None else (None if a.cmd == "sample" else REPO / "predictions")
        run(a.root, a.out, a.max_replays, getattr(a, "fields_verified", False), predictions=predictions,
            transformer=False if a.no_transformer else None)
        return 0
    except HostBlockedError as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 3
    except (IngestError, PipelineError, NotEnoughData, ContractError, InspectError, ProspectiveError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
