"""Step 3.3: the one table the ask box queries, written beside the possession table as Parquet.

One row per cell: a team or a player, in one situation, for one outcome, in one season. The numbers
are the tendency layer's own (`tendency_layer.candidates` and `estimate`), so a cell that is also on
a team page carries exactly the figures printed there. Two things are added:

* every season in the possession table gets its cells, not only the latest one;
* a cell too thin for the layer to estimate is still listed with its count (`estimated` false), so
  a question about it gets "12 of 31, too few to estimate" and not silence.

Each row also carries the cell's own rate with an interval (`rate`, `rate_lo`, `rate_hi`). The
deviation answers "against its baseline"; the rate is what two teams or two seasons are compared on.

Built at export time by the pipeline (it needs pandas and the layer). The service that reads it
(`ask.py`) needs only DuckDB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from . import tendency_layer as layer
from .summaries import UNIT, VERSUS
from .tendencies import SITUATIONS, outcome_specs

FILE = "ask_cells.parquet"
PLAYER_OUTCOME = "finisher"
PLAYER_LABEL = "Share of finishes while on the floor"
PLAYER_UNIT = "possessions on the floor"
COLUMNS = ["season", "scope", "team_id", "tricode", "team_name", "player_id", "player_name", "situation", "situation_label", "outcome",
           "outcome_label", "unit", "k", "n", "estimated", "baseline_kind", "baseline_is", "baseline_rate", "raw_rate", "shrunk_rate",
           "deviation", "dev_lo", "dev_hi", "excludes_zero", "rate", "rate_lo", "rate_hi", "drift_included", "n_games", "as_of_utc",
           "data_kind"]


def outcome_id(head: str, cls) -> str:
    return PLAYER_OUTCOME if head == "finisher" else f"{head}:{cls}"


def _directory(games: list[dict]) -> tuple[dict[int, dict], dict[int, str]]:
    """Team tricode and name, and player names, as the latest game has them (the same rule the export uses)."""
    teams: dict[int, dict] = {}
    names: dict[int, str] = {}
    for g in sorted(games, key=lambda x: x["tipoff_utc"]):
        for tid, t in g["teams"].items():
            teams[int(tid)] = {"tricode": t["tricode"], "name": t["name"]}
            for p in t["players"]:
                names[int(p["player_id"])] = p["name"]
    return teams, names


def build(poss: pd.DataFrame, games: list[dict]) -> pd.DataFrame:
    teams, names = _directory(games)
    kind = "synthetic" if games and games[0].get("synthetic") else "real"
    sit_label = dict(SITUATIONS)
    seasons = sorted(int(s) for s in poss["season"].unique())
    out = []
    for season in seasons:
        ps = poss[poss["season"] == season]
        earlier = [s for s in seasons if s < season]
        drift = layer.estimate_drift(poss[poss["season"] == max(earlier)]) if earlier else None      # never measured on the season described
        has_drift = bool(drift) and any(v > 0 for v in drift["dev"].values())
        labels = {outcome_id(h, c): lab for h, c, _, _, lab in outcome_specs(layer._eligible(ps))}
        n_games = ps.groupby("off_team_id")["game_id"].nunique()
        as_of = pd.Timestamp(ps["tipoff"].max()).strftime("%Y-%m-%dT%H:%M:%SZ")

        def common(team_id: int, scope: str, player_id, sid: str, head: str, cls) -> dict:
            t = teams.get(int(team_id), {"tricode": str(team_id), "name": str(team_id)})
            oid = outcome_id(head, cls)
            player = scope == "player"
            return {"season": season, "scope": scope, "team_id": int(team_id), "tricode": t["tricode"], "team_name": t["name"],
                    "player_id": int(player_id) if player else None, "player_name": names.get(int(player_id), str(int(player_id))) if player else None,
                    "situation": sid, "situation_label": sit_label[sid], "outcome": oid,
                    "outcome_label": PLAYER_LABEL if player else labels.get(oid, oid), "unit": PLAYER_UNIT if player else UNIT.get(head, "possessions"),
                    "drift_included": has_drift, "n_games": int(n_games.get(team_id, 0)), "as_of_utc": as_of, "data_kind": kind}

        est = layer.estimate(layer.candidates(ps), drift)
        seen = set()
        for c in est.itertuples():
            row, _ = layer._row_and_z(c, names)
            seen.add(c.key)
            out.append({**common(c.team_id, c.scope, c.player_id, c.sid, c.head, c.cls), "k": int(c.k), "n": int(c.n), "estimated": True,
                        "baseline_kind": c.kind, "baseline_is": VERSUS.get(c.kind, c.kind), "baseline_rate": row["baseline_rate"],
                        "raw_rate": row["raw_rate"], "shrunk_rate": row["shrunk_rate"], "deviation": row["deviation"],
                        "dev_lo": row["interval"][0], "dev_hi": row["interval"][1], "excludes_zero": row["excludes_zero"],
                        "rate": round(float(expit(c.rate_logit)), 4), "rate_lo": round(float(expit(c.rate_logit - layer.Z * c.rate_sd)), 4),
                        "rate_hi": round(float(expit(c.rate_logit + layer.Z * c.rate_sd)), 4)})
        thin = layer.cell_counts(ps).groupby("key", sort=False)[["k", "n"]].sum()
        for key, k, n in zip(thin.index, thin["k"].values, thin["n"].values):
            if key in seen or n <= 0:
                continue
            team_id, scope, player_id, sid, head, cls = key.split("|")
            out.append({**common(int(team_id), scope, int(player_id), sid, head, None if cls == "-" else cls), "k": int(k), "n": int(n),
                        "estimated": False, "raw_rate": round(float(k / n), 4)})
    df = pd.DataFrame(out, columns=COLUMNS)
    df["player_id"] = df["player_id"].astype("Int64")
    for c in ("estimated", "excludes_zero", "drift_included"):
        df[c] = df[c].astype("boolean")
    for c in ("baseline_rate", "raw_rate", "shrunk_rate", "deviation", "dev_lo", "dev_hi", "rate", "rate_lo", "rate_hi"):
        df[c] = df[c].astype("float64")
    return df.sort_values(["season", "scope", "tricode", "situation", "outcome", "player_id"], kind="stable").reset_index(drop=True)


def write(root: Path, poss: pd.DataFrame, games: list[dict], log=print) -> Path:
    df = build(poss, games)
    path = Path(root) / "processed" / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    est = int(df["estimated"].sum())
    log(f"ask table: {len(df):,} cells ({est:,} estimated, {len(df) - est:,} counted only) over {df['season'].nunique()} season(s) in {path}")
    return path
