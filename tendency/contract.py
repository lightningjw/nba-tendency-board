"""The data contract: load the JSON Schemas and validate exported files against them.

The app ships a small validator that understands only the keywords in
SUPPORTED_KEYWORDS, so the schemas are restricted to that subset (enforced by
tests/test_contract.py). That keeps "both sides validate" true without a
JavaScript dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "contract" / "schemas"
SCHEMAS = ["manifest", "teams", "team_report", "games", "replay", "scorecard", "tendency_validation", "hot_hand"]      # what the app reads
OPTIONAL_FILES = ["tendency_validation", "hot_hand"]       # contract 1.2; an older export has neither and is still valid
LOG_SCHEMAS = ["prediction_log"]                                                    # written by `predict`, read by `score`
SUPPORTED_KEYWORDS = {"$id", "$ref", "$defs", "title", "type", "properties", "required", "additionalProperties",
                      "items", "enum", "const", "minimum", "maximum", "minItems", "maxItems"}


class ContractError(ValueError):
    pass


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())


def validate(obj: dict, name: str, where: str = "") -> None:
    errors = sorted(Draft202012Validator(load_schema(name)).iter_errors(obj), key=lambda e: list(e.absolute_path))
    if errors:
        lines = [f"  at /{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}" for e in errors[:8]]
        raise ContractError(f"{where or name} violates {name}.schema.json ({len(errors)} errors):\n" + "\n".join(lines))


def validate_export_dir(out: Path) -> dict:
    """Validate every contract file in an export directory. Returns counts per schema."""
    counts = {}

    def check(path: Path, name: str) -> dict:
        obj = json.loads(path.read_text())
        validate(obj, name, str(path))
        counts[name] = counts.get(name, 0) + 1
        return obj

    manifest = check(out / "manifest.json", "manifest")
    files = manifest["files"]
    teams = check(out / files["teams"], "teams")
    games = check(out / files["games"], "games")
    check(out / files["scorecard"], "scorecard")
    kind = manifest["data_kind"]
    for name in OPTIONAL_FILES:
        if name in files and check(out / files[name], name)["data_kind"] != kind:
            raise ContractError(f"{files[name]} data_kind differs from the manifest")
    for t in teams["teams"]:
        rep = check(out / files["team_reports_dir"] / f"{t['team_id']}.json", "team_report")
        if rep["data_kind"] != kind:
            raise ContractError(f"team report {t['team_id']} data_kind differs from the manifest")
    for g in games["games"]:
        rep = check(out / files["replays_dir"] / f"{g['game_id']}.json", "replay")
        if rep["data_kind"] != kind:
            raise ContractError(f"replay {g['game_id']} data_kind differs from the manifest")
    return counts
