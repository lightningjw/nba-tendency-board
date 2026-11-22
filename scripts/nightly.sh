#!/bin/bash
# The daily run on your own machine: fetch, score yesterday, log today. See README, "Prospective log".
#
#   PUBLISH=none ./scripts/nightly.sh     # default: write locally, commit nothing
#   PUBLISH=sums ./scripts/nightly.sh     # commit only predictions/*/SHA256SUMS (timestamps the log, shows no names)
#   PUBLISH=full ./scripts/nightly.sh     # commit predictions/ whole, per-game files included
#
# Which PUBLISH value is right is the terms decision (roadmap step 1.0a). Nothing is pushed unless you set it.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
SEASON="${SEASON:-2026-27}"
PUBLISH="${PUBLISH:-none}"

python scripts/fetch_nba_api.py --season "$SEASON" || echo "nightly: fetch reported failures; continuing with what is on disk"
python -m tendency score   || echo "nightly: nothing scored (no log for yesterday, or its games are not on disk yet)"
python -m tendency predict || { echo "nightly: predict failed; no log written today"; exit 1; }

case "$PUBLISH" in
  sums) git add -f predictions/*/SHA256SUMS ;;
  full) git add predictions ;;
  *)    echo "nightly: PUBLISH=none, nothing committed. The log is only proof once it is committed before tip-off."; exit 0 ;;
esac
git diff --cached --quiet || { git commit -m "Prediction log $(TZ=America/New_York date +%F)" && git push; }
