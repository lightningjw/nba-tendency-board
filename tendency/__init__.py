"""Tendency Board pipeline: raw play-by-play -> possessions -> features -> models -> contract JSON."""
__version__ = "0.1.0"
# 1.1 adds the optional `prospective` block to scorecard.json.
# 1.2 adds tendency_validation.json and hot_hand.json (both optional, listed in manifest.files), and optional
#     team-report fields (threshold, drift_included, prob_beyond). 1.0 and 1.1 exports still validate and render.
CONTRACT_VERSION = "1.2"
