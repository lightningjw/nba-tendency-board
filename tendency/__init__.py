"""Tendency Board pipeline: raw play-by-play -> possessions -> features -> models -> contract JSON."""
__version__ = "0.1.0"
# 1.1 adds the optional `prospective` block to scorecard.json.
# 1.2 adds tendency_validation.json and hot_hand.json (both optional, listed in manifest.files), and optional
#     team-report fields (threshold, drift_included, prob_beyond). 1.0 and 1.1 exports still validate and render.
# 1.3 (step 2.2) adds two heads (action_style, tempo) to the manifest, scorecard and replays, a `labels` coverage
#     list to the manifest, step 2.2 fields on a replay possession's `actual`, and four outcome families on team
#     reports. All optional: 1.0 to 1.2 exports still validate and render.
# 1.4 (step 2.3) lets the scorecard carry a second model: an optional `models` list, and on each head an optional
#     `model_id` (the head's primary model) and `challengers` (the other model on the same possessions, with its
#     paired difference against the primary). An export made without PyTorch has neither and is still valid.
CONTRACT_VERSION = "1.7"
