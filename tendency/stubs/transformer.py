"""Phase 2: small transformer over possession tokens, one head per outcome.

Plugs in beside models.run_models: same Features in, same ModelRun out (a `preds` frame with
`<head>_model` probability lists and scorecard entries). The export step and the app need no change;
add a `model_id` field to the contract if both models should be shown side by side.
"""
from . import NotBuilt


def run_transformer(features):
    raise NotBuilt("The transformer predictor", "Phase 2")
