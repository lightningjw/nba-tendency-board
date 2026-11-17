"""Phase 2: partial pooling across team, player and situation (PyMC or empirical Bayes).

Replaces tendencies.team_reports. Keep the output shape (team_report.schema.json) and set
method.final = true so the app drops its "simple shrinkage" label.
"""
from . import NotBuilt


def fit_tendency_layer(possessions):
    raise NotBuilt("The partial-pooling tendency layer", "Phase 2")
