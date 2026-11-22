"""Tendency Board pipeline: raw play-by-play -> possessions -> features -> models -> contract JSON."""
__version__ = "0.1.0"
CONTRACT_VERSION = "1.1"       # 1.1 adds the optional `prospective` block to scorecard.json; 1.0 exports still validate
