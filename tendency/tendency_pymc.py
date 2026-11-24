"""The tendency model fitted in full with PyMC (NUTS), instead of by plug-in empirical Bayes.

Optional: `pip install -e ".[bayes]"`. Nothing in the pipeline imports this module; it exists so the
half-season validation can ask whether a full Bayesian fit predicts better than the fast one.

Same structure as `tendency_layer.py`, with three things the plug-in fit cannot do:

* exact binomial likelihoods for the situation AND for the baseline's raw material (the team's or
  player's other possessions), instead of normal approximations to empirical logits;
* uncertainty in the variance components carried into every interval, and the spread of
  team-by-situation interactions partially pooled across the 56 (situation, outcome) groups;
* players nested in teams: each team has its own scale for how far its players' situational shares
  stray from their lineup-adjusted usage, pooled across teams.

The lineup adjustment enters as an offset: the logit of the share his usage and the four teammates
on the floor predict (`candidates` computes it from choose-one strengths fitted on his other possessions).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.special import expit

SEED = 20261123


def fit(cand: pd.DataFrame, draws: int = 1000, tune: int = 1000, chains: int = 4, method: str = "nuts", progressbar: bool = False) -> dict:
    """Fit the model to `tendency_layer.candidates(...)`. Returns {"keys", "rate" (cells x draws), "seconds", "diagnostics"}.

    `method` is "nuts" or "advi". `rate` holds posterior draws of each cell's own rate, in `cand` row order.
    """
    import pymc as pm                                    # imported here so the package works without the extra

    main = cand[cand["kind"] == "league_same_situation"].reset_index(drop=True)
    sit = cand[cand["kind"] == "team_elsewhere_plus_league_shift"].reset_index(drop=True)
    ply = cand[cand["kind"] == "player_lineup_adjusted"].reset_index(drop=True)
    g_main, main_groups = pd.factorize(main["group"])
    g_sit, sit_groups = pd.factorize(sit["group"])
    g_ply, ply_groups = pd.factorize(ply["group"])
    t_ply, ply_teams = pd.factorize(ply["team_id"])
    out_center = sit.groupby(g_sit)["out_logit"].mean().values            # each group's typical rate elsewhere, as a prior centre only

    with pm.Model() as model:
        # team, all possessions: league rate + team effect
        mu = pm.Normal("mu_main", 0.0, 0.5, shape=len(main_groups))
        sd_main = pm.HalfNormal("sd_main", 0.5, shape=len(main_groups))
        z_main = pm.Normal("z_main", 0.0, 1.0, shape=len(main))
        eta_main = main["base_logit"].values + mu[g_main] + sd_main[g_main] * z_main
        pm.Binomial("k_main", n=main["n"].values.astype(int), logit_p=eta_main, observed=main["k"].values.astype(int))

        # team in a situation: its rate elsewhere + the league's contrast + the team-by-situation interaction
        a = pm.Normal("a_sit", out_center[g_sit], 1.5, shape=len(sit))            # wide: the other possessions decide it
        shift = pm.Normal("shift", 0.0, 1.0, shape=len(sit_groups))
        log_tau_mu = pm.Normal("log_tau_mu", np.log(0.1), 1.0)
        log_tau_sd = pm.HalfNormal("log_tau_sd", 1.0)
        tau = pm.Deterministic("tau_sit", pm.math.exp(log_tau_mu + log_tau_sd * pm.Normal("z_tau", 0.0, 1.0, shape=len(sit_groups))))
        b = tau[g_sit] * pm.Normal("z_sit", 0.0, 1.0, shape=len(sit))
        pm.Binomial("k_sit_out", n=sit["n_out"].values.astype(int), logit_p=a, observed=sit["k_out"].values.astype(int))
        eta_sit = a + shift[g_sit] + b
        pm.Binomial("k_sit", n=sit["n"].values.astype(int), logit_p=eta_sit, observed=sit["k"].values.astype(int))

        # player in a situation: lineup-adjusted offset + his own miss from it elsewhere + the situational deviation
        u = pm.Normal("u_ply", 0.0, 1.0, shape=len(ply))                          # wide: his other possessions decide it
        tau_p = pm.HalfNormal("tau_ply", 0.5, shape=len(ply_groups))
        lam_sd = pm.HalfNormal("lam_sd", 0.5)
        lam = pm.Deterministic("lam_team", pm.math.exp(lam_sd * pm.Normal("z_lam", 0.0, 1.0, shape=len(ply_teams))))
        c = tau_p[g_ply] * lam[t_ply] * pm.Normal("z_ply", 0.0, 1.0, shape=len(ply))
        pm.Binomial("k_ply_out", n=ply["n_out"].values.astype(int), logit_p=ply["out_logit"].values + u, observed=ply["k_out"].values.astype(int))
        eta_ply = ply["base_logit"].values + u + c
        pm.Binomial("k_ply", n=ply["n"].values.astype(int), logit_p=eta_ply, observed=ply["k"].values.astype(int))

        pm.Deterministic("eta_main", eta_main)
        pm.Deterministic("eta_sit", eta_sit)
        pm.Deterministic("eta_ply", eta_ply)

        t0 = time.time()
        diag = {"method": method, "n_parameters": int(sum(v.size for v in model.initial_point().values()))}
        if method == "advi":
            approx = pm.fit(n=30000, method="advi", random_seed=SEED, progressbar=progressbar)
            idata = approx.sample(draws * chains, random_seed=SEED)
        else:
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains, target_accept=0.9, random_seed=SEED,
                              progressbar=progressbar, compute_convergence_checks=False)
        seconds = time.time() - t0

    post = idata["posterior"]

    def flat(name: str) -> np.ndarray:
        v = np.asarray(post[name].values)
        return v.reshape(-1, v.shape[-1]).T                                      # cells x draws

    if method != "advi":
        import arviz as az
        try:
            rhat = az.rhat(idata, var_names=["sd_main", "tau_sit", "tau_ply", "lam_team", "shift"])
            diag["max_rhat"] = round(float(max(float(np.nanmax(np.asarray(rhat[v].values))) for v in rhat.data_vars)), 3)
        except Exception as e:                                                   # diagnostics must not lose a finished fit
            diag["max_rhat"] = None
            diag["rhat_error"] = f"{type(e).__name__}: {e}"[:200]
        try:
            diag["divergences"] = int(np.asarray(idata["sample_stats"]["diverging"].values).sum())
        except Exception:
            diag["divergences"] = None
    keys = np.concatenate([main["key"].values, sit["key"].values, ply["key"].values])
    rate = expit(np.concatenate([flat("eta_main"), flat("eta_sit"), flat("eta_ply")], axis=0))
    return {"keys": keys, "rate": rate, "seconds": round(seconds, 1), "diagnostics": diag,
            "tau_sit_median": {g: round(float(np.median(flat("tau_sit")[i])), 4) for i, g in enumerate(sit_groups)}}


def pred(cand: pd.DataFrame, **kw):
    """Fit and wrap as a `tendency_validation.Pred`, plus the fit's runtime and diagnostics."""
    from .tendency_validation import Pred
    f = fit(cand, **kw)
    return Pred(pd.Index(f["keys"]), f["rate"].mean(axis=1), f["rate"]), f
