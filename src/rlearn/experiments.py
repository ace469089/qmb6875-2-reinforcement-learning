"""
experiments.py -- WHY THESE SETTINGS?  Evidence for every hyperparameter choice.

Reinforcement learning has knobs (learning rate alpha, exploration eps, number
of episodes).  Instead of asserting "I used alpha = 0.1", this module runs the
alternatives and records what happens, so every choice in docs/DECISIONS.md is
backed by a table.

    python -m rlearn.experiments        (about 1-2 minutes)
"""

from __future__ import annotations

import numpy as np

from . import gold_rl, offer_rl


def offer_grid(alphas=(0.01, 0.05, 0.1, 0.3, "decay"), epss=(0.05, 0.2, 0.5),
               seeds=range(5), episodes: int = 20_000):
    """Error in the learned value of waiting, for every (alpha, eps) pair.

    error = |mean_s Q(s, reject) - V*|, averaged over seeds.
    good_rule = share of seeds whose learned cutoff is 0.625 or 0.675
                (the two levels within 0.002 of optimal in value).
    """
    V = offer_rl.true_value()
    rows = []
    for a in alphas:
        for e in epss:
            errs, good = [], 0
            for sd in seeds:
                Q, _ = offer_rl.learn("q", episodes=episodes, alpha=a, eps=e, seed=sd)
                errs.append(abs(Q[:, 1].mean() - V))
                good += offer_rl.learned_cutoff(Q) in (0.625, 0.675)
            rows.append({"alpha": a, "eps": e, "abs_error": float(np.mean(errs)),
                         "error_sd": float(np.std(errs, ddof=1)), "good_rule_share": good / len(seeds)})
    return rows


def offer_curves(episodes: int = 20_000, every: int = 250, seeds=range(5)):
    """Learning curves of the value of waiting for three learners."""
    setups = {"Q-learning, alpha=0.1": dict(method="q", alpha=0.1),
              "Q-learning, decaying alpha": dict(method="q", alpha="decay"),
              "SARSA, decaying alpha": dict(method="sarsa", alpha="decay")}
    out = {}
    for name, kw in setups.items():
        H = np.array([offer_rl.learn(episodes=episodes, seed=sd, record_every=every, **kw)[1]
                      for sd in seeds])
        out[name] = {"x": (np.arange(H.shape[1]) + 1) * every, "mean": H.mean(0),
                     "lo": H.min(0), "hi": H.max(0)}
    return out


def gold_curves(episodes: int = 200_000, every: int = 2_000, seeds=range(5)):
    """How the learned shipping cutoff settles as experience accumulates."""
    H = []
    for sd in seeds:
        _, _, hist = gold_rl.learn(episodes=episodes, seed=sd, record_every=every)
        H.append(hist)
    H = np.array(H, dtype=float)
    return {"x": (np.arange(H.shape[1]) + 1) * every, "cutoffs": H}


if __name__ == "__main__":
    print("alpha   eps   |error|   sd     good-rule share")
    for r in offer_grid():
        print(f"{str(r['alpha']):>6} {r['eps']:5.2f}  {r['abs_error']:.4f}  {r['error_sd']:.4f}  {r['good_rule_share']:.0%}")
