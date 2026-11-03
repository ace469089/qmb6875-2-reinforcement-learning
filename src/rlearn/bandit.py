"""
bandit.py -- THE SIMPLEST REINFORCEMENT LEARNING PROBLEM: WHICH AD DO I SHOW?

A "multi-armed bandit" is RL with only ONE state: every round you pick one of
k options ("arms", like slot-machine levers), you see a reward, and you try to
earn as much as possible.  Nothing you do changes the situation you face next,
so the ONLY hard part is the explore-vs-exploit tension:

    * exploit: show the ad that has looked best so far;
    * explore: show another ad, because your estimates might be wrong.

Story: 5 ad creatives with unknown click-through rates (CTRs).  Each time an ad
is shown, the user clicks (reward 1) or not (reward 0).  10,000 impressions.

Strategies compared (docs/DECISIONS.md, B3):
  * greedy        -- always show the best-looking ad (after one look at each)
  * epsilon-greedy-- explore at random with probability epsilon
  * UCB1          -- "optimism under uncertainty": add a bonus to ads you have
                     tried less, so you try them until you are sure they're worse
  * Thompson      -- keep a probability distribution (Beta) for each ad's CTR;
                     each round draw one plausible CTR per ad and show the ad
                     whose draw is highest.  Ads you are unsure about sometimes
                     draw high, so they get explored in proportion to the
                     chance they are actually the best.
  * A/B test      -- the classic marketing approach: show all ads equally for a
                     fixed budget, then commit to the winner forever

Score: REGRET = clicks you lost compared with someone who knew the best ad
from the start.  Lower is better.

Everything is vectorized across independent "runs" (whole 10,000-impression
experiments) so we can average over hundreds of them quickly.
"""

from __future__ import annotations

import numpy as np

CTRS = np.array([0.020, 0.030, 0.035, 0.040, 0.050])   # unknown to the agent
HORIZON = 10_000
RUNS = 500


def run(strategy: str, ctrs=CTRS, horizon: int = HORIZON, runs: int = RUNS,
        eps: float = 0.1, ab_budget: int = 2_000, seed: int = 6875):
    """Play `runs` independent experiments of `horizon` impressions.

    Returns dict with:
      regret[t]       average cumulative regret after t+1 impressions
      best_share[t]   fraction of runs showing the truly best ad at time t
      final_counts    average number of impressions per ad
    """
    rng = np.random.default_rng(seed)
    k = len(ctrs)
    best = int(np.argmax(ctrs))
    counts = np.zeros((runs, k))
    sums = np.zeros((runs, k))
    regret = np.zeros(horizon)
    best_share = np.zeros(horizon)
    cum = np.zeros(runs)
    idx = np.arange(runs)

    for t in range(horizon):
        est = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        if t < k and strategy not in ("ab", "thompson"):
            arm = np.full(runs, t)                        # look at each ad once
        elif strategy == "greedy":
            arm = _argmax_random_ties(est, rng)
        elif strategy == "egreedy":
            explore = rng.random(runs) < eps
            arm = np.where(explore, rng.integers(k, size=runs), _argmax_random_ties(est, rng))
        elif strategy == "ucb":
            bonus = np.sqrt(2.0 * np.log(t + 1) / counts)
            arm = _argmax_random_ties(est + bonus, rng)
        elif strategy == "thompson":
            # Beta(1 + clicks, 1 + non-clicks) is the posterior for a CTR that
            # started with a flat (uniform) prior.  One draw per ad per run.
            draws = rng.beta(1.0 + sums, 1.0 + counts - sums)
            arm = np.argmax(draws, axis=1)
        elif strategy == "ab":
            arm = np.full(runs, t % k) if t < ab_budget else _argmax_random_ties(est, rng)
        else:
            raise ValueError(strategy)
        reward = (rng.random(runs) < ctrs[arm]).astype(float)
        if strategy != "ab" or t < ab_budget:             # A/B stops learning after the test
            counts[idx, arm] += 1
            sums[idx, arm] += reward
        cum += ctrs[best] - ctrs[arm]                      # EXPECTED regret (less noisy)
        regret[t] = cum.mean()
        best_share[t] = np.mean(arm == best)
    return {"regret": regret, "best_share": best_share,
            "final_counts": counts.mean(axis=0) if strategy != "ab" else None}


def _argmax_random_ties(values: np.ndarray, rng) -> np.ndarray:
    """Row-wise argmax that breaks ties at random (plain argmax always picks the first)."""
    noise = rng.random(values.shape) * 1e-12
    return np.argmax(values + noise, axis=1)


def compare(horizon: int = HORIZON, runs: int = RUNS, seed: int = 6875):
    """Run every strategy on the same seed and summarize."""
    setups = {"greedy": dict(strategy="greedy"),
              "epsilon-greedy (0.1)": dict(strategy="egreedy", eps=0.1),
              "epsilon-greedy (0.01)": dict(strategy="egreedy", eps=0.01),
              "UCB1": dict(strategy="ucb"),
              "Thompson sampling": dict(strategy="thompson"),
              "A/B test then commit": dict(strategy="ab", ab_budget=2_000)}
    out = {}
    for name, kw in setups.items():
        res = run(horizon=horizon, runs=runs, seed=seed, **kw)
        out[name] = res
    return out


if __name__ == "__main__":
    res = compare()
    best_clicks = CTRS.max() * HORIZON
    print(f"Best ad expected clicks over {HORIZON:,} impressions: {best_clicks:.0f}")
    for name, r in res.items():
        print(f"{name:<24} regret {r['regret'][-1]:6.1f} clicks   "
              f"best ad shown at the end {r['best_share'][-1]:5.1%}")
