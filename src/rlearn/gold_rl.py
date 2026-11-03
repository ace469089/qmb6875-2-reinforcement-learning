"""
gold_rl.py -- THE GOLD TRADER LEARNS HIS OWN SELLING RULE

The professor's capstone story: a student used reinforcement learning, on his
mother's trading records, to find the gold trader's control-limit policy.

Presentation 1 SOLVED this problem because it knew the premium followed an
AR(1) with mu = 40, rho = 0.8, sigma = 20.  Here the agent is never told any
of that.  It only lives through weeks of the market:

    see the premium -> ship or hold -> get paid (or pay the holding cost) ->
    see next week's premium -> ...

and learns Q(premium bin, action) with Q-learning.

Environment (identical economics to Presentation 1, $ per ounce, 1 step = 1 week)
    premium s' = mu + rho (s - mu) + sigma * eps
    SHIP: reward s - K, episode ends
    HOLD: reward -h, discount beta, next week's premium arrives

What the agent sees
    The premium rounded into $5-wide bins from -$40 to $140, plus one bin for
    "below -40" and one for "above 140" (38 states).  State aggregation is
    explained in docs/DECISIONS.md (R2).

Ground truth for comparison
    `dp_cutoff()` re-solves the problem the Presentation 1 way (Tauchen grid +
    value iteration): ship iff s >= $50.49.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Market:
    mu: float = 40.0
    rho: float = 0.80
    sigma: float = 20.0
    K: float = 25.0
    X: float = 2500.0
    r: float = 0.05
    loss_annual: float = 0.01

    @property
    def beta(self) -> float:
        return float(np.exp(-self.r / 52.0))

    @property
    def h(self) -> float:
        q = 1.0 - (1.0 - self.loss_annual) ** (1.0 / 52.0)
        return self.X * (self.r / 52.0 + q)

    @property
    def stationary_sd(self) -> float:
        return self.sigma / np.sqrt(1.0 - self.rho**2)


MARKET = Market()
EDGES = np.arange(-40.0, 140.0 + 1e-9, 5.0)       # 37 edges -> 38 bins
CENTERS = np.concatenate(([EDGES[0] - 2.5], (EDGES[:-1] + EDGES[1:]) / 2, [EDGES[-1] + 2.5]))


def to_bin(s: float) -> int:
    """Map a premium to its bin index 0..37 (bins 0 and 37 are the open-ended tails)."""
    return int(np.searchsorted(EDGES, s, side="right"))


# ---------------------------------------------------------------------------
# Ground truth: the optimal-control answer (Tauchen + value iteration)
# ---------------------------------------------------------------------------
def dp_cutoff(m: Market = MARKET, n: int = 401) -> float:
    """Optimal shipping cutoff from dynamic programming (the Presentation 1 method)."""
    return dp_solution(m, n)["cutoff"]


def dp_solution(m: Market = MARKET, n: int = 401) -> dict:
    """Tauchen grid + value iteration.  Returns grid, sell, hold values and the cutoff."""
    sd = m.stationary_sd
    grid = np.linspace(m.mu - 4 * sd, m.mu + 4 * sd, n)
    step = grid[1] - grid[0]
    cm = m.mu + m.rho * (grid - m.mu)
    up = norm.cdf((grid[None, :] + step / 2 - cm[:, None]) / m.sigma)
    lo = norm.cdf((grid[None, :] - step / 2 - cm[:, None]) / m.sigma)
    P = up - lo
    P[:, 0] = up[:, 0]
    P[:, -1] = 1 - lo[:, -1]
    sell = grid - m.K
    V = np.maximum(sell, 0.0)
    for _ in range(100_000):
        V_new = np.maximum(sell, -m.h + m.beta * P @ V)
        if np.max(np.abs(V_new - V)) < 1e-10:
            break
        V = V_new
    hold = -m.h + m.beta * P @ V_new
    gap = sell - hold
    j = int(np.argmax(gap >= 0))
    cutoff = float(grid[j - 1] - gap[j - 1] * (grid[j] - grid[j - 1]) / (gap[j] - gap[j - 1]))
    return {"grid": grid, "sell": sell, "hold": hold, "cutoff": cutoff}


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------
def learn(episodes: int = 200_000, eps: float = 0.2, alpha="decay", m: Market = MARKET,
          seed: int = 6875, max_weeks: int = 500, record_every: int = 0):
    """Tabular Q-learning on the simulated market.

    Each episode starts in a 'typical week' (premium drawn from the long-run
    distribution) and ends when the agent ships (or after max_weeks).
    Actions: 0 = SHIP, 1 = HOLD.
    alpha = "decay" uses 1/(visits)^0.7 per (state, action); a number = constant.
    Returns (Q, visits, list of learned cutoffs over time if record_every > 0).
    """
    rnd = random.Random(seed)
    nb = len(CENTERS)
    Q = np.zeros((nb, 2))
    visits = np.zeros((nb, 2))
    history = []
    beta, h, K = m.beta, m.h, m.K
    for ep in range(episodes):
        s = m.mu + m.stationary_sd * rnd.gauss(0.0, 1.0)
        b = to_bin(s)
        for _ in range(max_weeks):
            if rnd.random() < eps:
                a = rnd.randrange(2)
            else:
                a = 0 if Q[b, 0] >= Q[b, 1] else 1
            visits[b, a] += 1
            lr = visits[b, a] ** -0.7 if alpha == "decay" else alpha
            if a == 0:                                     # SHIP
                Q[b, 0] += lr * ((s - K) - Q[b, 0])
                break
            s = m.mu + m.rho * (s - m.mu) + m.sigma * rnd.gauss(0.0, 1.0)
            b2 = to_bin(s)
            target = -h + beta * max(Q[b2, 0], Q[b2, 1])   # HOLD
            Q[b, 1] += lr * (target - Q[b, 1])
            b = b2
        if record_every and (ep + 1) % record_every == 0:
            history.append(learned_cutoff(Q))
    return Q, visits, history


def learned_cutoff(Q: np.ndarray) -> float:
    """Lowest premium (bin lower edge) above which the learned policy always ships.

    We scan from the top down and stop at the first bin where HOLD wins, so a
    stray noisy bin far below the cutoff cannot fool us.
    """
    ship = Q[:, 0] >= Q[:, 1]
    j = len(ship) - 1
    while j > 0 and ship[j - 1]:
        j -= 1
    if not ship[-1]:
        return float("inf")
    return float(EDGES[j - 1]) if j > 0 else float("-inf")


def evaluate(cutoff: float, m: Market = MARKET, n_paths: int = 20_000, seed: int = 6875,
             max_weeks: int = 2_000):
    """Average discounted $/oz of 'ship iff s >= cutoff' (same luck for every cutoff)."""
    rng = np.random.default_rng(seed)
    s = m.mu + m.stationary_sd * rng.standard_normal(n_paths)
    alive = np.ones(n_paths, bool)
    val = np.zeros(n_paths)
    disc = np.ones(n_paths)
    for _ in range(max_weeks):
        go = alive & (s >= cutoff)
        val[go] += disc[go] * (s[go] - m.K)
        alive &= ~go
        if not alive.any():
            break
        val[alive] -= disc[alive] * m.h
        disc[alive] *= m.beta
        s = m.mu + m.rho * (s - m.mu) + m.sigma * rng.standard_normal(n_paths)
    return float(val.mean()), float(val.std(ddof=1) / np.sqrt(n_paths))


if __name__ == "__main__":
    print(f"optimal-control cutoff (knows the model): ${dp_cutoff():.2f}")
    Q, visits, _ = learn()
    c = learned_cutoff(Q)
    print(f"Q-learning cutoff (never told the model):  ${c:.2f}  (bins are $5 wide)")
    v_opt, se = evaluate(dp_cutoff())
    v_rl, _ = evaluate(c)
    print(f"value of optimal rule ${v_opt:.2f} +/- {1.96*se:.2f};  learned rule ${v_rl:.2f}")
