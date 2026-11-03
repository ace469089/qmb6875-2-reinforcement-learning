"""
offer_rl.py -- LEARN WHICH OFFERS TO ACCEPT, WITHOUT BEING TOLD THE RULES

Same problem as Presentation 1 (offers arrive each period; accept and you are
paid, reject and you wait one period, discount beta = 0.9).  Optimal control
SOLVED it because it knew offers were Uniform(0,1).  Here the agent is never
told the distribution.  It only experiences (state, action, reward, next state)
and learns from them.

States and actions
------------------
The agent sees the offer rounded to one of N = 20 levels: 0.025, 0.075, ..., 0.975
(so we can keep a TABLE of values; docs/DECISIONS.md, Q2).
Actions: 0 = accept, 1 = reject.

Q-learning (Watkins, 1989)
--------------------------
Q(s,a) = value of taking action a with offer s, then behaving optimally.
After every step, nudge the guess toward what just happened:

    Q(s,a) <- Q(s,a) + alpha * [ target - Q(s,a) ]
    target = reward                       if the episode ended (accept)
           = reward + beta * max_a' Q(s',a')   otherwise (reject: reward 0)

The bracket is the TEMPORAL-DIFFERENCE (TD) ERROR, the "surprise".
alpha is the learning rate.  Behaviour is epsilon-greedy: with probability eps
take a random action (explore), otherwise the best-looking one (exploit).

SARSA (for comparison)
----------------------
Identical, except the target uses the action actually taken next, Q(s', a'),
instead of the best one.  SARSA therefore learns the value of the policy it is
FOLLOWING (exploration mistakes included); Q-learning learns the value of the
BEST policy while following another.  That is "on-policy" vs "off-policy".

The right answer to compare with
--------------------------------
With 20 equally likely offer levels the exact fixed point is
    V = beta * mean_i max(w_i, V)
which we compute exactly in `true_value()`: 0.6262, essentially the
continuous answer 0.6268.  Note the optimal rule on this grid is "accept from
0.675 up" because 0.625 is a hair BELOW 0.6262; accepting 0.625 too costs only
0.0001 (docs/FAQ.md).

SARSA has its own exact target, `epsilon_greedy_value()`: the value of waiting
when you will keep making random moves 20% of the time.
"""

from __future__ import annotations

import random

import numpy as np

BETA = 0.9
N_LEVELS = 20


def offer_levels(n: int = N_LEVELS) -> np.ndarray:
    """Midpoints of n equal bins on (0,1): what the agent can 'see'."""
    return (np.arange(n) + 0.5) / n


def true_value(n: int = N_LEVELS, beta: float = BETA, tol: float = 1e-14) -> float:
    """Exact value of waiting for the discretized problem (by value iteration)."""
    w = offer_levels(n)
    V = 0.0
    while True:
        V_new = beta * np.maximum(w, V).mean()
        if abs(V_new - V) < tol:
            return float(V_new)
        V = V_new


def continuous_value(beta: float = BETA) -> float:
    """Closed form for Uniform(0,1) offers, from Presentation 1."""
    return (1.0 - np.sqrt(1.0 - beta**2)) / beta


def epsilon_greedy_value(eps: float = 0.2, n: int = N_LEVELS, beta: float = BETA,
                         tol: float = 1e-14) -> float:
    """Value of waiting under the best EPSILON-GREEDY policy (what SARSA learns).

    With 2 actions, epsilon-greedy takes the better one with prob 1 - eps/2 and
    the worse one with prob eps/2.  Accepting is worth w, waiting is worth R, so
        R = beta * mean_i [ (1 - eps/2) max(w_i, R) + (eps/2) min(w_i, R) ].
    """
    w = offer_levels(n)
    R = 0.0
    while True:
        R_new = beta * np.mean((1 - eps / 2) * np.maximum(w, R) + (eps / 2) * np.minimum(w, R))
        if abs(R_new - R) < tol:
            return float(R_new)
        R = R_new


def learn(method: str = "q", episodes: int = 20_000, alpha=0.1, eps: float = 0.2,
          beta: float = BETA, n: int = N_LEVELS, seed: int = 6875, record_every: int = 0):
    """Tabular Q-learning ("q") or SARSA ("sarsa") on the offer problem.

    alpha: a number (constant learning rate) or "decay" for alpha = 1/(visits)^0.7,
           a schedule that satisfies the Robbins-Monro conditions (sum alpha = inf,
           sum alpha^2 < inf) needed for the convergence theorem.
    Returns (Q table [n x 2], history of mean Q(.,reject) if record_every > 0).
    """
    rnd = random.Random(seed)              # fast scalar random numbers (docs/DECISIONS.md, Q6)
    w = offer_levels(n)
    Q = np.zeros((n, 2))
    visits = np.zeros((n, 2))
    history = []

    def pick(s):
        if rnd.random() < eps:
            return rnd.randrange(2)
        q0, q1 = Q[s, 0], Q[s, 1]
        if q0 == q1:
            return rnd.randrange(2)          # break ties at random
        return 0 if q0 > q1 else 1

    def step_size(s, a):
        visits[s, a] += 1
        return visits[s, a] ** -0.7 if alpha == "decay" else alpha

    for ep in range(episodes):
        s = rnd.randrange(n)                 # the first offer of the episode
        a = pick(s)
        while True:
            if a == 0:                       # ACCEPT: paid w[s], episode over
                Q[s, 0] += step_size(s, 0) * (w[s] - Q[s, 0])
                break
            s2 = rnd.randrange(n)            # REJECT: next period's offer
            a2 = pick(s2)
            if method == "q":
                target = beta * max(Q[s2, 0], Q[s2, 1])    # best next action
            elif method == "sarsa":
                target = beta * Q[s2, a2]                  # action actually taken
            else:
                raise ValueError(method)
            Q[s, 1] += step_size(s, 1) * (target - Q[s, 1])
            s, a = s2, a2
        if record_every and (ep + 1) % record_every == 0:
            history.append(Q[:, 1].mean())
    return Q, np.array(history)


def learned_cutoff(Q: np.ndarray, n: int = N_LEVELS) -> float:
    """Lowest offer level at which the learned table says 'accept'."""
    accept = Q[:, 0] > Q[:, 1]
    return float(offer_levels(n)[accept].min()) if accept.any() else float("nan")


def policy_value(cutoff: float, beta: float = BETA, n: int = N_LEVELS) -> float:
    """Exact value of 'accept iff offer >= cutoff' on the 20-level problem."""
    w = offer_levels(n)
    take = w >= cutoff - 1e-12
    p = take.mean()
    return float(beta * (w[take].mean() * p) / (1.0 - beta * (1.0 - p))) if p > 0 else 0.0


def summary(method="q", seeds=range(10), **kw):
    """Run several seeds; return mean/sd of the learned value of waiting and cutoffs."""
    vals, cuts = [], []
    for sd in seeds:
        Q, _ = learn(method=method, seed=sd, **kw)
        vals.append(Q[:, 1].mean())
        cuts.append(learned_cutoff(Q))
    return {"value_mean": float(np.mean(vals)), "value_sd": float(np.std(vals, ddof=1)),
            "cutoffs": cuts}


if __name__ == "__main__":
    print(f"true value of waiting (20 levels): {true_value():.4f}   continuous: {continuous_value():.4f}")
    print(f"value of waiting under epsilon-greedy (eps=0.2): {epsilon_greedy_value():.4f}  (SARSA's target)")
    Q, _ = learn()
    print(f"Q-learning (alpha=0.1, eps=0.2, 20k episodes): value of waiting {Q[:,1].mean():.4f}, "
          f"accept from {learned_cutoff(Q):.3f}")
    Qs, _ = learn(method="sarsa")
    print(f"SARSA      (same settings):                  value of waiting {Qs[:,1].mean():.4f}, "
          f"accept from {learned_cutoff(Qs):.3f}")
    Qd, _ = learn(alpha="decay", episodes=100_000)
    print(f"Q-learning (decaying alpha, 100k episodes):   value of waiting {Qd[:,1].mean():.4f}")
