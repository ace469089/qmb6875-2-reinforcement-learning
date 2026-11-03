"""rlearn: reinforcement learning from zero, for QMB 6875.

Modules
-------
bandit       one-state RL: which ad to show (greedy, epsilon-greedy, UCB1, Thompson, A/B test)
offer_rl     Q-learning and SARSA rediscover the 0.627 offer cutoff without knowing the distribution
gold_rl      Q-learning learns the gold trader's shipping cutoff from simulated weeks
experiments  evidence for every hyperparameter choice (alpha, epsilon, episodes)
"""

__all__ = ["bandit", "offer_rl", "gold_rl", "experiments"]
