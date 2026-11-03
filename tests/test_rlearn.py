"""
Tests: the learners are checked against answers we know exactly.

    pytest -q        (about 20 seconds; RL needs experience)
"""

import numpy as np
import pytest

from rlearn import bandit, gold_rl, offer_rl


# --------------------------------------------------------------- bandits ----
@pytest.fixture(scope="module")
def bandit_results():
    return bandit.compare(horizon=4_000, runs=200, seed=1)


def test_exploring_beats_pure_greed(bandit_results):
    g = bandit_results["greedy"]["regret"][-1]
    assert bandit_results["epsilon-greedy (0.1)"]["regret"][-1] < g
    assert bandit_results["Thompson sampling"]["regret"][-1] < g


def test_regret_never_decreases(bandit_results):
    for r in bandit_results.values():
        assert np.all(np.diff(r["regret"]) >= -1e-12)


def test_ab_test_shows_each_ad_equally_during_the_test():
    res = bandit.run("ab", horizon=2_000, runs=50, ab_budget=2_000)
    assert res["best_share"][:2_000].mean() == pytest.approx(1 / len(bandit.CTRS), abs=1e-9)


# ---------------------------------------------------------- offer problem ----
def test_discrete_target_is_close_to_continuous_answer():
    assert offer_rl.true_value() == pytest.approx(offer_rl.continuous_value(), abs=1e-3)


def test_optimal_discrete_rule_achieves_the_true_value():
    assert offer_rl.policy_value(0.675) == pytest.approx(offer_rl.true_value(), abs=1e-12)
    assert offer_rl.policy_value(0.625) < offer_rl.true_value()


def test_exploring_policy_is_worth_less():
    assert offer_rl.epsilon_greedy_value(0.2) < offer_rl.true_value()


def test_q_learning_learns_the_value_of_waiting():
    Q, _ = offer_rl.learn("q", episodes=100_000, alpha="decay", seed=3)
    assert Q[:, 1].mean() == pytest.approx(offer_rl.true_value(), abs=0.005)
    assert offer_rl.learned_cutoff(Q) in (0.625, 0.675)


def test_q_learning_learns_accept_values_equal_to_the_offer():
    Q, _ = offer_rl.learn("q", episodes=100_000, alpha="decay", seed=4)
    w = offer_rl.offer_levels()
    high = w > 0.7                                  # visited often (greedy accepts them)
    assert np.allclose(Q[high, 0], w[high], atol=1e-6)


def test_sarsa_learns_the_value_of_the_exploring_policy():
    Q, _ = offer_rl.learn("sarsa", episodes=100_000, alpha="decay", eps=0.2, seed=5)
    assert Q[:, 1].mean() == pytest.approx(offer_rl.epsilon_greedy_value(0.2), abs=0.005)


# ---------------------------------------------------------- gold trader ----
def test_dp_reference_cutoff():
    assert gold_rl.dp_cutoff() == pytest.approx(50.49, abs=0.05)


def test_bins_are_assigned_correctly():
    assert gold_rl.to_bin(-100) == 0
    assert gold_rl.to_bin(1000) == len(gold_rl.CENTERS) - 1
    assert gold_rl.CENTERS[gold_rl.to_bin(52.0)] == pytest.approx(52.5)


def test_q_learning_finds_the_shipping_cutoff():
    Q, _, _ = gold_rl.learn(episodes=150_000, seed=11)
    assert 45.0 <= gold_rl.learned_cutoff(Q) <= 55.0


def test_learned_rule_is_nearly_as_good_as_optimal():
    opt, se = gold_rl.evaluate(gold_rl.dp_cutoff())
    assert gold_rl.evaluate(50.0)[0] > opt - 4 * se
    assert gold_rl.evaluate(-np.inf)[0] < opt - 10        # shipping at once is much worse
