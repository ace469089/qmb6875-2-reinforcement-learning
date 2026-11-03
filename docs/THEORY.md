# The Full Lesson: Reinforcement Learning From Zero

Assumes only: an average, an expected value, and the idea of a policy from Presentation 1 (a rule saying what to do in each situation).

**Contents**
1. [What RL is, and how it differs from optimal control](#1-what-rl-is)
2. [The vocabulary: agents, environments, MDPs](#2-the-vocabulary)
3. [Bandits: exploration vs. exploitation, isolated](#3-bandits)
4. [Learning values from experience: the incremental average](#4-learning-values-from-experience)
5. [The Q-function and Q-learning](#5-the-q-function-and-q-learning)
6. [SARSA: on-policy vs. off-policy](#6-sarsa-on-policy-vs-off-policy)
7. [The gold trader: RL on the professor's capstone](#7-the-gold-trader)
8. [When tables are not enough: function approximation and deep RL](#8-when-tables-are-not-enough)
9. [Summary](#9-summary)

---

## 1. What RL is

**Optimal control** (Presentation 1) assumed we **know**:
- how the state moves (the dynamics), and
- the probability distribution of the randomness.

With those, we wrote the Bellman equation and solved it.

**Reinforcement learning drops both assumptions.** The agent only gets to *act* and *observe what happens*:

> see the state → choose an action → receive a reward and the next state → update → repeat

**Why care?**
- Real decision makers rarely know the true distribution.
- The professor's capstone student had his mother's records of gold trades, not a formula for the premium.
- RL turns records or a simulator into a policy.

**What stays the same:** the goal (maximize expected discounted reward) and the Bellman logic. RL is "Bellman without the model," estimated from samples.

---

## 2. The vocabulary

| Term | Meaning | Gold trader |
|---|---|---|
| **Agent** | the decision maker | the trader |
| **Environment** | everything else; it responds to actions | the gold market |
| **State** $s$ | what the agent sees | this week's premium (in $5 bins) |
| **Action** $a$ | what the agent does | ship or hold |
| **Reward** $r$ | what the agent is paid this step | $s-K$ if shipped; $-h$ if held |
| **Episode** | one complete game, start to end | one shipment |
| **Policy** $\pi$ | a rule mapping states to actions | ship iff premium ≥ cutoff |
| **Discount** $\beta$ | value of $1 next step, today | 0.99904 per week |

Formally this is a **Markov decision process (MDP)**: states, actions, transition probabilities $P(s'\mid s,a)$, rewards $R(s,a)$, and $\beta$. "Markov" means the current state contains everything relevant about the past. Optimal control solves an MDP when $P$ is known; RL solves it when $P$ is unknown but can be sampled.

---

## 3. Bandits

*Code: [`src/rlearn/bandit.py`](../src/rlearn/bandit.py)*

A **multi-armed bandit** is RL with a **single state**. Your action never changes the situation you face next, so the only difficulty is **exploration vs. exploitation**. (The name comes from slot machines, the "one-armed bandits"; picture a row of them with unknown payout rates.)

**Our version:** 5 ad creatives with click-through rates of 2%, 3%, 3.5%, 4%, 5%. The agent doesn't know these. Each impression returns a click (1) or not (0).

### The core tension
- If you always show the ad with the best *estimated* rate (**exploit**), an unlucky start can convince you a good ad is bad, and you'll never show it again to find out.
- If you show ads at random (**explore**), you learn the truth but waste impressions on bad ads.

### Five strategies

1. **Greedy.** Show each ad once, then always the best-looking one. No exploration after the start.
2. **ε-greedy.** With probability ε, show a random ad; otherwise the best-looking. Simple and robust, but it explores forever at the same rate, even when it's sure.
3. **UCB1** (Upper Confidence Bound). Show the ad with the highest
   $$\hat p_i+\sqrt{\frac{2\ln t}{n_i}},$$
   i.e. the estimate plus an *uncertainty bonus* that shrinks as ad $i$ is shown more ($n_i$). This is "optimism in the face of uncertainty."
4. **Thompson sampling.** Keep a probability distribution for each ad's click rate. Starting from a flat prior, after $c$ clicks in $n$ impressions the posterior is $\text{Beta}(1+c,\,1+n-c)$. Each round, draw one plausible rate per ad and show the ad with the highest draw. Uncertain ads sometimes draw high, so each ad gets explored in proportion to the probability it's actually the best.
5. **A/B test then commit.** The classic marketing experiment: split 2,000 impressions evenly, then show the winner forever.

### Scoring: regret

$$\text{Regret}(T)=\sum_{t=1}^T\big(p_{\text{best}}-p_{a_t}\big)$$

That's the expected clicks lost versus someone who knew the best ad from the start. It's a cost of learning; lower is better. We average over 500 independent experiments.

### Results (10,000 impressions)

| Strategy | Regret | Best ad shown at the end |
|---|---:|---:|
| greedy | 117.9 | 32% |
| ε = 0.01 | 89.0 | 50% |
| ε = 0.1 | 54.2 | 74% |
| UCB1 | 129.4 | 30% |
| **Thompson** | **47.9** | **85%** |
| A/B then commit | 62.4 | 67% |

**Lessons:**
- **Not exploring is the worst mistake.** Greedy is wrong 68% of the time.
- **A/B testing is a crude form of exploration.** It wastes the test budget evenly on known-bad ads, and it can't recover if the test picked the wrong winner (33% of the time here).
- **Tuning matters.** UCB1 is mathematically optimal *asymptotically*, but its bonus is scaled for rewards that can be anywhere in [0, 1]. With click rates of 2–5% the bonus dwarfs the real differences, so it keeps exploring far too long. Thompson's Beta posterior automatically knows click rates are small.

---

## 4. Learning values from experience

*The one formula everything in RL is built on.*

Suppose you want the average of numbers $x_1, x_2, \dots$ that arrive one at a time. You don't need to store them all:

$$\bar x_n=\bar x_{n-1}+\frac1n\big(x_n-\bar x_{n-1}\big).$$

**new estimate = old estimate + step size × (new data − old estimate)**

Replace $\frac1n$ with a general **learning rate** $\alpha$:

$$\text{estimate}\leftarrow\text{estimate}+\alpha\,(\text{target}-\text{estimate}).$$

| $\alpha$ | Behavior |
|---|---|
| $1/n$ | exact running average |
| constant (e.g. 0.1) | weights recent data more; never fully settles; adapts if the world changes |
| $1/n^{0.7}$ | in between; satisfies the **Robbins–Monro conditions** $\sum\alpha=\infty$, $\sum\alpha^2<\infty$, which guarantee convergence |

---

## 5. The Q-function and Q-learning

*Code: [`src/rlearn/offer_rl.py`](../src/rlearn/offer_rl.py)*

### The Q-function

$$Q(s,a)=\text{value of taking action }a\text{ in state }s\text{, then behaving optimally forever.}$$

Think of it as a **report card**, one grade per action in every state. If you had it, the optimal policy is trivial: in each state, pick the action with the highest grade. And $V(s)=\max_aQ(s,a)$.

It satisfies a Bellman equation (Presentation 1, one grade per action):

$$Q(s,a)=R(s,a)+\beta\,E\Big[\max_{a'}Q(s',a')\Big].$$

**Why $Q$ instead of $V$?** To act on $V$ you need the model: "which action leads to the best next state?" requires $P(s'\mid s,a)$. To act on $Q$ you just compare two numbers. **No model needed.** That's why RL learns $Q$.

### Q-learning (Watkins, 1989)

We can't compute the expectation (no model), but each step gives **one sample** of what's inside it. So apply the running-average rule with that sample as the target:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\Big[\underbrace{r+\beta\max_{a'}Q(s',a')}_{\text{target: one sample of the right side}}-Q(s,a)\Big].$$

The bracket is the **temporal-difference (TD) error**: the surprise, what just happened minus what you expected.

**Behavior while learning:** ε-greedy. Mostly take the best-looking action, but explore with probability ε so every action keeps getting tried.

**Convergence theorem** (Watkins & Dayan, 1992): tabular Q-learning converges to the true $Q$ with probability 1 if every state–action pair is tried infinitely often and the learning rates satisfy Robbins–Monro. Both conditions are things *we* control: ε > 0 guarantees the first, $\alpha=1/n^{0.7}$ the second.

### The offer problem, learned

**Setup.** The agent sees offers rounded to 20 levels (0.025, 0.075, …, 0.975). Actions: accept (paid the offer, episode ends) or reject (reward 0, discount 0.9, next offer). The agent is never told offers are uniform.

**Exact target for the 20-level problem:**
$$V=\beta\cdot\tfrac1{20}\textstyle\sum_i\max(w_i,V)\ \Rightarrow\ V^*=0.6262$$
(The continuous answer is 0.6268.)

**Results:**

| Learner | Episodes | Learned value of waiting | Learned cutoff |
|---|---|---|---|
| Q-learning, α = 0.1 (10 seeds) | 20,000 | 0.6253 ± 0.0088 | 0.625 (all 10) |
| Q-learning, α = 1/n^0.7 (5 seeds) | 100,000 | **0.6260 ± 0.0010** | 0.625 or 0.675 |
| exact | — | 0.6262 | 0.675 |

**Why the cutoff comes out as 0.625 *or* 0.675.** The true value 0.6262 is a hair above the 0.625 level. So rejecting an offer of 0.625 is optimal, but only barely:

| Rule | Value |
|---|---|
| accept from 0.675 | 0.62620 |
| accept from 0.625 | 0.62609 |

A difference of 0.0001. The learned estimates have noise larger than that gap, so either answer is "right" to four decimals.

**The learned report card** (`figures/offer_rl.png`, middle):
- $Q(\cdot,\text{accept})$ learns to equal the offer exactly. Accepting pays the offer, with no future.
- $Q(\cdot,\text{reject})$ learns ≈ 0.626 in every state. Offers are independent, so today's offer says nothing about the future.

### Choosing α and ε: the evidence

Error $|\bar Q(\cdot,\text{reject})-V^*|$ after 20,000 episodes, averaged over 5 seeds:

| | ε = 0.05 | ε = 0.2 | ε = 0.5 |
|---|---:|---:|---:|
| α = 0.01 | 0.411 | 0.166 | 0.013 |
| α = 0.05 | 0.047 | 0.005 | 0.006 |
| α = 0.1 | 0.004 | 0.007 | 0.007 |
| α = 0.3 | 0.011 | 0.015 | 0.009 |
| **α = 1/n^0.7** | **0.002** | **0.002** | **0.002** |

How to read it:
- **α too small (0.01): too slow.** In 20,000 episodes the estimates barely move from their starting value of 0, especially with little exploration.
- **α too big (0.3): too jumpy.** Each estimate chases the latest sample.
- **Decaying α:** big steps early, small steps late. Best everywhere, and insensitive to ε.

---

## 6. SARSA: on-policy vs. off-policy

SARSA (State–Action–Reward–State–Action) changes **one thing**. The target uses the action you'll *actually take next*, not the best one:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\beta\,Q(s',a')-Q(s,a)\big],\qquad a'\ \text{chosen }\varepsilon\text{-greedily}.$$

| | Q-learning | SARSA |
|---|---|---|
| Learns the value of… | the **best** policy, while behaving ε-greedily | the policy it **follows**, exploration mistakes included |
| Name | **off-policy** | **on-policy** |
| Offer problem, ε = 0.2 | 0.6260 (target 0.6262) | **0.5768** (target **0.5762**) |

**Where SARSA's target comes from.** With two actions, ε-greedy takes the better one with probability $1-\varepsilon/2$ and the worse with $\varepsilon/2$. So the value of waiting while still exploring is

$$R=\beta\,\overline{\big[(1-\tfrac\varepsilon2)\max(w,R)+\tfrac\varepsilon2\min(w,R)\big]}=0.5762.$$

SARSA hits it.

**Which to use?**
- If you will keep exploring in the real world (live ads, a robot that can't stop experimenting), SARSA's lower, *honest* value is what you'll actually get.
- If exploration happens in simulation and you'll act greedily afterwards, Q-learning's value is the relevant one.

---

## 7. The gold trader

*Code: [`src/rlearn/gold_rl.py`](../src/rlearn/gold_rl.py)*

### The setup
- **Same economics as Presentation 1:** the premium follows an AR(1) ($\mu=40,\rho=0.8,\sigma=20$ \$/oz/week), shipping pays $s-K$ ($K=25$), holding costs $h=\$2.89$/week, $\beta=0.99904$.
- **The agent is told none of this.** It sees the premium in **$5 bins** from −$40 to $140 (plus two open-ended tail bins: 38 states), and learns $Q(\text{bin},\{\text{ship},\text{hold}\})$.
- Each episode starts in a typical week (the premium drawn from its long-run distribution) and ends when the agent ships.

### Results
- **All 5 independent agents learn to ship from $50**, the $5 bin containing optimal control's exact $50.49.
- The learned rule earns **$26.64/oz**, versus **$26.59 ± 0.40** for the exact rule (the difference is simulation noise) and $14.86 for shipping immediately.

**How the learned cutoff evolves:**

| Episodes | Cutoffs of 5 agents |
|---:|---|
| 2,000 | 135, 40, 45, 40, 40 |
| 10,000 | 135, 45, 50, 45, 45 |
| 50,000 | 50, 50, 50, 45, 50 |
| 200,000 | **50, 50, 50, 50, 50** |

**How flat the top is** (value of a cutoff rule, $/oz):

| cutoff | 40 | 45 | **50** | 55 | 60 |
|---|---|---|---|---|---|
| value | 25.54 | 26.22 | **26.64** | 26.24 | 25.09 |

### The lesson: data hunger

Getting there took about **737,000 simulated weeks**, roughly 14,000 years of one trader's life. The tail bins are visited rarely (`figures/gold_rl.png`, right), so values there stay noisy.

This is the honest trade-off:
- **Optimal control** needed the model and one second of computing.
- **RL** needed no model and an enormous amount of experience.

The capstone student could only do it because he had years of records *and* used them to build a simulator.

---

## 8. When tables are not enough

Tabular RL stores one number per (state, action), which breaks down fast:
- A gold trader who also tracks the gold price, interest rates and theft reports has millions of states.
- Most states are never visited, so their table entries are never learned.

**Function approximation** replaces the table with a formula: $Q(s,a)\approx f_\theta(s,a)$, a regression or a neural network with parameters $\theta$. The update becomes a gradient step that moves $\theta$ toward the TD target. Similar states then share what they learn.

That's **deep reinforcement learning** (DQN, policy gradients): the method behind game-playing systems like AlphaGo, and many robotics and pricing applications.

The trade-off: the clean convergence theorem no longer holds, and training can be unstable. That's why tricks like experience replay and target networks exist.

---

## 9. Summary

1. RL solves the same problem as optimal control (maximize expected discounted reward), **without knowing the model**, by learning from experience.
2. Everything rests on one update: *estimate ← estimate + α (target − estimate)*.
3. **Q-learning** learns the optimal report card $Q(s,a)$ off-policy; **SARSA** learns the value of the policy actually followed.
4. **Exploration is essential.** Greedy fails; Thompson sampling explores in proportion to doubt.
5. On every problem here, RL recovers the known optimum (0.6262; $50.49 → the $50 bin), at the cost of a lot of data.
