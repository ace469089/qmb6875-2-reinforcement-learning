# FAQ: Questions the Professor Is Likely to Ask

Answers are written to be said out loud. Numbers come from `results/results.json`.

**Jump to:** [Big picture](#big-picture) · [Bandits](#bandits) · [Q-learning](#q-learning) · [SARSA](#sarsa) · [Gold trader](#gold-trader) · [Beyond tables](#beyond-tables)

---

## Big picture

**What is reinforcement learning, in one sentence?**
Learning a good policy from experience (act, get paid, update) without being told how the world works.

**How is it different from optimal control?**
Same goal, same Bellman logic. Control assumes you know the transition probabilities and solves exactly. RL doesn't know them and estimates the Bellman equation from samples. In our offer problem, control used "offers are uniform" to get 0.6268; RL never used that fact and learned 0.6260 (exact discrete answer: 0.6262).

**How is it different from supervised learning (regression, classification)?**
Supervised learning gets the right answer for every example. RL only gets a *reward* for the action it took. It never learns what would have happened with the other action, and its own choices decide which data it sees. That's why exploration matters.

**When would you use RL instead of solving the model?**
When you can't write the model down, or it's too big to solve. You'd need data or a simulator, and a lot of it. If you know the model, solving it is faster and exact: one second vs. 737,000 simulated weeks for the gold trader.

**What's an MDP?**
A Markov decision process: states, actions, transition probabilities $P(s'\mid s,a)$, rewards, and a discount factor. "Markov" means the current state summarizes everything relevant about the past. It's the formal setting both control and RL work in.

---

## Bandits

**What's a multi-armed bandit?**
RL with one state: you repeatedly pick among options with unknown payoffs, and your choice doesn't change the situation. It isolates the explore/exploit trade-off.

**Why does greedy do so badly?**
After one impression per ad, most ads show zero clicks (CTRs are 2–5%). Greedy locks onto whichever ad got lucky early and never revisits the others. It ends up showing the best ad only 32% of the time.

**Why did UCB1 lose to plain ε-greedy?**
UCB1's bonus $\sqrt{2\ln t/n}$ is designed for rewards that could be anywhere between 0 and 1. With CTRs of 2–5%, that bonus is far bigger than the real differences between ads, so UCB1 keeps "giving every ad a chance" for a very long time. It's asymptotically optimal, but the constant is terrible at this scale. A version scaled to the reward variance (UCB-V, KL-UCB) fixes it. Thompson sampling handles it automatically because its Beta posterior "knows" the rates are small.

**Why does Thompson sampling win?**
It explores each ad with exactly the probability that the ad is the best, given the data so far. Clearly bad ads quickly stop being shown; close calls keep getting tested. It's also what large ad platforms use for creative testing.

**Isn't A/B testing the right way to do this?**
A/B testing answers "which ad is better?" with statistical confidence, but it pays for that by showing known-bad ads equally during the test, and it can't correct a wrong winner (here it picked wrong 33% of the time). Bandits *earn while learning*: regret 47.9 for Thompson vs. 62.4 for A/B. If you need an unbiased estimate of the difference, A/B. If you need clicks, a bandit.

**What is regret?**
The expected rewards lost compared with an oracle that knew the best option from the start. It's the cost of learning.

---

## Q-learning

**What is $Q(s,a)$?**
A report card: the value of taking action $a$ in state $s$ and then behaving optimally. $V(s)=\max_aQ(s,a)$.

**Why learn $Q$ instead of $V$?**
To act using $V$ you need the model: which action leads to the best next state? To act using $Q$ you just compare two numbers. That's why Q-learning is model-free.

**Walk me through the update.**
$Q(s,a)\leftarrow Q(s,a)+\alpha[r+\beta\max_{a'}Q(s',a')-Q(s,a)]$.
- The target is one sampled version of the Bellman right-hand side: what you got plus the discounted best you can do from where you landed.
- The bracket is the TD error, the surprise.
- You move a fraction $\alpha$ of the way toward the target. It's a running average of noisy samples of the Bellman equation.

**Why does it converge?**
Watkins & Dayan (1992): with a table, if every state–action pair is visited infinitely often and $\sum\alpha=\infty$, $\sum\alpha^2<\infty$, then $Q\to Q^*$ with probability 1. Intuitively, it's stochastic approximation to the fixed point of a contraction (the Bellman map from Presentation 1). ε > 0 handles the first condition and $\alpha=1/n^{0.7}$ the second.

**Why did you pick α = 0.1 / decaying, and ε = 0.2?**
I tested 15 combinations (`figures/offer_rl.png`, right).
- α = 0.01 is too slow (errors up to 0.41).
- α = 0.3 is too jumpy.
- A decaying α gives error 0.002 regardless of ε.
- ε = 0.2 means the "wrong-looking" action gets tried 10% of the time, enough to keep learning.

**Why do your agents give 0.625 sometimes and 0.675 other times?**
The exact $V^*$ = 0.6262 is a hair above the 0.625 level, so rejecting 0.625 is optimal, but only by 0.0001 in value (0.62620 vs 0.62609). The estimates' noise is bigger than that, so either cutoff is "right" to four decimals.

**Why is $Q(s,\text{reject})$ the same for every offer?**
Offers are independent. Today's offer tells you nothing about the next one, so the value of waiting doesn't depend on it. The agent *learns* this; it isn't told. In the gold problem the premium is persistent, so the hold value does depend on the state.

**How many episodes are enough?**
The learning curves flatten by about 5,000 episodes here with a decaying α. In general, you watch the learning curve until it's flat, and check across several seeds. A single run is one random draw.

**Why several seeds?**
One run with α = 0.1 gave 0.641 and another 0.602. Reporting mean ± sd across seeds shows both the typical result and how much it varies.

---

## SARSA

**What's the difference between SARSA and Q-learning?**
The target. Q-learning uses $\max_{a'}Q(s',a')$, the best next action. SARSA uses $Q(s',a')$ for the action actually taken next, exploration included.

**What does on-policy vs. off-policy mean?**
- **On-policy (SARSA):** learns the value of the policy it's following.
- **Off-policy (Q-learning):** learns the value of the optimal policy while following a different, exploring one.

**How do you know SARSA learned the right thing?**
I computed its exact target: the value of waiting when you keep exploring 20% of the time, $R=\beta\,\overline{[(1-\varepsilon/2)\max(w,R)+(\varepsilon/2)\min(w,R)]}=0.5762$. SARSA learned 0.5768 ± 0.0016. Q-learning learned 0.6260, the greedy optimum. Both are right; they answer different questions.

**When is SARSA's lower number the one you want?**
When you'll keep exploring in the real world (live campaigns, a robot that keeps experimenting). Then 0.576 is what you'll actually earn, and SARSA's policy is more cautious accordingly.

---

## Gold trader

**What does the agent know?**
Nothing about the AR(1), $\mu$, $\rho$ or $\sigma$. It sees the premium (in $5 bins), chooses ship or hold, and observes its payment and the next week's premium.

**Did it find the right rule?**
Optimal control says ship at $50.49. All 5 agents learn "ship from $50," the $5 bin that contains $50.49, which is the finest answer their bins allow. The learned rule earns $26.64/oz vs $26.59 ± 0.40 for the exact rule.

**Why does the learned rule earn slightly *more* than the optimal one?**
Simulation noise. The value curve is very flat at the top ($26.22 at 45, $26.64 at 50, $26.24 at 55), so rules a few dollars apart are statistically indistinguishable. Both use the same simulated premiums (common random numbers), which removes most of the noise, but not all.

**Why $5 bins?**
Tables need finite states. $5 bins give 38 states, each visited thousands of times. Finer bins would pin down the cutoff more precisely but need proportionally more experience. It's the bias–variance trade-off in RL form.

**How much data did it need?**
About 737,000 simulated weeks (200,000 episodes), roughly 14,000 years of one trader. That's the honest cost of not knowing the model, and why the capstone student needed years of records.

**How would you do it with real data instead of a simulator?**
Two routes:
1. Fit a model of the premium to the records, then solve (Presentation 1) or simulate and learn (this repo).
2. Use batch / offline RL: replay the historical (state, action, reward, next state) records through the Q update many times.

Route 2 can only learn about actions that appear in the data. If the family always shipped within a week, the data say little about waiting a month.

**Why does the learned hold value get noisy at high premiums?**
Few visits (`figures/gold_rl.png`, right): premiums above $120 are rare, and the agent usually ships before seeing many weeks there.

---

## Beyond tables

**What if the state were huge (price, rates, news, inventory…)?**
Replace the table with a function: $Q(s,a)\approx f_\theta(s,a)$ (linear regression, tile coding, or a neural network) and update $\theta$ by a gradient step toward the TD target. That's deep RL. Similar states share what they learn, but the clean convergence guarantee is lost and training can be unstable.

**Is this how AlphaGo or ChatGPT-style training works?**
AlphaGo combined deep RL with tree search. Language-model fine-tuning with human feedback uses policy-gradient RL with a learned reward model. Both rest on the same idea: improve a policy using rewards rather than correct answers.

**What's the biggest practical risk with RL?**
It optimizes exactly the reward you give it. A badly specified reward (clicks instead of profit, say) gets optimized very efficiently in the wrong direction.
