# Decision Log: Every Choice, Why, and the Evidence

Each entry: **Decision → Why → Alternatives rejected → Evidence** (numbers from `results/results.json`).
Code comments point here by ID (e.g. `DECISIONS, Q6`).

---

## Big picture

### D1. Three problems: bandit → offer → gold
- **Why:** each adds one difficulty.
  1. **Bandit:** exploration only (one state, no future).
  2. **Offer:** states and a future, but each state is independent of the last.
  3. **Gold:** states with *memory* (the premium is an AR(1)), and the professor's own example.
- **Rejected:**
  - Gridworld/maze examples (standard in textbooks, but unrelated to business).
  - CartPole-style control (needs extra libraries and physics).

### D2. Every learned answer is checked against a known truth
- **Why:** an RL result alone proves nothing; it might be a lucky seed or a bug. Each problem has an exact benchmark:
  - bandit: the true click rates;
  - offer: $V^*=0.6262$ by exact value iteration, and SARSA's own target 0.5762;
  - gold: the Presentation 1 dynamic-programming cutoff $50.49.
- **This is also the strongest answer to "how do you know it learned the right thing?"**

### D3. Reusing Presentation 1's problems
- **Why:** the audience already knows the answers (0.627, $50.49), so they can *watch* RL rediscover them. The three talks share one thread.

---

## Bandits

### B1. Bernoulli click rates 2%, 3%, 3.5%, 4%, 5%
- **Why:**
  - Realistic display-ad CTRs are in the low single-digit percent range.
  - The best two ads differ by one percentage point, so learning is genuinely hard (it takes thousands of impressions). That makes the strategies distinguishable.
- **Rejected:** Sutton & Barto's Gaussian testbed (10 arms, normal rewards). It's the standard, but abstract; ad clicks are concrete and match the binary-reward reality of marketing.

### B2. 10,000 impressions × 500 independent runs, expected regret
- **Why:**
  - 10,000 impressions is a small campaign.
  - 500 runs make the average regret stable to within about a click.
  - We add up the *expected* regret $p_{\text{best}}-p_{a_t}$ rather than realized clicks. That removes click-noise that has nothing to do with the strategy.

### B3. Which strategies
- **Greedy:** the baseline that shows why exploration matters.
- **ε-greedy (0.1 and 0.01):** the simplest exploration and the one used in Q-learning later. Two values show the trade-off.
- **UCB1:** the textbook "optimism" algorithm (Auer et al., 2002).
- **Thompson sampling:** the Bayesian approach; the natural fit for click data (Beta posterior).
- **A/B test then commit:** what most marketers actually do. Including it shows RL's practical relevance.
- **Evidence:** regrets 117.9 / 89.0 / 54.2 / 129.4 / **47.9** / 62.4.
  - UCB1's poor showing is reported, not hidden; its cause is explained in the FAQ.
  - A tuned UCB (scaled to the reward variance) would do better, but that's a refinement beyond a first lesson.

### B4. Vectorized across runs
- **Why:** all 500 experiments advance one impression at a time as NumPy arrays, so 5 million decisions per strategy take about 1.5 s instead of minutes in a Python loop.

---

## Q-learning on the offer problem

### Q1. Why tabular Q-learning (and not deep RL)
- **Why:**
  - The state is one number, so a table is exact and transparent: you can print the whole learned report card (`figures/offer_rl.png`, middle).
  - Convergence is guaranteed by theorem (Watkins & Dayan, 1992).
  - Neural networks would add complexity and remove the guarantee for no benefit here.

### Q2. 20 offer levels
- **Why:**
  - A table needs finitely many states, and 20 levels makes the discrete answer (0.6262) within 0.0006 of the continuous one (0.6268).
  - It's small enough that every level gets visited thousands of times.
- **Consequence:** the optimal discrete cutoff is 0.675 because 0.625 sits just under $V^*$. That's explained, not hidden (FAQ).

### Q3. Learning rate: constant 0.1 in the headline run, $1/n^{0.7}$ for the precise result
- **Evidence (error after 20k episodes, 5 seeds):**
  - α = 0.01 → error up to 0.41 (too slow);
  - α = 0.3 → 0.011–0.015 (too jumpy);
  - α = 0.1 → 0.004–0.007;
  - **decaying → 0.002 at every ε**.
- **Why keep 0.1 at all:** it is the most common setting in tutorials and in the slide deck. Showing it works *and* that decay is better teaches the trade-off.
- **Why the exponent 0.7:** any exponent in (0.5, 1] satisfies Robbins–Monro ($\sum\alpha=\infty$, $\sum\alpha^2<\infty$). Exactly 1 (a pure average) can be very slow when early targets are biased. 0.7 is a standard middle choice.

### Q4. Exploration ε = 0.2
- **Why:**
  - With only 2 actions, ε = 0.2 means the "wrong-looking" action is tried 10% of the time: enough to keep learning, not so much that learning is mostly noise.
  - The grid shows decaying-α Q-learning is insensitive to ε ∈ {0.05, 0.2, 0.5}.
  - With constant α, ε = 0.05 starves some (state, reject) pairs (error 0.047 at α = 0.05).

### Q5. Episodes: 20,000 (headline) and 100,000 (precise)
- **Evidence:** learning curves (`figures/offer_rl.png`, left) flatten by about 5,000 episodes with decaying α. 100,000 is used where we want the third decimal.

### Q6. Python's built-in `random` inside the learning loop
- **Why:**
  - RL updates happen one step at a time, so they can't be vectorized.
  - Python's `random.Random` draws a single number about 6× faster than calling NumPy for one value (measured: 0.016 s vs 0.091 s per 200,000 draws), which makes 200,000-episode runs take seconds.
  - Each learner gets its own seeded generator, so results are exactly reproducible.

### Q7. Ties broken at random
- **Why:** at the start every Q is 0. Always picking "action 0" on ties would bias early experience toward accepting. Random tie-breaking avoids building a bias into the algorithm.

### Q8. Reporting multiple seeds, not one run
- **Why:** a single RL run is a random draw. One seed with α = 0.1 gives 0.641, another 0.602. Reporting the mean and sd over seeds shows the typical result *and* the variability, so no one can say the result was cherry-picked.

### Q9. Including SARSA
- **Why:** "on-policy vs. off-policy" is the most common conceptual RL question. Showing SARSA converge to *a different, exactly computable number* (0.5762) proves the distinction rather than asserting it.

---

## The gold trader

### R1. Same economics as Presentation 1
- **Why:** so the learned cutoff can be compared with the exact $50.49, and the class sees the same answer reached two ways.

### R2. State aggregation: $5 bins from −$40 to $140, plus 2 tail bins (38 states)
- **Why:**
  - The premium is continuous, but a table needs finite states.
  - $5 is fine enough to locate the cutoff usefully, yet coarse enough that each bin gets thousands of visits.
  - The range runs from 2.4 long-run sd below the mean to 3 sd above it; the two open-ended bins catch the rest.
- **Trade-off:** the agent can only resolve the cutoff to within one bin ($5). It finds the bin containing $50.49, which is the best possible at this resolution. $1 bins would need roughly 5× more experience.
- **Rejected:** linear function approximation ($Q$ linear in $s$). It's elegant, but the true hold value is curved, so a linear $Q$ would be biased. Tile coding is a possible extension.

### R3. Episodes start from the long-run distribution
- **Why:** this exposes the agent to the whole range of premiums (exploring starts). Starting every episode at $40 would leave high premiums rarely seen.

### R4. 200,000 episodes, decaying α, ε = 0.2
- **Evidence:** cutoffs of 5 seeds at 2k / 10k / 50k / 200k episodes:
  - 2k: (135, 40, 45, 40, 40)
  - 10k: (135, 45, 50, 45, 45)
  - 50k: (50, 50, 50, 45, 50)
  - 200k: **(50, 50, 50, 50, 50)**

  200k is the point where every agent agrees.

### R5. Reading the cutoff top-down
- **Why:** in rarely visited low bins, noisy estimates can occasionally say "ship" by accident. Scanning from the top down for the first bin where "hold" wins reads the cutoff the policy actually uses, not a noise artifact far below it.

### R6. Evaluating policies by simulation with common random numbers
- **Why:**
  - The learned rule and the exact rule are compared on the same 20,000 simulated shipments (same seed).
  - Result: learned $26.64 vs exact $26.59 (95% interval ± 0.40). They're statistically identical.
  - The learned rule coming out a few cents *higher* is noise. The value curve is very flat near the optimum (cutoff 45: $26.22; 50: $26.64; 55: $26.24).

---

## Engineering

### E1. Package + tests + figure script, same as repo 1
- One implementation feeds the tests, the figures, the notebook and the README numbers.

### E2. 13 tests, about 20 seconds
- The RL tests use enough episodes to pass reliably (fixed seeds) but stay fast.
- They check against the *exact* targets with tolerances justified by the multi-seed spread (sd ≈ 0.001 at 100k episodes; the tolerance is 0.005).
