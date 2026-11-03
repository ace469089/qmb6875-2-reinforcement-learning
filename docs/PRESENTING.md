# How to Present This (15 minutes, or 60 seconds if cold-called)

## If you are cold-called: the 60-second version

> "Reinforcement learning solves the same problem as optimal control, but without knowing the rules. The agent acts, gets a reward, and updates.
> The core object is $Q(s,a)$, a report card for every action in every state. Q-learning nudges it toward what just happened: $Q\leftarrow Q+\alpha[r+\beta\max Q'-Q]$. The bracket is the surprise.
> You must explore, or you lock onto a bad choice. With five ads, greedy picks the wrong one two-thirds of the time.
> On the offer problem, Q-learning never learns the offers are uniform, but it still finds the value of waiting, 0.626, the same as Person 1.
> On your gold trader, it learns to ship from $50, the bin with the true $50.49. The catch: it took 700,000 simulated weeks. So use math when you know the model, RL when you don't."

## The 15-minute talk

| Min | Show | Say (key line) | Board |
|---|---|---|---|
| 0–1 | README top | "Person 1 knew the rules. Nobody tells a real trader the premium is an AR(1). RL learns from experience." | agent ⇄ environment loop |
| 1–3 | THEORY §2 table | "Agent, environment, state, action, reward, policy. Same pieces as control." | the table, gold column |
| 3–6 | `figures/bandit.png` | "One state, five ads. Greedy is wrong 68% of the time. Thompson explores in proportion to doubt and wins: 47.9 clicks lost vs 62.4 for an A/B test." | regret definition |
| 6–8 | THEORY §4 | "Every RL update is a running average: estimate ← estimate + α(target − estimate)." | the update line |
| 8–11 | `figures/offer_rl.png` | "Q is a report card. One sample of Bellman as the target. It learns 0.626 without knowing offers are uniform. The heatmap is why I chose these settings." | Q-learning update; label TD error |
| 11–12 | same, SARSA line | "SARSA learns 0.576 on purpose: the value of a policy that keeps exploring. On-policy vs off-policy." | the two targets side by side |
| 12–14 | `figures/gold_rl.png` | "Your gold trader: all five agents learn 'ship from $50', the bin with the true $50.49. Same money. But 737,000 simulated weeks." | — |
| 14–15 | THEORY §9 | "Use math when you know the model; learn when you don't. Person 3 shows how to compute either one." | 3-line summary |

## One update by hand (great for the board)

α = 0.5, β = 0.9, start Q(s, reject) = 0, next state's best grade = 0.8, so the target is 0 + 0.9 × 0.8 = 0.72:

0 → 0.36 → 0.54 → 0.63 → 0.675 → … → 0.72. Each step closes half the gap.

## Live demo (optional)

```bash
python -m rlearn.offer_rl    # Q-learning vs SARSA vs exact targets, under a second
python -m rlearn.gold_rl     # learns the shipping cutoff in about 5 seconds
```

## Hand-off line to Person 3

> "Both Person 1 and I relied on loops that run until the answer stops changing. Person 3 shows how to build those loops, how fast they converge, and how to trust them."

## The three questions most likely to come

1. **"Why Q instead of V?"** Acting on Q needs no model; acting on V does.
2. **"Why does it converge?"** It's a running average of samples of a contraction's fixed point. You need every pair visited and a shrinking α.
3. **"How did you choose α and ε?"** I tested 15 combinations; the heatmap is the evidence.

More in [FAQ.md](FAQ.md).
