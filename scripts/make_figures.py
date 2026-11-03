"""
Reproduce every figure in figures/ and every number in results/results.json.

    python scripts/make_figures.py      (about 1 minute)
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rlearn import bandit, experiments, gold_rl, offer_rl  # noqa: E402

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
INK, VIO, MINT, CORAL, LEM, GRAY = "#24203A", "#5B4BDB", "#0F8F6F", "#E5533D", "#C98F00", "#8A8A8A"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight"})
R = {}

# ------------------------------------------------------------- bandits ----
res = bandit.compare()
colors = {"greedy": GRAY, "epsilon-greedy (0.1)": VIO, "epsilon-greedy (0.01)": LEM,
          "UCB1": CORAL, "Thompson sampling": MINT, "A/B test then commit": INK}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
t = np.arange(1, bandit.HORIZON + 1)
for name, r in res.items():
    ls = "--" if name == "A/B test then commit" else "-"
    ax[0].plot(t, r["regret"], color=colors[name], lw=2, ls=ls, label=name)
    smooth = np.convolve(r["best_share"], np.ones(200) / 200, mode="valid")   # trailing 200-impression average
    ax[1].plot(t[199:], smooth, color=colors[name], lw=2, ls=ls)
ax[0].set_xlabel("impressions shown"); ax[0].set_ylabel("clicks lost vs. knowing the best ad")
ax[0].set_title("Regret (lower is better)"); ax[0].legend(frameon=False, fontsize=9)
ax[1].set_xlabel("impressions shown"); ax[1].set_ylabel("share of runs showing the best ad")
ax[1].set_title("How often the best ad is shown"); ax[1].set_ylim(0, 1)
ax[1].axhline(1 / len(bandit.CTRS), color=GRAY, lw=0.8, ls=":")
fig.savefig(os.path.join(FIG, "bandit.png")); plt.close(fig)
R["bandit"] = {"ctrs": bandit.CTRS.tolist(), "horizon": bandit.HORIZON, "runs": bandit.RUNS,
               "results": {n: {"regret": float(r["regret"][-1]),
                               "best_share_last_1000": float(r["best_share"][-1000:].mean())}
                           for n, r in res.items()}}

# ------------------------------------------------------------- offer ------
V = offer_rl.true_value()
Veg = offer_rl.epsilon_greedy_value()
curves = experiments.offer_curves()
grid = experiments.offer_grid()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
for (name, c), col in zip(curves.items(), [VIO, MINT, CORAL]):
    ax[0].plot(c["x"], c["mean"], color=col, lw=2, label=name)
    ax[0].fill_between(c["x"], c["lo"], c["hi"], color=col, alpha=0.15)
ax[0].axhline(V, color=INK, ls="--", lw=1.5); ax[0].text(300, V + 0.012, f"best possible {V:.4f}")
ax[0].axhline(Veg, color=CORAL, ls=":", lw=1.5); ax[0].text(2500, Veg - 0.035, f"value while exploring 20%: {Veg:.4f}", color=CORAL)
ax[0].set_xscale("log"); ax[0].set_ylim(0.3, 0.75)
ax[0].set_xlabel("episodes (log scale)"); ax[0].set_ylabel("learned value of waiting")
ax[0].set_title("Learning curves (band = 5 seeds)"); ax[0].legend(frameon=False, fontsize=9, loc="lower right")
Q, _ = offer_rl.learn(alpha="decay", episodes=100_000)
w = offer_rl.offer_levels()
ax[1].plot(w, Q[:, 0], "o-", color=MINT, label="learned Q(offer, accept)")
ax[1].plot(w, Q[:, 1], "o-", color=VIO, label="learned Q(offer, reject)")
ax[1].plot(w, w, color=MINT, lw=0.8, ls=":"); ax[1].axhline(V, color=VIO, lw=0.8, ls=":")
ax[1].axvline(offer_rl.learned_cutoff(Q) - 0.025, color=INK, ls="--", lw=1)
ax[1].set_xlabel("offer"); ax[1].set_ylabel("Q value"); ax[1].legend(frameon=False, fontsize=9)
ax[1].set_title("The learned report card (dotted = truth)")
alphas = [0.01, 0.05, 0.1, 0.3, "decay"]; epss = [0.05, 0.2, 0.5]
M = np.array([[next(r["abs_error"] for r in grid if r["alpha"] == a and r["eps"] == e) for e in epss] for a in alphas])
im = ax[2].imshow(np.log10(M), cmap="viridis_r", aspect="auto")
ax[2].set_xticks(range(3)); ax[2].set_xticklabels([f"ε={e}" for e in epss])
ax[2].set_yticks(range(5)); ax[2].set_yticklabels([f"α={a}" for a in alphas])
for i in range(5):
    for j in range(3):
        ax[2].text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                   color="white" if np.log10(M[i, j]) > -1.5 else "black", fontsize=10)
ax[2].set_title("Error after 20k episodes (5 seeds)"); ax[2].spines[:].set_visible(False)
fig.savefig(os.path.join(FIG, "offer_rl.png")); plt.close(fig)
q10 = offer_rl.summary("q", alpha=0.1)
qd = offer_rl.summary("q", alpha="decay", episodes=100_000, seeds=range(5))
sd = offer_rl.summary("sarsa", alpha="decay", episodes=100_000, seeds=range(5))
R["offer"] = {"true_value": V, "continuous_value": offer_rl.continuous_value(), "epsilon_greedy_value": Veg,
              "policy_value_0625": offer_rl.policy_value(0.625), "policy_value_0675": offer_rl.policy_value(0.675),
              "q_const_10seeds": q10, "q_decay_5seeds_100k": qd, "sarsa_decay_5seeds_100k": sd, "grid": grid}

# ------------------------------------------------------------- gold -------
dp = gold_rl.dp_solution()
Qg, visits, _ = gold_rl.learn()
c_rl = gold_rl.learned_cutoff(Qg)
gc = experiments.gold_curves()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
keep = (dp["grid"] > -45) & (dp["grid"] < 145)
ax[0].plot(dp["grid"][keep], dp["sell"][keep], color=CORAL, lw=1.5, label="ship (true)")
ax[0].plot(dp["grid"][keep], dp["hold"][keep], color=VIO, lw=1.5, label="hold (true)")
mid = slice(1, -1)
ax[0].plot(gold_rl.CENTERS[mid], Qg[mid, 0], "o", color=CORAL, ms=4, label="ship (learned)")
ok = visits[:, 1] > 50
ax[0].plot(gold_rl.CENTERS[ok], Qg[ok, 1], "s", color=VIO, ms=4, label="hold (learned)")
ax[0].axvline(dp["cutoff"], color=INK, ls="--", lw=1)
ax[0].set_xlabel("premium ($/oz)"); ax[0].set_ylabel("value ($/oz)"); ax[0].legend(frameon=False, fontsize=9)
ax[0].set_title("Learned values vs. the optimal-control answer")
for k in range(gc["cutoffs"].shape[0]):
    ax[1].plot(gc["x"], np.clip(gc["cutoffs"][k], 0, 140), color=MINT, alpha=0.6, lw=1.2, drawstyle="steps-post")
ax[1].axhline(dp["cutoff"], color=INK, ls="--", label=f"optimal ${dp['cutoff']:.2f}")
ax[1].axhspan(50, 55, color=LEM, alpha=0.2, label="the $5 bin containing it")
ax[1].set_xscale("log"); ax[1].set_ylim(0, 145)
ax[1].set_xlabel("episodes (log scale)"); ax[1].set_ylabel("learned shipping cutoff ($/oz)")
ax[1].set_title("Early guesses jump; all 5 agents settle on $50"); ax[1].legend(frameon=False, fontsize=9)
ax[2].bar(gold_rl.CENTERS, visits.sum(1), width=4.5, color=GRAY)
ax[2].set_yscale("log"); ax[2].set_xlabel("premium bin ($/oz)"); ax[2].set_ylabel("visits (log scale)")
ax[2].set_title("Experience is thin in the tails")
fig.savefig(os.path.join(FIG, "gold_rl.png")); plt.close(fig)
cut_final = gc["cutoffs"][:, -1]
v_opt, v_se = gold_rl.evaluate(dp["cutoff"])
R["gold"] = {"dp_cutoff": dp["cutoff"], "rl_cutoff_seed6875": c_rl,
             "final_cutoffs_5_seeds": cut_final.tolist(),
             "cutoffs_at": {str(int(x)): gc["cutoffs"][:, i].tolist() for i, x in enumerate(gc["x"])
                            if int(x) in (2000, 10000, 50000, 200000)},
             "value_optimal": v_opt, "value_se": v_se, "value_learned": gold_rl.evaluate(c_rl)[0],
             "value_by_cutoff": {str(c): gold_rl.evaluate(c)[0] for c in (40, 45, 50, 55, 60)},
             "total_weeks_experienced": float(visits.sum()), "h": gold_rl.MARKET.h, "beta": gold_rl.MARKET.beta}

with open(os.path.join(ROOT, "results", "results.json"), "w") as fh:
    json.dump(R, fh, indent=2, default=float)
print(json.dumps({k: v for k, v in R.items()}, indent=1, default=float)[:4000])
