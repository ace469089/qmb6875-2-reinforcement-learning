# Glossary: Every Term in Plain English

The same glossary appears in all three repos (optimal control, reinforcement learning, on a computer), so terms mean the same thing everywhere.

| Term | Plain English | Example from these repos |
|---|---|---|
| **Action / control** $u$, $a$ | The choice you make each period | feed amount; accept or reject; ship or hold |
| **Agent** | The decision maker | the gold trader |
| **AR(1)** | A number that drifts back toward its average, plus a random shock: $s'=\mu+\rho(s-\mu)+\sigma\varepsilon$ | the NY gold premium |
| **Bandit (multi-armed)** | An RL problem with one state: keep choosing among options with unknown payoffs | which ad creative to show |
| **Bellman equation** | Best value from here = today's payoff + discounted best value from where you land | $V(x)=\max_u\{R+\beta E[V(x')]\}$ |
| **Bellman map / operator** $T$ | The right-hand side of the Bellman equation, as a machine that turns a guess of $V$ into a better guess | $T(V)=0.9(1+V^2)/2$ |
| **Common random numbers** | Comparing two policies on the *same* simulated luck, so differences are due to the policy | every gold rule sees the same premiums |
| **Contraction** | A map that shrinks the distance between any two inputs by a fixed factor $<1$; repeated application converges to one point | Bellman map shrinks gaps by $\beta$ |
| **Control-limit policy** | "Act once the state crosses a line" | ship iff premium ≥ $50.49 |
| **Converge** | The guesses stop changing (within a tolerance) | value iteration after 38 steps |
| **Costate / shadow value** $\lambda$ | The value of one more unit of the state | the value of one more pound of heifer |
| **Curse of dimensionality** | Grids explode: 100 points per variable × 5 variables = 10 billion states | why deep RL exists |
| **Deterministic** | No randomness: today's state and action fully determine tomorrow | tree, heifer |
| **Discount factor** $\beta$ | What \$1 next period is worth today; $\beta=e^{-r\Delta t}$ | 0.9 in the offer problem |
| **Dynamics / law of motion / transition** | How the state moves | $\dot W=a\sqrt f-mW$ |
| **Dynamic programming** | Solving a problem over time by working with the value function (Bellman) | value and policy iteration |
| **Episode** | One complete play of a game, from start to the end (a sale, a theft, a deadline) | one gold shipment |
| **Euler–Maruyama** | Simulating a random differential equation in small time steps, adding $\sigma\sqrt{\Delta t}\,\varepsilon$ each step | noisy heifer weight |
| **Exploration vs. exploitation** | Trying something to learn about it vs. doing what has worked best so far | ε-greedy |
| **ε-greedy** | With probability ε act randomly; otherwise pick the best-looking action | ε = 0.2 |
| **Expected value** $E[\cdot]$ | Probability-weighted average | $E[\max(w,V)]=(1+V^2)/2$ |
| **Fixed point** | An input a map leaves unchanged: $T(V)=V$ | $V^*=0.627$ |
| **Grid / discretization** | Replacing a continuous variable with a list of points | 201 premium values |
| **Hamiltonian** $H$ | Profit flow now + value of the state you're building (continuous-time control) | $-cfe^{-rt}+\lambda(a\sqrt f-mW)$ |
| **HJB equation** | Continuous-time stochastic Bellman equation, with the Itô term $\tfrac12\sigma^2V''$ | Theory §6 (repo 1) |
| **Jacobian** | Matrix of partial derivatives of a vector function; the slope used by multi-equation Newton | 3×3 in the lognormal method of moments |
| **Learning rate** $\alpha$ | How far you move a guess toward new information | 0.1 = 10% of the way |
| **Markov chain** | States plus a table of probabilities of moving between them | Tauchen's 201×201 table |
| **Markov decision process (MDP)** | States, actions, transition probabilities, rewards, discount: the formal setting of control and RL | the gold problem on a grid |
| **Monte Carlo** | Estimating an expectation by simulating many times and averaging | value of a selling rule |
| **Newton's method** | Find where $f(x)=0$ by following the tangent line: $x_{k+1}=x_k-f(x_k)/f'(x_k)$ | tree age, lognormal moments |
| **Off-policy / on-policy** | Learning about the best policy while doing something else (Q-learning) vs. learning about the policy you're following (SARSA) | repo 2 |
| **Optimal stopping** | Control problems whose only decision is *when* to stop | tree, offer, gold |
| **Policy** | A rule: in each state, which action | accept iff $w\ge0.627$ |
| **Policy iteration** | Alternate between "value of current policy" and "improve the policy" | repo 3 |
| **Pontryagin's maximum principle** | Continuous-time conditions for an optimal plan: maximize $H$, costate equation, transversality | heifer |
| **Principle of optimality** | An optimal plan stays optimal from any point it reaches | basis of Bellman |
| **Q-function** $Q(s,a)$ | Value of taking action $a$ in state $s$ and then behaving optimally; a report card per action | $Q(w,\text{reject})=0.627$ |
| **Q-learning** | Learn $Q$ from experience: $Q\leftarrow Q+\alpha[r+\beta\max Q'-Q]$ | repo 2 |
| **Regularization** | Adding a small stabilizing term so a calculation can't blow up | $J+\lambda I$ in Newton |
| **Reinforcement learning (RL)** | Learning a good policy from rewards by trial and error, without knowing the dynamics | repo 2 |
| **Reservation value / wage** | The cutoff below which you reject an offer | 0.627 |
| **Reward** $R$, $r$ | What you get paid this period | the accepted offer |
| **Seed** | The starting number for a random generator; same seed = same "random" numbers = reproducible | 6875 |
| **Standard error** | How uncertain a simulated average is: sd / √(number of runs) | ±0.40 $/oz |
| **State** $x$, $s$ | Everything you need to know right now to decide | weight; premium; current offer |
| **Stochastic** | Involving randomness | offer, gold |
| **Tauchen's method** | Turning an AR(1) into a Markov chain on a grid | gold premium |
| **Temporal-difference (TD) error** | The "surprise": what just happened minus what you predicted | bracket in the Q update |
| **Transversality condition** | The end condition on the costate | $\lambda(T)=pe^{-rT}$ |
| **Value function** $V(x)$ | Best total discounted payoff from state $x$ | $V(s)$ for gold |
| **Value iteration** | Apply the Bellman map repeatedly until it stops changing | 38 steps (offer), 196 (gold) |
