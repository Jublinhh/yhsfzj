# Report (Topic 4: Stochastic + Adversarial Multi-Armed Bandits)

**Team member:** 赵朝彬  
**Reproducibility:** `python main.py` (run from project root)

---

## 1. Problem statement

We study the **multi-armed bandit** (MAB) problem with **K = 10** arms and horizon **T = 20000**.
At each time step \(t\), the agent chooses an arm \(I_t\) and only observes the reward of that arm:
\[
r_t \sim \mathrm{Bernoulli}(\mu_{I_t}).
\]
The goal is to maximize cumulative reward, equivalently minimize regret.

### Regret definition used in this project

We report the **empirical cumulative regret** computed step-by-step:
\[
R_t = \sum_{s=1}^{t} (\mu_s^{\*} - r_s),
\]
where \(r_s\) is the realized Bernoulli reward at step \(s\), and \(\mu_s^{\*}\) is the best achievable mean reward at step \(s\).

- **Env A (stationary):** \(\mu_s^{\*}=\mu^{\*}=0.60\) for all \(s\).  
- **Env B (non-stationary):** \(\mu_s^{\*}\) is **time-varying** because the best arm switches every 4000 steps; therefore, we evaluate algorithms using **dynamic-oracle regret** (best arm at each time step).

---

## 2. Environments

### 2.1 Env A (stationary Bernoulli, required)

- Best arm mean: **0.60**
- Other arms: **linearly spaced in [0.45, 0.58]**
- Rewards: \(r_t \sim \mathrm{Bernoulli}(\mu_{I_t})\)

### 2.2 Env B (non-stationary Bernoulli, required)

The horizon is divided into **5 stages** of length **4000**. For each run, we first sample baseline means:
\[
\mu_k^{base} \sim \mathrm{Uniform}(0.45, 0.58), \quad k=1,\dots,K.
\]
In each stage, a (different) arm is forced to be optimal by setting its mean to **0.60**, while all other arms keep their baseline means:
\[
\mu_{k,t} =
\begin{cases}
0.60, & k = k^{\*}(\mathrm{stage}(t)) \\
\mu_k^{base}, & \text{otherwise}.
\end{cases}
\]
Thus, Env B is non-stationary because the **identity of the best arm changes every 4000 steps**, while the suboptimal arms remain within \([0.45,0.58]\).

---

## 3. Algorithms

All agents are implemented in `src/agents.py`.

1. **Epsilon-Greedy (baseline)**  
   With probability \(\varepsilon\) explore a random arm, otherwise exploit the arm with the highest estimated mean.
   We use **\(\varepsilon = 0.1\)** (selected based on simple tuning in earlier experiments).

2. **UCB1 (Upper Confidence Bound)**  
   Choose the arm maximizing:
   \[
   \hat\mu_i(t) + c\sqrt{\frac{\log t}{n_i(t)}},
   \]
   where \(n_i(t)\) is the number of pulls of arm \(i\) up to time \(t\). We use **\(c=2\)**.

3. **Thompson Sampling (Beta–Bernoulli)**  
   Maintain a Beta posterior for each arm: \(\mathrm{Beta}(\alpha_i,\beta_i)\).  
   Sample \(\theta_i \sim \mathrm{Beta}(\alpha_i,\beta_i)\) and pull \(\arg\max_i \theta_i\).  
   We use the standard prior **\(\alpha_i=\beta_i=1\)**.

---

## 4. Experimental setup

- **Horizon:** \(T=20000\)
- **Arms:** \(K=10\)
- **Runs:** \(N\_RUNS=20\)
- We fix a global seed and generate per-run seeds so that each algorithm is evaluated on **the same random environments** for fair comparison.
- We plot the mean regret curve across runs and also report the final cumulative regret \(R_T\) as mean ± std.

---

## 5. Results

### 5.1 Performance in Env A (stationary)

**Table 1:** Final cumulative regret (Mean ± Std. Dev.) in Environment A

| Algorithm | Final Cumulative Regret |
|---|---:|
| **UCB1 (c=2)** | **223.23 ± 108.17** |
| Thompson Sampling | 395.75 ± 256.18 |
| EpsilonGreedy (ε=0.1) | 949.65 ± 89.06 |

**Figure 1:** Regret curves in Environment A

![Regret Curve for Env A](figures/regret_curve_Stable_Env_A.png)

**Observation.** In this stationary setting, **UCB1 achieves the lowest final regret**, and it also shows the fastest reduction of regret over time. Thompson Sampling performs reasonably but with higher variance, while ε-greedy is consistently worse because it keeps exploring at a constant rate.

---

### 5.2 Performance in Env B (non-stationary)

**Table 2:** Final cumulative regret (Mean ± Std. Dev.) in Environment B

| Algorithm | Final Cumulative Regret |
|---|---:|
| **UCB1 (c=2)** | **864.70 ± 270.98** |
| Thompson Sampling | 1072.40 ± 312.45 |
| EpsilonGreedy (ε=0.1) | 1102.55 ± 145.28 |

**Figure 2:** Regret curves in Environment B

![Regret Curve for Env B](figures/regret_curve_NonStationary_Env_B.png)

**Observation.** Regret increases substantially for all methods because the optimal arm changes every 4000 steps. Under this particular Env B construction (only the identity of the best arm changes; suboptimal means stay fixed in \([0.45,0.58]\)), **UCB1 still attains the lowest final regret** among the three tested methods.

---

### 5.3 Runtime

Wall-clock runtime measured on my run (20 repetitions per algorithm).

**Table 3:** Runtime in Env A

| Agent | total_seconds | seconds_per_run |
|---|---:|---:|
| EpsilonGreedy (e=0.1) | 2.4093 | 0.1205 |
| UCB1 (c=2) | 7.0801 | 0.3540 |
| ThompsonSampling | 9.1904 | 0.4595 |

**Table 4:** Runtime in Env B

| Agent | total_seconds | seconds_per_run |
|---|---:|---:|
| EpsilonGreedy (e=0.1) | 2.7571 | 0.1379 |
| UCB1 (c=2) | 7.4104 | 0.3705 |
| ThompsonSampling | 9.5167 | 0.4758 |

---

## 6. Discussion

1. **Stationary (Env A).**  
   UCB1 performs best here because its optimism-based exploration efficiently focuses on high-mean arms once enough evidence is collected. ε-greedy continues to explore at rate ε forever, which causes avoidable regret late in the horizon.

2. **Non-stationary (Env B).**  
   None of the tested algorithms is explicitly designed for non-stationarity (e.g., discounting, sliding windows, or restart mechanisms). Nevertheless, in this specific setting the environment changes only 4 times over the horizon (every 4000 steps), and suboptimal arm means remain in a narrow range. Empirically, UCB1 still achieves the lowest final regret. A natural extension would be to implement non-stationary variants such as discounted UCB / sliding-window UCB or change-point detection to better track the switching best arm.

3. **Efficiency trade-off.**  
   ε-greedy is fastest; UCB1 is moderately slower due to computing confidence bonuses; Thompson Sampling is slowest because it draws Beta samples for all arms at each step.

---

## 7. Conclusion

Across both required environments, **UCB1 (c=2)** achieves the lowest final regret in my experiments. Thompson Sampling is competitive but exhibits higher variance and higher runtime. The ε-greedy baseline is simple and fast but performs worse due to persistent random exploration. Future work should include algorithms tailored for non-stationary bandits to further reduce regret in Env B.

