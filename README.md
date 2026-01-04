# Optimization Mini-Project: Multi-Armed Bandits (Topic 4)

This repository contains my implementation and experimental report for **Topic 4: Stochastic + Adversarial Multi-Armed Bandits**.
We evaluate three classic bandit algorithms on **two required environments** (one stationary, one non-stationary) under the project setting:

- Number of arms: **K = 10**
- Horizon: **T = 20000**
- Repetitions: **N_RUNS = 20**
- Reward: **Bernoulli**

---

## Quick start

From the **project root directory**:

```bash
pip install -r requirements.txt
python main.py
```

Outputs are written to:
- `results/` (per-run cumulative regret CSVs + runtime CSVs)
- `figures/` (regret curves)

---

## Environments

### Env A (stationary Bernoulli, required)

- Best arm mean: **0.60**
- Other arms: **linearly spaced in [0.45, 0.58]**
- Reward model: \(r_t \sim \mathrm{Bernoulli}(\mu_{I_t})\)

### Env B (non-stationary Bernoulli, required)

- Best arm **switches every 4000 steps** (5 stages over T=20000)
- For each run, we first sample baseline means for all arms:
  \(\mu_k^{base} \sim \mathrm{Uniform}(0.45, 0.58)\).
- In each stage, one arm is set to **0.60** (the stage-optimal arm), while all other arms keep their baseline means.

So Env B is non-stationary because the **identity of the best arm changes**, while suboptimal arms stay within \([0.45,0.58]\).

---

## Algorithms

Implemented in `src/agents.py`:

- **EpsilonGreedy (ε=0.1)** (baseline)
- **UCB1 (c=2)** (implemented from scratch)
- **Thompson Sampling (Beta-Bernoulli)** (implemented from scratch)

---

## Metric: cumulative regret (empirical)

We report the **empirical cumulative regret** computed step-by-step:

\[
R_t = \sum_{s=1}^{t} (\mu_s^{\*} - r_s),
\]

where \(r_s\) is the realized Bernoulli reward and \(\mu_s^{\*}\) is the best achievable mean reward at step \(s\).

- **Env A (stationary):** \(\mu_s^{\*}=0.60\) for all \(s\)
- **Env B (non-stationary):** \(\mu_s^{\*}\) is **time-varying** due to the best-arm switch every 4000 steps (i.e., **dynamic-oracle regret**)

---

## Runtime (measured)

Wall-clock runtime measured from my run (20 repetitions per algorithm).

### Env A

| Agent | total_seconds | seconds_per_run |
|---|---:|---:|
| EpsilonGreedy (e=0.1) | 2.4093 | 0.1205 |
| UCB1 (c=2) | 7.0801 | 0.3540 |
| ThompsonSampling | 9.1904 | 0.4595 |

### Env B

| Agent | total_seconds | seconds_per_run |
|---|---:|---:|
| EpsilonGreedy (e=0.1) | 2.7571 | 0.1379 |
| UCB1 (c=2) | 7.4104 | 0.3705 |
| ThompsonSampling | 9.5167 | 0.4758 |

The raw CSVs are saved as:
- `results/runtime_Stable_Env_A.csv`
- `results/runtime_NonStationary_Env_B.csv`

---

## Project structure

```text
.
├── main.py
├── Project.md
├── report.md
├── README.md
├── requirements.txt
├── src/
│   ├── agents.py
│   └── environments.py
├── results/
├── figures/
└── xxx.txt   # run log (optional)
```
