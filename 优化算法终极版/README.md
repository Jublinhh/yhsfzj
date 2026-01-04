# Optimization Mini-Project: Multi-Armed Bandits

  

This repository contains the implementation and analysis for the Optimization Mini-Project, focusing on Topic 4. We implement and compare several multi-armed bandit algorithms in both stationary and non-stationary environments.

  

---

  

## Topic Chosen

**Topic 4: Stochastic + Adversarial Multi-Armed Bandits**

  

---

  

## Team Members & Contributions

*   **赵朝彬**: Led the entire project from conception to completion. Independently implemented all algorithms (Epsilon-Greedy, UCB1, Thompson Sampling) and both the stationary and non-stationary bandit environments. Designed and executed all experiments, performed in-depth analysis of the results, and authored the final comprehensive report.

  

---

  

## Project Summary

  

This project explores the multi-armed bandit problem, a classic reinforcement learning challenge focused on the exploration-exploitation trade-off. We implemented three core algorithms and evaluated their performance by measuring cumulative regret over 20,000 time steps.

  

### Implemented Methods ("Ours")

*   **UCB1**: An optimistic algorithm that uses an upper confidence bound to guide exploration toward less-explored, potentially high-reward arms.

*   **Thompson Sampling**: A Bayesian algorithm that samples from the posterior distribution of each arm's reward probability to make decisions, providing a sophisticated balance of exploration and exploitation.

  

### Baseline Method

*   **Epsilon-Greedy**: A simple and effective baseline that explores randomly with a small probability (`ε`) and otherwise exploits the currently known best arm.

  

---

  

## Reproducibility Instructions

  

This project is fully reproducible. The following instructions will regenerate all data in the `results/` directory and all figures in the `figures/` directory.

  

### 1. Environment Setup


**Commands:**

```bash

# 1. Clone this repository

git clone https://github.com/ZJUT-CS/2025optimization-tmp.git

cd 2025optimization-tmp

  

# 2. Create and activate the conda environment

conda create -n bandit_project python=3.10 -y

conda activate bandit_project

  

# 3. Install all required libraries from the requirements file

pip install -r requirements.txt
```

### 2. Run the Main Experiment

After setting up the environment, run the following single command from the project's root directory:

```bash

python main.py

```

This script will execute the full suite of experiments for both **Environment A (Stationary)** and **Environment B (Non-Stationary)**, running 20 trials for each of the three algorithms. It will automatically save all CSV results and PNG figures to their respective directories.

  

**Random Seed:** The experiments use a fixed random seed (`42`) to ensure that the exact results and figures can be reproduced.

  

---

  

## Repository Layout

The project is organized as follows:

```
.
├── README.md

├── report.md

├── requirements.txt

├── src/

│ ├── agents.py  

│ └── environments.py

├── results/

├── figures/

└── xxx.txt # this is my log
```
