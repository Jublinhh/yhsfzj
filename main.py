import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os # 导入os模块来操作文件夹
import time

# 从 src 中导入我们所有的“蓝图”
from src.environments import BernoulliBandit, NonStationaryBernoulliBandit
from src.agents import EpsilonGreedy, UCB1, ThompsonSampling

# --- 1. 全局参数设定 ---
K = 10
T = 20000
N_RUNS = 20
RANDOM_SEED = 42
OUTPUT_DIR = "results/"
FIGURE_DIR = "figures/"

# 确保输出文件夹存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# --- 2. 运行单次实验的函数 (这个函数不需要改变) ---
def run_experiment(env, agent):
    regret_history = []
    cumulative_regret = 0.0

    for t in range(T):
        # 注意：非平稳环境会在 env.step() 内部更新 best_prob
        # 所以必须在 step() 之前保存“当前时刻”的 μ*_t
        best_prob_t = env.best_prob

        chosen_arm = agent.choose_arm()
        reward = env.step(chosen_arm)
        agent.update(chosen_arm, reward)

        # Project.md (stochastic) 要求：μ* - r_t（逐步累加）
        instant_regret = best_prob_t - reward
        cumulative_regret += instant_regret
        regret_history.append(cumulative_regret)

    return np.array(regret_history)



# --- 3. 画图和保存结果的函数 (把这部分独立出来，方便复用) ---
def save_and_plot_results(results, env_name, runtimes=None):
    print(f"\n--- {env_name} 实验完成，正在保存和绘图... ---")
    
    final_regrets = {}
    for agent_name, agent_results in results.items():
        mean_regret = np.mean(agent_results, axis=0)
        std_regret = np.std(agent_results, axis=0)
        
        df = pd.DataFrame({
            'mean_regret': mean_regret,
            'std_regret': std_regret,
            'time_step': np.arange(T)
        })
        # 保存文件名中加入环境名以区分
        df.to_csv(f"{OUTPUT_DIR}{agent_name.replace(' ', '_')}_{env_name}.csv", index=False)
        
        final_regrets[agent_name] = f"{mean_regret[-1]:.2f} ± {std_regret[-1]:.2f}"

    print(f"结果已保存到 {OUTPUT_DIR} 文件夹。")
    print(f"\n{env_name} 环境下最终累计遗憾值 (T=20000):")
    print(pd.Series(final_regrets))
# ----
    # --- NEW: 保存与打印 runtime ---
    if runtimes is not None:
        runtime_rows = []
        for agent_name, total_sec in runtimes.items():
            runtime_rows.append({
                "agent": agent_name,
                "total_seconds": total_sec,
                "seconds_per_run": total_sec / N_RUNS
            })
        runtime_df = pd.DataFrame(runtime_rows).sort_values("total_seconds")
        runtime_path = f"{OUTPUT_DIR}runtime_{env_name}.csv"
        runtime_df.to_csv(runtime_path, index=False)

        print(f"\n{env_name} 环境下 runtime (wall-clock):")
        print(runtime_df.to_string(index=False))
        print(f"runtime 已保存到: {runtime_path}")

# ----
    plt.figure(figsize=(12, 8))
    for agent_name in results.keys():
        df = pd.read_csv(f"{OUTPUT_DIR}{agent_name.replace(' ', '_')}_{env_name}.csv")
        mean_regret = df['mean_regret']
        std_regret = df['std_regret']
        plt.plot(df['time_step'], mean_regret, label=agent_name)
        plt.fill_between(df['time_step'], 
                         mean_regret - std_regret, 
                         mean_regret + std_regret, 
                         alpha=0.15)

    plt.title(f"Regret Curve Comparison on {env_name} Bandit (20 runs)")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.grid(True)
    
    figure_path = f"{FIGURE_DIR}regret_curve_{env_name}.png"
    plt.savefig(figure_path)
    print(f"图片已保存到: {figure_path}")


# --- 4. 主程序入口 ---
if __name__ == "__main__":
    
    # 定义我们的参赛选手
    agents_dict = {
        "EpsilonGreedy (e=0.1)": lambda k: EpsilonGreedy(k, epsilon=0.1),
        "UCB1 (c=2)": lambda k: UCB1(k, c=2),
        "ThompsonSampling": lambda k: ThompsonSampling(k)
    }

    # ==================================================================
    #                      第一场比赛: 稳定环境 (Env A)
    # ==================================================================
    print("="*60)
    print("               开始运行第一场比赛: 稳定环境 (A)")
    print("="*60)
    
    np.random.seed(RANDOM_SEED)
    env_seeds = [np.random.randint(100000) for _ in range(N_RUNS)]
    results_A = {}
    # ----
    runtimes_A = {}

    # for agent_name, agent_builder in agents_dict.items():
    #     agent_regrets = []
    #     for i in tqdm(range(N_RUNS), desc=f"稳定环境: {agent_name}"):
    #         np.random.seed(env_seeds[i])
    #         best_prob = 0.6
    #         # other_probs = np.random.uniform(low=0.45, high=0.58, size=K-1)
    #         other_probs = np.linspace(0.45, 0.58, K-1)

    #         all_probs = np.append(other_probs, best_prob)
    #         np.random.shuffle(all_probs)
    #         env = BernoulliBandit(probs=all_probs)
    #         agent = agent_builder(K)
    #         regret_curve = run_experiment(env, agent)
    #         agent_regrets.append(regret_curve)
    #     results_A[agent_name] = np.array(agent_regrets)
        
    # save_and_plot_results(results_A, "Stable_Env_A")
    for agent_name, agent_builder in agents_dict.items():
        t0 = time.perf_counter()  # <-- NEW
        agent_regrets = []
        for i in tqdm(range(N_RUNS), desc=f"稳定环境: {agent_name}"):
            np.random.seed(env_seeds[i])
            best_prob = 0.6
            other_probs = np.linspace(0.45, 0.58, K-1)

            all_probs = np.append(other_probs, best_prob)
            np.random.shuffle(all_probs)
            env = BernoulliBandit(probs=all_probs)
            agent = agent_builder(K)
            regret_curve = run_experiment(env, agent)
            agent_regrets.append(regret_curve)
        results_A[agent_name] = np.array(agent_regrets)
        t1 = time.perf_counter()  # <-- NEW
        runtimes_A[agent_name] = t1 - t0  # <-- NEW
    save_and_plot_results(results_A, "Stable_Env_A", runtimes_A)



    # ==================================================================
    #                   第二场比赛: 非平稳环境 (Env B)
    # ==================================================================
    print("\n" * 3)
    print("="*60)
    print("              开始运行第二场比赛: 非平稳环境 (B)")
    print("="*60)

    # 任务要求：最好的臂每4000步切换一次
    # T=20000, 所以我们总共需要 20000 / 4000 = 5 个阶段
    change_points = [4000, 8000, 12000, 16000]
    results_B = {}
    runtimes_B = {}

    # for agent_name, agent_builder in agents_dict.items():
    #     agent_regrets = []
    #     for i in tqdm(range(N_RUNS), desc=f"非平稳环境: {agent_name}"):
    #         # 每次运行都重新生成一套会变的概率
    #         np.random.seed(env_seeds[i])
    #         probs_list = []
    #         base_probs = np.random.uniform(low=0.45, high=0.58, size=K)
    #         best_arm_indices = np.random.choice(K, size=5, replace=False) # 5个阶段，选5个不同的最佳臂

    #         for stage in range(5):
    #             stage_probs = base_probs.copy()
    #             stage_probs[best_arm_indices[stage]] = 0.6 # 在该阶段设置最佳臂
    #             probs_list.append(stage_probs)

    #         env = NonStationaryBernoulliBandit(probs_list, change_points)
    #         agent = agent_builder(K)
    #         regret_curve = run_experiment(env, agent)
    #         agent_regrets.append(regret_curve)
    #     results_B[agent_name] = np.array(agent_regrets)

    # save_and_plot_results(results_B, "NonStationary_Env_B")
    for agent_name, agent_builder in agents_dict.items():
        t0 = time.perf_counter()  # <-- NEW
        agent_regrets = []
        for i in tqdm(range(N_RUNS), desc=f"非平稳环境: {agent_name}"):
            np.random.seed(env_seeds[i])
            probs_list = []
            base_probs = np.random.uniform(low=0.45, high=0.58, size=K)
            best_arm_indices = np.random.choice(K, size=5, replace=False)

            for stage in range(5):
                stage_probs = base_probs.copy()
                stage_probs[best_arm_indices[stage]] = 0.6
                probs_list.append(stage_probs)

            env = NonStationaryBernoulliBandit(probs_list, change_points)
            agent = agent_builder(K)
            regret_curve = run_experiment(env, agent)
            agent_regrets.append(regret_curve)
        results_B[agent_name] = np.array(agent_regrets)
        t1 = time.perf_counter()  # <-- NEW
        runtimes_B[agent_name] = t1 - t0  # <-- NEW
    save_and_plot_results(results_B, "NonStationary_Env_B", runtimes_B)


    print("\n\n--- 所有比赛全部完成！ ---")