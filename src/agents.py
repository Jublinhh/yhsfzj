import numpy as np

class EpsilonGreedy:
    """
    ε-greedy (小比例贪心) 策略。
    
    这是我们的第一个“玩家”模型。
    """
    
    def __init__(self, k, epsilon=0.1):
        """
        初始化玩家。
        
        参数:
            k (int): 玩家知道有多少台老虎机（手臂数量）。
            epsilon (float): 探索的比例，一个0到1之间的小数。
                              例如 0.1 代表有 10% 的机会去随机探索。
        """
        self.k = k
        self.epsilon = epsilon
        
        # 我们需要一些“账本”来记录每个手臂的情况
        self.counts = np.zeros(k, dtype=int)     # 账本1: 记录每个手臂被拉了多少次
        self.values = np.zeros(k, dtype=float)   # 账本2: 记录每个手臂的“平均奖励”的估计值
        print(f"ε-greedy 玩家已创建，ε = {self.epsilon}")

    def choose_arm(self):
        """
        根据策略，决定这次要拉动哪个手臂。
        
        返回:
            arm (int): 决定要拉动的手臂编号。
        """
        # 生成一个 0 到 1 的随机数
        if np.random.rand() < self.epsilon:
            # 进入“探索”模式：随机选一个手臂
            return np.random.randint(0, self.k)
        else:
            # 进入“利用”模式：选择目前为止平均奖励最高的那个手臂
            # 如果有好几个手臂的平均奖励并列最高，np.argmax 会默认选择第一个
            return np.argmax(self.values)

    def update(self, arm, reward):
        """
        玩了一次之后，更新我们的“账本”。
        
        参数:
            arm (int): 刚刚拉动的手臂编号。
            reward (int): 得到的奖励 (1 或 0)。
        """
        # 1. 给被拉动的手臂的“次数”账本加 1
        self.counts[arm] += 1
        n = self.counts[arm]
        
        # 2. 更新这个手臂的“平均奖励”估计值
        # 这里用了一个经典的增量更新公式，效果等同于 (旧的总奖励 + 新奖励) / 新次数
        old_value = self.values[arm]
        new_value = old_value + (1/n) * (reward - old_value)
        self.values[arm] = new_value


class UCB1:
    """
    UCB1 (信心上界) 策略。
    
    一个更聪明的“玩家”，它会根据不确定性来指导探索。
    """
    
    def __init__(self, k, c=2):
        """
        初始化玩家。
        
        参数:
            k (int): 手臂数量。
            c (float): 探索参数，用来控制“好奇心”的权重。
                       c越大，就越倾向于探索。
        """
        self.k = k
        self.c = c
        self.t = 0  # 用来记录总共玩了多少次
        
        # 账本和 EpsilonGreedy 一样
        self.counts = np.zeros(k, dtype=int)
        self.values = np.zeros(k, dtype=float)
        print(f"UCB1 玩家已创建，探索参数 c = {self.c}")

    def choose_arm(self):
        """
        根据 UCB1 公式，决定这次要拉动哪个手臂。
        """
        self.t += 1 # 每次选择前，总次数加 1
        
        # 优先策略：如果有的手臂一次都还没被玩过，就先玩它
        # 这可以避免计算公式时分母为零
        never_played = np.where(self.counts == 0)[0]
        if len(never_played) > 0:
            return never_played[0]

        # 计算每个手臂的“好奇心加分”（也叫不确定性项）
        # self.t 是总次数，self.counts 是每个手臂被玩的次数
        # 玩得越少(counts小)，这一项的值就越大
        uncertainty = self.c * np.sqrt(np.log(self.t) / self.counts)
        
        # 综合分 = 平均奖励 + 好奇心加分
        ucb_scores = self.values + uncertainty
        
        # 选择综合分最高的那个手臂
        return np.argmax(ucb_scores)

    def update(self, arm, reward):
        """
        更新账本，和 EpsilonGreedy 的方法完全一样。
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        old_value = self.values[arm]
        new_value = old_value + (1/n) * (reward - old_value)
        self.values[arm] = new_value

class ThompsonSampling:
    """
    汤普森采样 (Thompson Sampling) 策略。
    
    一个基于贝叶斯思想的“玩家”，它为每个手臂维护一个概率分布。
    """
    
    def __init__(self, k):
        """
        初始化玩家。
        
        参数:
            k (int): 手臂数量。
        """
        self.k = k
        
        # 我们需要为每个手臂准备一个“信念账本”
        # 在伯努利环境中，这个信念最适合用 Beta 分布来描述
        # Beta分布由两个参数 α (alpha) 和 β (beta) 决定
        # 我们可以把 α 理解为“赢的次数+1”，β 理解为“输的次数+1”
        self.alphas = np.ones(k)  # 初始化每个臂的 alpha 为 1
        self.betas = np.ones(k)   # 初始化每个臂的 beta 为 1
        print(f"汤普森采样玩家已创建。")

    def choose_arm(self):
        """
        根据每个手臂的“信念”，采样一个值，并选择最高的一个。
        """
        # 从每个手臂的 Beta(alpha, beta) 分布中随机抽取一个样本
        # 这就是“顿悟”的过程
        samples = [np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
        
        # 选择样本值最大的那个手臂
        return np.argmax(samples)

    def update(self, arm, reward):
        """
        根据玩的结果，更新对应手臂的“信念账本”。
        """
        # 如果奖励是1（赢了），就给对应手臂的 alpha 加 1
        if reward == 1:
            self.alphas[arm] += 1
        # 如果奖励是0（输了），就给对应手臂的 beta 加 1
        else:
            self.betas[arm] += 1