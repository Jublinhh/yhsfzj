# 从我们安装的 numpy 神器里，导入 random 功能，用来生成随机数
import numpy as np

class BernoulliBandit:
    """
    一个伯努利多臂老虎机环境。

    这就是我们的“环境A”，一个拥有K个臂（老虎机）的稳定环境。
    每个臂的奖励是伯努利分布的，即每次拉动，要么奖励为1（赢），要么为0（输）。
    """
    
    def __init__(self, probs):
        """
        这是“诞生”方法，当我们创建一个老虎机群时，它会被调用。
        
        参数:
            probs (list or numpy array): 一个列表，包含了每个老虎机手臂的“中奖概率”。
                                          列表的长度就是老虎机的数量 K。
        """
        self.probs = np.array(probs)  # 把概率列表存起来
        self.k = len(probs)           # 老虎机的数量 K
        
        # 找到中奖概率最高的手臂是哪个，以及它对应的概率值
        self.best_arm = np.argmax(self.probs)
        self.best_prob = np.max(self.probs)
        print(f"老虎机环境已创建，共有 {self.k} 个臂。")
        print(f"各个臂的中奖概率是: {[round(p, 2) for p in self.probs]}")
        print(f"最好的臂是 {self.best_arm} 号，它的中奖概率是 {self.best_prob:.2f}")

    def step(self, arm):
        """
        模拟“拉动”一个手臂的动作。
        
        参数:
            arm (int): 要拉动的手臂的编号 (从0开始)。
        
        返回:
            reward (int): 获得的奖励，1 或者 0。
        """
        # 检查一下手臂编号是不是有效的
        if arm < 0 or arm >= self.k:
            raise ValueError(f"手臂编号 {arm} 无效，必须在 0 到 {self.k-1} 之间。")
        
        # 游戏规则：生成一个0到1之间的随机小数，如果这个数小于这台老虎机的中奖概率，就算赢了
        if np.random.rand() < self.probs[arm]:
            return 1  # 赢了，奖励为 1
        else:
            return 0  # 输了，奖励为 0


class NonStationaryBernoulliBandit:
    """
    一个非平稳的伯努利多臂老虎机环境 (环境B)。

    在这个环境中，“最好”的手臂会随着时间的推移而改变。
    """
    
    def __init__(self, probs_list, change_points):
        """
        初始化一个会变的环境。
        
        参数:
            probs_list (list of lists): 一个列表，每个元素是某一阶段的概率列表。
                                        例如: [[0.1, 0.9], [0.9, 0.1]]
            change_points (list): 一个列表，定义了在哪些时间点改变概率。
                                  例如: [10000] 表示在第10000步时，从第一套概率切换到第二套。
                                  它的长度必须比 probs_list 少一个。
        """
        self.probs_list = [np.array(p) for p in probs_list]
        self.change_points = change_points
        self.k = len(probs_list[0]) # 手臂数量
        
        self.t = 0 # 记录当前的时间步
        self.current_stage = 0
        self.probs = self.probs_list[self.current_stage]
        self.best_prob = np.max(self.probs)
        
        print(f"非平稳老虎机环境已创建，共有 {self.k} 个臂。")
        print(f"初始阶段的中奖概率是: {[round(p, 2) for p in self.probs]}")

    def step(self, arm):
        """
        模拟“拉动”一个手臂的动作，并检查是否需要切换环境。
        """
        # 1. 先根据当前的概率，计算奖励 (和稳定环境一样)
        if arm < 0 or arm >= self.k:
            raise ValueError(f"手臂编号 {arm} 无效，必须在 0 到 {self.k-1} 之间。")
        
        reward = 1 if np.random.rand() < self.probs[arm] else 0
        
        # 2. 时间步加1
        self.t += 1
        
        # 3. 检查是否到达了切换点
        # 如果还有未到达的切换点，并且当前时间等于下一个切换点的时间
        if self.current_stage < len(self.change_points) and self.t == self.change_points[self.current_stage]:
            # 进入下一个阶段
            self.current_stage += 1
            self.probs = self.probs_list[self.current_stage]
            self.best_prob = np.max(self.probs)
            print(f"\n注意：环境在时间步 {self.t} 发生改变！")
            print(f"新阶段的中奖概率是: {[round(p, 2) for p in self.probs]}")
        
        return reward