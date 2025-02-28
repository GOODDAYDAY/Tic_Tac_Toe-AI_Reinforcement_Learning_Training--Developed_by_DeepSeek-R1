import argparse
import logging
import os
import pickle
import random
import tkinter as tk
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("TicTacToe")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TicTacToeEnv:
    """可扩展的棋盘游戏环境，支持井字棋和类似规则的游戏"""

    def __init__(self, size=3):
        """
        初始化游戏环境
        :param size: 棋盘尺寸，默认为3（3x3的井字棋）
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 当前执棋玩家（1或2）
        self.winning_length = size if size <= 5 else 5  # 支持五子棋规则

    def reset(self):
        """重置游戏环境"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        return self._get_state()

    def _get_state(self):
        """获取当前棋盘的一维展开状态"""
        return self.board.flatten().copy()

    def get_valid_actions(self):
        """获取所有合法动作坐标列表"""
        return list(zip(*np.where(self.board == 0)))

    def step(self, action):
        """
        执行动作并返回环境反馈
        :param action: 动作坐标元组 (row, col)
        :return: (新状态, 奖励, 是否终止, 附加信息)
        """
        row, col = action
        if self.board[row, col] != 0:
            logger.error(f"无效动作: {action}")
            raise ValueError("Invalid action")

        acting_player = self.current_player  # 记录当前执行动作的玩家
        self.board[row, col] = acting_player  # 落子

        # 判断游戏是否结束
        winner = self._check_winner()
        done = winner is not None or len(self.get_valid_actions()) == 0

        # 计算奖励（从执行动作玩家的视角）
        reward = self._calculate_reward(acting_player, winner)

        # 切换执棋玩家（确保在计算奖励后再切换）
        self.current_player = 3 - self.current_player

        logger.debug(f"动作: {action}, 奖励: {reward}, 游戏结束: {done}")
        return self._get_state(), reward, done, {}

    def _check_line(self, line: np.ndarray, player: int) -> bool:
        """
        检查单行是否包含连续获胜序列
        :param line: 待检查的行/列/对角线
        :param player: 玩家编号（1或2）
        :return: 是否满足获胜条件
        """
        count = 0
        for cell in line:
            if cell == player:
                count += 1
                if count >= self.winning_length:
                    return True
            else:
                count = 0
        return False

    def _check_winner(self) -> Optional[int]:
        """
        检查当前棋盘是否有获胜者
        :return: 获胜玩家编号（1/2），平局返回None
        """
        # 遍历两个玩家进行检查
        for player in [1, 2]:
            # 检查所有行和列
            for i in range(self.size):
                if self._check_line(self.board[i, :], player) or self._check_line(
                        self.board[:, i], player
                ):
                    return player

            # 检查对角线（考虑不同偏移量）
            for offset in range(
                    -self.size + self.winning_length, self.size - self.winning_length + 1
            ):
                # 主对角线方向
                if self._check_line(np.diag(self.board, k=offset), player):
                    return player
                # 副对角线方向
                if self._check_line(np.diag(np.fliplr(self.board), k=offset), player):
                    return player
        return None

    def _calculate_reward(
            self, acting_player: int, winner: Optional[int]
    ) -> float:
        """
        计算当前玩家的即时奖励
        :param acting_player: 执行当前动作的玩家
        :param winner: 获胜玩家（None表示未决状态）
        :return: 奖励值
        """
        if winner is not None:
            return 1.0 if acting_player == winner else -1.0  # 胜负奖励
        if len(self.get_valid_actions()) == 0:
            return 0.0  # 平局奖励
        return 0.1  # 中间过程奖励


class DQN(nn.Module):
    """深度Q网络，支持自动选择最优动作"""

    def __init__(self, input_size: int, hidden_size: int = 512):
        """
        初始化网络结构
        :param input_size: 输入维度（棋盘状态空间）
        :param hidden_size: 隐藏层维度
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)  # 输出每个位置的Q值

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """经验回放缓冲区，用于存储和采样训练数据"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
            self,
            state: np.ndarray,
            action: int,
            next_state: np.ndarray,
            reward: float,
            done: bool,
    ):
        """存储转换经验"""
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.LongTensor([action]).to(device)
        reward = torch.FloatTensor([reward]).to(device)
        done = torch.FloatTensor([done]).to(device)

        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size: int):
        """随机采样批量数据"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.cat(actions),
            torch.stack(next_states),
            torch.cat(rewards),
            torch.cat(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """深度Q学习智能体，支持双网络和epsilon-greedy策略"""

    def __init__(
            self,
            env: TicTacToeEnv,
            gamma: float = 0.99,
            lr: float = 1e-4,
            target_update: int = 10,
    ):
        """
        初始化智能体
        :param env: 游戏环境
        :param gamma: 折扣因子
        :param lr: 学习率
        :param target_update: 目标网络更新间隔（单位：episode）
        """
        self.env = env
        self.input_size = env.size**2
        self.gamma = gamma
        self.target_update = target_update

        # 初始化策略网络和目标网络
        self.policy_net = DQN(self.input_size).to(device)
        self.target_net = DQN(self.input_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(100000)
        self.batch_size = 512

        # Epsilon-greedy参数
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(
            self, state: np.ndarray, training: bool = True
    ) -> tuple[int, int]:
        """
        选择动作（支持训练模式和推理模式）
        :param state: 当前棋盘状态
        :param training: 是否处于训练模式（启用探索）
        :return: 动作坐标 (row, col)
        """
        if training and random.random() < self.epsilon:
            action = random.choice(self.env.get_valid_actions())
            logger.debug(f"随机选择动作: {action}")
            return action

        state_tensor = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        valid_actions = self.env.get_valid_actions()
        valid_indices = [row * self.env.size + col for (row, col) in valid_actions]
        best_idx = torch.argmax(q_values[valid_indices]).item()
        action = valid_actions[best_idx]
        logger.debug(f"选择动作: {action}")
        return action

    def update_model(self) -> float:
        """执行一次模型更新，返回损失值"""
        if len(self.memory) < self.batch_size:
            logger.info("内存不足，跳过模型更新")
            return 0.0

        # 从内存中采样
        states, actions, next_states, rewards, dones = self.memory.sample(
            self.batch_size
        )

        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值（双DQN）
        with torch.no_grad():
            # 使用策略网络选择动作
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # 使用目标网络评估价值
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算并反向传播损失
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        logger.debug(f"模型更新完成，损失: {loss.item():.4f}, 探索率: {self.epsilon:.3f}")
        return loss.item()

    def update_target_net(self):
        """同步目标网络参数"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.debug("目标网络参数已更新")

    def save(self, path: str):
        """保存模型参数到文件"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )
        logger.info(f"模型已保存至 {path}")

    def load(self, path: str):
        """从文件加载模型参数"""
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        logger.info(f"已从 {path} 加载模型")

    def load_memory(self, path: str):
        """加载经验回放缓冲区"""
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.memory.buffer = deque(pickle.load(f), maxlen=self.memory.buffer.maxlen)
            logger.info(f"已加载经验回放缓冲区 {path}")
        else:
            logger.warning(f"经验回放文件 {path} 不存在")


class TicTacToeGUI(tk.Tk):
    """游戏图形界面，支持人机交互"""

    def __init__(self, env: TicTacToeEnv, agent: DQNAgent = None, cell_size: int = 100):
        super().__init__()
        self.env = env
        self.agent = agent
        self.cell_size = cell_size

        # 初始化窗口布局
        self.title(f"井字棋 {env.size}x{env.size}")
        self.geometry(f"{self.env.size * cell_size}x{self.env.size * cell_size + 50}")

        # 创建画布组件
        self.canvas = tk.Canvas(
            self,
            width=self.env.size * cell_size,
            height=self.env.size * cell_size,
            bg="white",
        )
        self.canvas.pack()

        # 创建状态栏和按钮
        self.status = tk.Label(self, text="玩家回合", font=("Arial", 12))
        self.status.pack()
        self.reset_btn = tk.Button(self, text="新游戏", command=self.reset)
        self.reset_btn.pack()

        # 绑定事件并初始化游戏
        self.canvas.bind("<Button-1>", self.on_click)
        self._draw_grid()
        self.reset()

    def _draw_grid(self):
        """绘制棋盘格子"""
        for i in range(1, self.env.size):
            self.canvas.create_line(
                0, i * self.cell_size, self.env.size * self.cell_size, i * self.cell_size
            )
            self.canvas.create_line(
                i * self.cell_size, 0, i * self.cell_size, self.env.size * self.cell_size
            )

    def reset(self):
        """重置游戏状态并刷新界面"""
        self.env.reset()
        self.canvas.delete("pieces")
        self.status.config(text="玩家回合")
        logger.info("游戏已重置")
        if self.agent and self.env.current_player == 2:
            self.ai_move()

    def draw_piece(self, row: int, col: int, player: int):
        """
        绘制棋子图形
        :param row: 行坐标
        :param col: 列坐标
        :param player: 玩家编号（1为X，2为O）
        """
        color = "blue" if player == 1 else "red"
        x = col * self.cell_size + self.cell_size // 2
        y = row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3

        if player == 1:
            # 绘制X符号
            self.canvas.create_line(
                x - radius, y - radius, x + radius, y + radius, width=3, tags="pieces", fill=color
            )
            self.canvas.create_line(
                x + radius, y - radius, x - radius, y + radius, width=3, tags="pieces", fill=color
            )
        else:
            # 绘制O符号
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius, outline=color, width=3, tags="pieces"
            )

    def on_click(self, event: tk.Event):
        """处理玩家点击事件"""
        if self.env.current_player != 1 or self.env._check_winner() is not None:
            logger.debug("无效点击：非玩家回合或游戏已结束")
            return

        # 转换点击坐标为棋盘索引
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if 0 <= row < self.env.size and 0 <= col < self.env.size:
            if self.env.board[row, col] == 0:
                logger.info(f"玩家落子位置: ({row}, {col})")
                self.env.step((row, col))
                self.draw_piece(row, col, 1)
                self.update_status()

                # 如果游戏未结束且存在AI代理，延迟触发AI移动
                if not self.env._check_winner() and self.agent is not None:
                    self.after(500, self.ai_move)

    def ai_move(self):
        """执行AI移动操作"""
        if self.env.current_player == 2 and self.env._check_winner() is None:
            state = self.env._get_state()
            action = self.agent.select_action(state, training=False)
            self.env.step(action)
            self.draw_piece(*action, 2)
            self.update_status()

    def update_status(self):
        """更新状态栏显示"""
        winner = self.env._check_winner()
        if winner is not None:
            text = "玩家获胜！" if winner == 1 else "AI获胜！"
            self.status.config(text=text)
        elif len(self.env.get_valid_actions()) == 0:
            self.status.config(text="平局！")
        else:
            text = "AI思考中..." if self.env.current_player == 2 else "玩家回合"
            self.status.config(text=text)


def train_agent(
        env: TicTacToeEnv,
        episodes: int = 10000,
        save_path: str = "tictactoe_model.pth",
        memory_path: str = "replay_buffer.pkl",
        ai_player: int = 1,
        opponent: str = "random",
):
    """
    训练AI智能体
    :param env: 游戏环境
    :param episodes: 训练回合数
    :param save_path: 模型保存路径
    :param memory_path: 经验回放缓冲区路径
    :param ai_player: AI执棋方（1或2）
    :param opponent: 对手类型（'random'随机策略，'self'自我对弈）
    """
    agent = DQNAgent(env)
    agent.load_memory(memory_path)  # 加载历史经验

    logger.info(f"开始训练，使用设备：{device}")
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # 根据当前执棋方选择动作
            if env.current_player == ai_player:
                action = agent.select_action(state)
            else:
                if opponent == "random":
                    action = random.choice(env.get_valid_actions())
                elif opponent == "self":
                    action = agent.select_action(state)
                else:
                    raise ValueError(f"未知对手类型: {opponent}")

            # 执行动作并存储经验
            next_state, reward, done, _ = env.step(action)

            # 仅记录AI玩家的经验（当使用自我对弈时双方经验都记录）
            if opponent == "self" or env.current_player == ai_player:
                action_idx = action[0] * env.size + action[1]
                agent.memory.push(state, action_idx, next_state, reward, done)
                total_reward += reward

            state = next_state

            # 执行模型更新
            if len(agent.memory) >= agent.batch_size:
                loss = agent.update_model()

        # 定期更新目标网络
        if episode % agent.target_update == 0:
            agent.update_target_net()

        # 定期打印训练进度
        if episode % 100 == 0:
            logger.info(
                f"回合 {episode:04d}/{episodes} | "
                f"累计奖励: {total_reward:.2f} | "
                f"探索率: {agent.epsilon:.3f} | "
                f"经验池: {len(agent.memory)}"
            )

    # 保存训练成果
    with open(memory_path, "wb") as f:
        pickle.dump(list(agent.memory.buffer), f)
    agent.save(save_path)
    logger.info(f"训练完成，模型已保存至 {save_path}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="井字棋AI训练与对弈系统")
    parser.add_argument("--size", type=int, default=3, help="棋盘尺寸（默认：3）")
    parser.add_argument("--train", action="store_true", default=True, help="启用训练模式")
    parser.add_argument("--episodes", type=int, default=10000, help="训练回合数（默认：10000）")
    parser.add_argument("--model", type=str, default="tictactoe_model.pth", help="模型文件路径")
    parser.add_argument("--memory", type=str, default="replay_buffer.pkl", help="经验回放缓冲区路径")
    parser.add_argument("--ai_player", type=int, choices=[1, 2], default=1,
                        help="AI执棋方（1=先手，2=后手，默认：1）")
    parser.add_argument("--opponent", choices=["random", "self"], default="random",
                        help="对手类型（'random'随机策略，'self'自我对弈，默认：random）")
    args = parser.parse_args()

    # 初始化环境
    env = TicTacToeEnv(args.size)

    if args.train:
        # 训练模式
        agent = train_agent(
            env,
            episodes=args.episodes,
            save_path=args.model,
            memory_path=args.memory,
            ai_player=args.ai_player,
            opponent=args.opponent,
        )
    else:
        # 对弈模式
        if os.path.exists(args.model):
            agent = DQNAgent(env)
            agent.load(args.model)
            logger.info(f"模型加载成功，使用设备：{device}")
        else:
            raise FileNotFoundError(f"模型文件 {args.model} 不存在")

    # 启动图形界面
    gui = TicTacToeGUI(env, agent)
    gui.mainloop()
