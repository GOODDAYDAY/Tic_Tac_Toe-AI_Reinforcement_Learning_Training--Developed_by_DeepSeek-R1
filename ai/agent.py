"""
强化学习智能体模块，包含DQN算法实现
"""

import logging
import os
import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from game.config import GameConfig
from utils.timer import timer
from .dqn import DQN

logger = logging.getLogger(__name__)


class RLAgent:
    """强化学习智能体，实现DQN算法和游戏交互逻辑"""

    def __init__(self,
                 n: int = GameConfig.DEFAULT_BOARD_SIZE,
                 use_gpu: bool = True,
                 training_mode: bool = False):
        """
        初始化强化学习智能体
        :param n: 棋盘尺寸
        :param use_gpu: 是否启用GPU加速
        :param training_mode: 是否处于训练模式
        """
        self.n = n
        self.input_size = n * n + 2  # 棋盘状态 + 当前玩家特征
        self.action_size = n * n
        self.use_gpu = use_gpu
        self.training_mode = training_mode

        # 初始化强化学习参数
        self._init_hyperparameters()
        self._init_model()
        self._ensure_model_directory()

        logger.info(f"Initialized RLAgent in {'training' if training_mode else 'inference'} mode")

    @timer
    def _init_hyperparameters(self) -> None:
        """初始化超参数"""
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.1  # 最小探索率
        self.epsilon_decay = 0.997  # 探索率衰减
        self.batch_size = 128  # 批量大小

        # 模型文件路径
        self.model_file = f"models/tictactoe_n{self.n}_v3.pth"

    @timer
    def _init_model(self) -> None:
        """初始化神经网络模型"""
        logger.debug("Building DQN model architecture")
        self.model = DQN(
            input_size=self.input_size,
            action_size=self.action_size,
            use_gpu=self.use_gpu
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 加载预训练模型（如果存在）
        if os.path.exists(self.model_file):
            self.load_model()
        else:
            logger.info("No pre-trained model found, initializing new model")

    @timer
    def _ensure_model_directory(self) -> None:
        """确保模型保存目录存在"""
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        logger.debug(f"Model directory verified: {os.path.dirname(self.model_file)}")

    @timer
    def _generate_symmetries(self, board: np.ndarray) -> list:
        """生成所有对称变换后的棋盘状态"""
        symmetries = []
        for k in range(4):
            # 原始旋转
            rotated = np.rot90(board, k)
            symmetries.append(rotated)
            # 镜像变换
            symmetries.append(np.fliplr(rotated))
        return symmetries

    @timer
    def _normalize_state(self, state: np.ndarray) -> list:
        """标准化状态处理（返回所有对称状态）"""
        board = state[:-2].reshape(self.n, self.n)
        player_feature = state[-2:]

        normalized_states = []
        for transformed in self._generate_symmetries(board):
            # 保持玩家特征不变
            normalized = np.concatenate([transformed.flatten(), player_feature])
            normalized_states.append(normalized)

            # 生成玩家反转的对称状态（正反方对称）
            reversed_player = np.array([player_feature[1], player_feature[0]])
            reversed_state = np.concatenate([transformed.flatten(), reversed_player])
            normalized_states.append(reversed_state)

        return normalized_states

    @timer
    def get_state(self, game) -> list:
        """获取所有对称状态"""
        raw_state = game.board.flatten().astype(np.float32)
        current_player = game.current_player
        player_feature = np.array([1.0 if current_player == 1 else 0.0,
                                   1.0 if current_player == -1 else 0.0])
        combined = np.concatenate([raw_state, player_feature])
        return [torch.FloatTensor(s).to(self.model.device) for s in self._normalize_state(combined)]

    @timer
    def act(self,
            states: list,  # 修改为接收多个状态
            valid_moves: List[Tuple[int, int]],
            training: bool = False) -> Tuple[int, int]:
        """基于所有对称状态选择最佳动作"""
        if training and np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        with torch.no_grad():
            # 综合所有对称状态的Q值
            q_values = torch.zeros(self.action_size, device=self.model.device)
            for state in states:
                q_values += self.model(state)

            valid_actions = [i * self.n + j for (i, j) in valid_moves]
            return valid_moves[torch.argmax(q_values[valid_actions]).item()]

    @timer
    def remember(self,
                 states: list,  # 修改为接收多个状态
                 action: int,
                 reward: float,
                 next_states: list,
                 done: bool) -> None:
        """存储所有对称状态的经验"""
        for s, ns in zip(states, next_states):
            self.memory.append((s, action, reward, ns, done))

    @timer
    def replay(self) -> None:
        """执行经验回放训练"""
        if len(self.memory) < self.batch_size:
            return

        # 从记忆库中采样
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 转换为张量
        states = torch.stack(states).to(self.model.device)
        actions = torch.LongTensor(actions).to(self.model.device)
        rewards = torch.FloatTensor(rewards).to(self.model.device)
        next_states = torch.stack(next_states).to(self.model.device)
        dones = torch.FloatTensor(dones).to(self.model.device)

        # 计算当前Q值和目标Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # 计算损失
        loss = F.mse_loss(current_q, target_q)

        # 反向传播优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        self.memory = deque(maxlen=10000)

    @timer
    def save_model(self) -> None:
        """保存模型到文件"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'input_size': self.input_size,
            'action_size': self.action_size
        }, self.model_file)
        logger.info(f"Model saved to {self.model_file}")

    @timer
    def load_model(self) -> None:
        """从文件加载模型"""
        try:
            checkpoint = torch.load(self.model_file)
            # 维度验证
            if (checkpoint.get('input_size') != self.input_size or
                    checkpoint.get('action_size') != self.action_size):
                raise ValueError("Model architecture mismatch")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            logger.info(f"Loaded model from {self.model_file} Epsilon: {self.epsilon}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
