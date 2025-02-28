# 导入模块
import logging
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from game.config import GameConfig
from .dqn import DQN

logger = logging.getLogger(__name__)


class RLAgent:
    """强化学习智能体，实现DQN算法和游戏交互逻辑"""

    def __init__(self, n: int = GameConfig.DEFAULT_BOARD_SIZE, use_gpu: bool = False):
        """
        初始化强化学习智能体
        :param n: 棋盘尺寸
        :param use_gpu: 是否启用GPU加速
        """
        logger.info(f"Initializing RLAgent for {n}x{n} board")
        self.n = n
        self.input_size = n * n + 2
        self.action_size = n * n
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.batch_size = 128
        self.model_file = f"models/tictactoe_model_n{n}.pth"
        self.use_gpu = use_gpu

        self._init_model()
        self._ensure_model_directory()
        logger.debug(f"Device in use: {self.model.device}")
        logger.info(f"模型初始化到设备：{self.model.device}")
        logger.debug(f"首个参数设备：{next(self.model.parameters()).device}")

    def _init_model(self) -> None:
        """初始化神经网络模型"""
        logger.debug("Building DQN model architecture")
        self.model = DQN(
            input_size=self.input_size,
            action_size=self.action_size,
            use_gpu=self.use_gpu
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        if os.path.exists(self.model_file):
            self.load_model()
        else:
            logger.info("No pre-trained model found, initializing new model")

    def _ensure_model_directory(self) -> None:
        """确保模型保存目录存在"""
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        logger.debug(f"Model directory verified: {os.path.dirname(self.model_file)}")

    def get_state(self, game):
        """获取增强版游戏状态（增加维度验证）"""
        state = game.board.flatten().astype(np.float32)
        # 验证棋盘维度
        if len(state) != self.n * self.n:
            raise ValueError(f"棋盘维度错误，预期{self.n}x{self.n}，实际长度{len(state)}")

        # 玩家特征（当前玩家和对手）
        current_player = game.current_player
        player_feature = np.array([
            1.0 if current_player == 1 else 0.0,
            1.0 if current_player == -1 else 0.0
        ], dtype=np.float32)

        combined_state = np.concatenate([state, player_feature])

        # 最终维度验证
        if len(combined_state) != self.input_size:
            raise ValueError(f"状态维度错误，预期{self.input_size}，实际{len(combined_state)}")

        return torch.FloatTensor(combined_state).to(self.model.device)

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves):
        """选择行动"""
        if np.random.rand() <= self.epsilon:
            logger.info("random 执行")
            return random.choice(valid_moves)
        else:
            logger.info("模型 执行")
            with torch.no_grad():
                q_values = self.model(state)
                valid_actions = [i * self.n + j for (i, j) in valid_moves]
                q_valid = q_values[valid_actions]
                return valid_moves[torch.argmax(q_valid).item()]

    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return

        device = self.model.device  # 获取当前模型设备
        minibatch = random.sample(self.memory, self.batch_size)

        # 重新组织数据并确保设备一致性
        states = torch.stack([x[0].to(device) for x in minibatch])
        actions = torch.LongTensor([x[1] for x in minibatch]).to(device)
        rewards = torch.FloatTensor([x[2] for x in minibatch]).to(device)
        next_states = torch.stack([x[3].to(device) for x in minibatch])
        dones = torch.FloatTensor([x[4] for x in minibatch]).to(device)

        # 计算当前Q值（确保在模型设备）
        current_q = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1))

        # 计算目标Q值（保持设备一致性）
        next_q_values = self.model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        # 计算损失
        loss = F.mse_loss(current_q.squeeze(), target_q)

        # 反向传播优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        """保存模型（包含完整维度信息）"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'input_size': self.input_size,  # 新增维度保存
            'action_size': self.action_size  # 新增动作空间保存
        }, self.model_file)

    def load_model(self):
        """加载模型（增强维度验证）"""
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file)
            # 双重维度验证
            if checkpoint.get('input_size') != self.input_size or \
                    checkpoint.get('action_size') != self.action_size:
                logger.warning(
                    f"模型维度不匹配（当前输入:{self.input_size} 动作:{self.action_size}，文件输入:{checkpoint.get('input_size')} 动作:{checkpoint.get('action_size')}）")
                return
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            logger.info(f"从 {self.model_file} 加载模型成功")
