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
        self.input_size = n * n
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.model_file = f"models/tictactoe_model_n{n}.pth"
        self.use_gpu = use_gpu

        self._init_model()
        self._ensure_model_directory()

        logger.debug(f"Device in use: {self.model.device}")

    def _init_model(self) -> None:
        """初始化神经网络模型"""
        logger.debug("Building DQN model architecture")
        self.model = DQN(self.input_size, use_gpu=self.use_gpu)
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
        """获取游戏状态的扁平化表示"""
        return torch.FloatTensor(game.board.flatten()).to(self.model.device)

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_moves):
        """选择行动"""
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        else:
            with torch.no_grad():
                state_tensor = state.view(-1)
                q_values = self.model(state_tensor)
                valid_actions = [i * self.n + j for (i, j) in valid_moves]
                q_valid = q_values[valid_actions]
                return valid_moves[torch.argmax(q_valid).item()]

    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.stack([x[0] for x in minibatch])
        actions = torch.LongTensor([x[1] for x in minibatch]).view(-1, 1)
        rewards = torch.FloatTensor([x[2] for x in minibatch])
        next_states = torch.stack([x[3] for x in minibatch])
        dones = torch.FloatTensor([x[4] for x in minibatch])

        current_q = self.model(states).gather(1, actions)
        next_q = self.model(next_states).max(1)[0].detach()
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, self.model_file)
        logging.info(f"Model saved to {self.model_file}")

    def load_model(self):
        """加载模型（增加维度检查）"""
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file)
            if checkpoint.get('input_size') != self.input_size:
                logging.warning("Model dimension mismatch, skipping load")
                return
        checkpoint = torch.load(self.model_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        logging.info(f"Loaded model from {self.model_file}")
