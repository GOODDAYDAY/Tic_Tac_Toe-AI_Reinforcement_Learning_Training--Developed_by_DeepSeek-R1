"""
AI 训练模块，包含训练循环和对抗逻辑
"""

import logging
import random

import torch

from ai.agent import RLAgent
from game.core import GameLogic

logger = logging.getLogger(__name__)


class AITrainer:
    """强化学习训练器，管理训练流程和双AI对抗"""

    def __init__(self, agent: RLAgent, n: int = 3):
        """
        初始化训练器
        :param agent: 需要训练的RL智能体
        :param n: 棋盘尺寸
        """
        self.agent = agent
        self.n = n
        logger.info(f"Initialized AI trainer for {n}x{n} board")

    def train(self, episodes: int = 1000) -> dict:
        """
        执行训练循环
        :param episodes: 训练轮次
        :return: 训练统计结果
        """
        stats = {'wins': 0, 'losses': 0, 'draws': 0}

        for episode in range(episodes):
            game = GameLogic(self.n)
            ai_player = random.choice([1, -1])
            game.current_player = ai_player

            while not game.game_over:
                self._train_step(game, ai_player)

            self._update_stats(game, ai_player, stats)
            self._log_progress(episode, stats)

        return stats

    def _train_step(self, game: GameLogic, ai_player: int) -> None:
        """单步训练流程"""
        state = self.agent.get_state(game)
        valid_moves = game.get_valid_moves()

        if not valid_moves:
            return

        # 保存执行动作前的状态
        prev_state = state.clone() if isinstance(state, torch.Tensor) else state

        # 选择并执行动作
        row, col = self.agent.act(state, valid_moves, training=True)
        move_success = game.make_move(row, col)

        if not move_success:
            return

        next_state = self.agent.get_state(game)
        done = game.game_over

        # 计算奖励
        reward = self._calculate_reward(game, ai_player, done)

        # 存储经验并回放
        action_index = row * self.n + col
        self.agent.remember(prev_state, action_index, reward, next_state, done)
        self.agent.replay()

    def _calculate_reward(self, game: GameLogic, ai_player: int, done: bool) -> float:
        """根据游戏状态计算奖励"""
        if not done:
            return 0.0

        if game.winner == ai_player:
            return 1.0
        if game.winner is not None:
            return -1.0
        return 0.1

    def _update_stats(self, game: GameLogic, ai_player: int, stats: dict) -> None:
        """更新训练统计信息"""
        if game.winner == ai_player:
            stats['wins'] += 1
        elif game.winner is not None:
            stats['losses'] += 1
        else:
            stats['draws'] += 1

    def _log_progress(self, episode: int, stats: dict) -> None:
        """记录训练进度"""
        if (episode + 1) % 1000 == 0:
            total = sum(stats.values())
            logger.info(
                f"Episode {episode + 1} | "
                f"Win Rate: {stats['wins'] / total:.2%} | "
                f"Loss Rate: {stats['losses'] / total:.2%} | "
                f"Draw Rate: {stats['draws'] / total:.2%}"
            )
            stats.update({'wins': 0, 'losses': 0, 'draws': 0})
