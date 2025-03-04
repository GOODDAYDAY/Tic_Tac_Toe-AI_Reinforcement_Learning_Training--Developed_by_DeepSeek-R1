"""
AI 训练模块，包含训练循环和对抗逻辑
"""
import copy
import logging
import random

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
            episode_experiences = []  # 存储本回合所有经验

            while not game.game_over:
                # 保存初始状态快照
                prev_state = self.agent.get_state(game)
                prev_game = copy.deepcopy(game)

                # 执行训练步骤
                row, col, reward, done = self._train_step(game, ai_player)

                # 存储经验（延迟计算奖励）
                action_index = row * self.n + col
                next_state = self.agent.get_state(game)
                episode_experiences.append(
                    (prev_state, action_index, 0, next_state, done)  # 初始奖励设为0
                )

                # 使用游戏副本回退到上一步来计算即时奖励
                instant_reward = self._calculate_instant_reward(prev_game, ai_player, row, col)
                episode_experiences[-1] = (prev_state, action_index, instant_reward, next_state, done)

                # 添加最终结果奖励
            final_reward = self._calculate_final_reward(game, ai_player)
            self._adjust_episode_rewards(episode_experiences, final_reward)

            # 存入经验池并回放
            for exp in episode_experiences:
                self.agent.remember(*exp)
            self.agent.replay()

            self._update_stats(game, ai_player, stats)
            self._log_progress(episode, stats)

        return stats

    def _train_step(self, game: GameLogic, ai_player: int) -> tuple:
        """单步训练流程，返回执行的动作和结果"""
        state = self.agent.get_state(game)
        valid_moves = game.get_valid_moves()

        if not valid_moves:
            return (0, 0, 0, True)

            # 选择并执行动作
        row, col = self.agent.act(state, valid_moves, training=True)
        move_success = game.make_move(row, col)

        done = game.game_over
        return (row, col, 0, done)

    def _calculate_instant_reward(self, game: GameLogic, ai_player: int, row: int, col: int) -> float:
        """计算即时奖励，包含潜在胜利路径奖励"""
        temp_game = copy.deepcopy(game)
        temp_game.make_move(row, col)

        # 最终结果奖励
        if temp_game.winner == ai_player:
            return 1.0
        elif temp_game.winner is not None:
            return -10.0

        # 1. 堵住敌方胜利的奖励（+10）
        block_reward = 0.0
        original_game = copy.deepcopy(game)
        opponent = -ai_player
        opponent_winning_moves = []

        # 检查原游戏中对手可获胜的位置
        for r in range(original_game.n):
            for c in range(original_game.n):
                if original_game.board[r][c] == 0:
                    temp_opponent = copy.deepcopy(original_game)
                    temp_opponent.current_player = opponent
                    if temp_opponent.make_move(r, c) and temp_opponent.winner == opponent:
                        opponent_winning_moves.append((r, c))

        if (row, col) in opponent_winning_moves:
            block_reward += 10.0

        # 2. 下一步必胜奖励（+100）
        winning_reward = 0.0
        for r in range(temp_game.n):
            for c in range(temp_game.n):
                if temp_game.board[r][c] == 0:
                    temp_win_check = copy.deepcopy(temp_game)
                    temp_win_check.current_player = ai_player  # 强制设置当前玩家
                    if temp_win_check.make_move(r, c) and temp_win_check.winner == ai_player:
                        winning_reward = 100.0
                        break
            if winning_reward > 0:
                break

        # 3. 优化后的威胁检测（基于棋局实际获胜条件）
        threat_reward = 0.0
        win_condition = temp_game.win_condition
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dx, dy in directions:
            # 生成完整方向线
            line = []
            # 正向延伸（包含当前点）
            step = 0
            while True:
                x, y = row + dx * step, col + dy * step
                if 0 <= x < self.n and 0 <= y < self.n:
                    line.append(temp_game.board[x][y])
                    step += 1
                else:
                    break
            # 反向延伸（不含当前点）
            step = -1
            while True:
                x, y = row + dx * step, col + dy * step
                if 0 <= x < self.n and 0 <= y < self.n:
                    line.insert(0, temp_game.board[x][y])
                    step -= 1
                else:
                    break

            # 滑动窗口检测威胁
            for i in range(len(line) - win_condition + 1):
                window = line[i:i + win_condition]
                ai_count = sum(1 for p in window if p == ai_player)
                opp_count = sum(1 for p in window if p == -ai_player)
                empty = window.count(0)

                # 检测己方威胁（差一棋胜利）
                if ai_count == win_condition - 1 and empty == 1:
                    threat_reward += 0.5  # 原0.2 -> 调整为0.5 * 0.5 = 0.25
                # 检测敌方威胁（差一棋失败）
                if opp_count == win_condition - 1 and empty == 1:
                    threat_reward -= 0.7  # 原0.3 -> 调整为0.7 * 0.5 = 0.35

        # 合并所有奖励（威胁奖励适当缩减）
        return (
                block_reward +
                winning_reward +
                threat_reward * 0.5  # 缩减系数
        )

    def _calculate_final_reward(self, game: GameLogic, ai_player: int) -> float:
        """根据最终结果计算奖励"""
        if game.winner == ai_player:
            return 1.0
        if game.winner is not None:
            return -1.0
        return 0.1

    def _adjust_episode_rewards(self, experiences: list, final_reward: float) -> None:
        """调整最终奖励并合成总奖励"""
        total_steps = len(experiences)
        discount = 1.0
        gamma = 0.5

        # 逆向传播最终奖励
        for i in reversed(range(total_steps)):
            state, action, reward, next_state, done = experiences[i]
            new_reward = reward + discount * final_reward
            discount *= gamma
            experiences[i] = (state, action, new_reward, next_state, done)

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
