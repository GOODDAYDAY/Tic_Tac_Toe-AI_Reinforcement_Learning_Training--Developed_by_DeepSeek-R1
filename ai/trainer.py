"""
AI 训练模块，包含训练循环和对抗逻辑
"""
import copy
import logging
import random

from ai.agent import RLAgent
from game.core import GameLogic
from utils.timer import timer

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

    @timer
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

    @timer
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

    @timer
    def _calculate_instant_reward(self, prev_game: GameLogic, ai_player: int, row: int, col: int) -> float:
        """优化后的即时奖励计算，避免深拷贝并优化检测逻辑"""
        # 生成下子后的棋盘副本（仅复制棋盘数据）
        board = [row.copy() for row in prev_game.board]
        board[row][col] = ai_player  # 应用当前落子
        n = prev_game.n
        win_condition = prev_game.win_condition
        opponent = -ai_player
        block_reward = 0.0
        winning_reward = 0.0
        threat_reward = 0.0

        # 1. 堵住敌方胜利的奖励（使用原游戏状态检测）
        opponent_winning_moves = []
        for r in range(prev_game.n):
            for c in range(prev_game.n):
                if prev_game.board[r][c] == 0 and self._is_winning_move(prev_game, r, c, opponent):
                    opponent_winning_moves.append((r, c))
        if (row, col) in opponent_winning_moves:
            block_reward += 10.0

        # 2. 下一步必胜奖励（使用新棋盘状态检测）
        for r in range(n):
            for c in range(n):
                if board[r][c] == 0 and self._is_winning_move_on_board(board, n, win_condition, r, c, ai_player):
                    winning_reward = 100.0
                    break
            if winning_reward > 0:
                break

        # 3. 基于连续性的威胁检测（优化后的算法）
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            # 检测己方威胁
            ai_continuous = self._count_continuous(board, n, row, col, dx, dy, ai_player)
            if ai_continuous >= win_condition - 1:
                threat_reward += 0.5

            # 检测敌方威胁（遍历棋盘检测敌方潜在胜利路径）
            if win_condition > 3:  # 仅在需要时检测（如五子棋模式）
                for r in range(n):
                    for c in range(n):
                        if board[r][c] == opponent and self._count_continuous(board, n, r, c, dx, dy,
                                                                              opponent) >= win_condition - 1:
                            threat_reward -= 0.7

        return block_reward + winning_reward + threat_reward * 0.5

    @timer
    def _is_winning_move(self, game: GameLogic, row: int, col: int, player: int) -> bool:
        """检查指定位置落子是否直接导致胜利"""
        board = game.board
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # 正向延伸
            x, y = row + dx, col + dy
            while 0 <= x < game.n and 0 <= y < game.n and board[x][y] == player:
                count += 1
                x += dx
                y += dy
            # 反向延伸
            x, y = row - dx, col - dy
            while 0 <= x < game.n and 0 <= y < game.n and board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
            if count >= game.win_condition:
                return True
        return False

    @timer
    def _is_winning_move_on_board(self, board: list, n: int, win_condition: int, row: int, col: int,
                                  player: int) -> bool:
        """在指定棋盘状态检查落子是否胜利"""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            x, y = row + dx, col + dy
            while 0 <= x < n and 0 <= y < n and board[x][y] == player:
                count += 1
                x += dx
                y += dy
            x, y = row - dx, col - dy
            while 0 <= x < n and 0 <= y < n and board[x][y] == player:
                count += 1
                x -= dx
                y -= dy
            if count >= win_condition:
                return True
        return False

    @timer
    def _count_continuous(self, board: list, n: int, row: int, col: int, dx: int, dy: int, player: int) -> int:
        """计算指定方向连续同色棋子数量"""
        count = 0
        # 正向延伸
        x, y = row, col
        while 0 <= x < n and 0 <= y < n and board[x][y] == player:
            count += 1
            x += dx
            y += dy
        # 反向延伸
        x, y = row - dx, col - dy
        while 0 <= x < n and 0 <= y < n and board[x][y] == player:
            count += 1
            x -= dx
            y -= dy
        return count

    @timer
    def _calculate_final_reward(self, game: GameLogic, ai_player: int) -> float:
        """根据最终结果计算奖励"""
        if game.winner == ai_player:
            return 1.0
        if game.winner is not None:
            return -1.0
        return 0.1

    @timer
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

    @timer
    def _calculate_reward(self, game: GameLogic, ai_player: int, done: bool) -> float:
        """根据游戏状态计算奖励"""
        if not done:
            return 0.0

        if game.winner == ai_player:
            return 1.0
        if game.winner is not None:
            return -1.0
        return 0.1

    @timer
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
