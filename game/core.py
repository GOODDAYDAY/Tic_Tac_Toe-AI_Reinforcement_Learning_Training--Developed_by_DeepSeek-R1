# game/core.py
"""
游戏核心逻辑模块
实现棋盘状态管理和胜负判断
"""

import logging
from typing import Optional, Tuple, List

import numpy as np

from game.config import GameConfig

logger = logging.getLogger(__name__)


class GameLogic:
    """游戏逻辑核心类，管理棋盘状态和游戏规则"""

    def __init__(self, n: int = 3):
        """
        初始化游戏逻辑
        :param n: 棋盘尺寸
        """
        logger.debug(f"Creating game logic for {n}x{n} board")
        self.n = n
        self.win_condition = GameConfig.get_win_condition(n)
        self.reset()
        # 新增回合状态属性
        self.is_human_turn = True
        logger.debug(f"Win condition set to {self.win_condition} in a row")

    def reset(self) -> None:
        """重置游戏到初始状态"""
        logger.debug("Resetting game state")
        self.board = np.zeros((self.n, self.n), dtype=int)
        self.current_player = 1
        self.winner: Optional[int] = None
        self.game_over = False
        self.is_human_turn = True

    def make_move(self, row: int, col: int) -> bool:
        """执行移动前添加额外验证"""
        if self.game_over:
            logger.warning("Game already ended")
            return False
        return self._original_make_move(row, col)

    def _original_make_move(self, row: int, col: int) -> bool:
        """
        执行移动操作，包含详细的合法性检查
        :return: 是否成功执行移动
        """
        if not (0 <= row < self.n and 0 <= col < self.n):
            logger.warning(f"Invalid move position: ({row}, {col})")
            return False

        if self.board[row][col] != 0:
            logger.debug(f"Position occupied: ({row}, {col})")
            return False

        self.board[row][col] = self.current_player

        if self._check_win(row, col):
            self._handle_win()
        elif self._check_draw():
            self._handle_draw()
        else:
            self._switch_player()

        return True

    def _check_win(self, row: int, col: int) -> bool:
        """检查当前移动是否导致胜利（私有方法）"""
        logger.debug(f"Checking win condition for ({row}, {col})")
        player = self.board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dx, dy in directions:
            count = 1
            for d in [1, -1]:
                step = 1
                while True:
                    x = row + dx * d * step
                    y = col + dy * d * step
                    if 0 <= x < self.n and 0 <= y < self.n:
                        if self.board[x][y] == player:
                            count += 1
                            step += 1
                        else:
                            break
                    else:
                        break
            if count >= self.win_condition:
                logger.debug(f"Player {player} wins with {count} in a row")
                return True
        return False

    def _check_draw(self) -> bool:
        """检查是否平局（私有方法）"""
        if np.all(self.board != 0):
            logger.debug("Game ended in a draw")
            return True
        return False

    def _handle_win(self) -> None:
        """处理胜利场景（私有方法）"""
        self.winner = self.current_player
        self.game_over = True

    def _handle_draw(self) -> None:
        """处理平局场景（私有方法）"""
        self.winner = None
        self.game_over = True
        logger.debug("Game ended in a draw")

    def _switch_player(self) -> None:
        """切换当前玩家（私有方法）"""
        self.current_player = -self.current_player
        logger.debug(f"Switched to player {self.current_player}")

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """获取所有合法移动位置"""
        return [(i, j) for i in range(self.n) for j in range(self.n) if self.board[i][j] == 0]
