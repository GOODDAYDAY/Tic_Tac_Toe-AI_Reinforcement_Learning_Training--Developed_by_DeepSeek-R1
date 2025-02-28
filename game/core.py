# 导入模块
import numpy as np

from game.config import GameConfig


class GameLogic:
    """游戏逻辑核心类"""

    def __init__(self, n=3):
        self.n = n
        self.reset()

    def reset(self):
        """重置游戏状态"""
        self.board = np.zeros((self.n, self.n), dtype=int)
        self.current_player = 1
        self.winner = None
        self.game_over = False

    def get_valid_moves(self):
        """获取所有合法移动位置"""
        return [(i, j) for i in range(self.n) for j in range(self.n) if self.board[i][j] == 0]

    def make_move(self, row, col):
        """执行移动操作"""
        if self.board[row][col] == 0 and not self.game_over:
            self.board[row][col] = self.current_player
            if self.check_win(row, col):
                self.winner = self.current_player
                self.game_over = True
            elif len(self.get_valid_moves()) == 0:
                self.game_over = True
            else:
                self.current_player = -self.current_player
            return True
        return False

    def check_win(self, row, col):
        """改进的胜利条件校验"""
        player = self.board[row][col]
        required = GameConfig.get_win_condition(self.n)

        # 动态步长检查
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # 双向检查
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
            if count >= required:
                return True
        return False
