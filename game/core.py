# 导入模块
import numpy as np


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
        """检查是否获胜"""
        player = self.board[row][col]

        # 根据棋盘尺寸决定胜利条件
        required = 3 if self.n < 5 else 5
        if required > self.n:
            required = self.n

        # 检查行列和对角线
        def check_line(dx, dy):
            count = 1
            for d in [-1, 1]:
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
            return count >= required

        # 检查水平、垂直、两个对角线
        return (check_line(0, 1) or  # 水平
                check_line(1, 0) or  # 垂直
                check_line(1, 1) or  # 主对角线
                check_line(1, -1))  # 副对角线
