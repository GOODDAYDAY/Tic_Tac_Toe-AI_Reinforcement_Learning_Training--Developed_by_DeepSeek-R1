class GameConfig:
    # 棋盘参数
    DEFAULT_BOARD_SIZE = 3
    CELL_SIZE = 100  # 新增单元格尺寸配置

    # 胜利条件
    @staticmethod
    def get_win_condition(n):
        return 5 if n >= 5 else 3

    # 颜色配置
    COLORS = {
        'player1': 'blue',
        'player2': 'red',
        'bg': 'white'
    }
