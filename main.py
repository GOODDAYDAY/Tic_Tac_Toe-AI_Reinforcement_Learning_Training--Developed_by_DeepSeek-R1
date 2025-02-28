import sys

from gui.interface import GameGUI

if __name__ == "__main__":
    BOARD_SIZE = 3  # 可修改棋盘尺寸
    USE_GPU = True

    try:
        game_gui = GameGUI(n=BOARD_SIZE)
        game_gui.run()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)
