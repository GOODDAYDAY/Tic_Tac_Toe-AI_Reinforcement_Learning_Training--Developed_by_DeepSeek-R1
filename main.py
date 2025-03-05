import logging
import sys

from gui.interface import GameGUI
from utils.logger import configure_logger
from utils.timer import timer


@timer
def main():
    """应用程序主入口"""
    # 初始化日志系统
    configure_logger()
    logger = logging.getLogger(__name__)
    logger.info("应用程序启动")

    # 配置参数
    BOARD_SIZE = 3  # 可修改棋盘尺寸
    USE_GPU = True

    try:
        logger.debug("初始化游戏界面")
        game_gui = GameGUI(n=BOARD_SIZE)
        logger.info("开始主事件循环")
        game_gui.run()
    except Exception as e:
        logger.critical("致命错误导致程序终止", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("应用程序正常退出")


if __name__ == "__main__":
    main()
