# 导入模块
import datetime
import logging


def configure_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"tictactoe_{datetime.now().strftime('%Y%m%d%H%M')}.log"),
            logging.StreamHandler()
        ]
    )
