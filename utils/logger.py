# 导入模块
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler


def configure_logger():
    """配置全局日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 确保日志目录存在
    log_dir = "logs"
    try:
        os.makedirs(log_dir, exist_ok=True)
    except PermissionError as e:
        print(f"无法创建日志目录: {str(e)}")
        return

    # 移除现有处理器避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 文件处理器（按天滚动）
    file_handler = RotatingFileHandler(
        filename=os.path.join(log_dir, f"tictactoe_{datetime.now().strftime('%Y%m%d')}.log"),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s][%(levelname)s] %(module)s.%(funcName)s: %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 捕获未处理异常
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("未捕获异常", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
