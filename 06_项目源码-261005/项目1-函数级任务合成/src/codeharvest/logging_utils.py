"""日志工具：提供带颜色控制台输出与文件输出的 logger。"""

import os
import logging
from rich.logging import RichHandler

def init_logger(name="codeharvest", level=logging.WARNING, log_file=None, console=True):
    
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    logger.propagate = False

    if console:
        
        rich_handler = RichHandler(rich_tracebacks=True)
        rich_handler.setLevel(level)
        logger.addHandler(rich_handler)

    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

if not os.path.exists("logs"):
    os.makedirs("logs")

logger = init_logger(name="codeharvest", log_file="logs/codeharvest.log")
slicer_logger = init_logger(name="slicer", log_file="logs/slicer.log")
exec_logger = init_logger(name="exec", log_file="logs/exec.log", console=False)

