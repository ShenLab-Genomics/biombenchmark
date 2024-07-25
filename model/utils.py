import logging
import numpy as np


def make_logger(name='training_logger', log_file='training.log', level="INFO"):
    # 创建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if level == "INFO" else logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO if level == "INFO" else logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if level == "INFO" else logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
