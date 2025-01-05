import logging
import random
import numpy as np
import torch
import time

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(filename)s.%(funcName)s :: %(message)s",
)


logging_flag = True


def debug_log(message):
    """Logs a debug message with standard formatting."""
    if logging_flag:
        logging.debug(message)


def info_log(message):
    """Logs an info message with standard formatting."""
    if logging_flag:
        logging.info(message)


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return f"Set all the seeds to {seed} successfully!"


def start_timer():
    return time.time()


def end_timer(start_time):
    return time.time() - start_time


def get_elapsed_time(start_time):
    return time.time() - start_time
