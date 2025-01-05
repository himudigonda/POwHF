import logging
import random
import numpy as np
import torch

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(filename)s.%(funcName)s :: %(message)s",
)


def debug_log(message):
    """Logs a debug message with standard formatting."""
    logging.debug(message)


def info_log(message):
    """Logs an info message with standard formatting."""
    logging.info(message)


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return f"Set all the seeds to {seed} successfully!"
