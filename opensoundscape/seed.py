""" seed.py: Set random state across different libraries for reproducibility
"""
import numpy as np
import torch
import random


def seed(seed, verbose=True):
    print(f"Random state set with seed {seed}")

    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
