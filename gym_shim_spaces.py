"""Loaded as `gymnasium/spaces.py`."""
import numpy as np


class Discrete:
    def __init__(self, n, seed=None, start=0):
        self.n = int(n)
        self.start = int(start)


class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
        self.shape = tuple(shape) if shape is not None else np.shape(low)
        self.dtype = np.dtype(dtype)
        self.low = np.full(self.shape, low, dtype=self.dtype)
        self.high = np.full(self.shape, high, dtype=self.dtype)
