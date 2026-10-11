"""
Minimal gymnasium replacement used in the browser build.

Loaded as `gymnasium/__init__.py`. Seeding matches gymnasium 1.x
(Env.reset -> SeedSequence -> PCG64).
"""
import numpy as np
from . import spaces  # noqa: F401

__version__ = "shim"


def _np_random(seed=None):
    seed_seq = np.random.SeedSequence(seed)
    return np.random.Generator(np.random.PCG64(seed_seq)), seed_seq.entropy


class Env:
    metadata = {"render_modes": []}
    render_mode = None
    spec = None
    _np_random = None
    _np_random_seed = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, self._np_random_seed = _np_random(seed)

    @property
    def np_random(self):
        if self._np_random is None:
            self._np_random, self._np_random_seed = _np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value):
        self._np_random = value

    @property
    def unwrapped(self):
        return self

    def close(self):
        pass
