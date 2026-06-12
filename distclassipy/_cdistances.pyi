import numpy as np
from numpy.typing import ArrayLike

CYTHON_METRICS: frozenset[str]

def cdist(XA: ArrayLike, XB: ArrayLike, metric: str) -> np.ndarray: ...
