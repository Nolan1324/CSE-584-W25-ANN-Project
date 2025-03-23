import numpy as np
import numpy.typing as npt

def uniform_attributes(n: int, d: int, seed: int, dtype: type, lower: float, upper: float) -> npt.NDArray:
    rng = np.random.default_rng(seed)
    return rng.uniform(lower, upper, (n, d)).astype(dtype)

def uniform_attributes_example(n: int):
    return uniform_attributes(n, 1, 584, np.int32, 0, 1000).squeeze()