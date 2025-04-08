import numpy as np
import numpy.typing as npt

def uniform_attributes(n: int, seed: int, dtype: type, lower: float, upper: float) -> npt.NDArray:
    rng = np.random.default_rng(seed)
    return rng.uniform(lower, upper, n).astype(dtype)

def geometric_attributes(n: int, seed: int, dtype: type, p: float) -> npt.NDArray:
    rng = np.random.default_rng(seed)
    return rng.geometric(p, n).astype(dtype)

def normal_attributes(n: int, seed: int, dtype: type, mean: float, std: float) -> npt.NDArray:
    rng = np.random.default_rng(seed)
    return (mean + std * rng.standard_normal(n)).astype(dtype)

def uniform_attributes_basic(n: int):
    return uniform_attributes(n, 584, np.int32, 0, 1000).squeeze()

def geometric_attributes_basic(n: int, p: float):
    return geometric_attributes(n, 584, np.int32, p).squeeze()

def normal_attributes_basic(n: int, mean: float, std: float):
    return normal_attributes(n, 584, np.int32, mean, std).squeeze()

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # attrs = geometric_attributes_basic(10000, 0.005)
    attrs = normal_attributes_basic(10000, 500, 150)
    plt.hist(attrs, bins=np.arange(0, 1000, 10))
    plt.show()
