import numpy as np


# https://gist.github.com/danoneata/49a807f47656fedbb389
def ivecs_read(filename):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    fv = fv.copy()
    return fv


def fvecs_read(filename):
    return ivecs_read(filename).view(np.float32)
