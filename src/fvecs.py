"""
This module provides functions to read `.ivecs`, `.fvecs`, and `.bvecs` file formats 
commonly used for storing vectors in machine learning and data science applications.

Taken from
 - https://gist.github.com/danoneata/49a807f47656fedbb389
 - https://gist.github.com/CharlesLiu7/0496773a8a21934a746e3fd496f11e0c
"""

import os

import numpy as np
import numpy.typing as npt


__all__ = ["ivecs_read", "fvecs_read", "bvecs_read"]


def ivecs_read(path: os.PathLike) -> npt.NDArray[np.int32]:
    fv = np.fromfile(path, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + path)
    fv = fv[:, 1:]
    fv = fv.copy()
    return fv


def fvecs_read(path: os.PathLike) -> npt.NDArray[np.float32]:
    return ivecs_read(path).view(np.float32)


def bvecs_read(path: os.PathLike, n: int = None) -> npt.NDArray[np.uint8]:
    d = np.fromfile(path, dtype=np.int32, count=1)[0]
    if n is None:
        x = np.fromfile(path, dtype=np.uint8)
    else:
        x = np.fromfile(path, dtype=np.uint8, count=n * (d + 4))
    return x.reshape(-1, d + 4)[:, 4:].copy()
