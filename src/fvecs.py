"""
This module provides functions to read `.ivecs`, `.fvecs`, and `.bvecs` file formats 
commonly used for storing vectors in machine learning and data science applications.

Taken from
 - https://gist.github.com/danoneata/49a807f47656fedbb389
 - https://gist.github.com/CharlesLiu7/0496773a8a21934a746e3fd496f11e0c
"""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt


__all__ = ["ivecs_read", "fvecs_read", "bvecs_read", "vecs_read"]


def ivecs_read(path: os.PathLike) -> npt.NDArray[np.int32]:
    """
    Reads an .ivecs file and returns its contents as a NumPy array.
    
    Parameters:
        path: The file path to the .ivecs file.
        
    Returns:
        A 2D NumPy array where each row corresponds to a vector from the file. 
        If the file is empty, returns an empty 2D array with shape (0, 0).
    """
    
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
    """
    Reads an .fvecs file and returns its contents as a NumPy array.
    
    Parameters:
        path: The file path to the .fvecs file.
        
    Returns:
        A 2D NumPy array where each row corresponds to a vector from the file. 
        If the file is empty, returns an empty 2D array with shape (0, 0).
    """
    
    return ivecs_read(path).view(np.float32)


def bvecs_read(path: os.PathLike, n: int = None) -> npt.NDArray[np.uint8]:
    """
    Reads an .bvecs file and returns its contents as a NumPy array.
    
    Parameters:
        path: The file path to the .bvecs file.
        
    Returns:
        A 2D NumPy array where each row corresponds to a vector from the file. 
        If the file is empty, returns an empty 2D array with shape (0, 0).
    """
    
    d = np.fromfile(path, dtype=np.int32, count=1)[0]
    if n is None:
        x = np.fromfile(path, dtype=np.uint8)
    else:
        x = np.fromfile(path, dtype=np.uint8, count=n * (d + 4))
    return x.reshape(-1, d + 4)[:, 4:].copy()


def vecs_read(path: os.PathLike, n: int = None) -> npt.NDArray[np.float32] | npt.NDArray[np.int32] | npt.NDArray[np.uint8]:
    """Reads a vector file and returns its contents as a NumPy array.
    
    The type is automatically determined from the file path.
    
    Args:
        path: The path to the vector file. The file extension must be one of `.fvecs`, `.ivecs`, or `.bvecs`.
        n: The number of vectors to read, applicable only for `.bvecs` files.

    Returns:
        A NumPy array containing the vectors from the file. The data type of the array
        depends on the file format:
        - `.fvecs`: np.float32
        - `.ivecs`: np.int32
        - `.bvecs`: np.uint8
    """
    
    
    extension = Path(path).suffix
    if extension == ".fvecs":
        return fvecs_read(path)
    elif extension == ".ivecs":
        return ivecs_read(path)
    elif extension == ".bvecs":
        return bvecs_read(path, n=n)
    else:
        raise ValueError(f"Unsupported file format: {extension}")
