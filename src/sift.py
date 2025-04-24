from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pymilvus import DataType

from fvecs import vecs_read


@dataclass
class Dataset:
    """
    A class representing a dataset for vector-based operations, including base vectors,
    query vectors, and ground truth data. This class also provides metadata about the
    dataset, such as the dimensionality of the vectors and the number of vectors.

    Attributes:
        base: The base dataset containing vectors. Default is None.
        ground_truth: The ground truth data for the dataset. Default is None.
        query: The query dataset containing vectors. Default is None.
        datatype: The datatype of the vectors, either FLOAT_VECTOR or FLOAT16_VECTOR.
                  Default is DataType.FLOAT_VECTOR.
        d: The dimensionality of the vectors. Automatically initialized based on the
           shape of the `query` or `base` dataset.
        num_base_vecs: The number of vectors in the base dataset. Automatically initialized.
        num_query_vecs: The number of vectors in the query dataset. Automatically initialized.
    """

    base: npt.NDArray = None
    ground_truth: npt.NDArray = None
    query: npt.NDArray = None
    datatype: DataType = DataType.FLOAT_VECTOR
    d: int = field(init=False)
    num_base_vecs: int = field(init=False)
    num_query_vecs: int = field(init=False)

    def __post_init__(self):
        self.d = (
            self.query.shape[1]
            if self.query is not None
            else self.base.shape[1]
            if self.base is not None
            else None
        )
        self.num_base_vecs = self.base.shape[0] if self.base is not None else None
        self.num_query_vecs = self.query.shape[0] if self.query is not None else None

    def __str__(self) -> str:
        str_datatype = "FLOAT16" if self.datatype == DataType.FLOAT16_VECTOR else "FLOAT32"
        return f'{self.__class__.__name__}(name="sift", num_base_vecs={self.num_base_vecs if self.num_base_vecs is not None else 0:,.0f}, num_query_vecs={self.num_query_vecs if self.num_query_vecs is not None else 0:,.0f}, d={self.d}, datatype={str_datatype})'


def load_sift_1b(directory: PathLike, n: int, base: bool) -> Dataset:
    """
    Loads a subset of the SIFT1B dataset.

    Args:
        directory: The directory containing the SIFT1B dataset files.
        n: The number of millions of vectors to load.
        base: A flag indicating whether to load the base vectors or the ground truth and query vectors.

    Returns:
        A Dataset object containing the requested subset of the SIFT1B dataset.
        If `base` is True, only the base vectors are loaded. If `base` is False,
        the ground truth and query vectors are loaded.
    """

    assert n in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    directory = Path(directory)
    return Dataset(
        base=vecs_read(directory / "bigann_base.bvecs", n * 1_000_000).astype(np.float32) if base else None,
        ground_truth=vecs_read(directory / "gnd" / f"idx_{n}M.ivecs") if not base else None,
        query=vecs_read(directory / "bigann_query.bvecs").astype(np.float32) if not base else None,
        # datatype=DataType.FLOAT16_VECTOR,
    )


def load_sift_small(directory: PathLike, base: bool) -> Dataset:
    """
    Loads the SIFT small dataset.

    Args:
        directory: The directory containing the SIFT small dataset files.
        base: A flag indicating whether to load the base vectors or the ground truth and query vectors.

    Returns:
        A Dataset object containing the SIFT small dataset.
        If `base` is True, only the base vectors are loaded. If `base` is False,
        the ground truth and query vectors are loaded.
    """

    directory = Path(directory)
    return Dataset(
        base=vecs_read(directory / "siftsmall_base.fvecs") if base else None,
        ground_truth=vecs_read(directory / "siftsmall_groundtruth.ivecs") if not base else None,
        query=vecs_read(directory / "siftsmall_query.fvecs") if not base else None,
    )


def load_sift_1m(directory: PathLike, base: bool) -> Dataset:
    """
    Loads the SIFT 1M dataset.

    Args:
        directory: The directory containing the SIFT 1M dataset files.
        base: A flag indicating whether to load the base vectors or the ground truth and query vectors.

    Returns:
        A Dataset object containing the SIFT 1M dataset.
        If `base` is True, only the base vectors are loaded. If `base` is False,
        the ground truth and query vectors are loaded.
    """

    directory = Path(directory)
    return Dataset(
        base=vecs_read(directory / "sift_base.fvecs") if base else None,
        ground_truth=vecs_read(directory / "sift_groundtruth.ivecs") if not base else None,
        query=vecs_read(directory / "sift_query.fvecs") if not base else None,
    )


if __name__ == "__main__":
    dataset = load_sift_1b("data/datasets/sift1b", 10)
    print(dataset.base.shape)
    print(dataset)
