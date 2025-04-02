from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pymilvus import DataType

from fvecs import vecs_read


@dataclass
class Dataset:
    base: npt.NDArray = None
    ground_truth: npt.NDArray = None
    query: npt.NDArray = None
    datatype: DataType = DataType.FLOAT_VECTOR
    d: int = field(init=False)
    num_base_vecs: int = field(init=False)
    num_query_vecs: int = field(init=False)
    
    def __post_init__(self):
        self.d = self.query.shape[1] if self.query is not None else self.base.shape[1] if self.base is not None else None
        self.num_base_vecs = self.base.shape[0] if self.base is not None else None
        self.num_query_vecs = self.query.shape[0] if self.query is not None else None
    
    def __str__(self) -> str:
        str_datatype = "FLOAT16" if self.datatype == DataType.FLOAT16_VECTOR else "FLOAT32"
        return f"{self.__class__.__name__}(name=\"sift\", num_base_vecs={self.num_base_vecs if self.num_base_vecs is not None else 0:,.0f}, num_query_vecs={self.num_query_vecs if self.num_query_vecs is not None else 0:,.0f}, d={self.d}, datatype={str_datatype})"


def load_sift_1b(directory: PathLike, n: int, base: bool) -> Dataset:
    assert n in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    directory = Path(directory)
    return Dataset(
        base=vecs_read(directory / "bigann_base.bvecs", n * 1_000_000).astype(np.float32) if base else None,
        ground_truth=vecs_read(directory / "gnd" / f"idx_{n}M.ivecs") if not base else None,
        query=vecs_read(directory / "bigann_query.bvecs").astype(np.float32) if not base else None,
        # datatype=DataType.FLOAT16_VECTOR,
    )


def load_sift_small(directory: PathLike, base: bool) -> Dataset:
    directory = Path(directory)
    return Dataset(
        base=vecs_read(directory / "siftsmall_base.fvecs") if base else None,
        ground_truth=vecs_read(directory / "siftsmall_groundtruth.ivecs") if not base else None,
        query=vecs_read(directory / "siftsmall_query.fvecs") if not base else None,
    )


def load_sift_1m(directory: PathLike, base: bool) -> Dataset:
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
