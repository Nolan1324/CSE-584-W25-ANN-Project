import numpy as np
import numpy.typing as npt
from collections import Counter

from attributes import uniform_attributes, uniform_attributes_basic, geometric_attributes_basic, normal_attributes_basic
from partitioner import RangePartitioner


def create_partitioner_uniform_intervals(data: npt.NDArray, num_partitions: int):
    start = data.min()
    end = data.max()
    interval = end - start + 1
    step = interval // num_partitions
    return RangePartitioner((i, i+step-1) for i in range(start, end+1, step))

def create_partitioner_uniform_sizes(data: npt.NDArray, desired_num_partitions: int):
    start = data.min()
    end = data.max()

    size = data.shape[0] // desired_num_partitions

    partitions = []
    cur_start = start
    cur_size = 0
    
    for attr, freq in sorted(Counter(data).items()):
        cur_size += freq
        if cur_size >= size:
            partitions.append((cur_start, attr))
            cur_start = attr + 1
            cur_size = 0
    if len(partitions) < desired_num_partitions:
        partitions.append((cur_start, end))

    return RangePartitioner(partitions)


if __name__ == '__main__':
    # attrs = uniform_attributes(5000, 584, np.int32, 0, 1000)
    # attrs = geometric_attributes_basic(10000, 0.005)
    attrs = normal_attributes_basic(10000, 500, 150)
    print(create_partitioner_uniform_intervals(attrs, 10))
    print(create_partitioner_uniform_sizes(attrs, 10))