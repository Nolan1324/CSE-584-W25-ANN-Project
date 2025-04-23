# CSE 584 WN25 ANN Project

## Usage

### Experiments

### Example

`example.py` provides a simple example of using the partitioning system. Run it with `python example.py` while running the Milvus docker container. It creates a collection of vectors with two attributes, `x` and `y`. It then characterizes a simple workload of three filters, builds a partition tree with 4 partitions, and then conducts a basic filtered vector search which is able to skip one of the partitions.

## Implementation

Below is a summary of the functions that each file implements. See the docstrings within each file for more information on the implementation.

- `workload_char.py` - This implements the following workload characterization function.
    - `counter_characterize_workload(workload: List[Predicate]) -> List[Tuple[Predicate, int]]` - Characterizes a workload by counting the frequencies of atomic predicates.
- `tree_partition_algo.py` - This implements the main greedy tree-based partitioning algorithm. The primary function is `build_tree`, which takes a dataset and workload characterization and builds the partition tree. It also implements `get_partitions_from_tree`, which takes the partition tree as input and summarizes each partition as a set of ranges of the attributes.
- `multirange_partitioner.py` - This implements a partitioner class that uses the partition tree generated from `build_tree`. It implements the following functions:
    - `get_partition(self, vals: Dict[str, int]) -> str` - Locates the partition a vector is contained in, given its scalar attributes.
    - `get_query_partitions(self, predicate: Predicate) -> Generator[str, None, None]` - Locates all partitions that either always or sometimes satisfy the filter predicate.
    - `add_partitions_to_collection(self, client: MilvusClient, collection_name: str)` - Adds all of the partitions to a Milvus collection.
    - `partition_names` - Property that gets the names of all the partitions.
- `tvl.py` - This implements three-value logic. It exports a `Maybe` singleton, similar to `True` and `False`. However, expression using `Maybe` must use `&`, `|`, `tvl_not` in place of `and`, `or`, `not`, respectively. Using the wrong operator with `Maybe` will raise an exception.
- `predicates.py` - This implements the data structures for defining predicate expression trees: for instance, `Atomic`, `And`, `Or`, and `Not`. Each of these predicates is a subclass of the `Predicate` abstract base class. `Predicate` defines the following interface:
    - `evaluate(self, vals: Dict[str, int]) -> bool` - determine if a data point satisfies the predicate or not
    - `range_may_satisfy(self, ranges: Dict[str, Range]) -> TVL` - determines if a range of data **may** satisfy the predicate. Returns a three-value logic truth value (`True`, `Maybe`, or `False`)
    - `atomics(self) -> Generator[Atomic, None, None]` - breaks the predicate down into atomic predicates
    - `to_filter_string(self) -> str` - converts the predicate to a string that can be passed as a filter to Milvus
