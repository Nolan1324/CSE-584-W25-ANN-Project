from collections import defaultdict
from typing import Any, Dict, List
import numpy as np
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from tqdm import tqdm
from client import get_client
from multirange_partitioner import MultiRangePartitioner
from predicates import Atomic, And, Or, Not, Operator, Predicate
from sift import Dataset, load_sift_small
from workload_char import counter_characterize_workload
from tree_partition_algo import build_tree, TreeAlgoParams
import numpy.typing as npt

GTE = Operator.GTE


def create_schema(client: MilvusClient, partitioner: MultiRangePartitioner):
    if client.has_collection("data"):
        client.drop_collection("data")

    id_schema = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
    )

    x_schema = FieldSchema(
        name="x",
        dtype=DataType.INT32,
    )

    y_schema = FieldSchema(
        name="y",
        dtype=DataType.INT32,
    )

    vector_schema = FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=128,
    )

    schema = CollectionSchema(
        fields=[id_schema, vector_schema, x_schema, y_schema],
        description="Example",
        enable_dynamic_field=False,
    )

    client.create_collection(
        collection_name="data",
        schema=schema,
    )

    partitioner.add_partitions_to_collection(client, "data")


def insert_data(
    client: MilvusClient,
    data_list: List[Dict[str, Any]],
    partitioner: MultiRangePartitioner,
):
    # Partition the data so that we can batch insert it
    partitions = defaultdict(list)
    for data in data_list:
        partitions[partitioner.get_partition(data)].append(data)
    # Insert each partition as a batch
    for partition_name, partition_data_list in partitions.items():
        client.insert(
            collection_name="data",
            data=partition_data_list,
            partition_name=partition_name,
        )


def create_partitioner(attribute_data_array: npt.NDArray):
    workload = [
        And(Atomic("x", GTE, 100), Not(Atomic("x", GTE, 500))),
        And(Atomic("x", GTE, 500)),
        And(Atomic("x", GTE, 500), Not(Atomic("y", GTE, 300))),
    ]
    atomics = counter_characterize_workload(workload)

    params = TreeAlgoParams(
        min_predicate_frequency=1,
        max_num_partitions=4,
    )
    tree = build_tree(attribute_data_array, ["x", "y"], atomics, params)
    partitioner = MultiRangePartitioner(tree)

    print(f"## Created partition tree:\n{tree}")

    return partitioner


def create_index(client: MilvusClient):
    client.flush("data")

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type="L2",
        index_type="FLAT",
        index_name="vector_index",
    )
    client.create_index(collection_name="data", index_params=index_params, sync=True)

    client.load_collection("data")

    client.flush("data")


def search(
    client: MilvusClient,
    partitioner: MultiRangePartitioner,
    vector,
    predicate: Predicate,
):
    partitions = list(partitioner.get_query_partitions(predicate))
    print(f"## Searching partitions {partitions}")
    result = client.search(
        collection_name="data",
        data=[vector],
        limit=100,
        output_fields=["id"],
        filter=predicate.to_filter_string(),
        partition_names=partitions,
    )
    return [data["id"] for data in result[0]]


def main():
    dataset_path = "../data/datasets/siftsmall"

    base_dataset = load_sift_small(dataset_path, base=True)
    rng = np.random.default_rng(584)
    attribute_data_array = rng.uniform(0, 1000, (base_dataset.num_base_vecs, 2)).astype(np.int32)

    partitioner = create_partitioner(attribute_data_array)

    data_list = [
        {"id": i, "vector": list(vector), "x": x, "y": y}
        for i, (vector, [x, y]) in enumerate(zip(base_dataset.base, attribute_data_array))
    ]

    client = get_client()
    create_schema(client, partitioner)
    insert_data(client, data_list, partitioner)
    create_index(client)

    query_dataset = load_sift_small(dataset_path, base=False)
    query_vector_index = 0
    query_vector = query_dataset.query[query_vector_index]
    predicate = Or(
        And(Atomic("y", GTE, 200), Not(Atomic("y", GTE, 250))),
        Not(Atomic("x", GTE, 400)),
    )
    result = search(client, partitioner, query_vector, predicate)
    print(f"Search result: {result}")

    ground_truth = [
        i for i in query_dataset.ground_truth[query_vector_index] if predicate.evaluate(data_list[i])
    ]
    print(f"Filtered ground truth: {ground_truth}")


if __name__ == "__main__":
    main()
