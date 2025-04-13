import time
from typing import List, Optional, Tuple
from logging import Logger

import numpy as np
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from pymilvus import utility
from tqdm import tqdm

from multirange_partitioner import MultiRangePartitioner
from predicates import Range
from sift import Dataset, load_sift_1m
from client import get_client
from partitioner import Partitioner, RangePartitioner
from attributes import uniform_attributes, uniform_attributes_basic
from utils import Timer

import logging


class Creator:
    def __init__(
        self,
        partitioner: Optional[MultiRangePartitioner] = None,
        attributes: List[str] = [],
        datatype: DataType = DataType.FLOAT_VECTOR,
        logger: Logger = None,
    ):
        self.client = get_client()
        self.partitioner = partitioner
        self.datatype = datatype
        self.attributes = attributes
        self.logger: Logger = (
            logger.getChild("Creator") if logger else logging.getLogger("null")
        )

    def create_collection_schema(self, name: str = "data"):
        if self.client.has_collection(name):
            self.client.drop_collection(name)

        id_schema = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
        )

        attrib_schemas = []
        for attr_name in self.attributes:
            attrib_schemas.append(
                FieldSchema(
                    name=attr_name,
                    dtype=DataType.INT32,
                )
            )

        vector_schema = FieldSchema(
            name="vector",
            dtype=self.datatype,
            dim=128,
        )

        schema = CollectionSchema(
            fields=[id_schema, vector_schema, *attrib_schemas],
            description="Test attributes on SIFT",
            enable_dynamic_field=False,
        )

        self.client.create_collection(
            collection_name=name,
            schema=schema,
        )

        if self.partitioner is not None:
            self.partitioner.add_partitions_to_collection(self.client, name)

    def populate_collection(
        self,
        dataset: Dataset,
        attributes_data: "np.ndarray[np.int32]",
        index_type: str = "HNSW",
        flush: bool = True,
        name: str = "data",
    ):
        assert len(attributes_data.shape) == 2
        assert dataset.num_base_vecs == attributes_data.shape[0]

        def _get_attributes_dict(row_index):
            return {
                name: value
                for name, value in zip(self.attributes, attributes_data[row_index])
            }

        if self.partitioner is None:
            CHUNK_SIZE = 10000

            for i in tqdm(range(0, dataset.num_base_vecs, CHUNK_SIZE)):
                print(f"{i}/{dataset.num_base_vecs}")

                data = [
                    {
                        "id": i + offset,
                        "vector": list(vector),
                    }
                    | _get_attributes_dict(i + offset)
                    for offset, vector in enumerate(
                        iter(dataset.base[i : i + CHUNK_SIZE])
                    )
                ]
                res = self.client.insert(collection_name=name, data=data)
        else:
            CHUNK_SIZE = 10_000

            def _insert_batch(partition_name: str, batch: List[int]):
                data = [
                    {
                        "id": index,
                        "vector": dataset.base[index],
                    }
                    | _get_attributes_dict(index)
                    for index in batch
                ]
                res = self.client.insert(
                    collection_name=name, data=data, partition_name=partition_name
                )

            partition_batches = {
                partition_name: []
                for partition_name in self.partitioner.partition_names
            }
            for i in tqdm(range(0, dataset.num_base_vecs)):
                partition_name = self.partitioner.get_partition(_get_attributes_dict(i))
                batch = partition_batches[partition_name]
                batch.append(i)

                if len(batch) >= CHUNK_SIZE:
                    _insert_batch(partition_name, batch)
                    batch.clear()
            for partition_name, batch in partition_batches.items():
                if batch:
                    _insert_batch(partition_name, batch)
                    batch.clear()

        with Timer(self.logger, "flush"):
            self.client.flush(name)

        with Timer(self.logger, "indexing"):
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="L2",
                index_type=index_type,
                index_name="vector_index",
            )
            self.client.create_index(
                collection_name=name, index_params=index_params, sync=True
            )

        with Timer(self.logger, "loading"):
            self.client.load_collection(name)

        if flush:
            with Timer(self.logger, "flushing"):
                self.client.flush(name)

        if self.partitioner is not None:
            for partition_name in self.partitioner.partition_names:
                self.logger.debug(
                    f"Partition states: {[self.client.get_load_state(name, partition_name) for partition_name in self.partitioner.partition_names]}"
                )
        # self.client.load_partitions(name, self.partitioner.partition_names)

    def cluster_compact(self, name):
        job = self.client.compact(name, is_clustering=True)
        while self.client.get_compaction_state(job) != "Completed":
            time.sleep(5.0)


if __name__ == "__main__":
    # creator = Creator(RangePartitioner([(0, 100), (101, 1000)]))
    # creator = Creator(attributes=["attr"])
    creator = Creator(
        partitioner=MultiRangePartitioner.from_partitions(
            {"0_100": {"x": Range(0, 100)}, "101_1000": {"x": Range(101, 1000)}}
        ),
        attributes=["x"],
    )

    dataset = load_sift_1m("../data/datasets/sift", True)
    print("Creating schema")
    creator.create_collection_schema()
    print("Populating collection")
    creator.populate_collection(
        dataset, uniform_attributes_basic(dataset.num_base_vecs)
    )
    # print('Cluster compaction')
    # creator.cluster_compact(name)
